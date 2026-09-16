/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include "slam/vpr/vpr_anyloc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

#include "common/log.h"
#include "slam/vpr/vpr_image.h"

namespace cuvslam::slam::vpr {

namespace {

constexpr uint32_t kBlobMagic = 0x41525056;  // "VPRA"
constexpr uint32_t kBlobVersion = 1;

/// ImageNet statistics the DINOv2 checkpoints were trained with, applied after dividing by 255.
constexpr std::array<float, 3> kPixelMean = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kPixelStd = {0.229f, 0.224f, 0.225f};

float Dot(const float* a, const float* b, int size) {
  float sum = 0.f;
  for (int i = 0; i < size; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

/// Scale `values` to unit L2 norm in place. A zero vector is left as it is.
void Normalize(float* values, int size) {
  const float norm = std::sqrt(Dot(values, values, size));
  if (!(norm > 1e-12f)) {
    return;
  }
  const float inv = 1.f / norm;
  for (int i = 0; i < size; ++i) {
    values[i] *= inv;
  }
}

/// Index of the center with the largest cosine similarity to `descriptor`.
/// Both sides are unit norm, so the cosine is a plain dot product.
int NearestCenter(const float* descriptor, const std::vector<float>& centers, int dim) {
  int best = 0;
  float best_similarity = -2.f;
  const int count = static_cast<int>(centers.size()) / dim;
  for (int c = 0; c < count; ++c) {
    const float similarity = Dot(descriptor, centers.data() + static_cast<size_t>(c) * dim, dim);
    if (similarity > best_similarity) {
      best_similarity = similarity;
      best = c;
    }
  }
  return best;
}

}  // namespace

struct VprAnyLoc::Network {
  explicit Network(const std::string& path)
      : env(ORT_LOGGING_LEVEL_WARNING, "cuvslam_vpr_anyloc"),
        // A default constructed SessionOptions leaves the intra-op thread count at the ONNX Runtime
        // default, which is one thread per core. Nothing here runs concurrently with the rest of
        // SLAM anyway: the caller holds the SLAM lock across the whole query.
        session(env, path.c_str(), Ort::SessionOptions{}),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::AllocatorWithDefaultOptions allocator;
    input_name = session.GetInputNameAllocated(0, allocator).get();
    output_name = session.GetOutputNameAllocated(0, allocator).get();

    const std::vector<int64_t> shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0) {
      // DINOv2 bakes the interpolated position encoding into constants at trace time, so a model
      // exported with a dynamic input size builds but fails at run time. Every model file is
      // therefore tied to one resolution, and that resolution is read from the file rather than
      // assumed, so swapping in a 224 model costs nothing but the path.
      throw std::runtime_error("VPR AnyLoc: " + path +
                               " does not declare a fixed [1, 3, H, W] input; DINOv2 must be exported per "
                               "resolution, see cuvslam_export_dinov2.");
    }
    input_height = static_cast<int>(shape[2]);
    input_width = static_cast<int>(shape[3]);
  }

  Ort::Env env;
  Ort::Session session;
  Ort::MemoryInfo memory_info;
  std::string input_name;
  std::string output_name;
  int input_width = 0;
  int input_height = 0;
};

VprAnyLoc::VprAnyLoc(const VprOptions& options)
    : model_path_(options.model_path),
      cluster_count_(options.vocabulary_size > 0 ? options.vocabulary_size : kDefaultVocabularySize),
      score_threshold_(options.score_threshold > 0.f ? options.score_threshold : kDefaultScoreThreshold),
      ratio_threshold_(options.ratio_threshold),
      max_entries_(options.max_entries) {
  // Reading the model is deferred to the first frame, because it costs 69 MB that a session which
  // never queries should not pay. Its *existence* is not deferred: this constructor runs on the
  // caller's thread inside Slam's, where a throw is a catchable error, while the first frame is
  // mapped on the SLAM worker, where one would be std::terminate.
  std::error_code ec;
  if (model_path_.empty() || !std::filesystem::is_regular_file(model_path_, ec)) {
    throw std::runtime_error(
        "VPR backend AnyLoc needs a DINOv2 ONNX model: set Slam::Config::vpr_model_path to one exported by "
        "cuvslam_export_dinov2. Tried \"" +
        model_path_ + "\".");
  }
}

VprAnyLoc::~VprAnyLoc() = default;

void VprAnyLoc::EnsureNetwork() {
  if (network_) {
    return;
  }
  std::error_code ec;
  if (model_path_.empty() || !std::filesystem::is_regular_file(model_path_, ec)) {
    throw std::runtime_error(
        "VPR backend AnyLoc needs a DINOv2 ONNX model: set Slam::Config::vpr_model_path to one exported by "
        "cuvslam_export_dinov2. Tried \"" +
        model_path_ + "\".");
  }
  network_ = std::make_unique<Network>(model_path_);
  TraceMessage("VPR AnyLoc: loaded %s, input %dx%d\n", model_path_.c_str(), network_->input_width,
               network_->input_height);
}

bool VprAnyLoc::Describe(const VprImage& image, std::vector<float>& patches) {
  patches.clear();
  if (image.Empty()) {
    return false;
  }
  EnsureNetwork();

  const VprImage resized = Resize(image, network_->input_width, network_->input_height);
  if (resized.Empty()) {
    return false;
  }

  // DINOv2 was trained on color and takes three channels, but everything this backend is handed is
  // single channel: cuvslam2.cpp converts color input to grayscale before it reaches VPR, and the
  // tracking pipeline below it is grayscale throughout. Replicating the gray channel into R, G and
  // B is a deliberate simplification rather than a color path that does not exist. It costs some
  // absolute descriptor quality, but not retrieval quality, because the mapped frames and the query
  // go through exactly the same transform and are therefore compared on equal terms.
  const size_t plane = resized.row.size();
  std::vector<float> input(3 * plane);
  for (int c = 0; c < 3; ++c) {
    const float scale = 1.f / (255.f * kPixelStd[c]);
    const float bias = -kPixelMean[c] / kPixelStd[c];
    float* dst = input.data() + static_cast<size_t>(c) * plane;
    for (size_t i = 0; i < plane; ++i) {
      dst[i] = static_cast<float>(resized.row[i]) * scale + bias;
    }
  }

  const std::array<int64_t, 4> shape = {1, 3, resized.height, resized.width};
  Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(network_->memory_info, input.data(), input.size(), shape.data(), shape.size());
  const char* input_names[] = {network_->input_name.c_str()};
  const char* output_names[] = {network_->output_name.c_str()};
  const std::vector<Ort::Value> outputs =
      network_->session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
  if (outputs.size() != 1 || !outputs[0].IsTensor()) {
    return false;
  }

  const std::vector<int64_t> out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  if (out_shape.size() != 3 || out_shape[1] <= 0 || out_shape[2] <= 0) {
    return false;
  }
  const int count = static_cast<int>(out_shape[1]);
  const int dim = static_cast<int>(out_shape[2]);
  if (descriptor_dim_ == 0) {
    descriptor_dim_ = dim;
  } else if (dim != descriptor_dim_) {
    TraceError("VPR AnyLoc: model emits %d dimensional patches but the map holds %d dimensional ones\n", dim,
               descriptor_dim_);
    return false;
  }

  const float* data = outputs[0].GetTensorData<float>();
  patches.assign(data, data + static_cast<size_t>(count) * dim);
  // The exporter already L2 normalizes each patch, but the VLAD residual is only the AnyLoc
  // residual if the descriptors are unit norm, so do not depend on a property of the model file.
  for (int i = 0; i < count; ++i) {
    Normalize(patches.data() + static_cast<size_t>(i) * dim, dim);
  }
  return true;
}

void VprAnyLoc::AddFrame(KeyFrameId node_id, const VprImage& image) {
  std::vector<float> patches;
  if (!Describe(image, patches)) {
    return;
  }

  const auto it = index_by_node_.find(node_id);
  if (it != index_by_node_.end()) {
    entries_[it->second].patches = std::move(patches);
    entries_[it->second].vlad.clear();
    dirty_ = true;
    return;
  }

  if (max_entries_ != 0 && entries_.size() >= max_entries_) {
    return;
  }

  index_by_node_.emplace(node_id, entries_.size());
  Entry entry;
  entry.node_id = node_id;
  entry.patches = std::move(patches);
  entries_.push_back(std::move(entry));
  dirty_ = true;
}

bool VprAnyLoc::HasNode(KeyFrameId node_id) const { return index_by_node_.count(node_id) != 0; }

void VprAnyLoc::RemoveNode(KeyFrameId node_id) {
  const auto it = index_by_node_.find(node_id);
  if (it == index_by_node_.end()) {
    return;
  }

  const size_t removed = it->second;
  const size_t last = entries_.size() - 1;
  if (removed != last) {
    entries_[removed] = std::move(entries_[last]);
    index_by_node_[entries_[removed].node_id] = removed;
  }
  entries_.pop_back();
  index_by_node_.erase(node_id);
}

void VprAnyLoc::ShiftNodeIds(KeyFrameId offset) {
  index_by_node_.clear();
  index_by_node_.reserve(entries_.size());
  for (size_t i = 0; i < entries_.size(); ++i) {
    entries_[i].node_id += offset;
    index_by_node_.emplace(entries_[i].node_id, i);
  }
}

void VprAnyLoc::TrainVocabulary() {
  if (descriptor_dim_ <= 0) {
    return;
  }

  size_t available = 0;
  for (const auto& entry : entries_) {
    available += entry.patches.size() / static_cast<size_t>(descriptor_dim_);
  }
  if (available == 0) {
    return;
  }

  // Fixed stride over the concatenation of every mapped frame's patches, not a random sample: two
  // runs over the same sequence have to produce the same vocabulary, and a strided walk also spreads
  // the training set evenly over the trajectory instead of over-weighting whichever frames an RNG
  // happened to favor.
  const size_t stride = std::max<size_t>(1, (available + kMaxTrainingDescriptors - 1) / kMaxTrainingDescriptors);
  std::vector<float> training;
  training.reserve(std::min(available, kMaxTrainingDescriptors) * static_cast<size_t>(descriptor_dim_));
  size_t seen = 0;
  for (const auto& entry : entries_) {
    const size_t count = entry.patches.size() / static_cast<size_t>(descriptor_dim_);
    for (size_t i = 0; i < count; ++i, ++seen) {
      if (seen % stride != 0) {
        continue;
      }
      const float* row = entry.patches.data() + i * static_cast<size_t>(descriptor_dim_);
      training.insert(training.end(), row, row + descriptor_dim_);
    }
  }

  const int dim = descriptor_dim_;
  const int points = static_cast<int>(training.size() / static_cast<size_t>(dim));
  const int clusters = std::min(cluster_count_, points);
  if (clusters <= 0) {
    return;
  }

  // Farthest point initialization: after the first center, repeatedly take the training descriptor
  // that is least similar to everything chosen so far. Deterministic, ties broken by the lower
  // index, and it spreads the centers over the whole descriptor cloud, which is what k-means++ buys
  // with randomness and what plain "take the first k" does not.
  std::vector<float> centers(static_cast<size_t>(clusters) * dim);
  std::copy(training.begin(), training.begin() + dim, centers.begin());
  std::vector<float> best_similarity(points, -2.f);
  for (int c = 1; c < clusters; ++c) {
    const float* previous = centers.data() + static_cast<size_t>(c - 1) * dim;
    int farthest = 0;
    float farthest_similarity = 2.f;
    for (int i = 0; i < points; ++i) {
      best_similarity[i] =
          std::max(best_similarity[i], Dot(training.data() + static_cast<size_t>(i) * dim, previous, dim));
      if (best_similarity[i] < farthest_similarity) {
        farthest_similarity = best_similarity[i];
        farthest = i;
      }
    }
    const float* row = training.data() + static_cast<size_t>(farthest) * dim;
    std::copy(row, row + dim, centers.begin() + static_cast<size_t>(c) * dim);
  }

  std::vector<int> labels(points, -1);
  std::vector<float> sums(centers.size());
  int iteration = 0;
  for (; iteration < kKMeansIterations; ++iteration) {
    bool changed = false;
    std::fill(sums.begin(), sums.end(), 0.f);
    for (int i = 0; i < points; ++i) {
      const float* row = training.data() + static_cast<size_t>(i) * dim;
      const int label = NearestCenter(row, centers, dim);
      if (label != labels[i]) {
        labels[i] = label;
        changed = true;
      }
      float* sum = sums.data() + static_cast<size_t>(label) * dim;
      for (int d = 0; d < dim; ++d) {
        sum[d] += row[d];
      }
    }
    for (int c = 0; c < clusters; ++c) {
      float* sum = sums.data() + static_cast<size_t>(c) * dim;
      // An empty cluster keeps its previous center: moving it somewhere arbitrary would only make
      // the vocabulary depend on the iteration it went empty in.
      if (Dot(sum, sum, dim) > 0.f) {
        Normalize(sum, dim);
        std::copy(sum, sum + dim, centers.begin() + static_cast<size_t>(c) * dim);
      }
    }
    if (!changed) {
      break;
    }
  }

  cluster_count_ = clusters;
  centers_ = std::move(centers);
  TraceMessage("VPR AnyLoc: trained %d clusters on %d of %zu patch descriptors in %d iterations\n", clusters, points,
               available, iteration + 1);
}

void VprAnyLoc::Encode(const std::vector<float>& patches, std::vector<float>& vlad) const {
  vlad.assign(static_cast<size_t>(cluster_count_) * descriptor_dim_, 0.f);
  const int dim = descriptor_dim_;
  const size_t count = patches.size() / static_cast<size_t>(dim);
  for (size_t i = 0; i < count; ++i) {
    const float* row = patches.data() + i * static_cast<size_t>(dim);
    const int label = NearestCenter(row, centers_, dim);
    const float* center = centers_.data() + static_cast<size_t>(label) * dim;
    float* block = vlad.data() + static_cast<size_t>(label) * dim;
    for (int d = 0; d < dim; ++d) {
      block[d] += row[d] - center[d];
    }
  }

  // Intra-normalization first, then a final normalization over the whole vector, both as AnyLoc
  // does them. The intra step is what keeps a repeated texture from deciding the score: a patch that
  // occurs a hundred times in one frame contributes a hundred near-identical residuals to one
  // cluster, and without the per-cluster normalization that block simply outweighs everything the
  // two images actually differ in. The final step makes a cosine a plain dot product.
  for (int c = 0; c < cluster_count_; ++c) {
    Normalize(vlad.data() + static_cast<size_t>(c) * dim, dim);
  }
  Normalize(vlad.data(), static_cast<int>(vlad.size()));
}

void VprAnyLoc::Finalize() {
  if (!dirty_) {
    return;
  }
  dirty_ = false;

  if (centers_.empty() && entries_.size() < kMinTrainingFrames) {
    // The vocabulary is fitted once and never re-fitted, so fitting it too early would freeze the
    // cluster centers on whatever the first frame or two happened to contain and encode the whole
    // session against them. A caller that queries on every frame reaches here long before the map
    // is representative; answering "no match" until there is something to fit is the honest outcome.
    dirty_ = true;
    return;
  }
  if (centers_.empty()) {
    TrainVocabulary();
  }
  if (centers_.empty()) {
    // Not enough material yet. The patches stay pending and the next Finalize() tries again.
    dirty_ = true;
    return;
  }

  // Frames added after the vocabulary exists are encoded against it rather than triggering a
  // retrain: re-fitting the centers would change the meaning of every descriptor already stored and
  // force a full re-encode of the map on every keyframe.
  for (auto& entry : entries_) {
    if (entry.patches.empty()) {
      continue;
    }
    Encode(entry.patches, entry.vlad);
    entry.patches.clear();
    entry.patches.shrink_to_fit();
  }
}

VprMatch VprAnyLoc::Query(const VprImage& image) {
  const std::vector<VprMatch> matches = QueryTopK(image, 1);
  return matches.empty() ? VprMatch{} : matches.front();
}

std::vector<VprMatch> VprAnyLoc::QueryTopK(const VprImage& image, size_t max_results) {
  std::vector<VprMatch> matches;
  Finalize();
  if (entries_.empty() || centers_.empty() || max_results == 0) {
    return matches;
  }

  std::vector<float> patches;
  if (!Describe(image, patches)) {
    return matches;
  }
  std::vector<float> query;
  Encode(patches, query);

  std::vector<std::pair<float, size_t>> scored;
  scored.reserve(entries_.size());
  for (size_t i = 0; i < entries_.size(); ++i) {
    const auto& entry = entries_[i];
    if (entry.vlad.size() != query.size()) {
      continue;
    }
    scored.emplace_back(Dot(query.data(), entry.vlad.data(), static_cast<int>(query.size())), i);
  }
  if (scored.empty()) {
    return matches;
  }

  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  if (scored.front().first < score_threshold_) {
    return matches;
  }

  const KeyFrameId best_node = entries_[scored.front().second].node_id;
  if (ratio_threshold_ > 0.f) {
    // Second best from a different part of the trajectory: an ambiguous place is dropped, see README.md.
    float rival = -1.f;
    for (const auto& [score, index] : scored) {
      const KeyFrameId node = entries_[index].node_id;
      const uint64_t distance = best_node > node ? best_node - node : node - best_node;
      if (distance > kNeighbourGuard) {
        rival = std::max(rival, score);
      }
    }
    if (rival > 0.f && scored.front().first < rival * ratio_threshold_) {
      return matches;
    }
  }

  // Only the winner is held to the threshold; the runners-up are there for a relocalizer to verify.
  const size_t count = std::min(max_results, scored.size());
  matches.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    VprMatch match;
    match.found = true;
    match.node_id = entries_[scored[i].second].node_id;
    match.score = std::min(1.f, std::max(0.f, scored[i].first));
    matches.push_back(match);
  }
  return matches;
}

void VprAnyLoc::Serialize(Blob& blob) {
  Finalize();

  BlobWriter writer(blob);
  writer.write(kBlobMagic);
  writer.write(kBlobVersion);
  writer.write(static_cast<int32_t>(cluster_count_));
  writer.write(static_cast<int32_t>(descriptor_dim_));
  writer.write(static_cast<uint64_t>(centers_.size()));
  if (!centers_.empty()) {
    writer.write(centers_.data(), centers_.size() * sizeof(float));
  }
  writer.write(static_cast<uint64_t>(entries_.size()));
  for (const auto& entry : entries_) {
    writer.write(static_cast<uint64_t>(entry.node_id));
    writer.write(static_cast<uint64_t>(entry.vlad.size()));
    if (!entry.vlad.empty()) {
      writer.write(entry.vlad.data(), entry.vlad.size() * sizeof(float));
    }
  }
}

bool VprAnyLoc::Deserialize(const BlobReader& reader) {
  uint32_t magic = 0;
  uint32_t version = 0;
  if (!reader.read(magic) || !reader.read(version) || magic != kBlobMagic || version != kBlobVersion) {
    return false;
  }

  int32_t cluster_count = 0;
  int32_t descriptor_dim = 0;
  uint64_t center_count = 0;
  if (!reader.read(cluster_count) || !reader.read(descriptor_dim) || !reader.read(center_count)) {
    return false;
  }
  if (cluster_count < 0 || descriptor_dim < 0 ||
      center_count != static_cast<uint64_t>(cluster_count) * static_cast<uint64_t>(descriptor_dim)) {
    return false;
  }

  if (center_count > reader.remaining() / sizeof(float)) {
    return false;
  }
  std::vector<float> centers(static_cast<size_t>(center_count));
  if (center_count != 0 && !reader.read(centers.data(), centers.size() * sizeof(float))) {
    return false;
  }

  uint64_t count = 0;
  if (!reader.read(count)) {
    return false;
  }

  // An entry is at least a node id and a size, so a count larger than the bytes left cannot be honest.
  constexpr size_t kMinEntryBytes = sizeof(uint64_t) + sizeof(uint64_t);
  if (count > reader.remaining() / kMinEntryBytes) {
    return false;
  }

  std::vector<Entry> entries;
  std::unordered_map<KeyFrameId, size_t> index_by_node;
  entries.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t node_id = 0;
    uint64_t vlad_size = 0;
    if (!reader.read(node_id) || !reader.read(vlad_size)) {
      return false;
    }
    if (vlad_size > reader.remaining() / sizeof(float)) {
      return false;
    }
    Entry entry;
    entry.node_id = static_cast<KeyFrameId>(node_id);
    entry.vlad.resize(static_cast<size_t>(vlad_size));
    if (vlad_size != 0 && !reader.read(entry.vlad.data(), entry.vlad.size() * sizeof(float))) {
      return false;
    }
    index_by_node.emplace(entry.node_id, entries.size());
    entries.push_back(std::move(entry));
  }

  cluster_count_ = cluster_count;
  descriptor_dim_ = descriptor_dim;
  centers_ = std::move(centers);
  entries_ = std::move(entries);
  index_by_node_ = std::move(index_by_node);
  // The vocabulary and every descriptor came out of the blob, so the map is queryable as it stands.
  dirty_ = false;
  return true;
}

}  // namespace cuvslam::slam::vpr
