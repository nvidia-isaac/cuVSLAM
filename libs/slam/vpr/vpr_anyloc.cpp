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

#include "common/include_eigen.h"
#include "common/log.h"
#include "slam/vpr/anyloc_network.h"
#include "slam/vpr/vpr_image.h"

namespace cuvslam::slam::vpr {

namespace {

constexpr uint32_t kBlobMagic = 0x41525056;  // "VPRA"
constexpr uint32_t kBlobVersion = 1;

/// DINOv2's ImageNet normalization, for pixels scaled to [0, 1].
constexpr std::array<float, 3> kPixelMean = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kPixelStd = {0.229f, 0.224f, 0.225f};

float Dot(const float* a, const float* b, int size) {
  return Eigen::Map<const Eigen::VectorXf>(a, size).dot(Eigen::Map<const Eigen::VectorXf>(b, size));
}

/// L2 normalize in place; a zero vector is left as is.
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

/// Most similar center; inputs are unit norm, so the dot product is the cosine.
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

VprAnyLoc::VprAnyLoc(const VprOptions& options)
    : model_path_(options.model_path),
      cluster_count_(options.vocabulary_size),
      score_threshold_(options.score_threshold),
      ratio_threshold_(options.ratio_threshold),
      max_entries_(options.max_entries) {
  // Load eagerly, so a bad engine fails Slam's constructor, not the SLAM worker.
  std::error_code ec;
  if (model_path_.empty() || !std::filesystem::is_regular_file(model_path_, ec)) {
    throw std::runtime_error(
        "VPR backend AnyLoc needs its TensorRT engine: set Slam::Config::vpr_model_path to the one the "
        "anyloc_engine CMake target builds. Tried \"" +
        model_path_ + "\".");
  }
  network_ = std::make_unique<AnyLocNetwork>(model_path_);
  descriptor_dim_ = network_->descriptor_dim();
  TraceMessage("VPR AnyLoc: loaded %s, input %dx%d, %d patches of %d\n", model_path_.c_str(), network_->input_width(),
               network_->input_height(), network_->patch_count(), descriptor_dim_);
}

VprAnyLoc::~VprAnyLoc() = default;

bool VprAnyLoc::Describe(const VprImage& image, std::vector<float>& patches) {
  patches.clear();
  if (image.Empty()) {
    return false;
  }

  const VprImage resized = Resize(image, network_->input_width(), network_->input_height());
  if (resized.Empty()) {
    return false;
  }

  // VPR input is gray: replicate it into planar R, G and B.
  const size_t plane = resized.row.size();
  float* input = network_->input();
  for (int c = 0; c < 3; ++c) {
    const float scale = 1.f / (255.f * kPixelStd[c]);
    const float bias = -kPixelMean[c] / kPixelStd[c];
    float* dst = input + static_cast<size_t>(c) * plane;
    for (size_t i = 0; i < plane; ++i) {
      dst[i] = static_cast<float>(resized.row[i]) * scale + bias;
    }
  }
  if (!network_->Run(patches)) {
    patches.clear();
    return false;
  }

  // Renormalize: VLAD needs unit norm, and FP16 output is only close to it.
  const int count = network_->patch_count();
  for (int i = 0; i < count; ++i) {
    Normalize(patches.data() + static_cast<size_t>(i) * descriptor_dim_, descriptor_dim_);
  }
  return true;
}

void VprAnyLoc::AddFrame(KeyFrameId node_id, const VprImage& image) {
  std::vector<float> patches;
  if (!Describe(image, patches)) {
    return;
  }

  auto it = index_by_node_.find(node_id);
  if (it == index_by_node_.end()) {
    if (max_entries_ != 0 && entries_.size() >= max_entries_) {
      return;
    }
    it = index_by_node_.emplace(node_id, entries_.size()).first;
    Entry entry;
    entry.node_id = node_id;
    entries_.push_back(std::move(entry));
  }
  const size_t index = it->second;
  entries_[index].vlad.clear();
  entries_[index].patches.clear();
  entries_[index].patches.shrink_to_fit();

  if (centers_.empty()) {
    size_t pending = patches.size();
    for (const auto& entry : entries_) {
      pending += entry.patches.size();
    }
    // Fit before raw patches outgrow the training set.
    if (pending > kMaxTrainingDescriptors * static_cast<size_t>(descriptor_dim_)) {
      Finalize();
    }
  }

  Entry& entry = entries_[index];
  if (!centers_.empty()) {
    Encode(patches, entry.vlad);
    return;
  }
  entry.patches = std::move(patches);
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

  // Fixed stride, not a random sample: deterministic and even over the trajectory.
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

  // Farthest point init: deterministic, and spreads centers like k-means++.
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
      // An empty cluster keeps its previous center.
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

  // Per-cluster then global L2 normalization, as AnyLoc does.
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
    dirty_ = true;
    return;
  }
  if (centers_.empty()) {
    TrainVocabulary();
  }
  if (centers_.empty()) {
    // Not trained yet; the next Finalize() retries.
    dirty_ = true;
    return;
  }

  // Encode pending frames; never refit, which would invalidate stored descriptors.
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
    // Ratio test against the best match elsewhere on the trajectory, see README.md.
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

  // Only the winner is thresholded; a relocalizer verifies the runners-up.
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

void VprAnyLoc::Serialize(BlobWriter& writer) {
  Finalize();

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
  // Zero centers is valid (saved before training); otherwise the full set.
  if (cluster_count < 0 || descriptor_dim < 0 ||
      (center_count != 0 &&
       center_count != static_cast<uint64_t>(cluster_count) * static_cast<uint64_t>(descriptor_dim))) {
    return false;
  }
  // 0 means saved before the first frame; otherwise it must match the engine.
  if (descriptor_dim != 0 && descriptor_dim != descriptor_dim_) {
    TraceError("VPR AnyLoc: the map holds %d dimensional descriptors but the engine produces %d dimensional ones\n",
               descriptor_dim, descriptor_dim_);
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

  // Reject a count the remaining bytes cannot hold.
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
  centers_ = std::move(centers);
  entries_ = std::move(entries);
  index_by_node_ = std::move(index_by_node);
  // Everything loaded is already encoded.
  dirty_ = false;
  return true;
}

}  // namespace cuvslam::slam::vpr
