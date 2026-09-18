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

#include "slam/vpr/vpr_dbow2.h"

#include <algorithm>

#include <DBoW2/DBoW2.h>
#include <opencv2/features2d.hpp>

#include "common/log.h"

namespace cuvslam::slam::vpr {

namespace {
constexpr uint32_t kBlobMagic = 0x44525056;  // "VPRD"
constexpr uint32_t kBlobVersion = 1;

/// DBoW2 addresses one descriptor as one 1x32 CV_8U cv::Mat, not as the Nx32 matrix cv::ORB
/// returns. Handing it the whole matrix compiles and runs but quantizes garbage, so every matrix
/// that crosses into DBoW2 goes through here first. The rows are views into `descriptors`, which
/// outlives every DBoW2 call that sees them.
std::vector<cv::Mat> SplitRows(const cv::Mat& descriptors) {
  std::vector<cv::Mat> rows;
  rows.reserve(static_cast<size_t>(descriptors.rows));
  for (int i = 0; i < descriptors.rows; ++i) {
    rows.push_back(descriptors.row(i));
  }
  return rows;
}
}  // namespace

/// DBoW2's ORB descriptor traits, with the one case that makes its k-means dereference a null
/// pointer taken out.
///
/// TemplatedVocabulary::create() clusters each tree level with Lloyd iterations: assign every
/// descriptor to its nearest center, recompute every center from the descriptors assigned to it,
/// repeat until the assignment stops changing. A center that moves can end up with nothing assigned
/// to it, and upstream then calls FORB::meanValue() with an empty descriptor list, which answers
/// with mean.release() -- a 0x0 cv::Mat. The next iteration measures every descriptor against that
/// center, and FORB::distance() reads through cv::Mat::ptr() without checking, so it dereferences
/// null and the process dies. Upstream believes the case cannot arise ("kmeans++ ensures all the
/// clusters has any feature associated with them", TemplatedVocabulary.h); kmeans++ only guarantees
/// that the initial centers are distinct, which says nothing about what happens once they move.
///
/// Leaving the center where it was is what k-means implementations that handle an empty cluster do.
/// The cluster stays empty for the rest of the run, so it ends up a leaf no training descriptor
/// reached, its IDF weight stays 0, and TemplatedVocabulary::transform() drops zero weight words,
/// which is the same as if the cluster had never been proposed.
struct SafeFORB : DBoW2::FORB {
  static void meanValue(const std::vector<pDescriptor>& descriptors, TDescriptor& mean) {
    if (descriptors.empty()) {
      return;
    }
    DBoW2::FORB::meanValue(descriptors, mean);
  }
};

struct VprDBoW2::Index {
  cv::Ptr<cv::ORB> orb = cv::ORB::create(VprDBoW2::kMaxFeatures);
  DBoW2::TemplatedVocabulary<SafeFORB::TDescriptor, SafeFORB> vocabulary{
      VprDBoW2::kBranchingFactor, VprDBoW2::kTreeLevels, DBoW2::TF_IDF, DBoW2::L1_NORM};
  DBoW2::TemplatedDatabase<SafeFORB::TDescriptor, SafeFORB> database{false};

  /// Pose graph node each database entry came from, indexed by DBoW2::EntryId.
  std::vector<KeyFrameId> node_by_entry;
  bool trained = false;
};

VprDBoW2::VprDBoW2(const VprOptions& options)
    : score_threshold_(options.score_threshold > 0.f ? options.score_threshold : kDefaultScoreThreshold),
      ratio_threshold_(options.ratio_threshold),
      max_entries_(options.max_entries),
      index_(std::make_unique<Index>()) {}

VprDBoW2::~VprDBoW2() = default;

cv::Mat VprDBoW2::ExtractDescriptors(const VprImage& image) const {
  if (image.Empty()) {
    return cv::Mat();
  }

  // VprImage pixels are single channel, tightly packed and row major, which is exactly a CV_8UC1
  // cv::Mat; the wrapper borrows them and cv::ORB only reads through it.
  const cv::Mat pixels(image.height, image.width, CV_8UC1, const_cast<uint8_t*>(image.row.data()));

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  index_->orb->detectAndCompute(pixels, cv::noArray(), keypoints, descriptors);

  if (descriptors.empty() || descriptors.type() != CV_8UC1 || descriptors.cols != DBoW2::FORB::L) {
    return cv::Mat();
  }
  // detectAndCompute allocated this itself, so it already owns its pixels and outlives `image`.
  return descriptors;
}

void VprDBoW2::AddFrame(KeyFrameId node_id, const VprImage& image) {
  cv::Mat descriptors = ExtractDescriptors(image);
  if (descriptors.empty()) {
    return;
  }

  const auto it = index_by_node_.find(node_id);
  if (it != index_by_node_.end()) {
    entries_[it->second].descriptors = std::move(descriptors);
    dirty_ = true;
    return;
  }

  if (max_entries_ != 0 && entries_.size() >= max_entries_) {
    return;
  }

  index_by_node_.emplace(node_id, entries_.size());
  entries_.push_back(Entry{node_id, std::move(descriptors)});
  dirty_ = true;
}

bool VprDBoW2::HasNode(KeyFrameId node_id) const { return index_by_node_.count(node_id) != 0; }

void VprDBoW2::RemoveNode(KeyFrameId node_id) {
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
  dirty_ = true;
}

void VprDBoW2::ShiftNodeIds(KeyFrameId offset) {
  index_by_node_.clear();
  index_by_node_.reserve(entries_.size());
  for (size_t i = 0; i < entries_.size(); ++i) {
    entries_[i].node_id += offset;
    index_by_node_.emplace(entries_[i].node_id, i);
  }
  // The vocabulary survives a renumbering, but the entry to node table does not, and rebuilding it
  // alone is not worth a second code path.
  dirty_ = true;
}

void VprDBoW2::Finalize() {
  if (!dirty_) {
    return;
  }
  dirty_ = false;
  index_->trained = false;
  index_->node_by_entry.clear();

  if (entries_.size() < kMinTrainingFrames) {
    // TF-IDF weights a word by log(N / n_i). On a handful of frames almost every word occurs in
    // almost every frame, the weights collapse to zero and the scores stop ordering the map at all.
    // Answering "no match" is the honest outcome: a fallback to raw descriptor distances would
    // answer on a different score scale, and the caller's score_threshold would quietly mean
    // something else below the frame count than above it.
    TraceWarning("VPR: DBoW2 needs %zu mapped frames to train a vocabulary, have %zu; queries will not match\n",
                 kMinTrainingFrames, entries_.size());
    return;
  }

  std::vector<std::vector<cv::Mat>> training;
  training.reserve(entries_.size());
  for (const auto& entry : entries_) {
    training.push_back(SplitRows(entry.descriptors));
  }

  index_->vocabulary.create(training, kBranchingFactor, kTreeLevels, DBoW2::TF_IDF, DBoW2::L1_NORM);
  if (index_->vocabulary.empty()) {
    TraceError("VPR: DBoW2 vocabulary training produced no words from %zu frames\n", entries_.size());
    return;
  }

  // No direct index: it exists to recover feature correspondences from a match, and this backend
  // reports a node id rather than correspondences.
  index_->database.setVocabulary(index_->vocabulary, false, 0);
  index_->node_by_entry.resize(entries_.size(), InvalidKeyFrameId);
  for (size_t i = 0; i < entries_.size(); ++i) {
    const DBoW2::EntryId entry_id = index_->database.add(training[i]);
    if (entry_id < index_->node_by_entry.size()) {
      index_->node_by_entry[entry_id] = entries_[i].node_id;
    }
  }

  index_->trained = true;
  TraceMessage("VPR: DBoW2 trained %u words over %zu frames\n", index_->vocabulary.size(), entries_.size());
}

VprMatch VprDBoW2::Query(const VprImage& image) {
  const std::vector<VprMatch> matches = QueryTopK(image, 1);
  return matches.empty() ? VprMatch{} : matches.front();
}

std::vector<VprMatch> VprDBoW2::QueryTopK(const VprImage& image, size_t max_results) {
  std::vector<VprMatch> matches;
  Finalize();
  if (!index_->trained || max_results == 0) {
    return matches;
  }

  const cv::Mat descriptors = ExtractDescriptors(image);
  if (descriptors.empty()) {
    return matches;
  }

  // The ratio test needs the best score of the rest of the map, which the top few hits cannot
  // supply: in a sequence they are all temporal neighbors of the winner. 0 means "all".
  const int requested =
      ratio_threshold_ > 0.f ? 0 : static_cast<int>(std::max<size_t>(max_results, static_cast<size_t>(kQueryResults)));
  DBoW2::QueryResults results;
  index_->database.query(SplitRows(descriptors), results, requested);
  if (results.empty()) {
    return matches;
  }

  // queryL1 leaves the results sorted best first, on a scale where 1 is an identical bag of words.
  const DBoW2::Result& best = results.front();
  if (best.Id >= index_->node_by_entry.size()) {
    return matches;
  }
  const KeyFrameId best_node = index_->node_by_entry[best.Id];
  if (best_node == InvalidKeyFrameId || best.Score < score_threshold_) {
    return matches;
  }

  if (ratio_threshold_ > 0.f) {
    // Second best from a different part of the trajectory: an ambiguous place is dropped, see README.md.
    double rival = -1.0;
    for (const auto& result : results) {
      if (result.Id >= index_->node_by_entry.size()) {
        continue;
      }
      const KeyFrameId node = index_->node_by_entry[result.Id];
      if (node == InvalidKeyFrameId) {
        continue;
      }
      const uint64_t distance = best_node > node ? best_node - node : node - best_node;
      if (distance <= kNeighbourGuard) {
        continue;
      }
      rival = std::max(rival, result.Score);
    }
    if (rival > 0.0 && best.Score < rival * static_cast<double>(ratio_threshold_)) {
      return matches;
    }
  }

  // Only the winner is held to the threshold; the runners-up are there for a relocalizer to verify.
  for (const auto& result : results) {
    if (matches.size() >= max_results) {
      break;
    }
    if (result.Id >= index_->node_by_entry.size()) {
      continue;
    }
    const KeyFrameId node = index_->node_by_entry[result.Id];
    if (node == InvalidKeyFrameId) {
      continue;
    }
    VprMatch match;
    match.found = true;
    match.node_id = node;
    match.score = std::min(1.f, std::max(0.f, static_cast<float>(result.Score)));
    matches.push_back(match);
  }
  return matches;
}

void VprDBoW2::Serialize(Blob& blob) {
  // Only the descriptors are written; Finalize() retrains the vocabulary from them on the loading
  // side. DBoW2 can only export a vocabulary through cv::FileStorage, which would mean a text file
  // nested inside a binary blob. The retrained tree is not bit identical, because DBoW2 seeds its
  // k-means from the global rand() state, so a reloaded map scores a few percent away from the
  // session that wrote it; the ranking, and with it the node a query resolves to, is unaffected.
  BlobWriter writer(blob);
  writer.write(kBlobMagic);
  writer.write(kBlobVersion);
  writer.write(static_cast<uint64_t>(entries_.size()));
  for (const auto& entry : entries_) {
    const cv::Mat descriptors = entry.descriptors.isContinuous() ? entry.descriptors : entry.descriptors.clone();
    writer.write(static_cast<uint64_t>(entry.node_id));
    writer.write(static_cast<int32_t>(descriptors.rows));
    writer.write(static_cast<int32_t>(descriptors.cols));
    writer.write(descriptors.data, descriptors.total() * descriptors.elemSize());
  }
}

bool VprDBoW2::Deserialize(const BlobReader& reader) {
  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t count = 0;
  if (!reader.read(magic) || !reader.read(version) || !reader.read(count)) {
    return false;
  }
  if (magic != kBlobMagic || version != kBlobVersion) {
    return false;
  }

  // Set before the loop, so that a blob that turns out to be truncated still leaves the backend in
  // a state Finalize() will rebuild from whatever was read.
  entries_.clear();
  index_by_node_.clear();
  index_->trained = false;
  dirty_ = true;
  entries_.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t node_id = 0;
    int32_t rows = 0;
    int32_t cols = 0;
    if (!reader.read(node_id) || !reader.read(rows) || !reader.read(cols)) {
      return false;
    }
    if (rows <= 0 || cols != DBoW2::FORB::L) {
      return false;
    }

    Entry entry;
    entry.node_id = static_cast<KeyFrameId>(node_id);
    entry.descriptors.create(rows, cols, CV_8UC1);
    if (!reader.read(entry.descriptors.data, entry.descriptors.total() * entry.descriptors.elemSize())) {
      return false;
    }
    index_by_node_.emplace(entry.node_id, entries_.size());
    entries_.push_back(std::move(entry));
  }
  return true;
}

}  // namespace cuvslam::slam::vpr
