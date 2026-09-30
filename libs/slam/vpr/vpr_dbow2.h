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

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "slam/vpr/ivpr.h"

namespace cuvslam::slam::vpr {

/// Bag of binary words place recognition, after Galvez-Lopez and Tardos (2012). Every mapped frame
/// is reduced to at most kMaxFeatures ORB descriptors, and the map to a vocabulary tree whose leaves
/// are the visual words they quantize to, which buys the viewpoint invariance Simple lacks.
///
/// The vocabulary is trained on the session's own frames rather than shipped as a hundred megabyte
/// pre-trained asset, which keeps the backend self contained and is what makes the descriptors
/// Serialize() writes enough to rebuild the whole index. The price is that every change to the map
/// invalidates it and Finalize() retrains from scratch on the next query: about 25 ms per mapped
/// frame on KITTI 07, against 12 ms for a query on a map that is holding still. Querying every frame
/// is therefore affordable against a loaded or finished map, not against one that is still growing.
class VprDBoW2 : public IVpr {
public:
  /// Score below which a query is reported as "no match" when the caller left score_threshold at 0.
  ///
  /// Measured on KITTI 07 with maps of 25, 60, 118 and 200 nodes, scoring 221 queries against the
  /// ground truth: a query within 8 m of the frame it matched scored 0.12 and up, one more than 25 m
  /// away never scored above 0.146. 0.15 accepted no wrong place at any of the four map sizes and
  /// still recognizes 90 to 97% of the true revisits. Precision is the side to err on: a caller
  /// relocalizing into a map acts on a match.
  ///
  /// That holds for a map of one sequence queried by the same sequence. Across two separate drives
  /// of one route (CODa 00 against CODa 05, 2500 frames) the correct and the wrong matches overlap
  /// almost completely - medians 0.128 and 0.123 - so no threshold separates them, and 0.15 keeps
  /// its precision by giving up recall: 10% of queries recognized, 1% wrong. This backend does not
  /// discriminate across sessions; see the evaluation in libs/slam/vpr/README.md.
  static constexpr float kDefaultScoreThreshold = 0.15f;

  /// Two nodes closer than this in creation order are treated as the same place by the ratio test.
  static constexpr uint64_t kNeighbourGuard = 10;

  /// ORB features kept per frame. 500 descriptors are 16 kB, so a ten thousand frame session costs
  /// 160 MB of map, and the features past the first few hundred are the low contrast corners that
  /// are the least repeatable between two passes anyway.
  static constexpr int kMaxFeatures = 500;

  /// Vocabulary tree shape, at most kBranchingFactor^kTreeLevels = 10000 words.
  ///
  /// k and L are fixed rather than derived from options.vocabulary_size: that option counts VLAD
  /// cluster centers for AnyLoc and defaults to 32, which is three orders of magnitude away from a
  /// usable word count, and a DBoW2 tree can only realize word counts that are powers of k anyway.
  /// 10000 words against the few hundred thousand descriptors of a session sized map leaves tens of
  /// descriptors per word, which is the ratio the words stay discriminative at. The pre-trained
  /// vocabularies shipped with ORB-SLAM use L=6 (a million words), but those are trained offline on
  /// ten thousand unrelated images; at L=6 a session map has fewer descriptors than leaves and every
  /// word degenerates into a single feature.
  static constexpr int kBranchingFactor = 10;
  static constexpr int kTreeLevels = 4;

  /// Below this many mapped frames the index is left untrained and every query answers "no match".
  static constexpr size_t kMinTrainingFrames = 8;

  /// Database entries a query inspects when the ratio test is off.
  static constexpr int kQueryResults = 10;

  explicit VprDBoW2(const VprOptions& options);
  ~VprDBoW2() override;

  const char* Name() const override { return "DBoW2"; }

  void AddFrame(KeyFrameId node_id, const VprImage& image) override;
  bool HasNode(KeyFrameId node_id) const override;
  void RemoveNode(KeyFrameId node_id) override;
  void ShiftNodeIds(KeyFrameId offset) override;
  size_t Size() const override { return entries_.size(); }
  void Finalize() override;
  VprMatch Query(const VprImage& image) override;
  std::vector<VprMatch> QueryTopK(const VprImage& image, size_t max_results) override;
  void Serialize(Blob& blob) override;
  bool Deserialize(const BlobReader& reader) override;

private:
  /// The ORB detector, the DBoW2 vocabulary and the DBoW2 database. Held behind a pointer so the
  /// OpenCV features2d and DBoW2 headers stay out of this one, which ivpr.cpp includes.
  struct Index;

  struct Entry {
    KeyFrameId node_id = InvalidKeyFrameId;
    cv::Mat descriptors;  ///< N x 32 CV_8U, one ORB descriptor per row
  };

  /// ORB descriptors of `image`, empty when the image yields no usable features.
  cv::Mat ExtractDescriptors(const VprImage& image) const;

  float score_threshold_ = kDefaultScoreThreshold;
  float ratio_threshold_ = 0.f;
  uint32_t max_entries_ = 0;

  std::vector<Entry> entries_;
  std::unordered_map<KeyFrameId, size_t> index_by_node_;

  std::unique_ptr<Index> index_;
  /// The map changed since the last Finalize(), so the vocabulary and the database are stale.
  bool dirty_ = false;
};

}  // namespace cuvslam::slam::vpr
