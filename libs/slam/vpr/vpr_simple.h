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

#include <unordered_map>
#include <vector>

#include "slam/vpr/ivpr.h"

namespace cuvslam::slam::vpr {

/// Thumbnail based place recognition: one heavily downscaled grayscale copy of the frame per node,
/// scored by zero mean normalized cross correlation, which costs one dot product per map entry and
/// makes the score invariant to the exposure and gain differences between two passes.
///
/// The baseline backend, with no vocabulary, training or model file, and the weakest: a thumbnail is
/// a global appearance descriptor with no viewpoint invariance, so it degrades quickly once the
/// second pass drives a different lane or the opposite direction.
class VprSimple : public IVpr {
public:
  /// Score below which a query is reported as "no match" when the caller left score_threshold at 0.
  static constexpr float kDefaultScoreThreshold = 0.5f;

  /// Two nodes closer than this in creation order are treated as the same place by the ratio test.
  static constexpr uint64_t kNeighbourGuard = 10;

  explicit VprSimple(const VprOptions& options);
  ~VprSimple() override = default;

  const char* Name() const override { return "Simple"; }

  void AddFrame(KeyFrameId node_id, const VprImage& image) override;
  bool HasNode(KeyFrameId node_id) const override;
  void RemoveNode(KeyFrameId node_id) override;
  void ShiftNodeIds(KeyFrameId offset) override;
  size_t Size() const override { return entries_.size(); }
  void Finalize() override {}
  VprMatch Query(const VprImage& image) override;
  std::vector<VprMatch> QueryTopK(const VprImage& image, size_t max_results) override;
  void Serialize(Blob& blob) override;
  bool Deserialize(const BlobReader& reader) override;

private:
  struct Entry {
    KeyFrameId node_id = InvalidKeyFrameId;
    std::vector<float> descriptor;  ///< zero mean, unit norm thumbnail
  };

  /// Downscale, resize to the map thumbnail size and normalize. Returns false for a flat image.
  bool MakeDescriptor(const VprImage& image, std::vector<float>& descriptor) const;

  int downscale_ = 8;
  float score_threshold_ = kDefaultScoreThreshold;
  float ratio_threshold_ = 0.f;
  uint32_t max_entries_ = 0;

  /// Thumbnail geometry, fixed by the first frame added to the map.
  mutable int thumb_width_ = 0;
  mutable int thumb_height_ = 0;

  std::vector<Entry> entries_;
  std::unordered_map<KeyFrameId, size_t> index_by_node_;
};

}  // namespace cuvslam::slam::vpr
