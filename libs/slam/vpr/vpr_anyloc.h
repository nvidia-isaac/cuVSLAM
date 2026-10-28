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
#include <string>
#include <unordered_map>
#include <vector>

#include "slam/vpr/ivpr.h"

namespace cuvslam::slam::vpr {

class AnyLocNetwork;

/// AnyLoc (Keetha et al., 2023): VLAD over DINOv2 patch features.
/// Loads VprOptions::vocabulary_path or fits a vocabulary per session; maps of two vocabularies are not comparable.
class VprAnyLoc : public IVpr {
public:
  /// Ratio test treats nodes this close in creation order as the same place.
  static constexpr uint64_t kNeighbourGuard = 10;

  /// Max spherical k-means iterations; stops early once assignments settle.
  static constexpr int kKMeansIterations = 25;

  /// @throws std::runtime_error if options.model_path is not a usable engine or options.vocabulary_path not a
  /// vocabulary of it; std::invalid_argument if the training options cannot hold min_training_keyframes frames.
  explicit VprAnyLoc(const VprOptions& options);
  ~VprAnyLoc() override;

  const char* Name() const override { return "AnyLoc"; }

  void AddFrame(KeyFrameId node_id, const VprImage& image) override;
  bool HasNode(KeyFrameId node_id) const override;
  void RemoveNode(KeyFrameId node_id) override;
  size_t Size() const override { return entries_.size(); }
  size_t TrainingCapacity() const override;
  void Finalize() override;
  VprMatch Query(const VprImage& image) override;
  std::vector<VprMatch> QueryTopK(const VprImage& image, size_t max_results) override;
  void Serialize(BlobWriter& writer) override;
  bool Deserialize(const BlobReader& reader) override;

private:
  struct Entry {
    KeyFrameId node_id = InvalidKeyFrameId;
    std::vector<float> vlad;     ///< cluster_count_ * descriptor_dim_ floats
    std::vector<float> patches;  ///< per-patch descriptors until the vocabulary is fitted
  };

  /// Fill `patches` with the unit norm per-patch descriptors of `image`.
  bool Describe(const VprImage& image, std::vector<float>& patches);

  /// Fit the VLAD centers to the patches of the frames mapped so far.
  void TrainVocabulary();

  /// VLAD-pool `patches` against the current centers.
  void Encode(const std::vector<float>& patches, std::vector<float>& vlad) const;

  /// Take the centers of vocabulary_path_.
  void LoadVocabulary();

  std::string model_path_;
  std::string vocabulary_path_;
  int cluster_count_ = 0;
  float score_threshold_ = 0.f;
  float ratio_threshold_ = 0.f;
  uint32_t max_entries_ = 0;
  size_t min_training_keyframes_ = 0;
  size_t max_training_descriptors_ = 0;

  std::unique_ptr<AnyLocNetwork> network_;

  int descriptor_dim_ = 0;
  /// cluster_count_ x descriptor_dim_ unit norm rows; empty until trained.
  std::vector<float> centers_;

  std::vector<Entry> entries_;
  std::unordered_map<KeyFrameId, size_t> index_by_node_;

  /// Some entries still hold unpooled patches.
  bool dirty_ = false;
};

}  // namespace cuvslam::slam::vpr
