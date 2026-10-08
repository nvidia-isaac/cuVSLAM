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

/// AnyLoc place recognition: VLAD aggregation of DINOv2 dense patch features, after Keetha et al.
/// (2023), "AnyLoc: Towards Universal Visual Place Recognition". The value projection of one
/// intermediate attention block gives one unit norm descriptor per image patch, and VLAD pools those
/// into one fixed length vector, which is what makes a plain cosine similarity a usable place score.
///
/// The strongest and the most expensive of the three backends: DINOv2 features are semantic rather
/// than photometric, so a place still matches under weather, season or illumination change that
/// defeats thumbnails and ORB corners. The price is a model file (`Slam::Config::vpr_model_path`),
/// tens of milliseconds of CPU per frame, and `vocabulary_size * 384` floats per map entry.
///
/// The vocabulary is fitted to the session's own frames rather than read from one of AnyLoc's domain
/// specific ones, so the saved map is self contained and always matches the domain being driven - at
/// the cost of two maps of the same place not being comparable to each other.
class VprAnyLoc : public IVpr {
public:
  /// Score below which a query is reported as "no match" when the caller left score_threshold at 0.
  ///
  /// Measured on KITTI 07 with the ViT-S/14 layer 9 value facet model at 322x322, a 64 node map of
  /// frames 0..299 and 68 queries scored against the ground truth: every query within 2 m of the node
  /// it matched scored 0.703 and up, while a query more than 90 m away from everything in the map
  /// never passed 0.655. 0.68 is the middle of that gap.
  ///
  /// It costs recall on the genuine second pass, where a correct match scores 0.62 to 0.82 and only
  /// the better half clears 0.68; the weak half scores in the same range as the wrong matches, so no
  /// threshold separates them. Precision is the side to err on: a relocalizing caller acts on a match.
  static constexpr float kDefaultScoreThreshold = 0.68f;

  /// Two nodes closer than this in creation order are treated as the same place by the ratio test.
  static constexpr uint64_t kNeighbourGuard = 10;

  /// VLAD cluster centers used when options.vocabulary_size is not set. This is AnyLoc's own
  /// default and the value its ablation settles on.
  static constexpr int kDefaultVocabularySize = 32;

  /// Patch descriptors the vocabulary is trained on, across all mapped frames.
  ///
  /// One 322x322 frame yields 529 of them, so an unbounded training set would cost a minute of
  /// k-means on a thousand frame sequence and grow without limit after that. The cap costs nothing
  /// in quality: 32 centers fitted to 100k points are already far past the point where more points
  /// move them, and the patches of a driving sequence are heavily redundant anyway.
  static constexpr size_t kMaxTrainingDescriptors = 100000;

  /// Lloyd iterations of the spherical k-means. It stops early when no assignment changes.
  static constexpr int kKMeansIterations = 25;

  /// Mapped frames the vocabulary needs before it is fitted. Below this a query answers "no match".
  ///
  /// The centers are fitted once and then frozen, because re-fitting them would change the meaning
  /// of every descriptor already stored. That makes fitting them too early expensive to undo: a
  /// caller that queries on its first keyframe would pin the whole session's vocabulary to one
  /// frame. The same guard exists in the DBoW2 backend for the same reason.
  static constexpr size_t kMinTrainingFrames = 8;

  explicit VprAnyLoc(const VprOptions& options);
  ~VprAnyLoc() override;

  const char* Name() const override { return "AnyLoc"; }

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
  /// The ONNX Runtime environment and session. Held behind a pointer so onnxruntime_cxx_api.h stays
  /// out of this header, which ivpr.cpp includes, and so the model is only loaded on first use.
  struct Network;

  struct Entry {
    KeyFrameId node_id = InvalidKeyFrameId;
    std::vector<float> vlad;     ///< pooled descriptor, cluster_count_ * descriptor_dim_ floats
    std::vector<float> patches;  ///< raw per-patch descriptors, only until Finalize() pools them
  };

  /// Load the model. Deferred to the first frame so a session that never hands VPR an image pays
  /// neither the 69 MB of model nor the failure of a missing file.
  /// @throws std::runtime_error when the model path is unset or does not name a readable file.
  void EnsureNetwork();

  /// Run the network on `image`. Returns the per-patch descriptors, `descriptor_dim_` floats each.
  bool Describe(const VprImage& image, std::vector<float>& patches);

  /// Fit the VLAD cluster centers to the patch descriptors of the frames added so far.
  void TrainVocabulary();

  /// Pool per-patch descriptors into one VLAD vector against the current centers.
  void Encode(const std::vector<float>& patches, std::vector<float>& vlad) const;

  std::string model_path_;
  int cluster_count_ = kDefaultVocabularySize;
  float score_threshold_ = kDefaultScoreThreshold;
  float ratio_threshold_ = 0.f;
  uint32_t max_entries_ = 0;

  std::unique_ptr<Network> network_;

  /// Width of one patch descriptor, read from the model output (384 for ViT-S/14).
  int descriptor_dim_ = 0;
  /// Cluster centers, cluster_count_ rows of descriptor_dim_ unit norm floats. Empty until trained.
  std::vector<float> centers_;

  std::vector<Entry> entries_;
  std::unordered_map<KeyFrameId, size_t> index_by_node_;

  /// Frames were added since the last Finalize(), so some entries still hold unpooled patches.
  bool dirty_ = false;
};

}  // namespace cuvslam::slam::vpr
