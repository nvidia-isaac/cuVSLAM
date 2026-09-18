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

#include <array>
#include <unordered_map>
#include <vector>

#include "slam/vpr/ivpr.h"
#include "slam/vpr/vpr_orb.h"

namespace cuvslam::slam::vpr {

/// Binary bag of words vocabulary: K cluster centers in Hamming space.
///
/// Flat rather than a tree, because K is a few hundred: quantizing a descriptor is K Hamming
/// distances, which is cheaper than walking a tree of the same size and has no branching to get
/// wrong. A tree only pays off at the tens of thousands of words a pre-trained vocabulary has.
class BowVocabulary {
public:
  using Descriptor = std::array<uint8_t, 32>;

  /// Fit `word_count` centers to `descriptors` by binary k-means with k-means++ seeding, bit wise
  /// majority vote for the update, and empty clusters reseeded from the worst fitted descriptor.
  /// Deterministic: the seeding uses a fixed generator, so two runs over the same input agree.
  void Train(const std::vector<Descriptor>& descriptors, int word_count);

  bool Ready() const { return !centers_.empty(); }
  int WordCount() const { return static_cast<int>(centers_.size()); }

  /// Nearest center to `descriptor`, in [0, WordCount()).
  int Assign(const Descriptor& descriptor) const;

  const std::vector<Descriptor>& Centers() const { return centers_; }
  void SetCenters(std::vector<Descriptor> centers) { centers_ = std::move(centers); }

  static int Hamming(const Descriptor& a, const Descriptor& b);

private:
  /// Lloyd iterations. It stops early when no assignment changes.
  static constexpr int kMaxIterations = 10;

  std::vector<Descriptor> centers_;
};

/// Bag of words place recognition over the in-tree ORB extractor, with no third-party dependency.
///
/// This is the DBoW2 backend's algorithm without DBoW2: ORB descriptors quantized against a binary
/// vocabulary, an inverted index over the words, and L1 normalized TF-IDF scoring. It exists because
/// DBoW2 makes OpenCV a hard dependency of a library that otherwise has none, which is a real cost
/// on an embedded target; the trade is a flat vocabulary of a few hundred words against DBoW2's tree
/// of ten thousand, so it discriminates less well on a large map.
///
/// The vocabulary is fitted once, to the first kMinTrainingFrames mapped frames, and then frozen
/// along with the inverse document frequencies. Refitting either would change the meaning of every
/// vector already stored and force a full re-encode of the map.
class VprBow : public IVpr {
public:
  /// Score below which a query is reported as "no match" when the caller left score_threshold at 0.
  ///
  /// The score is DBoW2's L1 similarity over L1 normalized TF-IDF vectors, so it is on the same
  /// scale as the DBoW2 backend's but not the same number: a flat vocabulary of a thousand words
  /// spreads its mass differently from a tree of ten thousand.
  ///
  /// Chosen from two measurements that pull in opposite directions. Replaying a sequence against a
  /// map of itself (KITTI 07), correct matches score from 0.32 up, so anything below that costs
  /// nothing. Across two separate drives of one route (CODa 00 against CODa 05, 2500 frames) the
  /// correct and the wrong matches overlap almost completely - medians 0.279 and 0.267 - so no
  /// threshold separates them and the only question is where to give up: 0.25 accepts 34% of
  /// queries correctly and 47% wrongly, while 0.30 accepts 8% correctly and 2% wrongly. 0.30 is the
  /// honest default, because a robot told confidently that it is somewhere it is not is worse off
  /// than one told nothing. What this backend cannot do is discriminate across sessions; see the
  /// evaluation in libs/slam/vpr/README.md.
  static constexpr float kDefaultScoreThreshold = 0.30f;

  /// Two nodes closer than this in creation order are treated as the same place by the ratio test.
  static constexpr uint64_t kNeighbourGuard = 10;

  /// Visual words.
  ///
  /// Fixed rather than taken from options.vocabulary_size: that option counts VLAD cluster centers
  /// for AnyLoc and defaults to 32, which is two orders of magnitude below anything a bag of words
  /// can discriminate with. Measured on KITTI 07, a 32 word vocabulary scores an unrelated street
  /// as high as a true revisit - every image activates every word - while 1024 separates them.
  static constexpr int kWordCount = 1024;

  /// Mapped frames the vocabulary and the inverse document frequencies are fitted to, and the
  /// minimum before a query is answered at all. The document frequencies are the sensitive half:
  /// estimated from a handful of frames they are pure noise, every word looks equally informative,
  /// and the scores stop ordering the map. Fifty is what the implementation this was ported from
  /// used, and measurements on KITTI 07 agree that eight is far too few.
  static constexpr size_t kMinTrainingFrames = 50;

  /// Descriptors the fit is allowed to see. Binary k-means is O(descriptors x words x iterations)
  /// and the frames of a driving sequence are heavily redundant, so the cap costs nothing in
  /// quality and is the difference between a fit that takes seconds and one that takes minutes.
  static constexpr size_t kMaxTrainingDescriptors = 20000;

  explicit VprBow(const VprOptions& options);
  ~VprBow() override = default;

  const char* Name() const override { return "Bow"; }

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
  /// One word of a frame's bag, with its L1 normalized TF-IDF weight.
  struct Word {
    int32_t id = 0;
    float weight = 0.f;
  };

  struct Entry {
    KeyFrameId node_id = InvalidKeyFrameId;
    std::vector<BowVocabulary::Descriptor> descriptors;  ///< kept only until the vocabulary is fitted
    std::vector<Word> words;                             ///< the bag, once it is
  };

  /// Quantize `descriptors` into an L1 normalized TF-IDF bag against the frozen vocabulary.
  std::vector<Word> Describe(const std::vector<BowVocabulary::Descriptor>& descriptors) const;

  /// Fit the vocabulary and the inverse document frequencies to the frames added so far.
  void TrainVocabulary();

  /// Rebuild the inverted index from the entries' bags.
  void RebuildIndex();

  float score_threshold_ = kDefaultScoreThreshold;
  float ratio_threshold_ = 0.f;
  uint32_t max_entries_ = 0;

  OrbExtractor orb_;
  BowVocabulary vocabulary_;
  std::vector<float> idf_;  ///< frozen with the vocabulary, one weight per word

  std::vector<Entry> entries_;
  std::unordered_map<KeyFrameId, size_t> index_by_node_;
  /// word id -> indices into entries_ whose bag contains it
  std::vector<std::vector<uint32_t>> inverted_index_;
  bool dirty_ = false;
};

}  // namespace cuvslam::slam::vpr
