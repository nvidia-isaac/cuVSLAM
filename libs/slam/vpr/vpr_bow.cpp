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

#include "slam/vpr/vpr_bow.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

#include "common/log.h"

namespace cuvslam::slam::vpr {

namespace {
constexpr uint32_t kBlobMagic = 0x57424f56;  // "VOBW"
constexpr uint32_t kBlobVersion = 1;
}  // namespace

int BowVocabulary::Hamming(const Descriptor& a, const Descriptor& b) {
  // Four 64 bit popcounts rather than thirty-two 8 bit ones. This is the inner loop of both the fit
  // and every quantization, so it is worth the memcpy the strict aliasing rules ask for.
  uint64_t lhs[4];
  uint64_t rhs[4];
  std::memcpy(lhs, a.data(), sizeof(lhs));
  std::memcpy(rhs, b.data(), sizeof(rhs));
  int distance = 0;
  for (int i = 0; i < 4; ++i) {
    distance += __builtin_popcountll(lhs[i] ^ rhs[i]);
  }
  return distance;
}

int BowVocabulary::Assign(const Descriptor& descriptor) const {
  int best = 0;
  int best_distance = std::numeric_limits<int>::max();
  for (int c = 0; c < static_cast<int>(centers_.size()); ++c) {
    const int distance = Hamming(descriptor, centers_[c]);
    if (distance < best_distance) {
      best_distance = distance;
      best = c;
    }
  }
  return best;
}

void BowVocabulary::Train(const std::vector<Descriptor>& descriptors, int word_count) {
  centers_.clear();
  const int k = std::min(word_count, static_cast<int>(descriptors.size()));
  if (k <= 0) {
    return;
  }

  // Seeding is what decides whether the words end up spread over the descriptor cloud, and a fixed
  // generator is what makes two runs over the same sequence produce the same map.
  std::mt19937 rng(0xdeadbeefU);
  centers_.resize(static_cast<size_t>(k));
  {
    std::uniform_int_distribution<size_t> pick(0, descriptors.size() - 1);
    centers_[0] = descriptors[pick(rng)];

    std::vector<int> nearest(descriptors.size(), std::numeric_limits<int>::max());
    std::vector<double> weights(descriptors.size());
    for (int c = 1; c < k; ++c) {
      for (size_t i = 0; i < descriptors.size(); ++i) {
        nearest[i] = std::min(nearest[i], Hamming(descriptors[i], centers_[c - 1]));
        weights[i] = static_cast<double>(nearest[i]) * nearest[i];
      }
      std::discrete_distribution<size_t> weighted(weights.begin(), weights.end());
      centers_[c] = descriptors[weighted(rng)];
    }
  }

  std::vector<int> assignment(descriptors.size(), 0);
  std::vector<std::array<int, 256>> bit_counts(static_cast<size_t>(k));
  std::vector<int> cluster_size(static_cast<size_t>(k), 0);
  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    bool changed = false;
    for (size_t i = 0; i < descriptors.size(); ++i) {
      const int best = Assign(descriptors[i]);
      if (assignment[i] != best) {
        assignment[i] = best;
        changed = true;
      }
    }
    if (!changed) {
      break;
    }

    // The mean of a set of bit strings under Hamming distance is the bit wise majority.
    for (auto& counts : bit_counts) {
      counts.fill(0);
    }
    std::fill(cluster_size.begin(), cluster_size.end(), 0);
    for (size_t i = 0; i < descriptors.size(); ++i) {
      const int c = assignment[i];
      ++cluster_size[c];
      for (int bit = 0; bit < 256; ++bit) {
        if (descriptors[i][bit / 8] & (1u << (bit % 8))) {
          ++bit_counts[c][bit];
        }
      }
    }
    for (int c = 0; c < k; ++c) {
      if (cluster_size[c] == 0) {
        continue;
      }
      const int half = cluster_size[c] / 2;
      centers_[c].fill(0);
      for (int bit = 0; bit < 256; ++bit) {
        if (bit_counts[c][bit] > half) {
          centers_[c][bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
        }
      }
    }

    // A cluster that lost its last member would keep an uncontested center forever and never win
    // another assignment, so the vocabulary would quietly shrink. Reseed it from the descriptor its
    // own center fits worst.
    for (int c = 0; c < k; ++c) {
      if (cluster_size[c] != 0) {
        continue;
      }
      size_t worst = 0;
      int worst_distance = -1;
      for (size_t i = 0; i < descriptors.size(); ++i) {
        const int distance = Hamming(descriptors[i], centers_[assignment[i]]);
        if (distance > worst_distance) {
          worst_distance = distance;
          worst = i;
        }
      }
      centers_[c] = descriptors[worst];
    }
  }
}

VprBow::VprBow(const VprOptions& options)
    : score_threshold_(options.score_threshold > 0.f ? options.score_threshold : kDefaultScoreThreshold),
      ratio_threshold_(options.ratio_threshold),
      max_entries_(options.max_entries) {}

void VprBow::AddFrame(KeyFrameId node_id, const VprImage& image) {
  const std::vector<OrbKeypoint> keypoints = orb_.Extract(image);
  if (keypoints.empty()) {
    return;
  }

  std::vector<BowVocabulary::Descriptor> descriptors;
  descriptors.reserve(keypoints.size());
  for (const OrbKeypoint& keypoint : keypoints) {
    descriptors.push_back(keypoint.descriptor);
  }

  const auto it = index_by_node_.find(node_id);
  if (it != index_by_node_.end()) {
    Entry& entry = entries_[it->second];
    entry.words = vocabulary_.Ready() ? Describe(descriptors) : std::vector<Word>{};
    entry.descriptors = vocabulary_.Ready() ? std::vector<BowVocabulary::Descriptor>{} : std::move(descriptors);
    dirty_ = true;
    return;
  }

  if (max_entries_ != 0 && entries_.size() >= max_entries_) {
    return;
  }

  Entry entry;
  entry.node_id = node_id;
  if (vocabulary_.Ready()) {
    entry.words = Describe(descriptors);
  } else {
    // Held until there are enough frames to fit a vocabulary to.
    entry.descriptors = std::move(descriptors);
  }
  index_by_node_.emplace(node_id, entries_.size());
  entries_.push_back(std::move(entry));
  dirty_ = true;
}

bool VprBow::HasNode(KeyFrameId node_id) const { return index_by_node_.count(node_id) != 0; }

void VprBow::RemoveNode(KeyFrameId node_id) {
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

void VprBow::ShiftNodeIds(KeyFrameId offset) {
  index_by_node_.clear();
  index_by_node_.reserve(entries_.size());
  for (size_t i = 0; i < entries_.size(); ++i) {
    entries_[i].node_id += offset;
    index_by_node_.emplace(entries_[i].node_id, i);
  }
}

std::vector<VprBow::Word> VprBow::Describe(const std::vector<BowVocabulary::Descriptor>& descriptors) const {
  std::vector<Word> words;
  if (descriptors.empty() || !vocabulary_.Ready()) {
    return words;
  }

  std::vector<float> term_frequency(static_cast<size_t>(vocabulary_.WordCount()), 0.f);
  for (const BowVocabulary::Descriptor& descriptor : descriptors) {
    term_frequency[static_cast<size_t>(vocabulary_.Assign(descriptor))] += 1.f;
  }

  const float inv_total = 1.f / static_cast<float>(descriptors.size());
  float sum = 0.f;
  for (int word = 0; word < vocabulary_.WordCount(); ++word) {
    if (term_frequency[static_cast<size_t>(word)] == 0.f) {
      continue;
    }
    const float weight = term_frequency[static_cast<size_t>(word)] * inv_total * idf_[static_cast<size_t>(word)];
    if (weight <= 0.f) {
      continue;
    }
    words.push_back({static_cast<int32_t>(word), weight});
    sum += weight;
  }

  // The loop above walks the word ids in order, so the bag is sorted and a query can binary search
  // it. L1 normalize as well, so the similarity below is DBoW2's and lands in [0, 1] whatever the
  // feature count.
  if (sum > 0.f) {
    for (Word& word : words) {
      word.weight /= sum;
    }
  }
  return words;
}

void VprBow::TrainVocabulary() {
  size_t available = 0;
  for (const Entry& entry : entries_) {
    available += entry.descriptors.size();
  }
  if (available == 0) {
    return;
  }

  // Fixed stride over the concatenation of every frame's descriptors, not a random sample: two runs
  // over the same sequence have to fit the same words, and a strided walk spreads the training set
  // evenly over the trajectory instead of over-weighting whichever frames an RNG happened to pick.
  const size_t stride = std::max<size_t>(1, (available + kMaxTrainingDescriptors - 1) / kMaxTrainingDescriptors);
  std::vector<BowVocabulary::Descriptor> training;
  training.reserve(std::min(available, kMaxTrainingDescriptors));
  size_t walked = 0;
  for (const Entry& entry : entries_) {
    for (const BowVocabulary::Descriptor& descriptor : entry.descriptors) {
      if (walked++ % stride == 0) {
        training.push_back(descriptor);
      }
    }
  }

  vocabulary_.Train(training, kWordCount);
  if (!vocabulary_.Ready()) {
    return;
  }

  // Inverse document frequency over the frames the vocabulary was fitted to, frozen from here on:
  // recomputing it as the map grows would change the meaning of every bag already stored.
  const int words = vocabulary_.WordCount();
  std::vector<int> document_frequency(static_cast<size_t>(words), 0);
  std::vector<bool> seen(static_cast<size_t>(words), false);
  for (const Entry& entry : entries_) {
    std::fill(seen.begin(), seen.end(), false);
    for (const BowVocabulary::Descriptor& descriptor : entry.descriptors) {
      const int word = vocabulary_.Assign(descriptor);
      if (!seen[static_cast<size_t>(word)]) {
        seen[static_cast<size_t>(word)] = true;
        ++document_frequency[static_cast<size_t>(word)];
      }
    }
  }

  const auto frames = static_cast<float>(entries_.size());
  idf_.assign(static_cast<size_t>(words), 0.f);
  for (int word = 0; word < words; ++word) {
    const int frequency = document_frequency[static_cast<size_t>(word)];
    // A word every frame contains says nothing about which frame this is, and gets weight zero.
    idf_[static_cast<size_t>(word)] = frequency > 0 ? std::log(frames / static_cast<float>(frequency)) : 0.f;
  }

  for (Entry& entry : entries_) {
    entry.words = Describe(entry.descriptors);
    entry.descriptors.clear();
    entry.descriptors.shrink_to_fit();
  }
  TraceMessage("VPR: Bow fitted %d words to %zu frames\n", words, entries_.size());
}

void VprBow::RebuildIndex() {
  inverted_index_.assign(static_cast<size_t>(std::max(0, vocabulary_.WordCount())), {});
  for (size_t i = 0; i < entries_.size(); ++i) {
    for (const Word& word : entries_[i].words) {
      inverted_index_[static_cast<size_t>(word.id)].push_back(static_cast<uint32_t>(i));
    }
  }
}

void VprBow::Finalize() {
  if (!dirty_) {
    return;
  }

  if (!vocabulary_.Ready()) {
    if (entries_.size() < kMinTrainingFrames) {
      // Too few frames to fit words to, and the inverse document frequencies would be degenerate.
      // Answering "no match" until there is something to fit is the honest outcome, and it is what
      // the DBoW2 backend does for the same reason.
      return;
    }
    TrainVocabulary();
    if (!vocabulary_.Ready()) {
      return;
    }
  }

  RebuildIndex();
  dirty_ = false;
}

VprMatch VprBow::Query(const VprImage& image) {
  const std::vector<VprMatch> matches = QueryTopK(image, 1);
  return matches.empty() ? VprMatch{} : matches.front();
}

std::vector<VprMatch> VprBow::QueryTopK(const VprImage& image, size_t max_results) {
  std::vector<VprMatch> matches;
  Finalize();
  if (!vocabulary_.Ready() || entries_.empty() || max_results == 0) {
    return matches;
  }

  const std::vector<OrbKeypoint> keypoints = orb_.Extract(image);
  if (keypoints.empty()) {
    return matches;
  }
  std::vector<BowVocabulary::Descriptor> descriptors;
  descriptors.reserve(keypoints.size());
  for (const OrbKeypoint& keypoint : keypoints) {
    descriptors.push_back(keypoint.descriptor);
  }
  const std::vector<Word> query = Describe(descriptors);
  if (query.empty()) {
    return matches;
  }

  // DBoW2's L1 similarity. For two L1 normalized vectors the L1 distance is
  // 2 + sum over the shared words of (|a - b| - |a| - |b|), so accumulating that one term per
  // shared word over the inverted index gives 1 - distance/2 directly, without touching the
  // entries that share nothing with the query.
  std::vector<float> score(entries_.size(), 0.f);
  for (const Word& word : query) {
    for (const uint32_t entry_index : inverted_index_[static_cast<size_t>(word.id)]) {
      const std::vector<Word>& bag = entries_[entry_index].words;
      const auto it = std::lower_bound(bag.begin(), bag.end(), word.id,
                                       [](const Word& candidate, int32_t id) { return candidate.id < id; });
      if (it == bag.end() || it->id != word.id) {
        continue;
      }
      score[entry_index] += std::fabs(word.weight - it->weight) - word.weight - it->weight;
    }
  }

  std::vector<std::pair<float, size_t>> ranked;
  ranked.reserve(entries_.size());
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (score[i] < 0.f) {
      ranked.emplace_back(-score[i] * 0.5f, i);
    }
  }
  if (ranked.empty()) {
    return matches;
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  if (ranked.front().first < score_threshold_) {
    return matches;
  }

  const KeyFrameId best_node = entries_[ranked.front().second].node_id;
  if (ratio_threshold_ > 0.f) {
    // Second best from a different part of the trajectory, see vpr/README.md.
    float rival = -1.f;
    for (const auto& [value, index] : ranked) {
      const KeyFrameId node = entries_[index].node_id;
      const uint64_t distance = best_node > node ? best_node - node : node - best_node;
      if (distance > kNeighbourGuard) {
        rival = std::max(rival, value);
      }
    }
    if (rival > 0.f && ranked.front().first < rival * ratio_threshold_) {
      return matches;
    }
  }

  const size_t count = std::min(max_results, ranked.size());
  matches.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    VprMatch match;
    match.found = true;
    match.node_id = entries_[ranked[i].second].node_id;
    match.score = std::min(1.f, std::max(0.f, ranked[i].first));
    matches.push_back(match);
  }
  return matches;
}

void VprBow::Serialize(Blob& blob) {
  Finalize();
  if (!vocabulary_.Ready()) {
    // The bags are what gets written, and there are none until the vocabulary exists. Saying so is
    // better than writing a file that loads into an empty map.
    TraceWarning(
        "VPR: Bow has only %zu frames, fewer than the %zu it needs to fit a vocabulary; the saved map "
        "will be empty\n",
        entries_.size(), kMinTrainingFrames);
  }

  BlobWriter writer(blob);
  writer.write(kBlobMagic);
  writer.write(kBlobVersion);
  writer.write(static_cast<int32_t>(vocabulary_.WordCount()));
  for (const BowVocabulary::Descriptor& center : vocabulary_.Centers()) {
    writer.write(center.data(), center.size());
  }
  if (!idf_.empty()) {
    writer.write(idf_.data(), idf_.size() * sizeof(float));
  }

  writer.write(static_cast<uint64_t>(entries_.size()));
  for (const Entry& entry : entries_) {
    writer.write(static_cast<uint64_t>(entry.node_id));
    writer.write(static_cast<uint64_t>(entry.words.size()));
    for (const Word& word : entry.words) {
      writer.write(word.id);
      writer.write(word.weight);
    }
  }
}

bool VprBow::Deserialize(const BlobReader& reader) {
  uint32_t magic = 0;
  uint32_t version = 0;
  int32_t words = 0;
  if (!reader.read(magic) || !reader.read(version) || !reader.read(words)) {
    return false;
  }
  if (magic != kBlobMagic || version != kBlobVersion || words < 0) {
    return false;
  }
  constexpr size_t kCenterBytes = sizeof(BowVocabulary::Descriptor) + sizeof(float);
  if (static_cast<size_t>(words) > reader.remaining() / kCenterBytes) {
    return false;
  }

  std::vector<BowVocabulary::Descriptor> centers(static_cast<size_t>(words));
  for (BowVocabulary::Descriptor& center : centers) {
    if (!reader.read(center.data(), center.size())) {
      return false;
    }
  }
  std::vector<float> idf(static_cast<size_t>(words));
  if (words != 0 && !reader.read(idf.data(), idf.size() * sizeof(float))) {
    return false;
  }

  uint64_t count = 0;
  if (!reader.read(count)) {
    return false;
  }
  constexpr size_t kMinEntryBytes = sizeof(uint64_t) + sizeof(uint64_t);
  if (count > reader.remaining() / kMinEntryBytes) {
    return false;
  }

  std::vector<Entry> entries;
  std::unordered_map<KeyFrameId, size_t> index_by_node;
  entries.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t node_id = 0;
    uint64_t word_count = 0;
    if (!reader.read(node_id) || !reader.read(word_count)) {
      return false;
    }
    if (word_count > reader.remaining() / (sizeof(int32_t) + sizeof(float))) {
      return false;
    }
    Entry entry;
    entry.node_id = static_cast<KeyFrameId>(node_id);
    entry.words.resize(static_cast<size_t>(word_count));
    for (Word& word : entry.words) {
      if (!reader.read(word.id) || !reader.read(word.weight)) {
        return false;
      }
      if (word.id < 0 || word.id >= words) {
        return false;
      }
    }
    // A query binary searches the bag, so a file whose words are out of order would silently score
    // every entry too low rather than fail.
    std::sort(entry.words.begin(), entry.words.end(), [](const Word& a, const Word& b) { return a.id < b.id; });
    index_by_node.emplace(entry.node_id, entries.size());
    entries.push_back(std::move(entry));
  }

  vocabulary_.SetCenters(std::move(centers));
  idf_ = std::move(idf);
  entries_ = std::move(entries);
  index_by_node_ = std::move(index_by_node);
  RebuildIndex();
  dirty_ = false;
  return true;
}

}  // namespace cuvslam::slam::vpr
