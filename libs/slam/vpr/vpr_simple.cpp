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

#include "slam/vpr/vpr_simple.h"

#include <algorithm>
#include <cmath>

#include "slam/vpr/vpr_image.h"

namespace cuvslam::slam::vpr {

namespace {
constexpr uint32_t kBlobMagic = 0x53525056;  // "VPRS"
constexpr uint32_t kBlobVersion = 1;
}  // namespace

VprSimple::VprSimple(const VprOptions& options)
    : downscale_(options.downscale > 0 ? options.downscale : 8),
      score_threshold_(options.score_threshold > 0.f ? options.score_threshold : kDefaultScoreThreshold),
      ratio_threshold_(options.ratio_threshold),
      max_entries_(options.max_entries) {}

bool VprSimple::MakeDescriptor(const VprImage& image, std::vector<float>& descriptor) const {
  descriptor.clear();
  if (image.Empty()) {
    return false;
  }

  VprImage thumb = Downscale(image, downscale_);
  if (thumb.Empty()) {
    return false;
  }
  if (thumb_width_ == 0) {
    thumb_width_ = thumb.width;
    thumb_height_ = thumb.height;
  } else if (thumb.width != thumb_width_ || thumb.height != thumb_height_) {
    thumb = Resize(thumb, thumb_width_, thumb_height_);
    if (thumb.Empty()) {
      return false;
    }
  }

  if (!NormalizePixels(thumb, descriptor)) {
    descriptor.clear();
    return false;
  }

  // NormalizePixels leaves a unit variance vector; scale it to unit L2 norm so a query is a plain
  // dot product whose value is the correlation coefficient.
  const float inv_norm = 1.f / std::sqrt(static_cast<float>(descriptor.size()));
  for (float& v : descriptor) {
    v *= inv_norm;
  }
  return true;
}

void VprSimple::AddFrame(KeyFrameId node_id, const VprImage& image) {
  std::vector<float> descriptor;
  if (!MakeDescriptor(image, descriptor)) {
    return;
  }

  const auto it = index_by_node_.find(node_id);
  if (it != index_by_node_.end()) {
    entries_[it->second].descriptor = std::move(descriptor);
    return;
  }

  if (max_entries_ != 0 && entries_.size() >= max_entries_) {
    return;
  }

  index_by_node_.emplace(node_id, entries_.size());
  entries_.push_back(Entry{node_id, std::move(descriptor)});
}

bool VprSimple::HasNode(KeyFrameId node_id) const { return index_by_node_.count(node_id) != 0; }

void VprSimple::RemoveNode(KeyFrameId node_id) {
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

void VprSimple::ShiftNodeIds(KeyFrameId offset) {
  index_by_node_.clear();
  index_by_node_.reserve(entries_.size());
  for (size_t i = 0; i < entries_.size(); ++i) {
    entries_[i].node_id += offset;
    index_by_node_.emplace(entries_[i].node_id, i);
  }
}

VprMatch VprSimple::Query(const VprImage& image) {
  const std::vector<VprMatch> matches = QueryTopK(image, 1);
  return matches.empty() ? VprMatch{} : matches.front();
}

std::vector<VprMatch> VprSimple::QueryTopK(const VprImage& image, size_t max_results) {
  std::vector<VprMatch> matches;
  if (entries_.empty() || max_results == 0) {
    return matches;
  }

  std::vector<float> query;
  if (!MakeDescriptor(image, query)) {
    return matches;
  }

  std::vector<std::pair<float, size_t>> scored;
  scored.reserve(entries_.size());
  for (size_t i = 0; i < entries_.size(); ++i) {
    const auto& entry = entries_[i];
    if (entry.descriptor.size() != query.size()) {
      continue;
    }
    float score = 0.f;
    for (size_t k = 0; k < query.size(); ++k) {
      score += query[k] * entry.descriptor[k];
    }
    scored.emplace_back(score, i);
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

void VprSimple::Serialize(Blob& blob) {
  BlobWriter writer(blob);
  writer.write(kBlobMagic);
  writer.write(kBlobVersion);
  writer.write(static_cast<int32_t>(thumb_width_));
  writer.write(static_cast<int32_t>(thumb_height_));
  writer.write(static_cast<int32_t>(downscale_));
  writer.write(static_cast<uint64_t>(entries_.size()));
  for (const auto& entry : entries_) {
    writer.write(static_cast<uint64_t>(entry.node_id));
    writer.write(static_cast<uint64_t>(entry.descriptor.size()));
    if (!entry.descriptor.empty()) {
      writer.write(entry.descriptor.data(), entry.descriptor.size() * sizeof(float));
    }
  }
}

bool VprSimple::Deserialize(const BlobReader& reader) {
  uint32_t magic = 0;
  uint32_t version = 0;
  if (!reader.read(magic) || !reader.read(version) || magic != kBlobMagic || version != kBlobVersion) {
    return false;
  }

  int32_t thumb_width = 0;
  int32_t thumb_height = 0;
  int32_t downscale = 0;
  uint64_t count = 0;
  if (!reader.read(thumb_width) || !reader.read(thumb_height) || !reader.read(downscale) || !reader.read(count)) {
    return false;
  }

  // An entry is at least a node id, so a count larger than the bytes left cannot be honest.
  constexpr size_t kMinEntryBytes = sizeof(uint64_t) + sizeof(uint64_t);
  if (count > reader.remaining() / kMinEntryBytes) {
    return false;
  }

  entries_.clear();
  index_by_node_.clear();
  entries_.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t node_id = 0;
    Entry entry;
    if (!reader.read(node_id)) {
      return false;
    }
    uint64_t descriptor_size = 0;
    if (!reader.read(descriptor_size)) {
      return false;
    }
    if (descriptor_size > reader.remaining() / sizeof(float)) {
      return false;
    }
    entry.descriptor.resize(static_cast<size_t>(descriptor_size));
    if (descriptor_size != 0 &&
        !reader.read(entry.descriptor.data(), static_cast<size_t>(descriptor_size) * sizeof(float))) {
      return false;
    }
    entry.node_id = static_cast<KeyFrameId>(node_id);
    index_by_node_.emplace(entry.node_id, entries_.size());
    entries_.push_back(std::move(entry));
  }

  thumb_width_ = thumb_width;
  thumb_height_ = thumb_height;
  downscale_ = downscale > 0 ? downscale : downscale_;
  return true;
}

}  // namespace cuvslam::slam::vpr
