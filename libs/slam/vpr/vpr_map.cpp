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

#include "slam/vpr/vpr_map.h"

#include "common/log.h"

namespace cuvslam::slam::vpr {

namespace {
constexpr uint32_t kBlobMagic = 0x504D5056;  // "VPMP"
constexpr uint32_t kBlobVersion = 2;
}  // namespace

VprMap::VprMap(const VprOptions& options) : VprMap(options, CreateVpr(options)) {}

VprMap::VprMap(const VprOptions& options, std::unique_ptr<IVpr> backend)
    : options_(options), backend_(std::move(backend)) {}

VprMap::~VprMap() = default;

const char* VprMap::BackendName() const { return backend_ ? backend_->Name() : "None"; }

size_t VprMap::Size() const { return backend_ ? backend_->Size() : 0; }

void VprMap::OnKeyframeAdded(KeyFrameId node_id, int64_t timestamp_ns) {
  if (!backend_) {
    return;
  }
  auto& info = nodes_[node_id];
  info.timestamp_ns = timestamp_ns;
}

void VprMap::OnKeyframeRemoved(KeyFrameId removed, KeyFrameId merged_into) {
  if (!backend_) {
    return;
  }
  // The survivor keeps its own image; it does not inherit this node's descriptor.
  (void)merged_into;
  nodes_.erase(removed);
  backend_->RemoveNode(removed);
}

bool VprMap::HasImage(KeyFrameId node_id) const {
  const auto it = nodes_.find(node_id);
  return it != nodes_.end() && it->second.has_image;
}

bool VprMap::AddFrame(KeyFrameId node_id, int64_t timestamp_ns, const VprImage& image) {
  if (!backend_ || node_id == InvalidKeyFrameId || image.Empty()) {
    return false;
  }

  auto& info = nodes_[node_id];
  if (info.timestamp_ns == 0) {
    info.timestamp_ns = timestamp_ns;
  }
  if (info.has_image) {
    return false;
  }
  if (options_.max_entries != 0 && backend_->Size() >= options_.max_entries) {
    return false;
  }

  backend_->AddFrame(node_id, image);
  info.has_image = backend_->HasNode(node_id);
  return info.has_image;
}

VprPlace VprMap::Query(const VprImage& image) {
  const std::vector<VprPlace> places = QueryTopK(image, 1);
  return places.empty() ? VprPlace{} : places.front();
}

std::vector<VprPlace> VprMap::QueryTopK(const VprImage& image, size_t max_results) {
  std::vector<VprPlace> places;
  if (!backend_ || image.Empty() || max_results == 0) {
    return places;
  }

  backend_->Finalize();
  const std::vector<VprMatch> matches = backend_->QueryTopK(image, max_results);
  places.reserve(matches.size());
  for (const VprMatch& match : matches) {
    if (!match.found) {
      continue;
    }
    VprPlace place;
    place.found = true;
    place.node_id = match.node_id;
    place.score = match.score;
    const auto it = nodes_.find(match.node_id);
    if (it != nodes_.end()) {
      place.timestamp_ns = it->second.timestamp_ns;
    }
    places.push_back(place);
  }
  return places;
}

bool VprMap::ToBlob(BlobWriter& writer) {
  if (!backend_) {
    return false;
  }

  uint64_t count = 0;
  for (const auto& [node_id, info] : nodes_) {
    if (info.has_image) {
      ++count;
    }
  }
  writer.write(kBlobMagic);
  writer.write(kBlobVersion);
  writer.write(count);
  for (const auto& [node_id, info] : nodes_) {
    if (info.has_image) {
      writer.write(static_cast<uint64_t>(node_id));
      writer.write(info.timestamp_ns);
    }
  }
  backend_->Finalize();
  backend_->Serialize(writer);
  TraceMessage("VPR: wrote %zu %s entries\n", static_cast<size_t>(count), backend_->Name());
  return true;
}

bool VprMap::FromBlob(const BlobReader& reader) {
  if (!backend_) {
    return false;
  }

  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t count = 0;
  if (!reader.read(magic) || !reader.read(version) || !reader.read(count)) {
    return false;
  }
  if (magic != kBlobMagic || version != kBlobVersion) {
    TraceError("VPR: the SLAM map holds no place recognition map of this version\n");
    return false;
  }

  // Reject a count the remaining bytes cannot hold.
  constexpr size_t kEntryBytes = sizeof(uint64_t) + sizeof(int64_t);
  if (count > reader.remaining() / kEntryBytes) {
    TraceError("VPR: the place recognition map declares %llu entries, more than its size allows\n",
               static_cast<unsigned long long>(count));
    return false;
  }

  std::unordered_map<KeyFrameId, NodeInfo> nodes;
  nodes.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t node_id = 0;
    NodeInfo info;
    if (!reader.read(node_id) || !reader.read(info.timestamp_ns)) {
      return false;
    }
    info.has_image = true;
    nodes[static_cast<KeyFrameId>(node_id)] = info;
  }

  if (!backend_->Deserialize(reader)) {
    TraceError("VPR: the place recognition map carries a payload %s cannot read\n", backend_->Name());
    return false;
  }
  nodes_ = std::move(nodes);
  TraceMessage("VPR: read %zu %s entries\n", nodes_.size(), backend_->Name());
  return true;
}

}  // namespace cuvslam::slam::vpr
