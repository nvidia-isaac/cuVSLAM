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

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "common/log.h"

namespace cuvslam::slam::vpr {

namespace {
constexpr uint32_t kFileMagic = 0x504D5056;  // "VPMP"
constexpr uint32_t kFileVersion = 1;
}  // namespace

VprMap::VprMap(const VprOptions& options) : options_(options), backend_(CreateVpr(options)) {}

VprMap::~VprMap() = default;

const char* VprMap::BackendName() const { return backend_ ? backend_->Name() : "None"; }

size_t VprMap::Size() const { return backend_ ? backend_->Size() : 0; }

void VprMap::OnKeyframeAdded(KeyFrameId node_id, int64_t timestamp_ns) {
  if (!backend_ || read_only_) {
    return;
  }
  auto& info = nodes_[node_id];
  info.timestamp_ns = timestamp_ns;
}

void VprMap::OnKeyframeRemoved(KeyFrameId removed, KeyFrameId merged_into) {
  if (!backend_) {
    return;
  }
  // The pose graph reduces itself by merging a keyframe into a neighbor. The descriptor of the
  // node that disappears is no longer addressable, because a match can only be resolved to a pose
  // through a node that still exists, so it is dropped. The surviving node keeps whatever image it
  // already had; it does not inherit this one, since the two frames were taken from different
  // viewpoints and the backend would need the pixels again to re-key the descriptor.
  (void)merged_into;
  nodes_.erase(removed);
  backend_->RemoveNode(removed);
}

bool VprMap::HasImage(KeyFrameId node_id) const {
  const auto it = nodes_.find(node_id);
  return it != nodes_.end() && it->second.has_image;
}

bool VprMap::AddFrame(KeyFrameId node_id, int64_t timestamp_ns, const VprImage& image) {
  if (!backend_ || read_only_ || node_id == InvalidKeyFrameId || image.Empty()) {
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
      place.pose = it->second.pose;
      place.imported = it->second.imported;
    }
    places.push_back(place);
  }
  return places;
}

void VprMap::UpdateNodePose(KeyFrameId node_id, const Isometry3T& pose) {
  if (!backend_) {
    return;
  }
  const auto it = nodes_.find(node_id);
  if (it == nodes_.end() || it->second.imported) {
    return;
  }
  it->second.pose = pose;
}

bool VprMap::Save(const std::string& folder) const {
  if (!backend_) {
    return false;
  }

  if (read_only_) {
    // A loaded map has had its node ids renumbered so they cannot collide with the ids of the live
    // pose graph, and the backend no longer knows which of its entries were renumbered. Writing it
    // out would produce a file whose ids shift again on every load.
    TraceError("VPR: a place recognition map that was loaded from disk cannot be saved again\n");
    return false;
  }

  std::error_code ec;
  std::filesystem::create_directories(folder, ec);

  Blob blob;
  BlobWriter writer(blob);
  writer.write(kFileMagic);
  writer.write(kFileVersion);
  writer.write(static_cast<uint32_t>(options_.type));

  // Only nodes that actually carry a descriptor are written; the rest are pose graph bookkeeping
  // that the loading session has no use for.
  uint64_t count = 0;
  for (const auto& [node_id, info] : nodes_) {
    if (info.has_image) {
      ++count;
    }
  }
  writer.write(count);
  for (const auto& [node_id, info] : nodes_) {
    if (!info.has_image) {
      continue;
    }
    writer.write(static_cast<uint64_t>(node_id));
    writer.write(info.timestamp_ns);
    writer.write_eigen(info.pose);
  }

  backend_->Finalize();
  backend_->Serialize(blob);

  const std::filesystem::path path = std::filesystem::path(folder) / kFileName;
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) {
    TraceError("VPR: cannot write %s\n", path.string().c_str());
    return false;
  }
  file.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
  if (!file) {
    TraceError("VPR: failed writing %s\n", path.string().c_str());
    return false;
  }
  TraceMessage("VPR: saved %zu %s entries to %s\n", static_cast<size_t>(count), backend_->Name(),
               path.string().c_str());
  return true;
}

bool VprMap::Load(const std::string& folder, LoadMode mode) {
  if (!backend_) {
    return false;
  }
  if (backend_->Size() != 0) {
    // Load() renumbers every id the backend holds, so it can only run on an empty map. SLAM loads
    // the map once, while constructing, which is the only supported moment.
    TraceError("VPR: a map can only be loaded into an empty place recognition map\n");
    return false;
  }

  const std::filesystem::path path = std::filesystem::path(folder) / kFileName;
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    TraceError("VPR: cannot read %s\n", path.string().c_str());
    return false;
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);
  Blob blob(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char*>(blob.data()), size)) {
    return false;
  }

  BlobReader reader(blob);
  uint32_t magic = 0;
  uint32_t version = 0;
  uint32_t type = 0;
  uint64_t count = 0;
  if (!reader.read(magic) || !reader.read(version) || !reader.read(type) || !reader.read(count)) {
    return false;
  }
  if (magic != kFileMagic || version != kFileVersion) {
    TraceError("VPR: %s is not a VPR map of this version\n", path.string().c_str());
    return false;
  }
  if (type != static_cast<uint32_t>(options_.type)) {
    TraceError("VPR: %s was built by backend %u but this session runs backend %u\n", path.string().c_str(), type,
               static_cast<uint32_t>(options_.type));
    return false;
  }

  // An entry is at least a node id, a timestamp and a pose, so a larger count cannot be honest.
  constexpr size_t kMinEntryBytes = sizeof(uint64_t) + sizeof(int64_t);
  if (count > reader.remaining() / kMinEntryBytes) {
    TraceError("VPR: %s declares %llu entries, more than its size allows\n", path.string().c_str(),
               static_cast<unsigned long long>(count));
    return false;
  }

  std::vector<std::pair<KeyFrameId, NodeInfo>> loaded;
  loaded.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t node_id = 0;
    NodeInfo info;
    if (!reader.read(node_id) || !reader.read(info.timestamp_ns) || !reader.read_eigen(info.pose)) {
      return false;
    }
    info.has_image = true;
    info.imported = true;
    loaded.emplace_back(static_cast<KeyFrameId>(node_id), info);
  }

  if (!backend_->Deserialize(reader)) {
    TraceError("VPR: %s carries a %s payload this backend cannot read\n", path.string().c_str(), backend_->Name());
    return false;
  }

  const KeyFrameId offset = mode == LoadMode::kStandalone ? kImportedNodeIdBase : 0;
  for (auto& [node_id, info] : loaded) {
    info.imported = mode == LoadMode::kStandalone;
    nodes_[node_id + offset] = info;
  }
  if (offset != 0) {
    backend_->ShiftNodeIds(offset);
  }
  read_only_ = mode == LoadMode::kStandalone;

  TraceMessage("VPR: loaded %zu %s entries from %s\n", static_cast<size_t>(count), backend_->Name(),
               path.string().c_str());
  return true;
}

}  // namespace cuvslam::slam::vpr
