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

#include "common/isometry.h"

#include "slam/vpr/ivpr.h"

namespace cuvslam::slam::vpr {

/// A place recognized from an image.
struct VprPlace {
  bool found = false;                        ///< true when the query matched a map entry
  KeyFrameId node_id = InvalidKeyFrameId;    ///< pose graph node the matched image was observed from
  int64_t timestamp_ns = 0;                  ///< timestamp of the matched map frame
  float score = 0.f;                         ///< similarity in [0, 1]
  Isometry3T pose = Isometry3T::Identity();  ///< world_from_rig of the matched node
  bool imported = false;                     ///< the entry came from a loaded map, not the live pose graph
};

/// The place recognition map: one image descriptor per pose graph node, plus the bookkeeping that
/// ties a descriptor back to a node. See README.md for why a match resolves to a node rather than to
/// a pose, and why a loaded map is read only.
///
/// The only poses stored here are the snapshot Save() writes, for a session that loads the map and
/// has no pose graph for those nodes.
///
/// Thread safety: not thread safe. Callers hold the SLAM lock.
class VprMap {
public:
  /// Node ids of an imported map start here. Live pose graph ids count up from 0 and a session that
  /// reaches this many keyframes has run out of memory long before.
  static constexpr KeyFrameId kImportedNodeIdBase = static_cast<KeyFrameId>(1) << 48;

  /// Name of the file Save() writes inside the map folder.
  static constexpr const char* kFileName = "vpr_map.bin";

  explicit VprMap(const VprOptions& options);
  ~VprMap();

  VprMap(const VprMap&) = delete;
  VprMap& operator=(const VprMap&) = delete;

  /// True when a backend is active. A disabled map accepts every call and does nothing.
  bool Enabled() const { return backend_ != nullptr; }

  /// True when this map came from disk in kStandalone mode: it answers queries and refuses to grow.
  bool ReadOnly() const { return read_only_; }

  /// Backend name, "None" when disabled.
  const char* BackendName() const;

  /// Number of mapped frames.
  size_t Size() const;

  /// Register a pose graph node. Called when SLAM appends a keyframe, before any image is attached.
  void OnKeyframeAdded(KeyFrameId node_id, int64_t timestamp_ns);

  /// Follow a pose graph reduction. `merged_into` may be InvalidKeyFrameId, which drops the entry.
  void OnKeyframeRemoved(KeyFrameId removed, KeyFrameId merged_into);

  /// True when `node_id` already carries an image descriptor.
  bool HasImage(KeyFrameId node_id) const;

  /// Attach `image` to `node_id`. Does nothing when the node already has an image or the map is read
  /// only, so a caller may offer every frame and still end up with exactly one descriptor per node.
  /// Returns true when the image was stored.
  bool AddFrame(KeyFrameId node_id, int64_t timestamp_ns, const VprImage& image);

  /// Find the mapped node that saw `image`. `pose` is the stored snapshot; the caller replaces it
  /// with the live pose graph pose for nodes that are not imported.
  VprPlace Query(const VprImage& image);

  /// The `max_results` best matches for `image`, best first.
  std::vector<VprPlace> QueryTopK(const VprImage& image, size_t max_results);

  /// Record the current pose of `node_id` so Save() can write a self contained map.
  void UpdateNodePose(KeyFrameId node_id, const Isometry3T& pose);

  /// Write the map to `folder`/vpr_map.bin, creating the folder if needed.
  /// Fails when the map itself was loaded from disk, see Load().
  bool Save(const std::string& folder) const;

  /// How a loaded map relates to the session that loads it.
  enum class LoadMode : uint8_t {
    /// The loading session keeps its own pose graph and this map is foreign to it. Node ids are
    /// shifted by kImportedNodeIdBase so they cannot collide, entries keep the poses stored in the
    /// file, and the map is read only. This is what Slam::Config::vpr_map_path does.
    kStandalone,
    /// The loading session IS the map: the pose graph was loaded from the same folder, so the node
    /// ids in the file are its own. Ids are kept as they are, poses resolve through that pose graph,
    /// and the map stays writable. This is what a relocalizer does.
    kWithPoseGraph,
  };

  /// Read a map previously written by Save(), into an empty map.
  /// In kStandalone mode a map that has been loaded cannot be saved again, because the renumbering
  /// is not reversible from the backend's side.
  bool Load(const std::string& folder, LoadMode mode = LoadMode::kStandalone);

private:
  struct NodeInfo {
    int64_t timestamp_ns = 0;
    Isometry3T pose = Isometry3T::Identity();
    bool has_image = false;
    bool imported = false;
  };

  VprOptions options_;
  std::unique_ptr<IVpr> backend_;
  std::unordered_map<KeyFrameId, NodeInfo> nodes_;
  bool read_only_ = false;
};

}  // namespace cuvslam::slam::vpr
