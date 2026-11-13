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

#include "common/isometry.h"

#include "slam/common/blob.h"
#include "slam/vpr/ivpr.h"

namespace cuvslam::slam::vpr {

struct VprPlace {
  bool found = false;
  KeyFrameId node_id = InvalidKeyFrameId;    ///< matched pose graph node
  int64_t timestamp_ns = 0;                  ///< of the matched map frame
  float score = 0.f;                         ///< similarity in [0, 1]
  Isometry3T pose = Isometry3T::Identity();  ///< world_from_rig of the matched node, set from the pose graph
};

/// One image descriptor per pose graph node, see README.md.
/// Not thread safe; callers hold the SLAM lock.
class VprMap {
public:
  explicit VprMap(const VprOptions& options);

  /// Uses `backend`, e.g. a test stand-in.
  VprMap(const VprOptions& options, std::unique_ptr<IVpr> backend);
  ~VprMap();

  VprMap(const VprMap&) = delete;
  VprMap& operator=(const VprMap&) = delete;

  /// A disabled map accepts every call and does nothing.
  bool Enabled() const { return backend_ != nullptr; }

  /// Backend name, "None" when disabled.
  const char* BackendName() const;

  /// Number of mapped frames.
  size_t Size() const;

  /// Register a new keyframe, before any image is attached.
  void OnKeyframeAdded(KeyFrameId node_id, int64_t timestamp_ns);

  /// Drop the descriptor of a keyframe the pose graph removed.
  void OnKeyframeRemoved(KeyFrameId removed, KeyFrameId merged_into);

  bool HasImage(KeyFrameId node_id) const;

  /// Store `image` for `node_id` unless it has one already; true if stored.
  bool AddFrame(KeyFrameId node_id, int64_t timestamp_ns, const VprImage& image);

  /// Best match for `image`; the caller sets `pose`.
  VprPlace Query(const VprImage& image);

  /// The `max_results` best matches for `image`, best first.
  std::vector<VprPlace> QueryTopK(const VprImage& image, size_t max_results);

  /// Write the map, e.g. into the SLAM database.
  bool ToBlob(BlobWriter& writer);

  /// Replace the map with one ToBlob() wrote; false leaves it unchanged.
  bool FromBlob(const BlobReader& reader);

private:
  struct NodeInfo {
    int64_t timestamp_ns = 0;
    bool has_image = false;
  };

  VprOptions options_;
  std::unique_ptr<IVpr> backend_;
  std::unordered_map<KeyFrameId, NodeInfo> nodes_;
};

}  // namespace cuvslam::slam::vpr
