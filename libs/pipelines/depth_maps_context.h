
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

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "map/depth_point_map.h"
#include "map/plane_map.h"
#include "pnp/multisensor_pose_estimator.h"

#include "pipelines/track.h"
#include "pipelines/track_lifter.h"
#include "pipelines/triangulator.h"

namespace cuvslam::pipelines {

// DepthMapsContext bundles the three depth/triangulation primitives used by the multisensor
// pipeline so the orchestrating solver can speak in terms of high-level operations:
//   * MulticamTriangulator   -- triangulate landmarks across cameras for new keyframes
//   * map::DepthPointMap     -- sparse map-frame depth points for the point-to-point ICP factor
//   * map::PlaneMap          -- planes detected at keyframes for the point-to-plane factor
//
// All operations are no-ops on a rig without depth cameras: plane and depth-point updates
// gracefully short-circuit when `depth_infos` is empty, and `build_keyframe_landmarks` falls back
// to multi-camera triangulation alone.
class DepthMapsContext {
public:
  explicit DepthMapsContext(const camera::Rig& rig);

  void reset();

  const std::vector<map::Plane>& planes() const { return plane_map_.get_planes(); }
  const std::vector<Vector3T>& depth_points() const { return depth_point_map_.get_points(); }

  // Refreshes the depth-point map after the per-frame pose has been finalized so the next frame's
  // ICP factor is anchored in the latest world frame. Returns true when new depth points were
  // added (so callers can log/visualize them).
  bool update_post_solve(const pnp::RGBDInfos& depth_infos, const Isometry3T& world_from_rig);

  // Builds the landmarks attached to a new keyframe. Always runs the multi-camera triangulator;
  // additionally lifts mono tracks via per-camera depth when depth is available.
  std::vector<Landmark> build_keyframe_landmarks(const Isometry3T& world_from_rig,
                                                 const std::vector<camera::Observation>& observations,
                                                 const pnp::RGBDInfos& depth_infos);

  // Updates the plane map at keyframe boundaries. No-op when depth is not available.
  void update_at_keyframe(const pnp::RGBDInfos& depth_infos, const Isometry3T& world_from_rig);

private:
  // Builds DepthCameraInfo records for cameras with depth so the plane / depth-point GPU passes
  // can run with up-to-date world poses. Empty input yields an empty result.
  std::vector<map::DepthCameraInfo> build_depth_camera_infos(const pnp::RGBDInfos& depth_infos,
                                                             const Isometry3T& world_from_rig) const;

  const camera::Rig& rig_;
  MulticamTriangulator triangulator_;
  TrackLifter track_lifter_;
  map::PlaneMap plane_map_;
  map::DepthPointMap depth_point_map_;
};

}  // namespace cuvslam::pipelines
