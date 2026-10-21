
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

#include <cstdint>
#include <vector>

#include "camera/rig.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

namespace cuvslam::sba {

// sorting:
// - all observations are ordered by key frame
// - within each key frame observations are ordered by point id
struct BundleAdjustmentProblem {
  // last num_fixed_points points are not allowed to move
  std::vector<Vector3T> points;
  int32_t num_fixed_points = 0;

  // // analog of observation_infos but indexed by point_ids
  std::vector<Matrix2T> info_matrix;
  std::vector<Vector2T> observation_xys;
  std::vector<Matrix2T> observation_infos;

  // index into points
  std::vector<int32_t> point_ids;
  // index into key_frame_poses
  std::vector<int32_t> pose_ids;

  // index into rig.camera_from_rig
  std::vector<int8_t> camera_ids;

  // Key frame pose or simply pose is a transformation
  // from world/map space to rig space.
  // Last num_fixed_key_frames are not allowed to move.
  std::vector<Isometry3T> rig_from_world;
  int32_t num_fixed_key_frames = 0;

  camera::Rig rig;

  // solver options
  int32_t max_iterations = 20;
  float robustifier_scale = 1.f;
  // GPU solver: add partial sums in a fixed order, so equal problems give equal bits
  bool deterministic = false;

  // solver statistic
  int32_t iterations = 0;
  float initial_cost = 0;
  float last_cost = 0;
};

}  // namespace cuvslam::sba
