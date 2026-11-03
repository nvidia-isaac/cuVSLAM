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

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <cunls/common/types.h>
#include <cunls/minimizer/levenberg_marquardt_minimizer.h>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "cuda_modules/image_pyramid.h"
#include "imu/imu_preintegration.h"
#include "map/plane_map.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

namespace cuvslam::pnp {

// Per-camera depth data for the RGBD pose estimator.
struct RGBDInfo {
  CameraId depth_id;
  const cuda::GaussianGPUImagePyramid& curr_depth;
};

// Camera-indexed vector sized to rig.num_cameras; nullptr means this camera has no depth for the frame.
using RGBDInfos = std::vector<const RGBDInfo*>;

// Weights and robust-loss scales for each factor type in the optimization.
struct FactorWeights {
  float reprojection = 1.0f;
  float point_to_point = 1e-1f;
  float point_to_plane = 1e-1f;
  float inertial = 2e-3f;

  float robust_reprojection = 1e4f;
  float robust_point_to_point = 5e-2f;
  float robust_point_to_plane = 5e-2f;
  float robust_inertial = 1.0f;
};

// Configuration for the Levenberg-Marquardt RGBD pose solver.
struct MultisensorSolverSettings {
  float lambda = 1e-3;

  int max_iteration = 15;
  bool verbose = false;     // TODO: these fields are currently unused
  float cost_thresh = 0.6;  // TODO: these fields are currently unused

  // Minimum filtered (landmark-matched) observations required to add a reprojection PnP factor.
  // When n < this AND no depth AND no IMU prior, the per-frame solve fails. Failure triggers a
  // full pipeline reset in MultisensorOdometry::track(), which on stereo-only sequences (no depth)
  // cascades: the post-reset map starts with a single keyframe whose landmarks have not yet
  // accumulated tracked observations, so the very next frame fails the same check. Lowering this
  // from 15 to 5 lets transient low-match frames produce a usable PnP-only solve and avoids the
  // reset cascade. Validated on EUROC V1_03 (4.60→1.72 %/m) and V2_03 (1.75→1.59 %/m, plus
  // eliminating ~58000 %/m catastrophic runs that the higher threshold occasionally produced).
  size_t min_observations = 5;

  FactorWeights factor_weights;
};

// Optional inertial prior carrying the previous-frame inertial state plus the IMU preintegration
// rolled since that frame. When supplied to MultisensorPoseEstimator::solve(), the per-frame cuNLS
// problem gains a pinned previous-pose state block and an inertial 2-pose factor whose Delta is
// derived from the IMU preintegration. The previous pose is held constant inside the LM so the
// previous estimate is not perturbed; velocities and biases are baked into the factor's Delta as
// constants and are not refined by this solve.
struct InertialPriorInput {
  // The previous frame's pose, in rig_from_world (matches the LM seed convention used elsewhere).
  Isometry3T prev_rig_from_world = Isometry3T::Identity();
  Vector3T prev_velocity = Vector3T::Zero();
  Vector3T prev_gyro_bias = Vector3T::Zero();
  Vector3T prev_acc_bias = Vector3T::Zero();
  // IMU preintegration accumulated between prev and the frame currently being solved. Borrowed
  // (must outlive solve()).
  const sba_imu::IMUPreintegration* preint = nullptr;
  // World-frame gravity (output of optimize_inertial / map_.set_gravity).
  Vector3T gravity_world = Vector3T::Zero();
  // The IMU-to-rig calibration used to convert between IMU-frame poses (which the preintegration
  // is expressed in) and rig poses (the cuNLS state holds world_from_rig).
  Isometry3T rig_from_imu = Isometry3T::Identity();
};

// Posterior inertial state recovered by the LM solve. Filled when the caller passes a non-null
// InertialPosteriorOutput pointer.
struct InertialPosteriorOutput {
  Vector3T velocity = Vector3T::Zero();
  Vector3T gyro_bias = Vector3T::Zero();
  Vector3T acc_bias = Vector3T::Zero();
};

// cuNLS-based RGBD pose estimator.
// Builds a single optimization problem containing:
//   - PnPFactorBatch                                 (reprojection residuals on rectified images)
//   - PointToPointICPBatch                           (depth-map point-to-point ICP residuals)
//   - PointToPlaneCostFunctionBatch                 (image-space point-to-plane residuals)
//   - ScaledLossFunctionBatch<CauchyLossFunctionBatch> (per-term weights + robust loss)
//   - inertial 2-pose factor (when imu_in is set)   (IMU-derived pose-to-pose constraint)
//
// All coordinates use the OpenCV convention (x-right, y-down, z-forward).
class MultisensorPoseEstimator {
public:
  MultisensorPoseEstimator(const camera::Rig& rig,
                           const MultisensorSolverSettings& settings = MultisensorSolverSettings());
  ~MultisensorPoseEstimator();

  bool solve(Isometry3T& rig_from_world, Matrix6T& static_info_exp,
             const std::vector<camera::Observation>& observations,
             const std::unordered_map<TrackId, Vector3T>& landmarks, const RGBDInfos& depth_infos = {},
             const std::vector<map::Plane>& planes = {}, const std::vector<Vector3T>& depth_points = {},
             const std::optional<InertialPriorInput>& imu_in = std::nullopt,
             InertialPosteriorOutput* imu_out = nullptr) const;

private:
  camera::Rig rig_;
  MultisensorSolverSettings settings_;
  mutable cuda::Stream stream_{false};

  // Pre-allocated GPU buffers (capacity = kMaxObsPerCamera * num_cameras).
  // Avoids cudaMalloc/cudaFree on every solve() call.
  mutable cuda::GPUArrayPinned<float2> gpu_observation_xy_;
  mutable cuda::GPUArrayPinned<float3> d_lm_world_;
  mutable cuda::GPUArrayPinned<cunls::SE3Transform> d_cam_from_rig_;
  mutable cuda::GPUArrayPinned<cunls::SE3Transform> d_pose_;
  mutable cuda::GPUArrayPinned<cuda::GPUPlane> d_planes_;

  // Bumped 5000 → 8000 to give margin for max_per_cam=1000 across multi-camera
  // rigs (e.g. 8 depth cameras). Single-camera ICL-NUIM only fills 1000.
  static constexpr size_t kMaxDepthPoints = 8000;
  mutable cuda::GPUArrayPinned<float3> d_depth_points_{kMaxDepthPoints};

  static constexpr size_t kMaxDepthCameras = camera::Rig::kMaxCameras;
  // Per-depth-camera cam_from_rig extrinsics, uploaded once per solve() and
  // shared by the point-to-point ICP and point-to-plane factors.
  mutable cuda::GPUArrayPinned<cunls::SE3Transform> d_depth_cam_from_rig_{kMaxDepthCameras};

  // Pre-allocated buffers for the optional IMU-between factor (singletons; capacity = 1).
  mutable cuda::GPUArrayPinned<cunls::SE3Transform> d_imu_delta_{1};
  mutable cuda::GPUArrayPinned<cunls::SE3Transform> d_prev_pose_{1};
  mutable cuda::GPUArrayPinned<int> d_const_id_{1};

  using ObservationRef = std::reference_wrapper<const camera::Observation>;
  mutable std::vector<std::vector<ObservationRef>> obs_per_camera_;
  mutable std::vector<camera::Observation> filtered_obs_;
  mutable std::vector<Vector3T> matched_landmarks_;

  // The cuNLS problem of the last solve, kept while the set of factor batches stays the same so the minimizer
  // can reuse its structure (MinimizerOptions::reuse_structure).
  struct CachedProblem;
  mutable std::unique_ptr<CachedProblem> cached_problem_;

  mutable cunls::LevenbergMarquardtMinimizer minimizer_;

  profiler::PnPProfiler::DomainHelper profiler_domain_ =
      profiler::PnPProfiler::DomainHelper("MultisensorPoseEstimator");
};

}  // namespace cuvslam::pnp
