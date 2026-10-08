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

#include "pnp/multisensor_pose_estimator.h"

#include <algorithm>
#include <optional>

#include <cunls/common/device_vector.h>
#include <cunls/common/log.h>
#include <cunls/factor/pnp_factor_batch.h>
#include <cunls/factor/se3_between_factor_batch.h>
#include <cunls/minimizer/levenberg_marquardt_minimizer.h>
#include <cunls/minimizer/problem.h>
#include <cunls/robustifier/cauchy_loss_function_batch.h>
#include <cunls/robustifier/scaled_loss_function_batch.h>
#include <cunls/state/se3_state_batch.h>

#include "common/rotation_utils.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "imu/imu_sba_problem.h"
#include "math/point_to_plane_factor.h"
#include "math/point_to_point_icp_factor.h"

namespace cuvslam::pnp {

namespace {

const size_t kMaxObsPerCamera = 270;
const size_t kMaxPlanes = 64;
constexpr int kP2PPixelStride = 8;

void IsometryToSE3Transform(const Isometry3T& iso, cunls::SE3Transform& out) {
  const auto& m = iso.matrix();
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      out[r * 4 + c] = m(r, c);
    }
  }
}

Isometry3T IsometryFromSE3Transform(const cunls::SE3Transform& se3) {
  Isometry3T iso = Isometry3T::Identity();
  auto& m = iso.matrix();
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      m(r, c) = se3[r * 4 + c];
    }
  }
  return iso;
}

void FilterObservations(const std::vector<camera::Observation>& observations,
                        const std::unordered_map<TrackId, Vector3T>& landmarks, int num_cameras,
                        std::vector<std::vector<std::reference_wrapper<const camera::Observation>>>& obs_per_camera,
                        std::vector<camera::Observation>& filtered_obs, std::vector<Vector3T>& matched_landmarks) {
  using obs_ref = std::reference_wrapper<const camera::Observation>;
  obs_per_camera.resize(num_cameras);
  for (auto& obs_vec : obs_per_camera) {
    obs_vec.clear();
  }

  for (const auto& obs : observations) {
    if (landmarks.count(obs.id)) {
      obs_per_camera[obs.cam_id].emplace_back(std::cref(obs));
    }
  }

  filtered_obs.clear();
  filtered_obs.reserve(observations.size());
  matched_landmarks.clear();
  matched_landmarks.reserve(observations.size());

  for (auto& obs_vec : obs_per_camera) {
    std::sort(obs_vec.begin(), obs_vec.end(),
              [](const obs_ref& a, const obs_ref& b) { return a.get().id < b.get().id; });

    const size_t cap = std::min(kMaxObsPerCamera, obs_vec.size());
    for (size_t i = 0; i < cap; i++) {
      const auto& obs = obs_vec[i].get();
      filtered_obs.push_back(obs);
      matched_landmarks.push_back(landmarks.at(obs.id));
    }
  }
}

struct CameraParams {
  float2 focal;
  float2 principal;
  int2 img_size;
  cudaTextureObject_t depth_tex;
};

CameraParams extract_camera_params(const camera::Rig& rig, CameraId cam_id, const pnp::RGBDInfo& info) {
  const auto& intrinsics = *rig.intrinsics[cam_id];
  Vector2T focal_v = intrinsics.getFocal();
  Vector2T principal_v = intrinsics.getPrincipal();
  return {{focal_v.x(), focal_v.y()},
          {principal_v.x(), principal_v.y()},
          make_int2(static_cast<int>(info.curr_depth[0].cols()), static_cast<int>(info.curr_depth[0].rows())),
          info.curr_depth[0].get_texture_filter_point()};
}

// Convert Plane to GPU representation. Plane equation: n . x + d = 0, so d = -(n . centroid).
cuda::GPUPlane PlaneToGPUPlane(const map::Plane& pl) {
  cuda::GPUPlane gp;
  gp.normal = {pl.normal.x(), pl.normal.y(), pl.normal.z()};
  gp.d = -pl.normal.dot(pl.centroid);
  return gp;
}

cunls::LevenbergMarquardtMinimizerOptions CreateLMMinimizerOptions(const MultisensorSolverSettings& settings) {
  cunls::LevenbergMarquardtMinimizerOptions lm_opts;
  lm_opts.base_options.max_num_iterations = settings.max_iteration;
  lm_opts.base_options.column_scaling = cunls::ColumnScaling::None;
  lm_opts.initial_lambda = settings.lambda;
  lm_opts.base_options.state_tolerance = 1e-5f;
  lm_opts.base_options.max_consecutive_rejected_steps = 5;
  lm_opts.base_options.cost_tolerance = 1e-5f;
  lm_opts.base_options.sparse_linear_solver_type = cunls::SparseLinearSolverType::DenseQR;
  return lm_opts;
}

}  // namespace

MultisensorPoseEstimator::MultisensorPoseEstimator(const camera::Rig& rig, const MultisensorSolverSettings& settings)
    : rig_(rig),
      settings_(settings),
      gpu_observation_xy_(kMaxObsPerCamera * rig.num_cameras),
      d_lm_world_(kMaxObsPerCamera * rig.num_cameras),
      d_cam_from_rig_(kMaxObsPerCamera * rig.num_cameras),
      d_pose_(1),
      d_planes_(kMaxPlanes),
      minimizer_(CreateLMMinimizerOptions(settings)) {}

bool MultisensorPoseEstimator::solve(Isometry3T& rig_from_world, Matrix6T& static_info_exp,
                                     const std::vector<camera::Observation>& observations,
                                     const std::unordered_map<TrackId, Vector3T>& landmarks,
                                     const RGBDInfos& depth_infos, const std::vector<map::Plane>& planes,
                                     const std::vector<Vector3T>& depth_points,
                                     const std::optional<InertialPriorInput>& imu_in,
                                     InertialPosteriorOutput* imu_out) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("solve");

  {
    TRACE_EVENT filter_ev = profiler_domain_.trace_event("filter");
    FilterObservations(observations, landmarks, rig_.num_cameras, obs_per_camera_, filtered_obs_, matched_landmarks_);
  }

  const size_t n = filtered_obs_.size();

  const size_t num_depth = std::count_if(depth_infos.begin(), depth_infos.end(),
                                         [](const RGBDInfo* depth_info) { return depth_info != nullptr; });

  if (n < settings_.min_observations && num_depth == 0 && !imu_in.has_value()) {
    static_info_exp.setZero();
    return false;
  }

  cudaStream_t s = stream_.get_stream();

  if (n >= settings_.min_observations) {
    TRACE_EVENT fill_ev = profiler_domain_.trace_event("fill");
    for (size_t i = 0; i < n; i++) {
      const auto& obs = filtered_obs_[i];
      const auto& lm = matched_landmarks_[i];
      gpu_observation_xy_[i] = {obs.xy.x(), obs.xy.y()};
      d_lm_world_[i] = {lm.x(), lm.y(), lm.z()};
      IsometryToSE3Transform(rig_.camera_from_rig[obs.cam_id], d_cam_from_rig_[i]);
    }

    gpu_observation_xy_.copy_top_n(cuda::ToGPU, n, s);
    d_lm_world_.copy_top_n(cuda::ToGPU, n, s);
    d_cam_from_rig_.copy_top_n(cuda::ToGPU, n, s);
  }

  using PnPFactor = cunls::PnPFactorBatch;
  using ScaledCauchy = cunls::ScaledLossFunctionBatch<cunls::CauchyLossFunctionBatch>;

  cunls::SE3Transform rig_from_world_se3;
  IsometryToSE3Transform(rig_from_world, rig_from_world_se3);

  d_pose_[0] = rig_from_world_se3;
  d_pose_.copy(cuda::ToGPU, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  cunls::SE3StateBatch se3_states(cublas_handle_, reinterpret_cast<const float*>(d_pose_.ptr()), 1);

  const auto& fw = settings_.factor_weights;

  std::optional<PnPFactor> pnp_cost;
  std::optional<ScaledCauchy> loss_reproj;
  pose_state_ptrs_.clear();

  cunls::Problem problem;
  // Depth cameras or an inertial prior do not guarantee residuals (no depth points, no planes yet, failed IMU
  // prediction), and cuNLS cannot minimize a problem without factors.
  size_t num_factor_batches = 0;

  if (n >= settings_.min_observations) {
    auto* obs_ptr = reinterpret_cast<const cunls::Vector<2>*>(gpu_observation_xy_.ptr());
    const auto* lm_world_ptr = reinterpret_cast<const cunls::Vector<3>*>(d_lm_world_.ptr());
    const auto* cam_from_rig_ptr = d_cam_from_rig_.ptr();
    const float z_threshold = 1e-3f;

    pnp_cost.emplace(obs_ptr, cam_from_rig_ptr, lm_world_ptr, n, z_threshold);
    // Per-residual normalization: scale = weight / num_residuals, so the total loss
    // contribution is independent of the observation count.
    loss_reproj.emplace(fw.reprojection / static_cast<float>(n), 1, fw.robust_reprojection);
    pose_state_ptrs_.assign(n, se3_states.StateBlockDevicePtr(0));
    problem.AddFactorBatch(&*pnp_cost, &*loss_reproj, pose_state_ptrs_);
    ++num_factor_batches;
  }

  problem.AddStateBatch(&se3_states);

  // ── Upload per-depth-camera cam_from_rig extrinsics (shared by ICP and P2P) ──
  if (num_depth > 0) {
    size_t idx = 0;
    for (CameraId cam_id = 0; cam_id < depth_infos.size(); ++cam_id) {
      const RGBDInfo* depth_info = depth_infos[cam_id];
      if (depth_info == nullptr) {
        continue;
      }
      IsometryToSE3Transform(rig_.camera_from_rig[cam_id], d_depth_cam_from_rig_[idx]);
      ++idx;
    }
    d_depth_cam_from_rig_.copy_top_n(cuda::ToGPU, num_depth, s);
  }

  // ── Point-to-point ICP factors: one per depth camera (level 0 only) ──
  const size_t num_dp = std::min(depth_points.size(), kMaxDepthPoints);
  std::vector<std::optional<math::PointToPointICPBatch>> icp_costs(num_depth);
  std::optional<ScaledCauchy> loss_icp_scaled;
  icp_state_ptrs_.clear();

  if (num_depth > 0 && num_dp > 0) {
    TRACE_EVENT icp_ev = profiler_domain_.trace_event("icp");

    for (size_t i = 0; i < num_dp; i++) {
      d_depth_points_[i] = {depth_points[i].x(), depth_points[i].y(), depth_points[i].z()};
    }
    d_depth_points_.copy_top_n(cuda::ToGPU, num_dp, s);

    const float icp_scale = fw.point_to_point / static_cast<float>(num_dp);
    loss_icp_scaled.emplace(icp_scale, 1, fw.robust_point_to_point);
    icp_state_ptrs_.assign(num_dp, se3_states.StateBlockDevicePtr(0));

    size_t idx = 0;
    for (CameraId cam_id = 0; cam_id < depth_infos.size(); ++cam_id) {
      const RGBDInfo* depth_info = depth_infos[cam_id];
      if (depth_info == nullptr) {
        continue;
      }
      auto cp = extract_camera_params(rig_, cam_id, *depth_info);

      const float* d_cfr_ptr = reinterpret_cast<const float*>(d_depth_cam_from_rig_.ptr()) + idx * 16;
      icp_costs[idx].emplace(reinterpret_cast<const float3*>(d_depth_points_.ptr()), d_cfr_ptr, cp.depth_tex, cp.focal,
                             cp.principal, cp.img_size, num_dp);
      problem.AddFactorBatch(&*icp_costs[idx], &*loss_icp_scaled, icp_state_ptrs_);
      ++num_factor_batches;

      ++idx;
    }
  }

  // ── Point-to-plane factors (image-space, per-camera pixel grid) ──
  const size_t num_gpu_planes = std::min(planes.size(), kMaxPlanes);
  std::vector<std::optional<math::PointToPlaneCostFunctionBatch>> p2p_costs(num_depth);
  std::vector<std::optional<ScaledCauchy>> loss_p2p(num_depth);
  p2p_state_ptrs_.clear();
  p2p_state_ptrs_.resize(num_depth);

  if (!planes.empty() && num_depth > 0) {
    TRACE_EVENT p2p_ev = profiler_domain_.trace_event("point_to_plane");

    for (size_t i = 0; i < num_gpu_planes; i++) {
      d_planes_[i] = PlaneToGPUPlane(planes[i]);
    }
    d_planes_.copy_top_n(cuda::ToGPU, num_gpu_planes, s);

    const int stride = kP2PPixelStride;
    size_t total_factors = 0;

    size_t cam_idx = 0;
    for (CameraId cam_id = 0; cam_id < depth_infos.size(); ++cam_id) {
      const RGBDInfo* depth_info = depth_infos[cam_id];
      if (depth_info == nullptr) {
        continue;
      }
      auto cp = extract_camera_params(rig_, cam_id, *depth_info);

      const int grid_w = (cp.img_size.x + stride - 1) / stride;
      const int grid_h = (cp.img_size.y + stride - 1) / stride;
      const size_t num_factors = static_cast<size_t>(grid_w) * grid_h;
      total_factors += num_factors;

      const float* d_cfr_ptr = reinterpret_cast<const float*>(d_depth_cam_from_rig_.ptr()) + cam_idx * 16;
      p2p_costs[cam_idx].emplace(d_planes_.ptr(), static_cast<int>(num_gpu_planes), stride, cp.depth_tex, cp.focal,
                                 cp.principal, cp.img_size, d_cfr_ptr);

      ++cam_idx;
    }

    if (total_factors > 0) {
      const float p2p_scale = fw.point_to_plane / static_cast<float>(total_factors);
      cam_idx = 0;
      for (CameraId cam_id = 0; cam_id < depth_infos.size(); ++cam_id) {
        if (depth_infos[cam_id] == nullptr) {
          continue;
        }
        const size_t nf = p2p_costs[cam_idx]->NumFactors();
        loss_p2p[cam_idx].emplace(p2p_scale, 1, fw.robust_point_to_plane);
        p2p_state_ptrs_[cam_idx].assign(nf, se3_states.StateBlockDevicePtr(0));
        problem.AddFactorBatch(&*p2p_costs[cam_idx], &*loss_p2p[cam_idx], p2p_state_ptrs_[cam_idx]);
        ++num_factor_batches;
        ++cam_idx;
      }
    }
  }

  // ── Inertial factor (IMU-derived 2-pose constraint with Delta from preintegration) ─────────
  // The inertial constraint is a 2-pose SE(3) between (T_i, T_j) with Delta = T_j_pred^{-1} *
  // T_i, where T_j_pred comes from `sba_imu::Pose::predict_pose(prev_pose, gravity, preint)`.
  // T_i is pinned constant so the previous-frame estimate is not perturbed.  Bias states /
  // bias-RW factors / zero-bias-prior factors were tested and removed: they did not influence
  // pose accuracy (the inertial factor uses prev biases as baked-in constants), and removing
  // them left the median ATE unchanged while simplifying the LM problem.
  std::unique_ptr<cunls::SE3StateBatch> prev_pose_state;
  std::unique_ptr<cunls::SE3BetweenFactorBatch> inertial_cost;
  std::optional<ScaledCauchy> inertial_loss;

  if (imu_in.has_value() && imu_in->preint != nullptr) {
    TRACE_EVENT inertial_ev = profiler_domain_.trace_event("inertial_factor");

    sba_imu::Pose prev_pose_imu;
    prev_pose_imu.w_from_imu = imu_in->prev_rig_from_world.inverse() * imu_in->rig_from_imu;
    prev_pose_imu.velocity = imu_in->prev_velocity;
    prev_pose_imu.gyro_bias = imu_in->prev_gyro_bias;
    prev_pose_imu.acc_bias = imu_in->prev_acc_bias;
    sba_imu::Pose predicted = prev_pose_imu;
    const bool pred_ok = prev_pose_imu.predict_pose(imu_in->gravity_world, *imu_in->preint, predicted);

    if (pred_ok) {
      const Isometry3T pred_rig_from_world = (predicted.w_from_imu * imu_in->rig_from_imu.inverse()).inverse();
      // The inertial residual is Log(Delta * T_left^{-1} * T_right).  For zero residual at the
      // IMU prediction we want Delta * T_i^{-1} * T_j_pred = I, i.e. Delta = T_j_pred^{-1} * T_i.
      const Isometry3T delta = pred_rig_from_world.inverse() * imu_in->prev_rig_from_world;

      cunls::SE3Transform prev_se3, delta_se3;
      IsometryToSE3Transform(imu_in->prev_rig_from_world, prev_se3);
      IsometryToSE3Transform(delta, delta_se3);
      d_prev_pose_[0] = prev_se3;
      d_imu_delta_[0] = delta_se3;
      d_const_id_[0] = 0;
      d_prev_pose_.copy(cuda::ToGPU, s);
      d_imu_delta_.copy(cuda::ToGPU, s);
      d_const_id_.copy(cuda::ToGPU, s);
      CUDA_CHECK(cudaStreamSynchronize(s));

      prev_pose_state = std::make_unique<cunls::SE3StateBatch>(
          cublas_handle_, reinterpret_cast<const float*>(d_prev_pose_.ptr()), 1, d_const_id_.ptr(), 1);
      problem.AddStateBatch(prev_pose_state.get());

      inertial_cost = std::make_unique<cunls::SE3BetweenFactorBatch>(d_imu_delta_.ptr(), 1);
      // 6 = SE3 DOF, normalising the weight per-residual to match the other factor scaling.
      inertial_loss.emplace(fw.inertial / 6.0f, 1, fw.robust_inertial);
      inertial_state_ptrs_.clear();
      inertial_state_ptrs_.push_back(prev_pose_state->StateBlockDevicePtr(0));
      inertial_state_ptrs_.push_back(se3_states.StateBlockDevicePtr(0));
      problem.AddFactorBatch(inertial_cost.get(), &*inertial_loss, inertial_state_ptrs_);
      ++num_factor_batches;
    }
  }

  if (num_factor_batches == 0) {
    static_info_exp.setZero();
    return false;
  }

  {
    TRACE_EVENT minimize_ev = profiler_domain_.trace_event("minimizer");
    minimizer_.Minimize(s, problem);
  }

  d_pose_.copy(cuda::ToCPU, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  rig_from_world = IsometryFromSE3Transform(d_pose_[0]);

  // Project to SO(3) via SVD to correct numerical drift from iterative optimization.
  rig_from_world.linear() = common::CalculateRotationFromSVD(rig_from_world.matrix());

  // Posterior inertial state.  The inertial factor uses prev_velocity / biases as baked-in
  // constants, so this LM does not refine them — surface the input values unchanged and let the
  // caller (ImuFusionContext) recompute velocity from the visual translation delta.
  if (imu_out != nullptr && imu_in.has_value()) {
    imu_out->velocity = imu_in->prev_velocity;
    imu_out->gyro_bias = imu_in->prev_gyro_bias;
    imu_out->acc_bias = imu_in->prev_acc_bias;
  }

  static_info_exp = Matrix6T::Identity();
  return true;
}

}  // namespace cuvslam::pnp
