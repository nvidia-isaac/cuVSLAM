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
#include <cunls/factor/between/se3_between_factor_batch.h>
#include <cunls/factor/pnp_factor_batch.h>
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
  lm_opts.base_options.sparse_linear_solver_type = cunls::SparseLinearSolverType::DenseLDLT;
  // The estimator keeps its problem, with fixed factor counts, while the set of factor batches is unchanged.
  lm_opts.base_options.reuse_structure = true;
  return lm_opts;
}

// Padding PnP slots see the world origin through a camera this far behind the rig, so the point has negative
// depth for any pose within that distance of the origin and the PnP factor gives it a zero residual and Jacobian.
constexpr float kPaddingCameraOffset = 1e7f;

// cunls::ScaledLossFunctionBatch fixes its scale at construction, while the per-residual normalization here
// follows the number of valid factors of each solve.
class ScaledCauchyLoss : public cunls::LossFunctionBatch {
public:
  explicit ScaledCauchyLoss(float c) : loss_(1.f, c) {}

  void SetScale(float scale) { scale_ = scale; }

  bool Evaluate(float* s, float3* out, int num_losses, cudaStream_t stream) const override {
    if (num_losses <= 0) {
      return true;
    }
    if (!loss_.Evaluate(s, out, num_losses, stream)) {
      return false;
    }
    cunls::ApplyScaling(scale_, out, num_losses, stream);
    return true;
  }

private:
  cunls::CauchyLossFunctionBatch loss_;
  float scale_ = 1.f;
};

// Which factor batches a solve has. The cached problem is rebuilt only when this changes.
struct ProblemLayout {
  bool pnp = false;
  bool icp = false;
  bool point_to_plane = false;
  bool inertial = false;
  std::vector<CameraId> depth_cameras;

  bool HasFactors() const { return pnp || icp || point_to_plane || inertial; }

  bool operator==(const ProblemLayout& other) const {
    return pnp == other.pnp && icp == other.icp && point_to_plane == other.point_to_plane &&
           inertial == other.inertial && depth_cameras == other.depth_cameras;
  }
};

}  // namespace

struct MultisensorPoseEstimator::CachedProblem {
  ProblemLayout layout;
  // Declared first, destroyed last: it only refers to the batches below.
  cunls::Problem problem;
  std::optional<cunls::SE3StateBatch> pose_state;

  std::optional<cunls::PnPFactorBatch> pnp;
  std::optional<ScaledCauchyLoss> pnp_loss;

  std::vector<std::unique_ptr<math::PointToPointICPBatch>> icp;
  std::optional<ScaledCauchyLoss> icp_loss;

  std::vector<std::unique_ptr<math::PointToPlaneCostFunctionBatch>> point_to_plane;
  std::optional<ScaledCauchyLoss> point_to_plane_loss;
  size_t num_point_to_plane_factors = 0;

  std::optional<cunls::SE3StateBatch> prev_pose_state;
  std::optional<cunls::SE3BetweenFactorBatch> inertial;
  std::optional<cunls::ScaledLossFunctionBatch<cunls::CauchyLossFunctionBatch>> inertial_loss;
};

MultisensorPoseEstimator::MultisensorPoseEstimator(const camera::Rig& rig, const MultisensorSolverSettings& settings)
    : rig_(rig),
      settings_(settings),
      gpu_observation_xy_(kMaxObsPerCamera * rig.num_cameras),
      d_lm_world_(kMaxObsPerCamera * rig.num_cameras),
      d_cam_from_rig_(kMaxObsPerCamera * rig.num_cameras),
      d_pose_(1),
      d_planes_(kMaxPlanes),
      minimizer_(CreateLMMinimizerOptions(settings)) {}

MultisensorPoseEstimator::~MultisensorPoseEstimator() = default;

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
  const size_t num_dp = std::min(depth_points.size(), kMaxDepthPoints);
  const size_t num_gpu_planes = std::min(planes.size(), kMaxPlanes);

  // ── Inertial prediction (host only); it decides whether the solve has an inertial factor ─────────
  // The inertial constraint is a 2-pose SE(3) between (T_i, T_j) with Delta = T_j_pred^{-1} *
  // T_i, where T_j_pred comes from `sba_imu::Pose::predict_pose(prev_pose, gravity, preint)`.
  // T_i is pinned constant so the previous-frame estimate is not perturbed.  Bias states /
  // bias-RW factors / zero-bias-prior factors were tested and removed: they did not influence
  // pose accuracy (the inertial factor uses prev biases as baked-in constants), and removing
  // them left the median ATE unchanged while simplifying the LM problem.
  std::optional<Isometry3T> imu_delta;
  if (imu_in.has_value() && imu_in->preint != nullptr) {
    TRACE_EVENT inertial_ev = profiler_domain_.trace_event("inertial_factor");

    sba_imu::Pose prev_pose_imu;
    prev_pose_imu.w_from_imu = imu_in->prev_rig_from_world.inverse() * imu_in->rig_from_imu;
    prev_pose_imu.velocity = imu_in->prev_velocity;
    prev_pose_imu.gyro_bias = imu_in->prev_gyro_bias;
    prev_pose_imu.acc_bias = imu_in->prev_acc_bias;
    sba_imu::Pose predicted = prev_pose_imu;
    if (prev_pose_imu.predict_pose(imu_in->gravity_world, *imu_in->preint, predicted)) {
      const Isometry3T pred_rig_from_world = (predicted.w_from_imu * imu_in->rig_from_imu.inverse()).inverse();
      // The inertial residual is Log(Delta * T_left^{-1} * T_right) on world_from_rig states.  For
      // zero residual at the IMU prediction we want Delta * T_i^{-1} * T_j_pred = I, i.e.
      // Delta = T_j_pred^{-1} * T_i = pred_rig_from_world * prev_rig_from_world^{-1}.
      imu_delta = pred_rig_from_world * imu_in->prev_rig_from_world.inverse();
    }
  }

  ProblemLayout layout;
  layout.pnp = n > 0 && n >= settings_.min_observations;
  for (CameraId cam_id = 0; cam_id < depth_infos.size(); ++cam_id) {
    if (depth_infos[cam_id] != nullptr) {
      layout.depth_cameras.push_back(cam_id);
    }
  }
  const bool has_depth = !layout.depth_cameras.empty();
  layout.icp = has_depth && num_dp > 0;
  layout.point_to_plane = has_depth && num_gpu_planes > 0;
  layout.inertial = imu_delta.has_value();
  if (!layout.depth_cameras.empty() && !layout.icp && !layout.point_to_plane) {
    layout.depth_cameras.clear();
  }

  // Depth cameras or an inertial prior do not guarantee residuals (no depth points, no planes yet, failed IMU
  // prediction), and cuNLS cannot minimize a problem without factors.
  if (!layout.HasFactors()) {
    static_info_exp.setZero();
    return false;
  }

  const auto& fw = settings_.factor_weights;
  const size_t pnp_capacity = kMaxObsPerCamera * rig_.num_cameras;
  const auto* d_depth_cam_from_rig = reinterpret_cast<const float*>(d_depth_cam_from_rig_.ptr());

  if (!cached_problem_ || !(cached_problem_->layout == layout)) {
    TRACE_EVENT build_ev = profiler_domain_.trace_event("build_problem");
    // Allocated before the old problem is freed, so the minimizer never sees a new problem at the old address.
    auto fresh = std::make_unique<CachedProblem>();
    CachedProblem& cp = *fresh;
    cp.layout = layout;

    cp.pose_state.emplace(reinterpret_cast<const float*>(d_pose_.ptr()), 1);
    cp.pose_state->SetNumActiveStates(1);
    float* pose_ptr = cp.pose_state->StateDevicePtr(0);

    if (layout.pnp) {
      cp.pnp.emplace(reinterpret_cast<const cunls::Vector<2>*>(gpu_observation_xy_.ptr()), d_cam_from_rig_.ptr(),
                     reinterpret_cast<const cunls::Vector<3>*>(d_lm_world_.ptr()), pnp_capacity, 1e-3f);
      cp.pnp->SetNumActiveFactors(pnp_capacity);
      cp.pnp_loss.emplace(fw.robust_reprojection);
      cp.problem.AddFactorBatch(&*cp.pnp, &*cp.pnp_loss, std::vector<float*>(pnp_capacity, pose_ptr));
    }

    if (layout.icp) {
      cp.icp_loss.emplace(fw.robust_point_to_point);
      for (size_t idx = 0; idx < layout.depth_cameras.size(); ++idx) {
        const CameraId cam_id = layout.depth_cameras[idx];
        const CameraParams params = extract_camera_params(rig_, cam_id, *depth_infos[cam_id]);
        cp.icp.push_back(std::make_unique<math::PointToPointICPBatch>(
            reinterpret_cast<const float3*>(d_depth_points_.ptr()), d_depth_cam_from_rig + idx * 16, params.depth_tex,
            params.focal, params.principal, params.img_size, kMaxDepthPoints));
        cp.problem.AddFactorBatch(cp.icp.back().get(), &*cp.icp_loss, std::vector<float*>(kMaxDepthPoints, pose_ptr));
      }
    }

    if (layout.point_to_plane) {
      cp.point_to_plane_loss.emplace(fw.robust_point_to_plane);
      for (size_t idx = 0; idx < layout.depth_cameras.size(); ++idx) {
        const CameraId cam_id = layout.depth_cameras[idx];
        const CameraParams params = extract_camera_params(rig_, cam_id, *depth_infos[cam_id]);
        cp.point_to_plane.push_back(std::make_unique<math::PointToPlaneCostFunctionBatch>(
            d_planes_.ptr(), static_cast<int>(num_gpu_planes), kP2PPixelStride, params.depth_tex, params.focal,
            params.principal, params.img_size, d_depth_cam_from_rig + idx * 16));
        const size_t num_factors = cp.point_to_plane.back()->NumActiveFactors();
        cp.num_point_to_plane_factors += num_factors;
        cp.problem.AddFactorBatch(cp.point_to_plane.back().get(), &*cp.point_to_plane_loss,
                                  std::vector<float*>(num_factors, pose_ptr));
      }
    }

    cp.problem.AddStateBatch(&*cp.pose_state);

    if (layout.inertial) {
      d_const_id_[0] = 0;
      d_const_id_.copy(cuda::ToGPU, stream_.get_stream());
      cp.prev_pose_state.emplace(reinterpret_cast<const float*>(d_prev_pose_.ptr()), 1, d_const_id_.ptr(), 1);
      cp.prev_pose_state->SetNumActiveStates(1, 1);
      cp.problem.AddStateBatch(&*cp.prev_pose_state);

      cp.inertial.emplace(d_imu_delta_.ptr(), 1);
      cp.inertial->SetNumActiveFactors(1);
      // 6 = SE3 DOF, normalising the weight per-residual to match the other factor scaling.
      cp.inertial_loss.emplace(fw.inertial / 6.0f, 1, fw.robust_inertial);
      cp.problem.AddFactorBatch(&*cp.inertial, &*cp.inertial_loss,
                                std::vector<float*>{cp.prev_pose_state->StateDevicePtr(0), pose_ptr});
    }

    cached_problem_ = std::move(fresh);
  }
  CachedProblem& cp = *cached_problem_;

  cudaStream_t s = stream_.get_stream();

  if (layout.pnp) {
    TRACE_EVENT fill_ev = profiler_domain_.trace_event("fill");
    for (size_t i = 0; i < n; i++) {
      const auto& obs = filtered_obs_[i];
      const auto& lm = matched_landmarks_[i];
      gpu_observation_xy_[i] = {obs.xy.x(), obs.xy.y()};
      d_lm_world_[i] = {lm.x(), lm.y(), lm.z()};
      IsometryToSE3Transform(rig_.camera_from_rig[obs.cam_id], d_cam_from_rig_[i]);
    }
    Isometry3T padding_cam_from_rig = Isometry3T::Identity();
    padding_cam_from_rig.translation() = Vector3T(0.f, 0.f, -kPaddingCameraOffset);
    for (size_t i = n; i < pnp_capacity; i++) {
      gpu_observation_xy_[i] = {0.f, 0.f};
      d_lm_world_[i] = {0.f, 0.f, 0.f};
      IsometryToSE3Transform(padding_cam_from_rig, d_cam_from_rig_[i]);
    }

    gpu_observation_xy_.copy_top_n(cuda::ToGPU, pnp_capacity, s);
    d_lm_world_.copy_top_n(cuda::ToGPU, pnp_capacity, s);
    d_cam_from_rig_.copy_top_n(cuda::ToGPU, pnp_capacity, s);
    // Per-residual normalization: scale = weight / num_residuals, so the total loss
    // contribution is independent of the observation count.
    cp.pnp_loss->SetScale(fw.reprojection / static_cast<float>(n));
  }

  // ── Per-depth-camera cam_from_rig extrinsics (shared by ICP and P2P) ──
  if (!layout.depth_cameras.empty()) {
    for (size_t idx = 0; idx < layout.depth_cameras.size(); ++idx) {
      IsometryToSE3Transform(rig_.camera_from_rig[layout.depth_cameras[idx]], d_depth_cam_from_rig_[idx]);
    }
    d_depth_cam_from_rig_.copy_top_n(cuda::ToGPU, layout.depth_cameras.size(), s);
  }

  // ── Point-to-point ICP factors: one per depth camera (level 0 only) ──
  if (layout.icp) {
    TRACE_EVENT icp_ev = profiler_domain_.trace_event("icp");
    for (size_t i = 0; i < num_dp; i++) {
      d_depth_points_[i] = {depth_points[i].x(), depth_points[i].y(), depth_points[i].z()};
    }
    d_depth_points_.copy_top_n(cuda::ToGPU, num_dp, s);

    for (size_t idx = 0; idx < layout.depth_cameras.size(); ++idx) {
      const CameraId cam_id = layout.depth_cameras[idx];
      const CameraParams params = extract_camera_params(rig_, cam_id, *depth_infos[cam_id]);
      cp.icp[idx]->SetFrame(params.depth_tex, params.focal, params.principal, params.img_size, num_dp);
    }
    cp.icp_loss->SetScale(fw.point_to_point / static_cast<float>(num_dp));
  }

  // ── Point-to-plane factors (image-space, per-camera pixel grid) ──
  if (layout.point_to_plane) {
    TRACE_EVENT p2p_ev = profiler_domain_.trace_event("point_to_plane");
    for (size_t i = 0; i < num_gpu_planes; i++) {
      d_planes_[i] = PlaneToGPUPlane(planes[i]);
    }
    d_planes_.copy_top_n(cuda::ToGPU, num_gpu_planes, s);

    for (size_t idx = 0; idx < layout.depth_cameras.size(); ++idx) {
      const CameraId cam_id = layout.depth_cameras[idx];
      const CameraParams params = extract_camera_params(rig_, cam_id, *depth_infos[cam_id]);
      cp.point_to_plane[idx]->SetFrame(params.depth_tex, static_cast<int>(num_gpu_planes));
    }
    cp.point_to_plane_loss->SetScale(fw.point_to_plane / static_cast<float>(cp.num_point_to_plane_factors));
  }

  if (layout.inertial) {
    IsometryToSE3Transform(imu_in->prev_rig_from_world.inverse(), d_prev_pose_[0]);
    IsometryToSE3Transform(*imu_delta, d_imu_delta_[0]);
    d_prev_pose_.copy(cuda::ToGPU, s);
    d_imu_delta_.copy(cuda::ToGPU, s);
  }

  // cuNLS pose states are world_from_rig; the estimator's interface stays rig_from_world.
  IsometryToSE3Transform(rig_from_world.inverse(), d_pose_[0]);
  d_pose_.copy(cuda::ToGPU, s);

  {
    TRACE_EVENT minimize_ev = profiler_domain_.trace_event("minimizer");
    minimizer_.Minimize(s, cp.problem);
  }

  d_pose_.copy(cuda::ToCPU, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  Isometry3T world_from_rig = IsometryFromSE3Transform(d_pose_[0]);

  // Project to SO(3) via SVD to correct numerical drift from iterative optimization. Done before
  // inverting, since Isometry3T::inverse() assumes an orthonormal rotation.
  world_from_rig.linear() = common::CalculateRotationFromSVD(world_from_rig.matrix());
  rig_from_world = world_from_rig.inverse();

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
