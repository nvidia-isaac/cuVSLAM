
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

#include "pipelines/imu_fusion_context.h"

#include "common/include_eigen.h"
#include "common/vector_2t.h"
#include "epipolar/camera_selection.h"

#include "pipelines/service_sba.h"

namespace cuvslam::pipelines {

using Vector15T = Eigen::Matrix<float, 15, 1>;

namespace {

// Reports whether the IMU stream covers the frame interval reliably. Counts samples in the
// preintegration vs the nominal expected count (calib.frequency * dT), absorbing a small jitter
// margin so normal sub-sample phase offsets don't flag as drops. The previous check truncated the
// expected count to int and divided by it, so on short frame intervals (e.g. 100 Hz IMU with
// 30 fps frames, expected count ~3) a single sample landing just outside [start_ts, end_ts] read
// as a 33% drop — even though the IMU stream was healthy.
bool check_imu_drops(sba_imu::IMUPreintegration& preint, const imu::ImuCalibration& calib, int64_t start_time_ns,
                     int64_t end_time_ns) {
  // First frame: prev_pose_ts_ns is sentinel (-1); no integration interval to validate.
  if (start_time_ns < 0 || end_time_ns <= start_time_ns) {
    return true;
  }
  const float freq = calib.frequency();
  if (freq <= 0.f) {
    return true;
  }
  const float dT_s = static_cast<float>((end_time_ns - start_time_ns) * 1e-9f);
  const float expected = freq * dT_s;
  const float actual = static_cast<float>(preint.size());
  // Allow up to ~2 samples of slack for phase/jitter (IMU timestamps are rarely periodic to within
  // a fraction of the nominal period — see e.g. multi_rgbd indoor where sample gaps span 5–20 ms
  // around a 10 ms nominal). Bound the slack by half the expected count so short intervals can't
  // hide a genuine dropout (otherwise e.g. expected=2 with actual=0 would still report healthy).
  constexpr float kJitterMarginSamples = 2.f;
  const float effective_jitter = std::min(kJitterMarginSamples, expected * 0.5f);
  const float missing = std::max(0.f, expected - actual - effective_jitter);
  const float drop_ratio = missing / expected;
  if (drop_ratio > 0.1f) {
    TraceWarning("Lost IMU msgs: %d, Frame time delta = %f [s], drop ratio = %f [%]", static_cast<int>(missing + 0.5f),
                 dT_s, drop_ratio * 100);
    return false;
  }
  return true;
}

// Run the cuNLS MultisensorPoseEstimator with the new IMU preintegration factor. The factor jointly
// optimizes pose + velocity + biases on the curr side, anchored to the prev state via per-block
// constants on the prev side. Velocity and biases are returned via InertialPosteriorOutput, so
// the caller no longer needs to recompute velocity from the visual translation delta.
bool run_rgbd_inertial_solve(const pnp::MultisensorPoseEstimator& estimator, const imu::ImuCalibration& calib,
                             const std::unordered_map<TrackId, Vector3T>& landmarks,
                             const std::vector<camera::Observation>& observations, const pnp::RGBDInfos& depth_infos,
                             const std::vector<map::Plane>& planes, const std::vector<Vector3T>& depth_points,
                             const Vector3T& gravity, sba_imu::Pose& prev_pose, sba_imu::Pose& curr_pose) {
  // Seed the LM with an IMU-integrated pose prediction for fast convergence.
  sba_imu::Pose predicted = curr_pose;
  if (!prev_pose.predict_pose(gravity, prev_pose.preintegration, predicted)) {
    return false;
  }
  const Isometry3T imu_from_rig = calib.rig_from_imu().inverse();
  Isometry3T rig_from_world = (predicted.w_from_imu * imu_from_rig).inverse();
  Matrix6T info;

  pnp::InertialPriorInput imu_in;
  imu_in.prev_rig_from_world = (prev_pose.w_from_imu * imu_from_rig).inverse();
  imu_in.prev_velocity = prev_pose.velocity;
  imu_in.prev_gyro_bias = prev_pose.gyro_bias;
  imu_in.prev_acc_bias = prev_pose.acc_bias;
  imu_in.preint = &prev_pose.preintegration;
  imu_in.gravity_world = gravity;
  imu_in.rig_from_imu = calib.rig_from_imu();

  pnp::InertialPosteriorOutput imu_out;
  const bool ok = estimator.solve(rig_from_world, info, observations, landmarks, depth_infos, planes, depth_points,
                                  imu_in, &imu_out);
  if (!ok) {
    return false;
  }

  curr_pose.w_from_imu = rig_from_world.inverse() * imu_from_rig.inverse();
  curr_pose.velocity = imu_out.velocity;
  curr_pose.gyro_bias = imu_out.gyro_bias;
  curr_pose.acc_bias = imu_out.acc_bias;
  curr_pose.info.setZero();
  curr_pose.info.block<6, 6>(0, 0) = info;
  return true;
}

}  // namespace

ImuFusionContext::ImuFusionContext(map::UnifiedMap& map, const camera::Rig& rig, const imu::ImuCalibration& calib,
                                   const pnp::MultisensorPoseEstimator& pose_estimator, bool use_gpu_bundler)
    : map_(map),
      rig_(rig),
      calib_(calib),
      pose_estimator_(pose_estimator),
      imu_storage_(1e5),
      optimizer_(10),
      init_bundler_(calib, use_gpu_bundler),
      sm_() {
  prev_pose_.w_from_imu = calib_.rig_from_imu();
}

void ImuFusionContext::register_gravity_callback() {
  sm_.set_gravity_init_fn([this](size_t num_kfs) { return gravity_estimation_callback(num_kfs); });
}

void ImuFusionContext::reset() {
  imu_storage_.clear();
  sm_.reset();
  is_first_run_ = true;
  integrated_ = false;
  prev_pose_ts_ns_ = -1;

  prev_pose_ = sba_imu::Pose{};
  prev_pose_.w_from_imu = calib_.rig_from_imu();
  curr_pose_ = sba_imu::Pose{};
  last_valid_pose_ = sba_imu::Pose{};
  integ_kf_ = sba_imu::Pose{};

  curr_pose_.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  prev_pose_.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  last_valid_pose_.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  integ_kf_.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  last_kf_preint_.Initialize(Vector3T::Zero(), Vector3T::Zero());
  last_frame_preint_.Initialize(Vector3T::Zero(), Vector3T::Zero());
}

void ImuFusionContext::add_measurement(const imu::ImuMeasurement& m) {
  imu_storage_.push_back(m);

  last_frame_preint_.IntegrateNewMeasurement(calib_, m);
  last_valid_pose_.preintegration.IntegrateNewMeasurement(calib_, m);
  last_kf_preint_.IntegrateNewMeasurement(calib_, m);
  integ_kf_.preintegration.IntegrateNewMeasurement(calib_, m);
}

ImuFusionContext::FramePrep ImuFusionContext::prepare_frame(int64_t time_ns) {
  if (is_first_run_) {
    Vector15T diag = Vector15T::Zero();
    diag.segment<6>(0).setConstant(1e6);
    prev_pose_.info = diag.asDiagonal();

    last_valid_pose_ = prev_pose_;
    integ_kf_ = prev_pose_;

    last_valid_pose_.preintegration = last_frame_preint_;
    integ_kf_.preintegration = last_frame_preint_;
    last_kf_preint_ = last_frame_preint_;
  }

  prev_pose_.preintegration = last_frame_preint_;
  curr_pose_ = prev_pose_;

  FramePrep prep;
  prep.imu_state = sm_.state();
  prep.predicted_world_from_rig = prev_pose_.w_from_imu * calib_.rig_from_imu().inverse();
  prep.no_drops = check_imu_drops(prev_pose_.preintegration, calib_, prev_pose_ts_ns_, time_ns);
  return prep;
}

bool ImuFusionContext::solve_inertial(const std::unordered_map<TrackId, Vector3T>& landmarks,
                                      const std::vector<camera::Observation>& observations,
                                      const pnp::RGBDInfos& depth_infos, const std::vector<map::Plane>& planes,
                                      const std::vector<Vector3T>& depth_points, int64_t time_ns) {
  std::optional<Vector3T> maybe_gravity = map_.get_gravity();
  if (!maybe_gravity) {
    return false;
  }

  TRACE_EVENT ev = profiler_domain_.trace_event("solve_inertial");
  // The cuNLS LM uses cunls::SE3BetweenFactorBatch with an IMU-derived Delta; velocity / biases
  // are baked in as constants in the Delta, so the solve does not refine them.  Pose is the only
  // state the LM moves.
  if (!run_rgbd_inertial_solve(pose_estimator_, calib_, landmarks, observations, depth_infos, planes, depth_points,
                               *maybe_gravity, prev_pose_, curr_pose_)) {
    return false;
  }
  if (!is_first_run_ && prev_pose_ts_ns_ > 0) {
    float dt = static_cast<float>((time_ns - prev_pose_ts_ns_) * 1e-9);
    if (dt > 1e-6f) {
      curr_pose_.velocity = (curr_pose_.w_from_imu.translation() - prev_pose_.w_from_imu.translation()) / dt;
    }
  } else {
    curr_pose_.velocity.setZero();
  }
  return true;
}

bool ImuFusionContext::solve_visual(const std::unordered_map<TrackId, Vector3T>& landmarks,
                                    const std::vector<camera::Observation>& observations,
                                    const pnp::RGBDInfos& depth_infos, const std::vector<map::Plane>& planes,
                                    const std::vector<Vector3T>& depth_points, int64_t time_ns) {
  TRACE_EVENT ev = profiler_domain_.trace_event("solve_visual");
  const Isometry3T imu_from_rig = calib_.rig_from_imu().inverse();
  Isometry3T rig_from_world = (prev_pose_.w_from_imu * imu_from_rig).inverse();
  Matrix6T info;
  const bool ok =
      pose_estimator_.solve(rig_from_world, info, observations, landmarks, depth_infos, planes, depth_points);
  if (!ok) {
    return false;
  }

  curr_pose_.w_from_imu = rig_from_world.inverse() * imu_from_rig.inverse();
  curr_pose_.info.setZero();
  curr_pose_.info.block<6, 6>(0, 0) = info;

  curr_pose_.gyro_bias = prev_pose_.gyro_bias;
  curr_pose_.acc_bias = prev_pose_.acc_bias;
  if (!is_first_run_ && prev_pose_ts_ns_ > 0) {
    float dt = static_cast<float>((time_ns - prev_pose_ts_ns_) * 1e-9);
    if (dt > 1e-6f) {
      Vector3T dp = (prev_pose_.w_from_imu * curr_pose_.w_from_imu.inverse()).translation();
      curr_pose_.velocity = dp / dt;
    }
  } else {
    curr_pose_.velocity.setZero();
  }
  return true;
}

ImuFusionContext::FrameResult ImuFusionContext::finalize_frame(bool pnp_result, bool no_drops, int64_t time_ns,
                                                               sof::FrameState fs,
                                                               const StateMachineSettings& sm_settings) {
  prev_pose_ts_ns_ = time_ns;

  StateMachine::State imu_state;
  {
    TRACE_EVENT ev = profiler_domain_.trace_event("update_frame_state");
    bool is_keyframe = fs == sof::FrameState::Key;
    StateMachine::Event event;
    if (!no_drops) {
      // IMU samples were lost across this frame — preintegration is unreliable.
      event = StateMachine::Event::FrameDropped;
    } else {
      event = pnp_result ? StateMachine::Event::FrameOk : StateMachine::Event::FrameFailed;
    }
    imu_state = sm_.on_event(event, time_ns, is_keyframe, sm_settings);
  }

  if (pnp_result && imu_state == StateMachine::State::Ok) {
    last_valid_pose_ = curr_pose_;
    last_valid_pose_.preintegration = sba_imu::IMUPreintegration(curr_pose_.gyro_bias, curr_pose_.acc_bias);
  }

  integrated_ = !pnp_result && imu_state == StateMachine::State::Ok;

  // Re-anchor integ_kf_ to the latest keyframe so IMU integration always starts from a known-good
  // visual anchor instead of accumulating frame-to-frame error.
  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  if (!map_.empty()) {
    auto recent_map = map_.get_recent_submap(1);
    if (!recent_map.consecutive_keyframes.empty()) {
      auto it = recent_map.consecutive_keyframes.rbegin();
      State s = it->keyframe->get_state();
      sba_imu::IMUPreintegration preint(calib_, imu_storage_, s.gyro_bias, s.acc_bias, it->keyframe->time_ns(),
                                        time_ns);
      integ_kf_ = {s.rig_from_w.inverse() * rig_from_imu, s.velocity, s.gyro_bias, s.acc_bias, preint};
    }
  }

  if (integrated_) {
    std::optional<Vector3T> maybe_gravity = map_.get_gravity();
    if (maybe_gravity) {
      integ_kf_.predict_pose(*maybe_gravity, integ_kf_.preintegration, curr_pose_);
    } else {
      integrated_ = false;
    }
  }

  FrameResult result;
  result.valid_pose = pnp_result || imu_state == StateMachine::State::Ok;
  if (result.valid_pose) {
    const Isometry3T imu_from_rig = rig_from_imu.inverse();
    result.world_from_rig = curr_pose_.w_from_imu * imu_from_rig;
    result.static_info_exp = curr_pose_.info.block<6, 6>(0, 0);
    std::swap(prev_pose_, curr_pose_);
  } else {
    result.world_from_rig = prev_pose_.w_from_imu * rig_from_imu.inverse();
  }

  // Pull SBA-refined biases back into prev_pose so the next frame's PnP starts from the latest
  // estimate.
  if (!map_.empty()) {
    auto recent_map = map_.get_recent_submap(1);
    if (!recent_map.consecutive_keyframes.empty()) {
      State s = recent_map.consecutive_keyframes.rbegin()->keyframe->get_state();
      prev_pose_.gyro_bias = s.gyro_bias;
      prev_pose_.acc_bias = s.acc_bias;
    }
  }

  last_frame_preint_ = sba_imu::IMUPreintegration(prev_pose_.gyro_bias, prev_pose_.acc_bias);

  result.lost = !is_first_run_ && !pnp_result && !integrated_;
  if (!result.lost && is_first_run_) {
    is_first_run_ = false;
  }
  return result;
}

map::State ImuFusionContext::build_keyframe_state(const Isometry3T& rig_from_world, bool lost) const {
  const sba_imu::Pose& pose = lost ? last_valid_pose_ : prev_pose_;
  return map::State{rig_from_world, pose.velocity, pose.acc_bias, pose.gyro_bias};
}

void ImuFusionContext::on_keyframe_committed(bool lost) {
  const sba_imu::Pose& pose = lost ? last_valid_pose_ : prev_pose_;
  last_kf_preint_ = sba_imu::IMUPreintegration(pose.gyro_bias, pose.acc_bias);
}

std::optional<Vector3T> ImuFusionContext::get_gravity() const {
  auto gravity = map_.get_gravity();
  if (!gravity) {
    return std::nullopt;
  }
  Isometry3T rig_from_w = calib_.rig_from_imu() * prev_pose_.w_from_imu.inverse();
  return rig_from_w.linear() * (*gravity);
}

void ImuFusionContext::snap_to_keyframe_pose(const sba_imu::Pose& last_kf_pose) {
  auto apply = [&](sba_imu::Pose& pose) {
    pose.velocity = last_kf_pose.velocity;
    pose.gyro_bias = last_kf_pose.gyro_bias;
    pose.acc_bias = last_kf_pose.acc_bias;
    pose.preintegration.SetNewBias(last_kf_pose.gyro_bias, last_kf_pose.acc_bias);
    pose.preintegration.Reintegrate(calib_);
  };
  apply(curr_pose_);
  apply(prev_pose_);
  apply(integ_kf_);
  apply(last_valid_pose_);
}

bool ImuFusionContext::gravity_estimation_callback(size_t num_kfs) {
  TRACE_EVENT ev = profiler_domain_.trace_event("gravity_estimation_callback");
  auto recent_map = map_.get_recent_submap(map_.capacity());

  if (recent_map.consecutive_keyframes.size() < num_kfs) {
    return false;
  }

  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  Matrix3T Rgravity = Matrix3T::Identity();
  std::vector<sba_imu::Pose> gravity_cache;
  for (const auto& kf : recent_map.consecutive_keyframes) {
    State s = kf.keyframe->get_state();
    gravity_cache.push_back({s.rig_from_w.inverse() * rig_from_imu, s.velocity, s.gyro_bias, s.acc_bias,
                             kf.preintegration ? *kf.preintegration : sba_imu::IMUPreintegration()});
  }

  std::vector<std::vector<camera::Observation>> observations_per_kf;
  observations_per_kf.reserve(recent_map.consecutive_keyframes.size());
  for (const auto& lm_obs : recent_map.landmark_and_obs) {
    std::vector<camera::Observation> kf_obs;
    for (const auto& lo : lm_obs) {
      for (const auto& obs : lo.observations) {
        kf_obs.push_back(obs);
      }
    }
    observations_per_kf.push_back(std::move(kf_obs));
  }

  // optimize_inertial already does the staged init: SolveGyroBias (gyro-bias first), then
  // LinearAlignment (joint velocities + gravity + acc_bias via a linear system), then
  // RefineGravity (tangent-space gravity + acc_bias refinement), then LM polish. The
  // OptimizeInertialAdaptive variant is preferred for stereo+IMU because of its post-opt
  // rotation-from-IMU + epipolar-translation refinement, but that refinement consistently
  // regresses the multisensor path (verified empirically on EUROC V2_03), so we stay on
  // optimize_inertial here.
  if (!optimizer_.optimize_inertial(gravity_cache, calib_, Rgravity, 1e-3)) {
    return false;
  }

  TRACE_EVENT ev1 = profiler_domain_.trace_event("update map");
  const Vector3T gravity_for_sba = Rgravity * optimizer_.get_default_gravity();

  if (enable_imu_sba_init_) {
    run_imu_sba_init(recent_map, gravity_for_sba, gravity_cache, observations_per_kf);
  }

  // Write back per-keyframe biases / velocities / preintegrations from the refined cache.
  int id = 0;
  for (const auto& kf : recent_map.consecutive_keyframes) {
    const sba_imu::Pose& pose = gravity_cache[id];
    kf.keyframe->set_gyro_bias(pose.gyro_bias);
    kf.keyframe->set_acc_bias(pose.acc_bias);
    kf.keyframe->set_velocity(pose.velocity);
    kf.keyframe->set_pose((pose.w_from_imu * rig_from_imu.inverse()).inverse());
    if (kf.preintegration) {
      kf.preintegration->SetNewBias(pose.gyro_bias, pose.acc_bias);
      kf.preintegration->Reintegrate(calib_);
    }
    id++;
  }

  // Snap live tracking state to the last keyframe pose (closest temporal match for the next
  // tracking frame).
  snap_to_keyframe_pose(gravity_cache.back());

  map_.set_gravity(gravity_for_sba);
  return true;
}

void ImuFusionContext::run_imu_sba_init(const map::UnifiedMap::SubMap& recent_map, const Vector3T& gravity,
                                        std::vector<sba_imu::Pose>& gravity_cache,
                                        const std::vector<std::vector<camera::Observation>>& observations_per_kf) {
  const size_t n = recent_map.consecutive_keyframes.size();
  if (gravity_cache.size() != n || observations_per_kf.size() != n) {
    return;
  }

  sba_imu::ImuBAProblem problem;
  problem.rig = rig_;
  problem.gravity = gravity;
  problem.robustifier_scale = 1.5f;
  problem.robustifier_scale_pose = -1.0f;
  problem.prior_acc = 1e1;
  problem.prior_gyro = 0;
  problem.imu_penalty = 1e-2f;
  problem.boundary_imu_penalty = 1e-2f;
  problem.acc_rw_penalty = 0.1f;

  constexpr float visual_noise_px = 1.0f;
  constexpr float info_scale = (3.0f / visual_noise_px) * (3.0f / visual_noise_px);
  problem.reintegration_thresh = 1e-3f;

  problem.max_iterations = 10;
  problem.num_fixed_key_frames = CalcNumFixedKeyframes(n, 1);
  if (problem.num_fixed_key_frames < 1) {
    return;
  }

  std::vector<std::vector<Isometry3T>> world_from_cam(n, std::vector<Isometry3T>(rig_.num_cameras));
  for (size_t i = 0; i < n; i++) {
    Isometry3T world_from_rig_i = gravity_cache[i].w_from_imu * calib_.rig_from_imu().inverse();
    for (CameraId cam_id = 0; cam_id < static_cast<CameraId>(rig_.num_cameras); cam_id++) {
      world_from_cam[i][cam_id] = world_from_rig_i * rig_.camera_from_rig[cam_id].inverse();
    }
  }

  struct TrackObs {
    size_t kf_idx;
    CameraId cam_id;
    Vector2T xy;
    Matrix2T xy_info;
  };
  std::unordered_map<TrackId, std::vector<TrackObs>> track_obs_map;
  for (size_t i = 0; i < n; i++) {
    for (const auto& obs : observations_per_kf[i]) {
      track_obs_map[obs.id].push_back({i, obs.cam_id, obs.xy, obs.xy_info});
    }
  }

  std::unordered_map<TrackId, map::LandmarkPtr> track_to_landmark;
  for (size_t i = 0; i < n; i++) {
    for (const auto& lo : recent_map.landmark_and_obs[i]) {
      if (!lo.landmark->get_pose()) {
        continue;
      }
      for (const auto& obs : lo.observations) {
        track_to_landmark[obs.id] = lo.landmark;
      }
    }
  }

  std::unordered_map<TrackId, int> track_point_ids;
  std::unordered_map<map::LandmarkPtr, int> landmark_point_ids;

  for (auto& [track_id, obs_list] : track_obs_map) {
    auto it = track_to_landmark.find(track_id);
    if (it == track_to_landmark.end()) {
      continue;
    }

    bool triangulated = false;
    Vector3T new_point;

    for (size_t a = 0; a < obs_list.size() && !triangulated; a++) {
      for (size_t b = a + 1; b < obs_list.size() && !triangulated; b++) {
        if (obs_list[a].cam_id == obs_list[b].cam_id) {
          continue;
        }

        try {
          const auto& oa = obs_list[a];
          const auto& ob = obs_list[b];
          const Isometry3T& wfc_a = world_from_cam[oa.kf_idx][oa.cam_id];
          const Isometry3T& wfc_b = world_from_cam[ob.kf_idx][ob.cam_id];
          Isometry3T cam_a_from_cam_b = wfc_a.inverse() * wfc_b;

          Vector3T xyz_in_cam_a;
          float pm;
          epipolar::TriangulationState ts;
          if (epipolar::OptimalTriangulation(cam_a_from_cam_b, oa.xy, ob.xy, xyz_in_cam_a, pm, ts)) {
            new_point = wfc_a * xyz_in_cam_a;
            triangulated = true;
          }
        } catch (const std::exception&) {
          continue;
        }
      }
    }

    if (!triangulated) {
      size_t kf_idx = obs_list[0].kf_idx;
      const auto& kf = recent_map.consecutive_keyframes[kf_idx].keyframe;
      Isometry3T old_world_from_rig = kf->get_pose().inverse();
      Isometry3T new_world_from_rig = gravity_cache[kf_idx].w_from_imu * calib_.rig_from_imu().inverse();
      Isometry3T correction = new_world_from_rig * old_world_from_rig.inverse();
      new_point = correction * (*it->second->get_pose());
    }

    int pid = static_cast<int>(problem.points.size());
    track_point_ids[track_id] = pid;
    landmark_point_ids[it->second] = pid;
    problem.points.push_back(new_point);
  }
  if (problem.points.empty()) {
    return;
  }

  for (size_t i = 0; i < n; i++) {
    problem.rig_poses.push_back(gravity_cache[i]);
  }

  size_t total_observations = 0;
  for (const auto& obs_kf : observations_per_kf) {
    total_observations += obs_kf.size();
  }
  problem.observation_xys.reserve(total_observations);
  problem.observation_infos.reserve(total_observations);
  problem.point_ids.reserve(total_observations);
  problem.pose_ids.reserve(total_observations);
  problem.camera_ids.reserve(total_observations);

  for (size_t i = 0; i < n; i++) {
    for (const auto& obs : observations_per_kf[i]) {
      auto it = track_point_ids.find(obs.id);
      if (it == track_point_ids.end()) {
        continue;
      }
      problem.observation_xys.push_back(obs.xy);
      problem.observation_infos.push_back(obs.xy_info * info_scale);
      problem.point_ids.push_back(it->second);
      problem.pose_ids.push_back(static_cast<int32_t>(i));
      problem.camera_ids.push_back(obs.cam_id);
    }
  }

  if (!init_bundler_.solve(problem)) {
    return;
  }

  for (size_t i = 0; i < n; i++) {
    const auto& pose = problem.rig_poses[i];
    gravity_cache[i] = pose;

    const auto& [kf, preint] = recent_map.consecutive_keyframes[i];
    kf->set_state({calib_.rig_from_imu() * pose.w_from_imu.inverse(), pose.velocity, pose.acc_bias, pose.gyro_bias});
    if (preint) {
      *preint = pose.preintegration;
    }
  }
  for (auto& [l, pid] : landmark_point_ids) {
    l->set_pose(problem.points[pid]);
  }
}

}  // namespace cuvslam::pipelines
