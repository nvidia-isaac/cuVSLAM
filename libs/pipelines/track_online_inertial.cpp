
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

#include "pipelines/track_online_inertial.h"

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/frame_id.h"
#include "common/include_eigen.h"
#include "common/isometry.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "imu/inertial_optimization.h"
#include "sof/sof_create.h"

#include "common/rerun.h"
#include "epipolar/camera_selection.h"
#include "pipelines/inertial_pnp.h"
#include "pipelines/service_sba.h"
#include "pipelines/track.h"
#include "pipelines/visualizer.h"
#ifdef USE_CUDA
#include "pipelines/service_sba_gpu.h"
#endif

namespace cuvslam::pipelines {

using Vector15T = Eigen::Matrix<float, 15, 1>;

// IMU SBA refinement step used exclusively during IMU initialization.
// Runs after optimize_inertial to jointly refine poses, 3D landmarks, velocities,
// and biases (including acc_bias) using visual reprojection + IMU residuals.
static void RunImuSbaInit(const map::UnifiedMap::SubMap& recent_map, const Vector3T& gravity, const camera::Rig& rig,
                          const imu::ImuCalibration& calib, std::vector<sba_imu::Pose>& gravity_cache,
                          sba_imu::IMUBundlerCpuFixedVel& bundler,
                          const std::vector<std::vector<camera::Observation>>& observations_per_kf) {
  const size_t n = recent_map.consecutive_keyframes.size();
  if (gravity_cache.size() != n || observations_per_kf.size() != n) return;

  sba_imu::ImuBAProblem problem;
  problem.rig = rig;
  problem.gravity = gravity;
  problem.robustifier_scale = 1.5f;
  problem.robustifier_scale_pose = -1.0f;
  problem.prior_acc = 1e1;
  problem.prior_gyro = 0;
  problem.imu_penalty = 1e-2f;
  problem.boundary_imu_penalty = 1e-2f;
  problem.acc_rw_penalty = 0.1f;

  // Scale observation info from the default σ=3px to σ=1px (9x stronger visual weight).
  constexpr float visual_noise_px = 1.0f;
  constexpr float info_scale = (3.0f / visual_noise_px) * (3.0f / visual_noise_px);
  problem.reintegration_thresh = 1e-3f;

  problem.max_iterations = 10;
  problem.num_fixed_key_frames = CalcNumFixedKeyframes(n, 1);
  if (problem.num_fixed_key_frames < 1) return;

  // Precompute world_from_cam transforms using IMU-refined poses (gravity_cache)
  std::vector<std::vector<Isometry3T>> world_from_cam(n, std::vector<Isometry3T>(rig.num_cameras));
  for (size_t i = 0; i < n; i++) {
    Isometry3T world_from_rig_i = gravity_cache[i].w_from_imu * calib.rig_from_imu().inverse();
    for (CameraId cam_id = 0; cam_id < static_cast<CameraId>(rig.num_cameras); cam_id++) {
      world_from_cam[i][cam_id] = world_from_rig_i * rig.camera_from_rig[cam_id].inverse();
    }
  }

  // Group observations by TrackId across all keyframes for re-triangulation
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

  // Build TrackId -> LandmarkPtr mapping from the map (for write-back)
  std::unordered_map<TrackId, map::LandmarkPtr> track_to_landmark;
  for (size_t i = 0; i < n; i++) {
    for (const auto& lo : recent_map.landmark_and_obs[i]) {
      if (!lo.landmark->get_pose()) continue;
      for (const auto& obs : lo.observations) {
        track_to_landmark[obs.id] = lo.landmark;
      }
    }
  }

  // Re-triangulate each track using IMU-refined poses
  std::unordered_map<TrackId, int> track_point_ids;
  std::unordered_map<map::LandmarkPtr, int> landmark_point_ids;  // for write-back
  int retriang_count = 0, total_tracks = 0;

  for (auto& [track_id, obs_list] : track_obs_map) {
    auto it = track_to_landmark.find(track_id);
    if (it == track_to_landmark.end()) continue;  // no landmark for this track
    total_tracks++;

    bool triangulated = false;
    Vector3T new_point;

    // Try pairs from different cameras
    for (size_t a = 0; a < obs_list.size() && !triangulated; a++) {
      for (size_t b = a + 1; b < obs_list.size() && !triangulated; b++) {
        if (obs_list[a].cam_id == obs_list[b].cam_id) continue;

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
            retriang_count++;
          }
        } catch (const std::exception& e) {
          // Skip this pair if matrix inversion fails
          continue;
        }
      }
    }

    if (!triangulated) {
      // Fallback: transform landmark from old visual-only frame to IMU-refined frame
      size_t kf_idx = obs_list[0].kf_idx;
      const auto& kf = recent_map.consecutive_keyframes[kf_idx].keyframe;
      Isometry3T old_world_from_rig = kf->get_pose().inverse();
      Isometry3T new_world_from_rig = gravity_cache[kf_idx].w_from_imu * calib.rig_from_imu().inverse();
      Isometry3T correction = new_world_from_rig * old_world_from_rig.inverse();
      new_point = correction * (*it->second->get_pose());
    }

    int pid = static_cast<int>(problem.points.size());
    track_point_ids[track_id] = pid;
    landmark_point_ids[it->second] = pid;
    problem.points.push_back(new_point);
  }
  if (problem.points.empty()) return;
  TraceMessage("init_sba: re-triangulated %d/%d tracks", retriang_count, total_tracks);

  for (size_t i = 0; i < n; i++) {
    problem.rig_poses.push_back(gravity_cache[i]);
  }

  // Build SBA observations using TrackId-based point_ids
  size_t total_observations = 0;
  for (const auto& obs_kf : observations_per_kf) total_observations += obs_kf.size();
  problem.observation_xys.reserve(total_observations);
  problem.observation_infos.reserve(total_observations);
  problem.point_ids.reserve(total_observations);
  problem.pose_ids.reserve(total_observations);
  problem.camera_ids.reserve(total_observations);

  for (size_t i = 0; i < n; i++) {
    for (const auto& obs : observations_per_kf[i]) {
      auto it = track_point_ids.find(obs.id);
      if (it == track_point_ids.end()) continue;
      problem.observation_xys.push_back(obs.xy);
      problem.observation_infos.push_back(obs.xy_info * info_scale);
      problem.point_ids.push_back(it->second);
      problem.pose_ids.push_back(static_cast<int32_t>(i));
      problem.camera_ids.push_back(obs.cam_id);
    }
  }

  TraceMessage("init_sba: n=%zu pts=%zu obs=%zu fixed=%d", n, problem.points.size(), problem.observation_xys.size(),
               problem.num_fixed_key_frames);

  if (!bundler.solve(problem)) {
    TraceMessage("init_sba: solve FAILED");
    return;
  }
  TraceMessage("init_sba: solve OK, iters=%d, init_cost=%.2f", problem.iterations, problem.initial_cost);

  // Check how much poses moved
  float max_pos_diff = 0, max_vel_diff = 0, max_acc_diff = 0;
  for (size_t i = 0; i < n; i++) {
    float pd = (problem.rig_poses[i].w_from_imu.translation() - gravity_cache[i].w_from_imu.translation()).norm();
    float vd = (problem.rig_poses[i].velocity - gravity_cache[i].velocity).norm();
    float ad = (problem.rig_poses[i].acc_bias - gravity_cache[i].acc_bias).norm();
    if (pd > max_pos_diff) max_pos_diff = pd;
    if (vd > max_vel_diff) max_vel_diff = vd;
    if (ad > max_acc_diff) max_acc_diff = ad;
  }
  TraceMessage("init_sba: max_delta pos=%.4f vel=%.4f acc_bias=%.4f", max_pos_diff, max_vel_diff, max_acc_diff);

  for (size_t i = 0; i < n; i++) {
    const auto& pose = problem.rig_poses[i];
    gravity_cache[i] = pose;

    const auto& [kf, preint] = recent_map.consecutive_keyframes[i];
    kf->set_state({calib.rig_from_imu() * pose.w_from_imu.inverse(), pose.velocity, pose.acc_bias, pose.gyro_bias});
    if (preint) *preint = pose.preintegration;
  }
  for (auto& [l, pid] : landmark_point_ids) {
    l->set_pose(problem.points[pid]);
  }
}

SolverSfMInertial::SolverSfMInertial(map::UnifiedMap& map, const camera::Rig& rig, const sba::Settings& sba_settings,
                                     const imu::ImuCalibration& calib, bool debug_imu_mode,
                                     bool disable_fusion_except_gravity)
    : imu_storage_(1e5),
      calib_(calib),
      rig_(rig),
      debug_imu_mode_(debug_imu_mode),
      disable_fusion_except_gravity_(disable_fusion_except_gravity),
      map_(map),
      optimizer_(10),
      imu_init_bundler_(calib),
      stereo_pnp_(rig),
      triangulator(rig) {
  sba::Mode sba_mode;
  if (disable_fusion_except_gravity_ && rig_.num_cameras > 2) {
    sba_mode = sba::OriginalGPU;
  } else {
    sba_mode = sba_settings.mode;  // InertialCPU
  }
  if (sba_mode != sba::InertialCPU && sba_mode != sba::InertialGPU && sba_mode != sba::Disabled) {  // Add GPU version
    TraceDebug("Cant launch fusion with regular SBA");
  }

  switch (sba_mode) {
    case sba::InertialCPU:
      sba_service_ = std::make_unique<ImuSbaCPUService>(sba_settings, rig, calib, map_);
      break;
    case sba::OriginalCPU:
      sba_service_ = std::make_unique<CpuSbaService>(sba_settings, rig, map_);
      break;
#ifdef USE_CUDA
    case sba::InertialGPU:
      sba_service_ = std::make_unique<ImuSbaGPUService>(sba_settings, rig, calib, map_);
      break;
    case sba::OriginalGPU:
      sba_service_ = std::make_unique<GpuSbaService>(sba_settings, rig, map_);
      break;
#endif
    default:
      sba_service_ = nullptr;
      break;
  }

  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  prev_pose.w_from_imu = rig_from_imu;

  auto gravity_estimation_callback = [this](size_t num_kfs) {
    TRACE_EVENT ev = profiler_domain_.trace_event("gravity_estimation_callback");
    // Fetch all available keyframes (up to map capacity) to maximize data for optimization.
    // num_kfs is only the minimum threshold — if the map hasn't filled yet, bail out.
    auto recent_map = map_.get_recent_submap(map_.capacity());

    if (recent_map.consecutive_keyframes.size() < num_kfs) {
      TraceDebug("Not enough keyframes in map, (%d, %d)", map_.size(), num_kfs);
      return false;
    }
    const Isometry3T& rig_from_imu = calib_.rig_from_imu();
    Matrix3T Rgravity = Matrix3T::Identity();
    std::vector<sba_imu::Pose> gravity_cache;

    for (const auto& kf : recent_map.consecutive_keyframes) {
      State s = kf.keyframe->get_state();
      gravity_cache.push_back({s.rig_from_w.inverse() * rig_from_imu, s.velocity, s.gyro_bias, s.acc_bias,
                               kf.preintegration ? *kf.preintegration : IMUPreintegration()});
    }

    bool imu_init_result = false;
    bool config_use_nec_init = false;

    // Extract per-keyframe observations from the submap for NEC gyro bias estimation
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

    if (config_use_nec_init) {
      imu_init_result = optimizer_.OptimizeInertialAdaptive(gravity_cache, calib_, Rgravity, 1e-3, observations_per_kf,
                                                            rig_.camera_from_rig[0]);
    } else {
      imu_init_result = optimizer_.optimize_inertial(gravity_cache, calib_, Rgravity, 1e-3);
    }

    if (imu_init_result) {
      TRACE_EVENT ev1 = profiler_domain_.trace_event("update map");

      const Vector3T gravity_for_sba = Rgravity * optimizer_.get_default_gravity();

      // this step can improve accuracy, but may use some time, by default it is enabled.
      if (enable_imu_sba_init_) {
        RunImuSbaInit(recent_map, gravity_for_sba, rig_, calib_, gravity_cache, imu_init_bundler_, observations_per_kf);
      }

      {
        // update biases, velocities, and preintegrations per-keyframe from SBA-refined results.
        int id = 0;
        for (const auto& kf : recent_map.consecutive_keyframes) {
          const sba_imu::Pose& pose = gravity_cache[id];

          kf.keyframe->set_gyro_bias(pose.gyro_bias);
          kf.keyframe->set_acc_bias(pose.acc_bias);
          kf.keyframe->set_velocity(pose.velocity);
          // Write back optimized pose (rotation from IMU propagation + translation from epipolar)
          kf.keyframe->set_pose((pose.w_from_imu * rig_from_imu.inverse()).inverse());

          if (kf.preintegration) {
            kf.preintegration->SetNewBias(pose.gyro_bias, pose.acc_bias);
            kf.preintegration->Reintegrate(calib_);
          }

          id++;
        }

        // Use the last keyframe's refined state as the best estimate for
        // current tracking frames, which are temporally closest to it.
        const sba_imu::Pose& last_kf_pose = gravity_cache.back();

        curr_pose.velocity = last_kf_pose.velocity;
        curr_pose.gyro_bias = last_kf_pose.gyro_bias;
        curr_pose.acc_bias = last_kf_pose.acc_bias;
        curr_pose.preintegration.SetNewBias(last_kf_pose.gyro_bias, last_kf_pose.acc_bias);
        curr_pose.preintegration.Reintegrate(calib_);

        prev_pose.velocity = last_kf_pose.velocity;
        prev_pose.gyro_bias = last_kf_pose.gyro_bias;
        prev_pose.acc_bias = last_kf_pose.acc_bias;
        prev_pose.preintegration.SetNewBias(last_kf_pose.gyro_bias, last_kf_pose.acc_bias);
        prev_pose.preintegration.Reintegrate(calib_);

        integ_kf.velocity = last_kf_pose.velocity;
        integ_kf.gyro_bias = last_kf_pose.gyro_bias;
        integ_kf.acc_bias = last_kf_pose.acc_bias;
        integ_kf.preintegration.SetNewBias(last_kf_pose.gyro_bias, last_kf_pose.acc_bias);
        integ_kf.preintegration.Reintegrate(calib_);

        last_valid_pose.velocity = last_kf_pose.velocity;
        last_valid_pose.gyro_bias = last_kf_pose.gyro_bias;
        last_valid_pose.acc_bias = last_kf_pose.acc_bias;
        last_valid_pose.preintegration.SetNewBias(last_kf_pose.gyro_bias, last_kf_pose.acc_bias);
        last_valid_pose.preintegration.Reintegrate(calib_);
      }

      map_.set_gravity(gravity_for_sba);

      const Vector3T& gyro_bias = gravity_cache.back().gyro_bias;
      const Vector3T& acc_bias = gravity_cache.back().acc_bias;
      const Vector3T& last_velocity = gravity_cache.back().velocity;
      TraceMessage("[IMU INIT DONE] gravity=[%.4f,%.4f,%.4f] |g|=%.4f", gravity_for_sba.x(), gravity_for_sba.y(),
                   gravity_for_sba.z(), gravity_for_sba.norm());
      TraceMessage("[IMU INIT DONE] gyro_bias=[%.6f,%.6f,%.6f] acc_bias=[%.6f,%.6f,%.6f]", gyro_bias.x(), gyro_bias.y(),
                   gyro_bias.z(), acc_bias.x(), acc_bias.y(), acc_bias.z());
      TraceMessage("[IMU INIT DONE] last_vel=[%.3f,%.3f,%.3f] |v|=%.3f num_kfs=%zu", last_velocity.x(),
                   last_velocity.y(), last_velocity.z(), last_velocity.norm(), gravity_cache.size());

      return true;
    }
    return false;
  };
  imu_sm_.set_gravity_init_fn(gravity_estimation_callback);
}

// OUT: pose, prev_pose - updated poses if success, otherwise it's guaranteed to be unchanged
bool runInertialPnP(const imu::ImuCalibration& calib, const InertialPnP& solver,
                    const std::unordered_map<TrackId, Track>& tracks3d,
                    const std::vector<camera::Observation>& observations, const camera::Rig& rig,
                    const Vector3T& gravity, const InertialPnPSettings& settings,
                    sba_imu::Pose& prev_pose,  // non-const because of velocities updates
                    sba_imu::Pose& curr_pose) {
  return solver.Solve(calib, tracks3d, observations, rig, gravity, settings, prev_pose, curr_pose);
}

// OUT: curr_pose - current pose if success otherwise curr_pose is unchanged
// landmark_buffer is caller-owned scratch storage for TrackId -> position.
bool runStereoPnP(pnp::PNPSolver& solver, const std::unordered_map<TrackId, Track>& tracks3d,
                  const std::vector<camera::Observation>& observations, map::Map<TrackId, Vector3T>& landmark_buffer,
                  const Isometry3T& imu_from_rig, const pnp::PNPSettings& settings,
                  sba_imu::Pose& prev_pose,  // non-const because of velocities updates
                  sba_imu::Pose& curr_pose) {
  /* logic here:
   * world_from_rig = prev_pose.w_from_imu * imu_from_rig;
   * world_from_rig = prev_rig_from_world_.inverse();
   * prev_pose.w_from_imu * imu_from_rig = prev_rig_from_world_.inverse()
   * prev_rig_from_world_ = (prev_pose.w_from_imu * imu_from_rig).inverse()  */

  landmark_buffer.clear();
  landmark_buffer.reserve(tracks3d.size());
  for (const auto& [track_id, track] : tracks3d) {
    if (track.hasLocation()) {
      landmark_buffer.insert({track_id, track.getLocation3D()});
    }
  }

  Isometry3T rig_from_world = (prev_pose.w_from_imu * imu_from_rig).inverse();
  Matrix6T info;
  const bool result = solver.solve(rig_from_world, info, observations, landmark_buffer, settings);

  if (result) {
    /* logic here:
     * prev_pose.w_from_imu * imu_from_rig = rig_from_world.inverse()
     * prev_pose.w_from_imu = rig_from_world.inverse() * imu_from_rig.inverse() */
    curr_pose.w_from_imu = rig_from_world.inverse() * imu_from_rig.inverse();
    curr_pose.info.setZero();
    curr_pose.info.block<6, 6>(0, 0) = info;
  }

  return result;
}

const camera::Rig& SolverSfMInertial::getRig() const { return rig_; }

void SolverSfMInertial::reset() {
  imu_storage_.clear();
  triangulator.reset();
  if (sba_service_) {
    sba_service_->restart();
  }
  imu_sm_.reset();
  is_first_run = true;

  prev_pose_ts_ns = -1;

  curr_pose.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  prev_pose.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());

  last_valid_pose.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  integ_kf.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());

  last_kf_preint.Initialize(Vector3T::Zero(), Vector3T::Zero());
  last_frame_preint_.Initialize(Vector3T::Zero(), Vector3T::Zero());
}

bool SolverSfMInertial::predict_pose(Isometry3T& pose) const {
  (void)pose;
  return false;
}

void SolverSfMInertial::set_verbose(bool verbose) { verbose_ = verbose; }

// Reports whether the IMU stream covers the frame interval reliably. See the matching helper in
// imu_fusion_context.cpp for the rationale — short frame intervals previously read as 33%-drop
// false positives from normal IMU phase jitter.
bool check_imu_drops(sba_imu::IMUPreintegration& preint, const imu::ImuCalibration& calib, int64_t start_time_ns,
                     int64_t end_time_ns) {
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
  // Bound the slack by half the expected count so short intervals can't hide a genuine dropout
  // (otherwise e.g. expected=2 with actual=0 would still report healthy). Matches the helper in
  // imu_fusion_context.cpp.
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

bool SolverSfMInertial::solveNextFrame(int64_t time_ns, const sof::FrameState& frameState,
                                       const MulticamObservations& observations, Isometry3T& world_from_rig,
                                       Matrix6T& static_info_exp, const SolverPerFrameSettings& solver_settings,
                                       std::vector<Track2D>* tracks2d, Tracks3DMap* tracks3d) {
  TRACE_EVENT ev = profiler_domain_.trace_event("solveNextFrame()");

  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  if (is_first_run) {
    Vector15T diag = Vector15T::Zero();
    diag.segment<6>(0).setConstant(1e6);  // prev_pose is fixed
    // diag.segment<6>(9).setConstant(1e6); // biases are fixed
    prev_pose.info = diag.asDiagonal();  // first pose is fixed, but velocities and biases are free

    last_valid_pose = prev_pose;
    integ_kf = prev_pose;

    last_valid_pose.preintegration = last_frame_preint_;
    integ_kf.preintegration = last_frame_preint_;
    last_kf_preint = last_frame_preint_;
  }

  prev_pose.preintegration = last_frame_preint_;

  curr_pose = prev_pose;
  // TODO: use last_valid_kf_pose to integrate pose, not last
  // find 2 last tracking_state with optimized_by_sba = true and timedelta > 0.5 s,
  // then compare it with last_kf_valid_pose.frameid and update it

  const Isometry3T imu_from_rig = rig_from_imu.inverse();

  bool pnp_result = false;
  StateMachine::State imu_state;
  if (disable_fusion_except_gravity_ && rig_.num_cameras > 2) {
    imu_state = StateMachine::State::Uninitialized;
  } else {
    imu_state = imu_sm_.state();
  }

  world_from_rig = prev_pose.w_from_imu * imu_from_rig;
  Isometry3T rig_from_w = world_from_rig.inverse();

  static_info_exp.setZero();
  std::optional<Vector3T> maybe_gravity = map_.get_gravity();

  if (debug_imu_mode_) {
    last_valid_pose.predict_pose(*maybe_gravity, last_valid_pose.preintegration, curr_pose);

    curr_pose.w_from_imu.translation().setZero();
    curr_pose.velocity.setZero();

    if (is_first_run) {
      is_first_run = false;
    }
    std::swap(prev_pose, curr_pose);
    return true;
  }

  obs_vector_.clear();
  size_t num_observations = 0;
  for (const auto& cam_observations : observations) {
    num_observations += cam_observations.size();
  }
  obs_vector_.reserve(num_observations);
  for (const auto& obs : observations) {
    std::copy(obs.begin(), obs.end(), std::back_inserter(obs_vector_));
  }

  RERUN(logObservations, obs_vector_, rig_, "world/camera_0/images/observations", Color(255, 165, 0));

  bool no_drops = check_imu_drops(prev_pose.preintegration, calib_, prev_pose_ts_ns, time_ns);

  if (!map_.empty()) {
    map_.get_recent_landmarks(recent_landmarks_);
    pnp_landmarks_.clear();
    pnp_landmarks_.reserve(recent_landmarks_.size());
    for (const auto& [track_id, point_3d] : recent_landmarks_) {
      Track track;
      track.setLocation3D(point_3d, TrackState::kTriangulated);
      pnp_landmarks_.insert({track_id, track});
    }

    if (imu_state == StateMachine::State::Ok) {
      assert(maybe_gravity != std::nullopt);
      {
        auto t_prev = prev_pose.w_from_imu.translation();
        TraceMessage("[PRE-PnP] landmarks=%d obs=%d prev_t=[%.4f,%.4f,%.4f] vel=[%.3f,%.3f,%.3f]",
                     (int)pnp_landmarks_.size(), (int)obs_vector_.size(), t_prev.x(), t_prev.y(), t_prev.z(),
                     prev_pose.velocity.x(), prev_pose.velocity.y(), prev_pose.velocity.z());
      }
      TRACE_EVENT ev2 = profiler_domain_.trace_event("runInertialPnP");
      pnp_result = runInertialPnP(calib_, inertial_pnp_, pnp_landmarks_, obs_vector_, rig_, *maybe_gravity,
                                  solver_settings.imu_pnp, prev_pose, curr_pose);
      TraceMessage("[INERTIAL PnP] result=%d gyro=[%.6f,%.6f,%.6f] acc=[%.6f,%.6f,%.6f] vel=[%.3f,%.3f,%.3f]",
                   (int)pnp_result, curr_pose.gyro_bias.x(), curr_pose.gyro_bias.y(), curr_pose.gyro_bias.z(),
                   curr_pose.acc_bias.x(), curr_pose.acc_bias.y(), curr_pose.acc_bias.z(), curr_pose.velocity.x(),
                   curr_pose.velocity.y(), curr_pose.velocity.z());
    } else {
      {
        TRACE_EVENT ev2 = profiler_domain_.trace_event("runStereoPnP");
        pnp_result = runStereoPnP(stereo_pnp_, pnp_landmarks_, obs_vector_, landmark_buffer_, imu_from_rig,
                                  solver_settings.inertial_stereo_pnp, prev_pose, curr_pose);
      }
      if (pnp_result) {
        curr_pose.gyro_bias = prev_pose.gyro_bias;
        curr_pose.acc_bias = prev_pose.acc_bias;
        if (!is_first_run) {
          float dt = static_cast<float>((time_ns - prev_pose_ts_ns) * 1e-9);
          Vector3T dp = (prev_pose.w_from_imu * curr_pose.w_from_imu.inverse()).translation();
          curr_pose.velocity = dp / dt;
        } else {
          curr_pose.velocity.setZero();
        }
      }
      //            TraceMessage("runStereoPnP result: %d", pnp_result);
    }
  }

  prev_pose_ts_ns = time_ns;

  {
    TRACE_EVENT ev2 = profiler_domain_.trace_event("update_frame_state");
    bool is_keyframe = frameState == sof::FrameState::Key;
    StateMachine::Event event;
    if (!no_drops) {
      event = StateMachine::Event::FrameDropped;
    } else {
      event = pnp_result ? StateMachine::Event::FrameOk : StateMachine::Event::FrameFailed;
    }
    imu_sm_.on_event(event, time_ns, is_keyframe,
                     solver_settings.sm);  // estimates gravity and biases through callback
  }

  if (pnp_result && imu_state == StateMachine::State::Ok) {
    // inertial pnp succeeded
    last_valid_pose = curr_pose;
    last_valid_pose.preintegration = sba_imu::IMUPreintegration(curr_pose.gyro_bias, curr_pose.acc_bias);
  }

  const bool no_observations = obs_vector_.empty();
  const bool integrated_from_blackout =
      !pnp_result && no_observations && no_drops && !is_first_run && maybe_gravity.has_value();
  integrated = !pnp_result && (imu_state == StateMachine::State::Ok || integrated_from_blackout);
  TraceMessage(
      "Frame: pnp=%d integrated=%d imu_state=%d obs=%d vel=[%.3f,%.3f,%.3f] gbias=[%.4f,%.4f,%.4f] "
      "abias=[%.4f,%.4f,%.4f]",
      pnp_result, integrated, (int)imu_state, (int)obs_vector_.size(), prev_pose.velocity.x(), prev_pose.velocity.y(),
      prev_pose.velocity.z(), prev_pose.gyro_bias.x(), prev_pose.gyro_bias.y(), prev_pose.gyro_bias.z(),
      prev_pose.acc_bias.x(), prev_pose.acc_bias.y(), prev_pose.acc_bias.z());
  if (!map_.empty()) {
    auto recent_map = map_.get_recent_submap(1);
    if (!recent_map.consecutive_keyframes.empty()) {
      auto it = recent_map.consecutive_keyframes.rbegin();
      State s = it->keyframe->get_state();

      // Always rebuild integ_kf from the latest KF with a fresh full preintegration
      // to current time. This ensures IMU propagation always starts from a known-good
      // visual anchor, avoiding accumulated error from chaining frame-to-frame steps.
      sba_imu::IMUPreintegration preint(calib_, imu_storage_, s.gyro_bias, s.acc_bias, it->keyframe->time_ns(),
                                        time_ns);
      integ_kf = {s.rig_from_w.inverse() * rig_from_imu, s.velocity, s.gyro_bias, s.acc_bias, preint};
    }
  }

  if (integrated) {
    if (integrated_from_blackout) {
      prev_pose.predict_pose(*maybe_gravity, prev_pose.preintegration, curr_pose);
    } else {
      integ_kf.predict_pose(*maybe_gravity, integ_kf.preintegration, curr_pose);
    }
    TraceDebug("Pose was integrated!");
  }
  if (pnp_result || integrated) {
    // either we successfully converged, or successfully integrated the pose
    world_from_rig = curr_pose.w_from_imu * imu_from_rig;
    rig_from_w = world_from_rig.inverse();
    static_info_exp = curr_pose.info.block<6, 6>(0, 0);

    auto t = curr_pose.w_from_imu.translation();
    auto t_integ = integ_kf.w_from_imu.translation();
    TraceMessage("[POSE OUT] t=[%.4f,%.4f,%.4f] integ=[%.4f,%.4f,%.4f] diff=%.4f pnp=%d integrated=%d", t.x(), t.y(),
                 t.z(), t_integ.x(), t_integ.y(), t_integ.z(), (t - t_integ).norm(), pnp_result, integrated);

    std::swap(prev_pose, curr_pose);
  }
  RERUN(logTrajectory, rig_from_w, "world/trajectories/vo_trajectory", Color(0, 255, 0), TrajectoryType::VO);

  // Propagate SBA-refined bias back to live tracking state AFTER the swap,
  // so the updated bias survives into prev_pose for the next frame's PnP.
  if (!map_.empty()) {
    auto recent_map = map_.get_recent_submap(1);
    if (!recent_map.consecutive_keyframes.empty()) {
      State s = recent_map.consecutive_keyframes.rbegin()->keyframe->get_state();
      prev_pose.gyro_bias = s.gyro_bias;
      prev_pose.acc_bias = s.acc_bias;
      TraceMessage("[SBA BIAS FEEDBACK] gyro=[%.6f,%.6f,%.6f] acc=[%.6f,%.6f,%.6f]", s.gyro_bias.x(), s.gyro_bias.y(),
                   s.gyro_bias.z(), s.acc_bias.x(), s.acc_bias.y(), s.acc_bias.z());
    }
  }

  last_frame_preint_ = sba_imu::IMUPreintegration(prev_pose.gyro_bias, prev_pose.acc_bias);

  bool lost = !is_first_run && !pnp_result && !integrated;
  if (lost) {
    return false;
  }

  if (is_first_run) {
    is_first_run = false;
  }

  // IMU-only fallback poses bridge short visual outages, but they are not visual
  // measurements. Committing them as keyframes can poison the map with anchors
  // that have no reliable visual pose constraint, which makes later IMU-only
  // propagation jump from a bad keyframe state. Keep the initial no-PnP bootstrap
  // path intact; it is needed before the map has enough landmarks for PnP.
  if (frameState == sof::FrameState::Key && !integrated) {
    auto tr_landmarks = triangulator.triangulate(world_from_rig, obs_vector_);

    State state = {rig_from_w, !lost ? prev_pose.velocity : last_valid_pose.velocity,
                   !lost ? prev_pose.acc_bias : last_valid_pose.acc_bias,
                   !lost ? prev_pose.gyro_bias : last_valid_pose.gyro_bias};
    map_.add_keyframe(time_ns, state,
                      last_kf_preint,  // preintegration
                      obs_vector_, tr_landmarks);
    if (sba_service_) {
      static_cast<SbaServiceBase*>(sba_service_.get())->trigger(solver_settings.sba);
    }
    if (!lost) {
      last_kf_preint = sba_imu::IMUPreintegration(prev_pose.gyro_bias, prev_pose.acc_bias);
    } else {
      last_kf_preint = sba_imu::IMUPreintegration(last_valid_pose.gyro_bias, last_valid_pose.acc_bias);
    }
  }

  if (tracks2d && tracks3d) {
    exportTracks(obs_vector_, *tracks2d, *tracks3d, rig_from_w);
  }

  return !lost;
}

// Exports observations in left camera along with corresponding 3d points
// out_tracks2d - output 2d track coordinates in pixels
// out_tracks3d - in rig space
void SolverSfMInertial::exportTracks(const std::vector<camera::Observation>& observations,
                                     std::vector<Track2D>& out_tracks2d, Tracks3DMap& out_tracks3d,
                                     const Isometry3T& rig_from_world) const {
  out_tracks2d.clear();
  out_tracks3d.clear();

  // export 2d tracks
  for (const camera::Observation& obs : observations) {
    const camera::ICameraModel& camera = *rig_.intrinsics[obs.cam_id];
    Vector2T uv;  // in pixels
    if (camera.denormalizePoint(obs.xy, uv)) {
      out_tracks2d.push_back({obs.cam_id, obs.id, uv});
    }
  }

  // export 3d tracks
  auto map_landmarks = map_.get_recent_landmarks();
  for (const camera::Observation& obs : observations) {
    if (map_landmarks.find(obs.id) != map_landmarks.end()) {
      const Vector3T& point_3d = map_landmarks.at(obs.id);
      out_tracks3d[obs.id] = rig_from_world * point_3d;
    }
  }
}

void SolverSfMInertial::add_imu_measurement(const imu::ImuMeasurement& m) {
  imu_storage_.push_back(m);

  last_frame_preint_.IntegrateNewMeasurement(calib_, m);
  last_valid_pose.preintegration.IntegrateNewMeasurement(calib_, m);
  last_kf_preint.IntegrateNewMeasurement(calib_, m);
  integ_kf.preintegration.IntegrateNewMeasurement(calib_, m);
}

std::optional<Vector3T> SolverSfMInertial::get_gravity() const {
  auto gravity = map_.get_gravity();
  if (!gravity) {
    return std::nullopt;
  }
  Isometry3T rig_from_w = calib_.rig_from_imu() * prev_pose.w_from_imu.inverse();  // compare with curr_pose.w_from_imu

  Vector3T gravity_rig = rig_from_w.linear() * (*gravity);

  return gravity_rig;
}

std::optional<SolverSfMInertial::ImuState> SolverSfMInertial::GetImuState() const {
  if (is_first_run) return std::nullopt;
  return ImuState{prev_pose.velocity, prev_pose.gyro_bias, prev_pose.acc_bias};
}

}  // namespace cuvslam::pipelines
