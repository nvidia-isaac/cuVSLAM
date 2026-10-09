
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

#include "pipelines/track_online_multisensor.h"

#include <algorithm>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/frame_id.h"
#include "common/include_eigen.h"
#include "common/isometry.h"
#include "common/rerun.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "pipelines/visualizer.h"

#include "pipelines/service_sba.h"
#ifdef USE_CUDA
#include "pipelines/service_sba_gpu.h"
#endif

namespace cuvslam::pipelines {

namespace {

void flatten_observations(const MulticamObservations& multicam_obs, std::vector<camera::Observation>& obs_vector) {
  obs_vector.clear();
  size_t num_observations = 0;
  for (const auto& cam_observations : multicam_obs) {
    num_observations += cam_observations.size();
  }
  obs_vector.reserve(num_observations);
  for (const auto& observations : multicam_obs) {
    std::copy(observations.begin(), observations.end(), std::back_inserter(obs_vector));
  }
}

void warn_if_non_pinhole(const camera::Rig& rig) {
  for (int32_t i = 0; i < rig.num_cameras; ++i) {
    if (dynamic_cast<const camera::PinholeCameraModel*>(rig.intrinsics[i]) == nullptr) {
      TraceWarning("SolverSfMMultisensor supports only pinhole cameras. Camera %d uses a different model.", i);
    }
  }
}

}  // namespace

SolverSfMMultisensor::SolverSfMMultisensor(map::UnifiedMap& map, const camera::Rig& rig,
                                           const sba::Settings& sba_settings)
    : rig_(rig), map_(map), pose_estimator_(rig), depth_maps_(rig_), with_imu_(false) {
  warn_if_non_pinhole(rig_);

  const auto& sba_mode = sba_settings.mode;
  if (sba_mode != sba::OriginalCPU && sba_mode != sba::OriginalGPU && sba_mode != sba::Disabled) {
    TraceError("Original VO cant run with inertial SBA");
  }

  switch (sba_mode) {
    case sba::OriginalCPU:
      sba_service_ = std::make_unique<CpuSbaService>(sba_settings, rig, map_);
      break;
#ifdef USE_CUDA
    case sba::OriginalGPU:
      sba_service_ = std::make_unique<GpuSbaService>(sba_settings, rig, map_);
      break;
#endif
    default:
      sba_service_ = nullptr;
      break;
  }
}

SolverSfMMultisensor::SolverSfMMultisensor(map::UnifiedMap& map, const camera::Rig& rig,
                                           const sba::Settings& sba_settings, const cuvslam::imu::ImuCalibration& calib)
    : rig_(rig), map_(map), pose_estimator_(rig), depth_maps_(rig_), with_imu_(true) {
  warn_if_non_pinhole(rig_);

  const auto& sba_mode = sba_settings.mode;
  switch (sba_mode) {
    case sba::InertialCPU:
      sba_service_ = std::make_unique<ImuSbaCPUService>(sba_settings, rig, calib, map_);
      break;
#ifdef USE_CUDA
    case sba::InertialGPU:
      sba_service_ = std::make_unique<ImuSbaGPUService>(sba_settings, rig, calib, map_);
      break;
#endif
    default:
      sba_service_ = nullptr;
      break;
  }

  imu_ = std::make_unique<ImuFusionContext>(map_, rig_, calib, pose_estimator_, sba_mode == sba::InertialGPU);
  imu_->register_gravity_callback();
}

const camera::Rig& SolverSfMMultisensor::getRig() const { return rig_; }

void SolverSfMMultisensor::reset() {
  depth_maps_.reset();
  if (sba_service_) {
    sba_service_->restart();
  }
  if (imu_) {
    imu_->reset();
  }
  prev_rig_from_world_ = Isometry3T::Identity();
  prev_static_info_exp_.setZero();
}

bool SolverSfMMultisensor::solveNextFrame(int64_t time_ns, const sof::FrameState& frameState,
                                          const MulticamObservations& observations, const pnp::RGBDInfos& depth_infos,
                                          Isometry3T& world_from_rig, Matrix6T& static_info_exp,
                                          std::vector<Track2D>* tracks2d, Tracks3DMap* tracks3d) {
  if (with_imu_) {
    return solveNextFrameInertial(time_ns, frameState, observations, depth_infos, world_from_rig, static_info_exp,
                                  tracks2d, tracks3d);
  }
  return solveNextFrameVisualOnly(time_ns, frameState, observations, depth_infos, world_from_rig, static_info_exp,
                                  tracks2d, tracks3d);
}

bool SolverSfMMultisensor::solveNextFrameVisualOnly(int64_t time_ns, const sof::FrameState& frameState,
                                                    const MulticamObservations& observations,
                                                    const pnp::RGBDInfos& depth_infos, Isometry3T& world_from_rig,
                                                    Matrix6T& static_info_exp, std::vector<Track2D>* tracks2d,
                                                    Tracks3DMap* tracks3d) {
  TRACE_EVENT ev = profiler_domain_.trace_event("SolverSfMMultisensor::solveNextFrame()", profiler_color_);

  flatten_observations(observations, obs_vector_);
  bool result = true;

  if (map_.empty()) {
    static_info_exp.setZero();
  } else {
    map_.get_recent_landmarks(recent_landmarks_);

    Isometry3T rig_from_world = prev_rig_from_world_;
    bool res;
    {
      TRACE_EVENT pnp_ev = profiler_domain_.trace_event("pose_estimator.solve");
      res = pose_estimator_.solve(rig_from_world, static_info_exp, obs_vector_, recent_landmarks_, depth_infos,
                                  depth_maps_.planes(), depth_maps_.depth_points());
    }

    RERUN(logTrajectory, rig_from_world, "world/trajectories/vo_trajectory", Color(0, 255, 0), TrajectoryType::VO);

    if (res) {
      prev_rig_from_world_ = rig_from_world;
      prev_static_info_exp_ = static_info_exp;
    } else {
      static_info_exp = prev_static_info_exp_;
      result = false;
    }

    {
      TRACE_EVENT dpm_ev = profiler_domain_.trace_event("depth_point_map.update");
      if (depth_maps_.update_post_solve(depth_infos, prev_rig_from_world_.inverse())) {
        RERUN(logPoints3D, depth_maps_.depth_points(), "world/depth_points", Color(0, 200, 255), 0.008f);
      }
    }
  }

  world_from_rig = prev_rig_from_world_.inverse();
  RERUN(logCameraFrustums, rig_, world_from_rig, "world/frustums");

  if (frameState == sof::FrameState::Key) {
    map::State state{prev_rig_from_world_};
    process_keyframe(time_ns, world_from_rig, obs_vector_, depth_infos, state, sba_imu::IMUPreintegration{});
  }

  if (tracks2d && tracks3d) {
    TRACE_EVENT export_ev = profiler_domain_.trace_event("exportTracks");
    exportTracks(obs_vector_, *tracks2d, *tracks3d, prev_rig_from_world_);
  }

  return result;
}

bool SolverSfMMultisensor::solveNextFrameInertial(int64_t time_ns, const sof::FrameState& frameState,
                                                  const MulticamObservations& observations,
                                                  const pnp::RGBDInfos& depth_infos, Isometry3T& world_from_rig,
                                                  Matrix6T& static_info_exp, std::vector<Track2D>* tracks2d,
                                                  Tracks3DMap* tracks3d) {
  TRACE_EVENT ev = profiler_domain_.trace_event("SolverSfMMultisensor::solveNextFrame(inertial)", profiler_color_);

  flatten_observations(observations, obs_vector_);
  RERUN(logObservations, obs_vector_, rig_, "world/camera_0/images/observations", Color(255, 165, 0));

  auto prep = imu_->prepare_frame(time_ns);

  bool pnp_result = false;
  if (!map_.empty()) {
    map_.get_recent_landmarks(recent_landmarks_);

    if (prep.imu_state == StateMachine::State::Ok) {
      pnp_result = imu_->solve_inertial(recent_landmarks_, obs_vector_, depth_infos, depth_maps_.planes(),
                                        depth_maps_.depth_points(), time_ns);
    } else {
      pnp_result = imu_->solve_visual(recent_landmarks_, obs_vector_, depth_infos, depth_maps_.planes(),
                                      depth_maps_.depth_points(), time_ns);
    }
  }

  auto frame = imu_->finalize_frame(pnp_result, prep.no_drops, time_ns, frameState, sm_settings_);
  world_from_rig = frame.world_from_rig;
  static_info_exp = frame.static_info_exp;
  prev_rig_from_world_ = world_from_rig.inverse();

  RERUN(logTrajectory, prev_rig_from_world_, "world/trajectories/vo_trajectory", Color(0, 255, 0), TrajectoryType::VO);

  // Visualise IMU-fusion intermediates anchored at the rig origin in world frame:
  //   - velocity (cyan) at true magnitude (m/s ↔ m on the world axes)
  //   - gravity (red) scaled by 0.1 so a 9.81 m/s² vector renders as a ~1 m arrow
  if (auto state = imu_->has_imu_state() ? std::optional<Vector3T>{imu_->velocity()} : std::nullopt; state) {
    [[maybe_unused]] const Vector3T origin = world_from_rig.translation();
    RERUN(logVector3D, *state, origin, "world/imu/velocity", Color(0, 200, 200), 1.0f);
    if (auto g = map_.get_gravity(); g) {
      RERUN(logVector3D, *g, origin, "world/imu/gravity", Color(220, 60, 60), 0.1f);
    }
  }

  if (frame.lost) {
    return false;
  }

  // Refresh depth-point map under the latest pose so the next frame's ICP factor is anchored
  // correctly. No-op for depth-less rigs.
  {
    TRACE_EVENT dpm_ev = profiler_domain_.trace_event("depth_point_map.update");
    if (depth_maps_.update_post_solve(depth_infos, world_from_rig)) {
      RERUN(logPoints3D, depth_maps_.depth_points(), "world/depth_points", Color(0, 200, 255), 0.008f);
    }
  }

  RERUN(logCameraFrustums, rig_, world_from_rig, "world/frustums");

  if (frameState == sof::FrameState::Key) {
    map::State state = imu_->build_keyframe_state(prev_rig_from_world_, frame.lost);
    process_keyframe(time_ns, world_from_rig, obs_vector_, depth_infos, state, imu_->last_kf_preint());
    imu_->on_keyframe_committed(frame.lost);
  }

  if (tracks2d && tracks3d) {
    TRACE_EVENT export_ev = profiler_domain_.trace_event("exportTracks");
    exportTracks(obs_vector_, *tracks2d, *tracks3d, prev_rig_from_world_);
  }

  return true;
}

void SolverSfMMultisensor::process_keyframe(int64_t time_ns, const Isometry3T& world_from_rig,
                                            const std::vector<camera::Observation>& obs,
                                            const pnp::RGBDInfos& depth_infos, const map::State& state,
                                            const sba_imu::IMUPreintegration& preint) {
  TRACE_EVENT kf_ev = profiler_domain_.trace_event("keyframe_processing");

  std::vector<Landmark> landmarks;
  {
    TRACE_EVENT tri_ev = profiler_domain_.trace_event("triangulate_and_lift");
    landmarks = depth_maps_.build_keyframe_landmarks(world_from_rig, obs, depth_infos);
  }

  map_.add_keyframe(time_ns, state, preint, obs, landmarks);

  {
    TRACE_EVENT pm_ev = profiler_domain_.trace_event("plane_map_update");
    depth_maps_.update_at_keyframe(depth_infos, world_from_rig);
  }
  RERUN(logPlanes, depth_maps_.planes(), "world/planes");

  if (sba_service_) {
    TRACE_EVENT sba_ev = profiler_domain_.trace_event("sba_notify");
    sba_service_->notify();
  }
}

void SolverSfMMultisensor::exportTracks(const std::vector<camera::Observation>& observations,
                                        std::vector<Track2D>& out_tracks2d, Tracks3DMap& out_tracks3d,
                                        const Isometry3T& rig_from_world) const {
  out_tracks2d.clear();
  out_tracks3d.clear();

  for (const camera::Observation& obs : observations) {
    const camera::ICameraModel& camera = *rig_.intrinsics[obs.cam_id];
    Vector2T uv;
    if (camera.denormalizePoint(obs.xy, uv)) {
      out_tracks2d.push_back({obs.cam_id, obs.id, uv});
    }
  }

  auto map_landmarks = map_.get_recent_landmarks();
  for (const camera::Observation& obs : observations) {
    if (map_landmarks.find(obs.id) != map_landmarks.end()) {
      const Vector3T& point_3d = map_landmarks.at(obs.id);
      out_tracks3d[obs.id] = rig_from_world * point_3d;
    }
  }
}

void SolverSfMMultisensor::add_imu_measurement(const cuvslam::imu::ImuMeasurement& m) {
  if (!imu_) {
    return;
  }
  imu_->add_measurement(m);
}

std::optional<Vector3T> SolverSfMMultisensor::get_gravity() const {
  if (!imu_) {
    return std::nullopt;
  }
  return imu_->get_gravity();
}

std::optional<SolverSfMMultisensor::ImuState> SolverSfMMultisensor::GetImuState() const {
  if (!imu_ || !imu_->has_imu_state()) {
    return std::nullopt;
  }
  return ImuState{imu_->velocity(), imu_->gyro_bias(), imu_->acc_bias()};
}

}  // namespace cuvslam::pipelines
