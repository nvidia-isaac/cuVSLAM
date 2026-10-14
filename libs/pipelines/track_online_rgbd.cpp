
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

#include "pipelines/track_online_rgbd.h"

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/frame_id.h"
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

SolverSfMRGBD::SolverSfMRGBD(map::UnifiedMap& map, const camera::Rig& rig, const sba::Settings& sba_settings)
    : rig_(rig), map_(map), triangulator(rig), visual_icp_(rig) {
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

const camera::Rig& SolverSfMRGBD::getRig() const { return rig_; }

void SolverSfMRGBD::reset() {
  triangulator.reset();
  if (sba_service_) {
    sba_service_->restart();
  }

  // map will be cleared outside of this function
}

bool SolverSfMRGBD::solveNextFrame(int64_t time_ns, const sof::FrameState& frameState, const SFMInputs& inputs,
                                   Isometry3T& world_from_rig, Matrix6T& static_info_exp,
                                   const SolverPerFrameSettings& solver_settings, std::vector<Track2D>* tracks2d,
                                   Tracks3DMap* tracks3d) {
  TRACE_EVENT ev = profiler_domain_.trace_event("SolverSfMRGBD::solveNextFrame()", profiler_color_);

  obs_vector_.clear();
  size_t num_observations = 0;
  for (const auto& cam_observations : inputs.observations) {
    num_observations += cam_observations.size();
  }
  obs_vector_.reserve(num_observations);
  for (const auto& obs : inputs.observations) {
    std::copy(obs.begin(), obs.end(), std::back_inserter(obs_vector_));
  }

  RERUN(logObservations, obs_vector_, rig_, "world/camera_0/images/observations", Color(255, 165, 0));

  bool result = true;

  if (map_.empty()) {
    static_info_exp.setZero();
  } else {
    map_.get_recent_landmarks(recent_landmarks_);
    RERUN(logLandmarks3D, recent_landmarks_, "world/camera_0/images/sba_landmarks", Color(255, 255, 0), 0.01f);
    RERUN(logLandmarks, recent_landmarks_, prev_rig_from_world_, *rig_.intrinsics[0], "world/camera_0/images/landmarks",
          Color(255, 255, 0));

    Isometry3T rig_from_world = prev_rig_from_world_;  // try to optimize copy, use result if success only

    bool res = visual_icp_.solve(rig_from_world, static_info_exp, obs_vector_, recent_landmarks_, solver_settings.icp,
                                 inputs.depth_info);

    RERUN(logLandmarks, recent_landmarks_, rig_from_world, *rig_.intrinsics[0], "world/camera_0/images/landmarks",
          Color(255, 255, 0));

    RERUN(logTrajectory, rig_from_world, "world/trajectories/vo_trajectory", Color(0, 255, 0), TrajectoryType::VO);

    if (res) {
      prev_rig_from_world_ = rig_from_world;
      prev_static_info_exp_ = static_info_exp;
    } else {
      static_info_exp = prev_static_info_exp_;
      result = false;
    }
  }

  world_from_rig = prev_rig_from_world_.inverse();

  if (frameState == sof::FrameState::Key) {
    auto tr_landmarks = triangulator.triangulate(world_from_rig, obs_vector_);

    if (inputs.depth_info) {
      std::vector<cuvslam::pipelines::Landmark> mono_landmarks;
      visual_icp_.lift_mono_tracks(*inputs.depth_info, world_from_rig, obs_vector_, mono_landmarks);

      std::move(mono_landmarks.begin(), mono_landmarks.end(), std::back_inserter(tr_landmarks));
    }

    map_.add_keyframe(time_ns, {prev_rig_from_world_}, {},  // preintegration
                      obs_vector_, tr_landmarks);
    if (sba_service_) {
      static_cast<SbaServiceBase*>(sba_service_.get())->trigger(solver_settings.sba);
    }
  }

  if (tracks2d && tracks3d) {
    exportTracks(obs_vector_, *tracks2d, *tracks3d, prev_rig_from_world_);
  }

  return result;
}

void SolverSfMRGBD::exportTracks(const std::vector<camera::Observation>& observations,
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

}  // namespace cuvslam::pipelines
