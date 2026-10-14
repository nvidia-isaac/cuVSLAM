
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
#include <optional>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/imu_calibration.h"
#include "common/imu_measurement.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "map/map.h"
#include "map/service.h"
#include "pnp/multisensor_pose_estimator.h"
#include "profiler/profiler.h"
#include "sba/sba_config.h"

#include "pipelines/depth_maps_context.h"
#include "pipelines/imu_fusion_context.h"
#include "pipelines/sfm_solver_interface.h"

namespace cuvslam::pipelines {

// Pipeline solver using cuNLS-based MultisensorPoseEstimator for the pose estimation step.
//
// The solver supports four rig configurations through one code path:
//   * pure multi-camera RGB (no depth) without IMU
//   * pure multi-camera RGB (no depth) with IMU
//   * mono / multi-camera RGBD with IMU
//   * mono / multi-camera RGBD without IMU
//
// IMU fusion is enabled by passing an ImuCalibration to the dedicated constructor; without it the
// solver falls back to a pure-visual RGBD pipeline. All IMU-side state lives in ImuFusionContext;
// all depth/triangulation/plane state lives in DepthMapsContext.
class SolverSfMMultisensor {
public:
  SolverSfMMultisensor(map::UnifiedMap& map, const camera::Rig& rig, const sba::Settings& sba_settings);

  SolverSfMMultisensor(map::UnifiedMap& map, const camera::Rig& rig, const sba::Settings& sba_settings,
                       const cuvslam::imu::ImuCalibration& calib);

  const camera::Rig& getRig() const;

  bool solveNextFrame(int64_t time_ns, const sof::FrameState& frameState, const MulticamObservations& observations,
                      const pnp::RGBDInfos& depth_infos, Isometry3T& world_from_rig, Matrix6T& static_info_exp,
                      std::vector<Track2D>* tracks2d = nullptr, Tracks3DMap* tracks3d = nullptr);

  void reset();

  // IMU integration (no-op when constructed without IMU calibration).
  void add_imu_measurement(const cuvslam::imu::ImuMeasurement& m);
  std::optional<Vector3T> get_gravity() const;

  struct ImuState {
    Vector3T velocity;
    Vector3T gyro_bias;
    Vector3T acc_bias;
  };
  std::optional<ImuState> GetImuState() const;

  bool has_imu() const { return with_imu_; }

private:
  bool solveNextFrameVisualOnly(int64_t time_ns, const sof::FrameState& frameState,
                                const MulticamObservations& observations, const pnp::RGBDInfos& depth_infos,
                                Isometry3T& world_from_rig, Matrix6T& static_info_exp, std::vector<Track2D>* tracks2d,
                                Tracks3DMap* tracks3d);

  bool solveNextFrameInertial(int64_t time_ns, const sof::FrameState& frameState,
                              const MulticamObservations& observations, const pnp::RGBDInfos& depth_infos,
                              Isometry3T& world_from_rig, Matrix6T& static_info_exp, std::vector<Track2D>* tracks2d,
                              Tracks3DMap* tracks3d);

  // Triangulate, lift, push the keyframe to the map, refresh planes, and notify SBA.
  void process_keyframe(int64_t time_ns, const Isometry3T& world_from_rig, const std::vector<camera::Observation>& obs,
                        const pnp::RGBDInfos& depth_infos, const map::State& state,
                        const sba_imu::IMUPreintegration& preint);

  // Exports per-camera observations along with corresponding 3d points
  // out_tracks2d - output 2d track coordinates in pixels
  // out_tracks3d - in rig space
  void exportTracks(const std::vector<camera::Observation>& observations, std::vector<Track2D>& out_tracks2d,
                    Tracks3DMap& out_tracks3d, const Isometry3T& rig_from_world) const;

  camera::Rig rig_;
  map::UnifiedMap& map_;

  std::unique_ptr<map::ServiceBase> sba_service_;

  pnp::MultisensorPoseEstimator pose_estimator_;
  DepthMapsContext depth_maps_;

  // Visual-only path state.
  Isometry3T prev_rig_from_world_{Isometry3T::Identity()};
  Matrix6T prev_static_info_exp_{Matrix6T::Zero()};

  // Inertial path state. Allocated only when constructed with an IMU calibration.
  bool with_imu_ = false;
  std::unique_ptr<ImuFusionContext> imu_;

  // StateMachine settings. The inertial path forwards these to ImuFusionContext::finalize_frame
  // every frame so that runtime updates from ApplyPersistentInternalParameters take immediate effect. Multisensor
  // currently uses defaults; full ApplyPersistentInternalParameters wiring (via SolverPerFrameSettings) will be
  // added when this solver is converted to the ISFMSolver interface.
  StateMachineSettings sm_settings_;

  std::vector<camera::Observation> obs_vector_;
  map::Map<TrackId, Vector3T> recent_landmarks_;

  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0x00FF00;
};

}  // namespace cuvslam::pipelines
