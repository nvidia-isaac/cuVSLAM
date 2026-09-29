
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
#include "common/frame_id.h"
#include "common/imu_calibration.h"
#include "common/include_eigen.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "imu/imu_sba.h"
#include "imu/imu_sba_problem.h"
#include "imu/inertial_optimization.h"
#include "imu/linear_filter.h"
#include "map/map.h"
#include "map/service.h"
#include "pnp/multicam_pnp.h"
#include "profiler/profiler.h"
#include "sba/sba_config.h"

#include "pipelines/inertial_pnp.h"
#include "pipelines/sfm_solver_interface.h"
#include "pipelines/tracker_state_machine.h"
#include "pipelines/triangulator.h"

namespace cuvslam::pipelines {

class SolverSfMInertial : public ISFMSolver {
public:
  SolverSfMInertial(map::UnifiedMap& map, const camera::Rig& rig, const sba::Settings& sba_settings,
                    const imu::ImuCalibration& calib, bool debug_imu_mode, bool disable_fusion_except_gravity);
  ~SolverSfMInertial() override = default;

  const camera::Rig& getRig() const override;

  void reset() override;

  // @see base class for comment
  virtual bool solveNextFrame(int64_t time_ns, const sof::FrameState& frameState,
                              const MulticamObservations& observations, Isometry3T& world_from_rig,
                              Matrix6T& static_info_exp, const SolverPerFrameSettings& solver_settings,
                              std::vector<Track2D>* tracks2d = nullptr, Tracks3DMap* tracks3d = nullptr) override;

  void add_imu_measurement(const imu::ImuMeasurement& m);
  void set_verbose(bool verbose);

  bool predict_pose(Isometry3T& pose) const;

  std::optional<Vector3T> get_gravity() const;

  struct ImuState {
    Vector3T velocity;
    Vector3T gyro_bias;
    Vector3T acc_bias;
  };
  std::optional<ImuState> GetImuState() const;

private:
  imu::ImuMeasurementStorage imu_storage_;
  imu::ImuCalibration calib_;

  camera::Rig rig_;

  bool debug_imu_mode_;

  //    Disable Inertial PNP and Inertial SBA by default for multicamera mode to prevent errors in IMU fusion.
  //    Inertial optimizer still runs to estimate gravity.
  bool disable_fusion_except_gravity_;
  bool verbose_ = false;

  map::UnifiedMap& map_;

  std::unique_ptr<map::ServiceBase> sba_service_;

  sba_imu::InertialOptimizer optimizer_;
  sba_imu::IMUBundlerCpuFixedVel imu_init_bundler_;

  InertialPnP inertial_pnp_;
  pnp::PNPSolver stereo_pnp_;

  MulticamTriangulator triangulator;

  StateMachine imu_sm_;

  bool is_first_run = true;
  bool enable_imu_sba_init_ = true;

  sba_imu::Pose curr_pose;  // keep valid preintegration
  sba_imu::Pose prev_pose;  // keep valid preintegration
  int64_t prev_pose_ts_ns = -1;

  sba_imu::Pose last_valid_pose;  // keep valid preintegration since last valid pose
  sba_imu::Pose integ_kf;

  sba_imu::IMUPreintegration last_kf_preint;

  sba_imu::IMUPreintegration last_frame_preint_;  // since last frame for next frame
  bool integrated;

  std::vector<camera::Observation> obs_vector_;
  map::Map<TrackId, Vector3T> recent_landmarks_;
  std::unordered_map<TrackId, Track> pnp_landmarks_;
  map::Map<TrackId, Vector3T> landmark_buffer_;

  // Exports observations in left camera along with corresponding 3d points
  // out_tracks2d - output 2d track coordinates in pixels
  // out_tracks3d - in rig space
  void exportTracks(const std::vector<camera::Observation>& observations, std::vector<Track2D>& out_tracks2d,
                    Tracks3DMap& out_tracks3d, const Isometry3T& camera_from_world) const;

  sba_imu::LinearFilter3 filter_acc;
  sba_imu::LinearFilter3 filter_gyro;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("SolverSfMInertial");
};

}  // namespace cuvslam::pipelines
