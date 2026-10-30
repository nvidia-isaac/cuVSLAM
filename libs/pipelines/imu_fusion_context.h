
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

#include <optional>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/imu_calibration.h"
#include "common/imu_measurement.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "imu/imu_bundler.h"
#include "imu/imu_preintegration.h"
#include "imu/imu_sba.h"
#include "imu/imu_sba_problem.h"
#include "imu/inertial_optimization.h"
#include "map/depth_point_map.h"
#include "map/keyframe.h"
#include "map/map.h"
#include "map/plane_map.h"
#include "pnp/multisensor_pose_estimator.h"
#include "profiler/profiler.h"
#include "sof/sof_multicamera_interface.h"

#include "pipelines/tracker_state_machine.h"

namespace cuvslam::pipelines {

// ImuFusionContext owns the IMU-side state and operations that wrap the visual RGBD pose estimator
// for SolverSfMMultisensor. It encapsulates:
//   * IMU measurement buffer + preintegrations (frame, last-valid, last-keyframe, keyframe-anchor)
//   * sba_imu::Pose snapshots (prev/curr/last_valid/integ_kf)
//   * StateMachine driving gravity initialization
//   * The gravity-estimation callback (NEC + SBA-init refinement of map keyframes)
//
// The same code path serves both:
//   * pure multi-RGB rigs (depth_infos is empty -> IMU SE3 prior tethers the cuNLS solve)
//   * mono / multi-camera RGBD rigs (depth/plane factors + IMU SE3 prior together)
class ImuFusionContext {
public:
  ImuFusionContext(map::UnifiedMap& map, const camera::Rig& rig, const imu::ImuCalibration& calib,
                   const pnp::MultisensorPoseEstimator& pose_estimator, bool use_gpu_bundler);

  // Wires the gravity-estimation callback into the internal StateMachine. Must be called once after
  // construction (kept separate so the lambda does not capture `this` inside the constructor).
  void register_gravity_callback();

  // Clears IMU state. Called from SolverSfMMultisensor::reset().
  void reset();

  // Pushes a new IMU measurement and integrates it into all live preintegrations.
  void add_measurement(const imu::ImuMeasurement& m);

  struct FramePrep {
    Isometry3T predicted_world_from_rig = Isometry3T::Identity();
    bool no_drops = false;
    StateMachine::State imu_state = StateMachine::State::Uninitialized;
  };
  // Re-anchors prev_pose to the latest frame preintegration, snapshots curr_pose, and reports the
  // IMU-predicted world_from_rig and per-frame drop status.
  FramePrep prepare_frame(int64_t time_ns);

  // Solve visual + IMU SE3 prior with the cuNLS RGBD estimator. Updates curr_pose's translation
  // (from the solve) and velocity/biases (from prev_pose + visual delta).
  bool solve_inertial(const std::unordered_map<TrackId, Vector3T>& landmarks,
                      const std::vector<camera::Observation>& observations, const pnp::RGBDInfos& depth_infos,
                      const std::vector<map::Plane>& planes, const std::vector<Vector3T>& depth_points,
                      int64_t time_ns);

  // Pure-visual RGBD solve while IMU is still uninitialized. Same data flow, just without the IMU
  // SE3 prior factor.
  bool solve_visual(const std::unordered_map<TrackId, Vector3T>& landmarks,
                    const std::vector<camera::Observation>& observations, const pnp::RGBDInfos& depth_infos,
                    const std::vector<map::Plane>& planes, const std::vector<Vector3T>& depth_points, int64_t time_ns);

  struct FrameResult {
    Isometry3T world_from_rig = Isometry3T::Identity();
    Matrix6T static_info_exp = Matrix6T::Zero();
    bool valid_pose = false;  // pnp succeeded or IMU integration produced a pose
    bool lost = false;        // both visual and IMU paths failed
  };
  // Drives the StateMachine, refreshes last_valid_pose, performs IMU-only integration if the visual
  // solve failed, swaps prev/curr, refreshes biases from the latest map keyframe, and updates the
  // per-frame preintegration anchor.
  FrameResult finalize_frame(bool pnp_result, bool no_drops, int64_t time_ns, sof::FrameState fs,
                             const StateMachineSettings& sm_settings);

  // Builds the State to attach to a new keyframe (uses last_valid_pose when the current frame was
  // lost so the map remains consistent).
  map::State build_keyframe_state(const Isometry3T& rig_from_world, bool lost) const;

  // Returns the preintegration to attach to the keyframe that is being committed.
  const sba_imu::IMUPreintegration& last_kf_preint() const { return last_kf_preint_; }

  // Resets last_kf_preint_ to a fresh preintegration starting from the current bias estimate.
  void on_keyframe_committed(bool lost);

  std::optional<Vector3T> get_gravity() const;

  bool has_imu_state() const { return !is_first_run_; }
  Vector3T velocity() const { return prev_pose_.velocity; }
  Vector3T gyro_bias() const { return prev_pose_.gyro_bias; }
  Vector3T acc_bias() const { return prev_pose_.acc_bias; }

private:
  // Body of the SBA-init step run after gravity optimization. Re-triangulates landmarks using
  // IMU-refined poses and refines (poses, points, biases, velocities) jointly with cuNLS-CPU.
  void run_imu_sba_init(const map::UnifiedMap::SubMap& recent_map, const Vector3T& gravity,
                        std::vector<sba_imu::Pose>& gravity_cache,
                        const std::vector<std::vector<camera::Observation>>& observations_per_kf);

  // Body of the gravity-estimation callback registered with imu_sm_.
  bool gravity_estimation_callback(size_t num_kfs);

  // Convenience: snapshot bias/velocity from the most recent keyframe state to all live poses and
  // re-integrate their preintegrations. Mirrors the post-gravity-init bookkeeping.
  void snap_to_keyframe_pose(const sba_imu::Pose& last_kf_pose);

  map::UnifiedMap& map_;
  const camera::Rig& rig_;
  imu::ImuCalibration calib_;
  const pnp::MultisensorPoseEstimator& pose_estimator_;
  imu::ImuMeasurementStorage imu_storage_;

  sba_imu::InertialOptimizer optimizer_;
  sba_imu::IMUBundler init_bundler_;
  StateMachine sm_;

  bool enable_imu_sba_init_ = true;
  bool is_first_run_ = true;
  bool integrated_ = false;
  int64_t prev_pose_ts_ns_ = -1;

  sba_imu::Pose prev_pose_;        // last accepted rig pose, used as the IMU integration anchor
  sba_imu::Pose curr_pose_;        // working pose for the current frame
  sba_imu::Pose last_valid_pose_;  // last rig pose with valid inertial PnP — fallback for keyframes
  sba_imu::Pose integ_kf_;         // pose anchored at the latest keyframe for IMU-only integration

  sba_imu::IMUPreintegration last_kf_preint_;     // preint accumulated since the last committed keyframe
  sba_imu::IMUPreintegration last_frame_preint_;  // preint accumulated since the last frame

  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("ImuFusionContext");
};

}  // namespace cuvslam::pipelines
