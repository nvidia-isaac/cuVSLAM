
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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "common/imu_calibration.h"
#include "pipelines/inertial_pnp.h"
#include "pipelines/tracker_state_machine.h"
#include "pnp/multicam_pnp.h"
#include "sba/sba_config.h"
#include "sof/sof_config.h"
#ifdef USE_CUDA
#include "pnp/visual_icp.h"
#endif

namespace cuvslam::odom {

struct KeyFrameSettings {
  // The current frame becomes keyframe if the percent of survivor tracks
  // from the last keyframe is less than this value
  float survivor_from_last = 41.f;

  // maximum timedelta between consecutive keyframes in seconds
  int64_t max_timedelta_between_kfs_s = 60;

  // Explicit per-frame keyframe decision. std::nullopt uses automatic keyframe checks.
  std::optional<bool> override_frame_selection;
};

// Bundles all per-frame setting overrides into a single struct
struct TrackPerFrameSettings {
  sof::Settings sof;
  KeyFrameSettings kf;
  sba::Settings sba;
  pipelines::StateMachineSettings sm;
  pnp::PNPSettings vo_pnp;
  pnp::PNPSettings inertial_stereo_pnp = pnp::PNPSettings::InertialSettings();
  pipelines::InertialPnPSettings imu_pnp;
#ifdef USE_CUDA
  pnp::ICPSettings icp;
#endif
};

// Configuration of the available sensor set for MultisensorOdometry.
//
// MultisensorOdometry supports any mix of:
//   * one or more plain RGB cameras
//   * one or more RGB-D cameras (depth available per camera)
//   * a single optional IMU
// The minimum rig has at least one RGB-D camera or one overlapping camera pair. Any
// valid visual configuration may additionally include one IMU.
// Construction-time fields here describe which inputs the solver should expect.
struct MultisensorSettings {
  // True if a single IMU is part of the rig and IMU measurements will be supplied via
  // add_imu_measurement(). When true, sba_settings.mode must be InertialCPU or
  // InertialGPU; the launcher and public API enforce this automatically. Default: false.
  bool with_imu = false;

  // Camera ids that provide depth measurements. Empty = no depth (pure multi-camera-RGB).
  // Each listed camera will deliver a depth image alongside its RGB frame at track() time;
  // any camera not in this list is treated as RGB-only. Mixed rigs (some RGB, some RGB-D)
  // are supported by listing only the depth-capable cameras here.
  std::vector<int32_t> depth_camera_ids;

  // Scale factor for depth measurements (denominator to convert raw depth values to meters).
  // Applied uniformly across all depth cameras. Used only when depth_camera_ids is non-empty.
  // Default: 1.0.
  float depth_scale_factor = 1.f;

  // Allow stereo 2D tracking between depth-aligned cameras and other cameras.
  // Used only when depth_camera_ids is non-empty. Default: true (multisensor mode benefits
  // from cross-camera 2D tracks; matches the public API default).
  bool enable_depth_stereo_tracking = true;
};

struct Settings {
  sba::Settings sba_settings;
  sof::Settings sof_settings;
  cuvslam::imu::ImuCalibration imu_calibration;
  KeyFrameSettings kf_settings;
  pipelines::StateMachineSettings sm_settings;
  MultisensorSettings multisensor_settings;
  bool verbose = false;
  bool use_prediction = true;
};

}  // namespace cuvslam::odom
