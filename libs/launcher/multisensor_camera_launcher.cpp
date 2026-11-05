
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

#include "launcher/multisensor_camera_launcher.h"

#include <sstream>
#include <string>

#include "gflags/gflags.h"

// Multisensor flags. The depth-id list and stereo-tracking-for-depth toggle are NOT here —
// they're owned by libs/launcher/multi_camera_launcher_base.cpp as -fig_depth_camera_ids and
// -allow_stereo_track_for_depth (single source of truth for every launcher that builds a FIG).
DEFINE_bool(multisensor_use_imu, false,
            "Treat the rig as having a single IMU. When true the launcher auto-switches "
            "sba_mode to inertial and registers the IMU callback.");
DEFINE_double(multisensor_depth_scale, 1.0, "Depth scale factor (raw depth divided by this value yields meters)");

namespace cuvslam::launcher {

MultisensorCameraLauncher::MultisensorCameraLauncher(ICameraRig& cameraRig, const odom::Settings& svo_settings)
    // Multisensor mode always wants stereo tracking between depth-aligned cameras and other
    // cameras, so force the FIG to keep those edges regardless of the global flag.
    : MultiCameraBaseLauncher(cameraRig, svo_settings, /*auto_allow_stereo_track_for_depth=*/true) {
  TraceMessage("Multisensor launcher is selected");
}

void MultisensorCameraLauncher::SetupTracker(const odom::Settings& svo_settings, bool use_gpu) {
  // Resolve only the non-FIG flags here. depth_camera_ids and enable_depth_stereo_tracking are
  // already baked into the FIG (built in the base ctor) and into the depth-ingestion path
  // (BaseLauncher::launch -> isDepthCamera). We mirror them into MultisensorSettings so
  // MultisensorOdometry sees the exact same set the FIG and the launcher were built with.
  odom::Settings effective = svo_settings;
  odom::MultisensorSettings& ms = effective.multisensor_settings;

  if (!gflags::GetCommandLineFlagInfoOrDie("multisensor_use_imu").is_default) {
    ms.with_imu = FLAGS_multisensor_use_imu;
  } else if (effective.sba_settings.mode == sba::InertialCPU || effective.sba_settings.mode == sba::InertialGPU) {
    // Backwards compatibility: caller explicitly chose an inertial sba_mode but didn't set
    // multisensor_use_imu — treat that as a request to enable IMU fusion.
    ms.with_imu = true;
  }

  if (!gflags::GetCommandLineFlagInfoOrDie("multisensor_depth_scale").is_default) {
    ms.depth_scale_factor = static_cast<float>(FLAGS_multisensor_depth_scale);
  }

  // Sync depth_camera_ids with the FIG/ingestion source of truth (resolved once in the base ctor).
  ms.depth_camera_ids.clear();
  ms.depth_camera_ids.reserve(depth_ids_.size());
  for (CameraId id : depth_ids_) {
    ms.depth_camera_ids.push_back(static_cast<int32_t>(id));
  }
  // Multisensor mode always allows stereo 2D tracking between depth-aligned cameras and other
  // cameras (the launcher passed auto_allow_stereo_track_for_depth=true to the base ctor; mirror
  // that here so the odometry settings agree with the FIG topology).
  ms.enable_depth_stereo_tracking = true;

  // Auto-switch SBA mode to inertial when an IMU is part of the configured sensor set.
  // Pure-visual multisensor leaves the mode untouched (caller-selected, e.g. OriginalGPU).
  if (ms.with_imu) {
    if (effective.sba_settings.mode != sba::InertialCPU && effective.sba_settings.mode != sba::InertialGPU) {
      TraceMessage("Multisensor: IMU present, auto-switching sba_mode to InertialCPU");
      effective.sba_settings.mode = sba::InertialCPU;
    }
  }

  {
    std::stringstream s;
    s << "Multisensor sensor set: imu=" << (ms.with_imu ? "yes" : "no") << ", depth_cameras=[";
    for (size_t i = 0; i < ms.depth_camera_ids.size(); ++i) {
      if (i > 0) {
        s << ",";
      }
      s << ms.depth_camera_ids[i];
    }
    s << "], depth_scale=" << ms.depth_scale_factor
      << ", stereo_depth_tracking=" << (ms.enable_depth_stereo_tracking ? "on" : "off");
    TraceMessage("%s", s.str().c_str());
  }

  // Persist effective settings back into the launcher's stored copy so downstream code paths
  // (per-frame settings, statistics, etc.) see the resolved configuration.
  svo_settings_ = effective;

  tracker = std::make_unique<odom::MultisensorOdometry>(rig, fig, effective, use_gpu);
  if (ms.with_imu) {
    cameraRig_.registerIMUCallback(
        [this](const imu::ImuMeasurement& measurement) { tracker->add_imu_measurement(measurement); });
  }
  tracker->enable_stat(true);
}

bool MultisensorCameraLauncher::launch_vo(Isometry3T& delta, Matrix6T& pose_info,
                                          const odom::TrackPerFrameSettings& per_frame) {
  return tracker->track(curr_sources, depth_sources, curr_image_ptrs, prev_image_ptrs, masks_sources, delta, pose_info,
                        per_frame);
}

const odom::IVisualOdometry::VOFrameStat& MultisensorCameraLauncher::last_vo_stat() {
  return *tracker->get_last_stat();
}
}  // namespace cuvslam::launcher
