
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

#include "launcher/launcher_create.h"

#include <stdexcept>

#include "gflags/gflags.h"

#include "launcher/monocular_launcher.h"
#include "launcher/multi_camera_launcher.h"
#include "launcher/visual_inertial_launcher.h"
#ifdef USE_CUDA
#include "launcher/rgbd_camera_launcher.h"
#endif
#ifdef USE_CUNLS
#include "launcher/multisensor_camera_launcher.h"
#endif

DEFINE_string(mode, "multicamera", "Choose from, 'mono', 'multicamera', 'inertial', 'rgbd', 'multisensor'");

namespace cuvslam::launcher {

std::unique_ptr<BaseLauncher> CreateLauncher(ICameraRig& rig, const odom::Settings& svo_settings) {
  if (FLAGS_mode == "mono") {
    return std::make_unique<MonocularLauncher>(rig, svo_settings);
  } else if (FLAGS_mode == "multicamera") {
    return std::make_unique<MultiCameraLauncher>(rig, svo_settings);
  } else if (FLAGS_mode == "inertial") {
    TraceWarningIf(svo_settings.sba_settings.mode != sba::Mode::InertialCPU,
                   "For inertial mode only -sba_mode=imu is supported.");
    return std::make_unique<VisualInertialLauncher>(rig, svo_settings);
  } else if (FLAGS_mode == "rgbd") {
#ifdef USE_CUDA
    return std::make_unique<RGBDCameraLauncher>(rig, svo_settings);
#else
    throw std::invalid_argument{"Unsupported mode. The rgbd mode requires build with cuda support."};
#endif
  } else if (FLAGS_mode == "multisensor") {
#ifdef USE_CUNLS
    return std::make_unique<MultisensorCameraLauncher>(rig, svo_settings);
#else
    throw std::invalid_argument{"Unsupported mode. The multisensor mode requires build with USE_CUDA and USE_CUNLS."};
#endif
  }
  throw std::invalid_argument{"Unsupported mode."};
}

}  // namespace cuvslam::launcher
