
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

#include "gflags/gflags.h"

#include "sba/sba_config.h"

DEFINE_bool(async_sba, false, "Use SBA in async mode");
DEFINE_string(sba_mode, "gpu", "SBA mode (none|cpu|gpu|imu|imugpu)");
DEFINE_int32(num_sba_frames, 7, "Number of key frames in SBA");
DEFINE_int32(num_fixed_sba_frames, 3, "Number of fixed key frames in SBA");
DEFINE_int32(num_sba_iterations, 7, "Maximum number of iterations for SBA");
DEFINE_double(robustifier_scale, 0.5, "Huber loss threshold");
DEFINE_bool(use_sba_winsorizer, false, "Toggle winsorization in SBA");

namespace cuvslam::sba {

bool ParseSettings(Settings& settings) {
  if (FLAGS_sba_mode == "cpu") {
    settings.mode = sba::Mode::OriginalCPU;
  } else if (FLAGS_sba_mode == "gpu") {
    settings.mode = sba::Mode::OriginalGPU;
  } else if (FLAGS_sba_mode == "imu") {
    settings.mode = sba::Mode::InertialCPU;
  } else if (FLAGS_sba_mode == "imugpu") {
    settings.mode = sba::Mode::InertialGPU;
  } else if (FLAGS_sba_mode == "none") {
    settings.mode = sba::Mode::Disabled;
  } else {
    return false;
  }
  settings.mode = SupportedMode(settings.mode);
  settings.num_sba_frames = FLAGS_num_sba_frames;
  settings.num_fixed_sba_frames = FLAGS_num_fixed_sba_frames;
  settings.num_sba_iterations = FLAGS_num_sba_iterations;
  settings.robustifier_scale = static_cast<float>(FLAGS_robustifier_scale);
  settings.use_sba_winsorizer = FLAGS_use_sba_winsorizer;
  settings.async = FLAGS_async_sba;

  return true;
}

}  // namespace cuvslam::sba
