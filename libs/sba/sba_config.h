
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
#include <string>

#include "params/params.h"

namespace cuvslam::sba {

enum Mode { Disabled, OriginalCPU, OriginalGPU, InertialCPU, InertialGPU };

struct Settings {
  bool async = true;

  // number of key frames in new SBA
  int32_t num_sba_frames = 7;

  // number of key frames in Inertial SBA
  int32_t num_inertial_sba_frames = 10;

  // number of fixed key frames in SBA
  int32_t num_fixed_sba_frames = 3;

  int32_t num_inertial_fixed_sba_frames = 1;

  // maximum number of iterations for new SBA
  int32_t num_sba_iterations = 7;

  // robustifier scale in new SBA
  float robustifier_scale = 5e-1;

  Mode mode = OriginalCPU;

  bool use_sba_winsorizer = false;
};

}  // namespace cuvslam::sba

CUVSLAM_PARAM_ENUM_BEGIN(cuvslam::sba::Mode)
CUVSLAM_PARAM_ENUM_VALUE("none", Disabled)
CUVSLAM_PARAM_ENUM_VALUE("cpu", OriginalCPU)
CUVSLAM_PARAM_ENUM_VALUE("gpu", OriginalGPU)
CUVSLAM_PARAM_ENUM_VALUE("imu", InertialCPU)
CUVSLAM_PARAM_ENUM_VALUE("imugpu", InertialGPU)
CUVSLAM_PARAM_ENUM_END()

// Tunable fields of sba::Settings, addressed as `sba.<name>`. `mode` stays out: it follows from
// the odometry mode and the SBA backend at construction, rather than being tuned per run.
CUVSLAM_PARAMS_BEGIN(cuvslam::sba::Settings)
CUVSLAM_PARAM(async, "Run SBA asynchronously")
CUVSLAM_PARAM_BOUNDED(num_sba_frames, "Keyframes in the SBA window", NonNegative())
CUVSLAM_PARAM_BOUNDED(num_inertial_sba_frames, "Keyframes in the inertial SBA window", NonNegative())
CUVSLAM_PARAM_BOUNDED(num_fixed_sba_frames, "Keyframes held fixed in SBA", NonNegative())
CUVSLAM_PARAM_BOUNDED(num_inertial_fixed_sba_frames, "Keyframes held fixed in inertial SBA", NonNegative())
CUVSLAM_PARAM_BOUNDED(num_sba_iterations, "Maximum solver iterations per SBA run", NonNegative())
CUVSLAM_PARAM_BOUNDED(robustifier_scale, "Huber loss threshold", NonNegative())
CUVSLAM_PARAM(use_sba_winsorizer, "Winsorize residuals in SBA")
CUVSLAM_PARAMS_END()
