
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

// GPU bundlers exist only in CUDA builds; there a GPU mode is replaced by the CPU bundler of the same kind.
inline Mode SupportedMode(Mode mode) {
#ifndef USE_CUDA
  if (mode == OriginalGPU) {
    return OriginalCPU;
  }
  if (mode == InertialGPU) {
    return InertialCPU;
  }
#endif
  return mode;
}

inline Mode SelectMode(bool inertial, bool use_gpu) {
  if (inertial) {
    return SupportedMode(use_gpu ? InertialGPU : InertialCPU);
  }
  return SupportedMode(use_gpu ? OriginalGPU : OriginalCPU);
}

}  // namespace cuvslam::sba
