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

#include "slam/vpr/ivpr.h"

#include <stdexcept>

#ifdef USE_TENSORRT
#include "slam/vpr/vpr_anyloc.h"
#endif

namespace cuvslam::slam::vpr {

std::unique_ptr<IVpr> CreateVpr(const VprOptions& options) {
  if (!options.enabled) {
    return nullptr;
  }
#ifdef USE_TENSORRT
  return std::make_unique<VprAnyLoc>(options);
#else
  throw std::runtime_error("VPR backend AnyLoc is not available: cuVSLAM was built with USE_TENSORRT=OFF.");
#endif
}

}  // namespace cuvslam::slam::vpr
