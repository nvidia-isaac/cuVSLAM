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

#include "slam/vpr/vpr_bow.h"
#include "slam/vpr/vpr_simple.h"

#ifdef USE_DBOW2
#include "slam/vpr/vpr_dbow2.h"
#endif

#ifdef USE_ONNXRUNTIME
#include "slam/vpr/vpr_anyloc.h"
#endif

namespace cuvslam::slam::vpr {

const char* ToString(VprType type) {
  switch (type) {
    case VprType::kNone:
      return "None";
    case VprType::kSimple:
      return "Simple";
    case VprType::kDBoW2:
      return "DBoW2";
    case VprType::kAnyLoc:
      return "AnyLoc";
    case VprType::kBow:
      return "Bow";
  }
  return "Unknown";
}

std::unique_ptr<IVpr> CreateVpr(const VprOptions& options) {
  switch (options.type) {
    case VprType::kNone:
      return nullptr;
    case VprType::kSimple:
      return std::make_unique<VprSimple>(options);
    case VprType::kDBoW2:
#ifdef USE_DBOW2
      return std::make_unique<VprDBoW2>(options);
#else
      throw std::runtime_error(
          "VPR backend DBoW2 is not available: cuVSLAM was built with USE_DBOW2=OFF, which happens when OpenCV is "
          "not found at configure time.");
#endif
    case VprType::kBow:
      return std::make_unique<VprBow>(options);
    case VprType::kAnyLoc:
#ifdef USE_ONNXRUNTIME
      return std::make_unique<VprAnyLoc>(options);
#else
      throw std::runtime_error("VPR backend AnyLoc is not available: cuVSLAM was built with USE_ONNXRUNTIME=OFF.");
#endif
  }
  throw std::runtime_error("Unknown VPR backend");
}

}  // namespace cuvslam::slam::vpr
