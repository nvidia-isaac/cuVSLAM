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

#include "slam/vpr/vpr_types.h"

namespace cuvslam::slam::vpr {

enum class VprPixelFormat : uint8_t {
  kMono8,
  kRgb8,   ///< interleaved
  kFloat,  ///< values in [0, 255]
};

/// Copy `data` into an 8 bit gray VprImage; empty on invalid arguments.
/// `pitch` is the row stride in bytes, 0 for tightly packed rows.
VprImage MakeVprImage(const void* data, int width, int height, int pitch, VprPixelFormat format);

/// Bilinear resize to exactly `width` x `height`.
VprImage Resize(const VprImage& image, int width, int height);

}  // namespace cuvslam::slam::vpr
