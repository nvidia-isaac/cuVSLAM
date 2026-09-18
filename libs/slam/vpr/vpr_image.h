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

/// Pixel layout of a buffer handed to MakeVprImage().
enum class VprPixelFormat : uint8_t {
  kMono8,  ///< one uint8 per pixel
  kRgb8,   ///< three interleaved uint8 per pixel
  kFloat,  ///< one float per pixel, values in [0, 255]
};

/// Copy an external image into a VprImage, converting to single channel 8 bit.
///
/// `pitch` is the distance between two rows in bytes; pass 0 for tightly packed rows.
/// Returns an empty image when the arguments do not describe a readable buffer.
VprImage MakeVprImage(const void* data, int width, int height, int pitch, VprPixelFormat format);

/// Box filter downscale by an integer factor. A factor of 1 copies the image.
/// The output keeps at least one pixel in each dimension.
VprImage Downscale(const VprImage& image, int factor);

/// Bilinear resize to an exact size. Used by the backends that feed a fixed size network input.
VprImage Resize(const VprImage& image, int width, int height);

/// Zero mean, unit variance normalization of the pixels, written to `out`.
/// Returns false when the image is constant, in which case `out` is left zeroed.
bool NormalizePixels(const VprImage& image, std::vector<float>& out);

}  // namespace cuvslam::slam::vpr
