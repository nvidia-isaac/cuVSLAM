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

#include <array>
#include <cstdint>
#include <vector>

#include "slam/vpr/vpr_types.h"

namespace cuvslam::slam::vpr {

/// One ORB keypoint and its 256 bit rotated BRIEF descriptor.
struct OrbKeypoint {
  float x = 0.f;          ///< x position in level 0 image coordinates
  float y = 0.f;          ///< y position in level 0 image coordinates
  float angle_rad = 0.f;  ///< orientation from the intensity centroid
  int level = 0;          ///< pyramid level the keypoint was detected at

  std::array<uint8_t, 32> descriptor{};
};

/// ORB feature extractor written against nothing but the standard library.
///
/// cuVSLAM's own tracker extracts Shi-Tomasi features and describes them with its own descriptor,
/// neither of which is a binary word a bag of words can quantize. OpenCV would supply ORB, but it is
/// a dependency the core library does not otherwise have; this is the reason the Bow backend exists
/// at all, so it brings its own.
///
/// Per image: build a Gaussian pyramid, detect FAST-9 corners at each level, rank them by the Harris
/// measure and keep the best per level, orient each by the intensity centroid, and describe it with
/// a rotated BRIEF pattern.
class OrbExtractor {
public:
  /// Pyramid levels, and the scale reduction between them.
  static constexpr int kNumLevels = 8;
  static constexpr float kScaleFactor = 1.2f;
  /// FAST contrast threshold, and the keypoint budget shared across the levels.
  static constexpr int kFastThreshold = 20;
  static constexpr int kMaxFeatures = 500;

  OrbExtractor() = default;

  /// Keypoints of a single channel 8 bit image. Empty for an image too small to hold a patch.
  std::vector<OrbKeypoint> Extract(const VprImage& image) const;
};

}  // namespace cuvslam::slam::vpr
