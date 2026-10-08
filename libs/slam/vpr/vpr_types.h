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
#include <vector>

#include "slam/common/slam_common.h"

namespace cuvslam::slam::vpr {

/// Visual Place Recognition backend selection.
enum class VprType : uint8_t {
  kNone = 0,    ///< VPR disabled
  kSimple = 1,  ///< downscaled grayscale thumbnail, brute force nearest neighbor
  kDBoW2 = 2,   ///< DBoW2 bag of binary words over ORB features
  kAnyLoc = 3,  ///< AnyLoc: VLAD aggregation of DINOv2 dense patch features
  kBow = 4,     ///< in-tree ORB and a binary bag of words, with no third-party dependency
};

/// Convert a VprType to its printable name.
const char* ToString(VprType type);

/// Single channel 8 bit image owned by the VPR module. Pixels are always copied in, never borrowed
/// from the tracking pipeline, so a map entry outlives the frame that produced it.
struct VprImage {
  int width = 0;             ///< image width in pixels
  int height = 0;            ///< image height in pixels
  std::vector<uint8_t> row;  ///< row major pixels, size() == width * height

  bool Empty() const { return width <= 0 || height <= 0 || row.size() != static_cast<size_t>(width) * height; }
};

/// VPR configuration.
struct VprOptions {
  /// Backend to instantiate. kNone disables place recognition entirely.
  VprType type = VprType::kNone;

  /// Downscale factor applied to the input image before the descriptor is computed.
  /// Used by the Simple backend; the other backends resize to their own working resolution.
  int downscale = 8;

  /// Minimum similarity in [0, 1] for a query to be reported as a match.
  /// 0 means "use the backend default", see kDefaultScoreThreshold in the backend sources.
  float score_threshold = 0.f;

  /// Reject a match whose similarity is not at least this factor above the second best match
  /// coming from a different part of the trajectory. 0 disables the ratio test.
  float ratio_threshold = 0.f;

  /// Path to the backend model file. AnyLoc reads a DINOv2 ONNX model from here.
  std::string model_path;

  /// Number of vocabulary entries: VLAD cluster centers for AnyLoc, leaf nodes for DBoW2.
  int vocabulary_size = 32;

  /// Upper bound on the number of frames kept in the map. 0 means unlimited.
  uint32_t max_entries = 0;
};

/// Result of a place recognition query.
struct VprMatch {
  bool found = false;                      ///< true when a map entry passed the score and ratio tests
  KeyFrameId node_id = InvalidKeyFrameId;  ///< pose graph node the matched image was observed from
  float score = 0.f;                       ///< similarity in [0, 1], 1 means identical
};

}  // namespace cuvslam::slam::vpr
