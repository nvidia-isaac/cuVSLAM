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

/// 8 bit grayscale image; always owns a copy of its pixels.
struct VprImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> row;  ///< row major pixels, size() == width * height

  bool Empty() const { return width <= 0 || height <= 0 || row.size() != static_cast<size_t>(width) * height; }
};

struct VprOptions {
  /// Run place recognition (AnyLoc: VLAD over DINOv2 patch features).
  bool enabled = false;

  /// Min similarity in [0, 1] to report a match; tuned on KITTI 07 to favor precision over recall.
  float score_threshold = 0.68f;

  /// Ratio test vs. the best match elsewhere on the trajectory; 0 disables.
  float ratio_threshold = 0.f;

  /// Backend model file: the AnyLoc TensorRT engine.
  std::string model_path;

  /// Number of VLAD cluster centers AnyLoc fits.
  int vocabulary_size = 32;

  /// Vocabulary file VocabularyFitter wrote; empty fits one per session.
  std::string vocabulary_path;

  /// Frames mapped before a session fits its vocabulary; queries miss until then.
  size_t min_training_keyframes = 8;

  /// Cap on the patch descriptors a session's vocabulary is fitted to, and on those held until then.
  size_t max_training_descriptors = 100000;

  /// Max frames kept in the map; 0 means unlimited.
  uint32_t max_entries = 0;
};

struct VprMatch {
  bool found = false;                      ///< passed the score and ratio tests
  KeyFrameId node_id = InvalidKeyFrameId;  ///< matched pose graph node
  float score = 0.f;                       ///< similarity in [0, 1]
};

}  // namespace cuvslam::slam::vpr
