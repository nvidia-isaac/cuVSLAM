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

#include <memory>
#include <string>
#include <vector>

namespace cuvslam::slam::vpr {

/// AnyLoc's TensorRT engine (see models/anyloc). Not thread safe.
class AnyLocNetwork {
public:
  /// Load, check and warm up the engine at `engine_path`.
  /// @throws std::runtime_error if the engine is missing or incompatible.
  explicit AnyLocNetwork(const std::string& engine_path);
  ~AnyLocNetwork();

  AnyLocNetwork(const AnyLocNetwork&) = delete;
  AnyLocNetwork& operator=(const AnyLocNetwork&) = delete;

  int input_width() const;
  int input_height() const;
  int patch_count() const;
  /// Width of one patch descriptor (384 for ViT-S/14).
  int descriptor_dim() const;

  /// Pinned input for Run(): 3 * H * W floats, planar RGB, ImageNet normalized.
  float* input();

  /// Run on input() into `patches`; logs and returns false on failure, never throws.
  bool Run(std::vector<float>& patches);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cuvslam::slam::vpr
