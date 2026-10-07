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

/// The DINOv2 patch descriptor network of the AnyLoc backend, run as a TensorRT engine on CUDA device 0.
///
/// The engine is the one the `anyloc_engine` CMake target builds (tools/anyloc_model): input `image`, fp32
/// [1, 3, H, W], ImageNet normalized; output `patch_descriptors`, fp32 [1, N, D], one L2 normalized row per image
/// patch. A serialized engine only loads into the TensorRT version, and only runs on the GPU architecture, that
/// built it.
///
/// Not thread safe; the backend calls it under the SLAM lock.
class AnyLocNetwork {
public:
  /// Load the engine at `engine_path`, check its inputs and outputs, and run it once so the first frame does not pay
  /// for the warm up.
  /// @throws std::runtime_error when the file is missing, is not an engine this TensorRT and GPU can run, or does not
  ///         have the inputs and outputs above.
  explicit AnyLocNetwork(const std::string& engine_path);
  ~AnyLocNetwork();

  AnyLocNetwork(const AnyLocNetwork&) = delete;
  AnyLocNetwork& operator=(const AnyLocNetwork&) = delete;

  int input_width() const;
  int input_height() const;
  /// Rows of the output, one per image patch.
  int patch_count() const;
  /// Width of one patch descriptor (384 for ViT-S/14).
  int descriptor_dim() const;

  /// The pinned host buffer `Run` reads the network input from: 3 * input_height() * input_width() floats, planar
  /// RGB, written by the caller.
  float* input();

  /// Run the network on input() and copy its output, patch_count() * descriptor_dim() floats, into `patches`.
  /// Never throws: a CUDA or TensorRT failure, or an output that is not finite, is logged and returns false.
  bool Run(std::vector<float>& patches);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cuvslam::slam::vpr
