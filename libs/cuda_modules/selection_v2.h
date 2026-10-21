
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

#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/gftt.h"
#include "cuda_modules/gradient_pyramid.h"

#define NUM_BINS_X 8
#define NUM_BINS_Y 8

namespace cuvslam::cuda {

class GPUSelection {
public:
  // deterministic: order tied keypoints by position and add GFTT bins in a fixed order
  explicit GPUSelection(bool deterministic);

  // burnHalfSize and alreadySelectedBurnHalfSize should depend on image width (i.e. aperture in pixels)
  void computeGFTTAndSelectFeatures(const GPUGradientPyramid& gradientPyramid, int border_top, int border_bottom,
                                    int border_left, int border_right, const ImageMatrix<uint8_t>* input_mask,
                                    const std::vector<Vector2T>& AliveFeatures, size_t num_desired_new_features,
                                    std::vector<Vector2T>& newSelectedFeatures, cudaStream_t& stream);

  ~GPUSelection();

private:
  uint2 num_bins_ = {NUM_BINS_X, NUM_BINS_Y};
  GPUArrayPinned<float> bins_array_{2 + NUM_BINS_X * NUM_BINS_Y};  // 2 head elements for kpCount_ and kpIndex_
  uint2 bin_size_ = {0, 0};

  size_t max_tracks_;
  GFTT gftt_;
  uint2 gftt_size_ = {0, 0};
  uint2 downsampled_size_ = {0, 0};
  std::unique_ptr<GPUImageT> gftt_value_ = nullptr;
  std::unique_ptr<GPUImageT> gftt_maximums_ = nullptr;
  std::unique_ptr<GPUImageT> downsampled_maximums_ = nullptr;
  std::unique_ptr<GPUImage<uint2>> indices_ = nullptr;
  std::unique_ptr<GPUArrayPinned<Keypoint>> keypoints_ = nullptr;
  void* sort_temp_buffer_ = nullptr;
  size_t sort_temp_buffer_size_ = 0;
  bool deterministic_ = false;
  std::unique_ptr<GPUOnlyArray<float>> gftt_tile_sums_ = nullptr;  // deterministic only

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0xFF0000;
};

}  // namespace cuvslam::cuda
