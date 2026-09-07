
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
#include <mutex>
#include <vector>

#include "common/image.h"

#ifdef USE_CUDA
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/gradient_pyramid.h"
#include "cuda_modules/image_cast.h"
#include "cuda_modules/image_pyramid.h"
#endif

#include "sof/gradient_pyramid.h"
#include "sof/image_pyramid_float.h"
#include "sof/image_pyramid_u8.h"

namespace cuvslam::sof {

class ImageContext {
public:
  ImageContext(const ImageShape& shape, bool use_gpu, bool support_depth);

  void set_image_meta(const ImageMeta& meta);
  const ImageMeta& get_image_meta() const;

  bool build_cpu_image_pyramid(const ImageSource& source, bool blur_filter);
  bool build_cpu_gradient_pyramid(bool horizontal);

  const ImagePyramidT& cpu_image_pyramid() const;
  const ImagePyramidU8& cpu_image_pyramid_u8() const;
  const GradientPyramidT& cpu_gradient_pyramid() const;

  bool support_depth() const;

  void reset();

  bool process_mask_cpu(const ImageSource& mask_source, ImageMatrix<uint8_t>& mask_resized);

#ifdef USE_CUDA
  const cuda::GPUImageT& gpu_image() const;

  bool build_gpu_image_pyramid(const ImageSource& source, bool blur_filter, cudaStream_t s);
  bool build_gpu_gradient_pyramid(bool horizontal, cudaStream_t s);

  bool build_gpu_depth_pyramid(const ImageSource& source, cudaStream_t s, const ImageSource* mask_source = nullptr);

  using DepthOptRef = std::optional<std::reference_wrapper<const cuda::GaussianGPUImagePyramid>>;
  DepthOptRef gpu_depth_pyramid() const;

  const cuda::GaussianGPUImagePyramid& gpu_image_pyramid() const;
  const cuda::GPUGradientPyramid& gpu_gradient_pyramid() const;

  bool process_mask_gpu(const ImageSource& mask_source, ImageMatrix<uint8_t>& mask_resized, cudaStream_t s);
#endif
private:
  ImageMeta meta_;
  ImagePyramidT cpu_image_pyramid_;
  ImagePyramidU8 cpu_image_pyramid_u8_;
  GradientPyramidT cpu_gradient_pyramid_;

  bool cpu_image_pyramid_initialized_ = false;
  bool cpu_gradient_pyramid_initialized_ = false;

  int slot_index_ = 0;

  std::mutex mutex_;

  ImageMatrixT cast_rgb2gs_cpu(void* data, const ImageShape& image_shape);
  ImageMatrix<uint8_t> resize_mask_cpu(void* data);

  const bool support_depth_;

#ifdef USE_CUDA
  cuda::GPUImageT gpu_image_;
  cuda::ImageCast gpu_image_cast_;
  cuda::GaussianGPUImagePyramid gpu_image_pyramid_;
  cuda::GPUGradientPyramid gpu_gradient_pyramid_;

  // despite these fields are present, no GPU allocations occur until .init method is called.
  cuda::GPUImageT gpu_depth_;
  cuda::GaussianGPUImagePyramid gpu_depth_pyramid_;

  cuda::GPUImage8 gpu_mask_;
  std::unique_ptr<cuda::GPUImage8> gpu_mask_source_;

  bool gpu_image_pyramid_initialized_ = false;
  bool gpu_gradient_pyramid_initialized_ = false;

  bool gpu_depth_pyramid_initialized_ = false;

#endif
};

using ImageContextPtr = std::shared_ptr<ImageContext>;

// Camera-indexed vector sized to rig.num_cameras; nullptr means this camera is absent in the frame.
using Images = std::vector<sof::ImageContextPtr>;
}  // namespace cuvslam::sof
