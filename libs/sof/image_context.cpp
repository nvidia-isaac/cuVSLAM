
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

#include "sof/image_context.h"

namespace cuvslam::sof {

ImageContext::ImageContext(const ImageShape& shape, bool use_gpu, bool support_depth) : support_depth_(support_depth) {
  if (use_gpu) {
#ifdef USE_CUDA
    gpu_image_.init(shape.width, shape.height);
    gpu_image_pyramid_.init(shape.width, shape.height);
    gpu_gradient_pyramid_.init(shape.width, shape.height);

    if (support_depth) {
      // perform neccessary gpu allocations here
      gpu_depth_.init(shape.width, shape.height);
      gpu_depth_pyramid_.init(shape.width, shape.height);
    }

    gpu_mask_.init(shape.width, shape.height);
#else
    (void)shape;
    (void)use_gpu;
    (void)support_depth;
#endif
  }
}

void ImageContext::set_image_meta(const ImageMeta& meta) {
  std::lock_guard<std::mutex> guard(mutex_);
  meta_ = meta;

  cpu_image_pyramid_initialized_ = false;
  cpu_gradient_pyramid_initialized_ = false;

#ifdef USE_CUDA
  gpu_image_pyramid_initialized_ = false;
  gpu_gradient_pyramid_initialized_ = false;

  gpu_depth_pyramid_initialized_ = false;
#endif
}

const ImageMeta& ImageContext::get_image_meta() const { return meta_; }

ImageMatrixT ImageContext::cast_rgb2gs_cpu(void* data, const ImageShape& image_shape) {
  Eigen::Map<ImageMatrix<uint8_t>> cpu_image_rgb =
      Eigen::Map<ImageMatrix<uint8_t>>(static_cast<uint8_t*>(data), image_shape.height, image_shape.width * 3);
  ImageMatrixT cpu_image_grayscale = ImageMatrixT(image_shape.height, image_shape.width);
  for (int i = 0; i < image_shape.height; i++) {
    for (int j = 0; j < image_shape.width; j++) {
      uint8_t red = cpu_image_rgb(i, j * 3);
      uint8_t green = cpu_image_rgb(i, j * 3 + 1);
      uint8_t blue = cpu_image_rgb(i, j * 3 + 2);
      float value = 0.299 * red + 0.587 * green + 0.114 * blue;
      cpu_image_grayscale(i, j) = value;
    }
  }
  return cpu_image_grayscale;
}

ImageMatrix<uint8_t> ImageContext::resize_mask_cpu(void* data) {
  Eigen::Map<ImageMatrix<uint8_t>> cpu_mask =
      Eigen::Map<ImageMatrix<uint8_t>>(static_cast<uint8_t*>(data), meta_.mask_shape.height, meta_.mask_shape.width);
  ImageMatrix<uint8_t> resized_mask(meta_.shape.height, meta_.shape.width);
  float scale_x = static_cast<float>(meta_.mask_shape.width) / static_cast<float>(meta_.shape.width);
  float scale_y = static_cast<float>(meta_.mask_shape.height) / static_cast<float>(meta_.shape.height);
  for (int y = 0; y < meta_.shape.height; ++y) {
    for (int x = 0; x < meta_.shape.width; ++x) {
      int orig_x = static_cast<int>(static_cast<float>(x) * scale_x);
      int orig_y = static_cast<int>(static_cast<float>(y) * scale_y);

      orig_x = std::min(orig_x, meta_.mask_shape.width - 1);
      orig_y = std::min(orig_y, meta_.mask_shape.height - 1);

      resized_mask(y, x) = cpu_mask(orig_y, orig_x);
    }
  }
  return resized_mask;
}

bool ImageContext::build_cpu_image_pyramid(const ImageSource& source, bool blur_filter) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (cpu_image_pyramid_initialized_) {
    return true;
  }

  if (source.data == nullptr || source.memory_type != ImageSource::Host) {
    TraceError("Invalid image source");
    return false;
  }
  if (source.image_encoding == ImageEncoding::RGB8) {
    ImageMatrix<uint8_t> grayscale = cast_rgb2gs_cpu(source.data, meta_.shape).cast<uint8_t>();
    ImageSource grayscale_source;
    grayscale_source.data = grayscale.data();
    grayscale_source.type = ImageSource::U8;
    grayscale_source.image_encoding = ImageEncoding::MONO8;
    cpu_image_pyramid_u8_.build(grayscale_source, meta_.shape, blur_filter);
  } else {
    cpu_image_pyramid_u8_.build(source, meta_.shape, blur_filter);
  }

  const int numLevels = cpu_image_pyramid_u8_.num_levels();

  cpu_image_pyramid_.setLevelsCount(numLevels);
  for (int k = 0; k < numLevels; ++k) {
    const std::pair<ImageSource, ImageShape>& pyramid_level_u8 = cpu_image_pyramid_u8_[k];

    const ImageSource& s = pyramid_level_u8.first;
    const ImageShape& shape = pyramid_level_u8.second;
    cpu_image_pyramid_[k] = s.as<uint8_t>(shape).cast<float>();
  }

  cpu_image_pyramid_initialized_ = true;
  return true;
}

bool ImageContext::build_cpu_gradient_pyramid(bool horizontal) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (cpu_gradient_pyramid_initialized_) {
    return true;
  }
  if (!cpu_image_pyramid_initialized_) {
    return false;
  }

  const int numLevels = cpu_image_pyramid_.getLevelsCount();
  cpu_gradient_pyramid_.setNumLevels(numLevels);
  bool res = cpu_gradient_pyramid_.set(cpu_image_pyramid_, horizontal);
  if (res) {
    cpu_gradient_pyramid_initialized_ = true;
  }
  return res;
}

const ImagePyramidT& ImageContext::cpu_image_pyramid() const {
  assert(cpu_image_pyramid_initialized_);
  return cpu_image_pyramid_;
}

const ImagePyramidU8& ImageContext::cpu_image_pyramid_u8() const {
  assert(cpu_image_pyramid_initialized_);
  return cpu_image_pyramid_u8_;
}

const GradientPyramidT& ImageContext::cpu_gradient_pyramid() const {
  assert(cpu_gradient_pyramid_initialized_);
  return cpu_gradient_pyramid_;
}

bool ImageContext::has_cpu_image() const { return cpu_image_pyramid_initialized_; }

bool ImageContext::has_gpu_image() const {
#ifdef USE_CUDA
  return gpu_image_pyramid_initialized_;
#else
  return false;
#endif
}

bool ImageContext::support_depth() const { return support_depth_; }

void ImageContext::reset() {
  std::lock_guard<std::mutex> guard(mutex_);
  cpu_image_pyramid_initialized_ = false;
  cpu_gradient_pyramid_initialized_ = false;

#ifdef USE_CUDA
  gpu_image_pyramid_initialized_ = false;
  gpu_gradient_pyramid_initialized_ = false;

  gpu_depth_pyramid_initialized_ = false;
#endif
}

bool ImageContext::process_mask_cpu(const ImageSource& mask_source, ImageMatrix<uint8_t>& mask_resized) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (mask_source.data == nullptr) {
    TraceError("Invalid mask source");
    return false;
  }
  if (mask_source.memory_type == ImageSource::Host && mask_source.image_encoding == ImageEncoding::MONO8) {
    if (meta_.mask_shape.width == meta_.shape.width && meta_.mask_shape.height == meta_.shape.height) {
      mask_resized = mask_source.as<uint8_t>(meta_.mask_shape);
      return true;
    } else {
      mask_resized = resize_mask_cpu(mask_source.data);
      return true;
    }
  } else {
    TraceError("Invalid mask source");
    return false;
  }
}

#ifdef USE_CUDA

const cuda::GPUImageT& ImageContext::gpu_image() const { return gpu_image_; }

bool ImageContext::build_gpu_image_pyramid(const ImageSource& source, bool blur_filter, cudaStream_t s) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (gpu_image_pyramid_initialized_) {
    return true;
  }
  if (source.data == nullptr) {
    TraceError("Invalid image source");
    return false;
  }
  if (source.memory_type == ImageSource::Host) {
    gpu_image_cast_.cast(source.data, source.image_encoding, meta_.shape, gpu_image_, s);
  } else {
    uint2 size = {static_cast<decltype(uint2::x)>(gpu_image_.cols()),
                  static_cast<decltype(uint2::y)>(gpu_image_.rows())};
    if (source.image_encoding == ImageEncoding::MONO8) {
      CUDA_CHECK(cuda::cast_image(static_cast<const uint8_t*>(source.data), source.pitch, gpu_image_.ptr(),
                                  gpu_image_.pitch(), size, s));
    } else if (source.image_encoding == ImageEncoding::RGB8) {
      CUDA_CHECK(cuda::cast_image_rgb(static_cast<const uint8_t*>(source.data), source.pitch, gpu_image_.ptr(),
                                      gpu_image_.pitch(), size, s));
    }
  }
  gpu_image_pyramid_.build(gpu_image_, blur_filter, s);
  gpu_image_pyramid_initialized_ = true;
  return true;
}

bool ImageContext::build_gpu_gradient_pyramid(bool horizontal, cudaStream_t s) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (gpu_gradient_pyramid_initialized_) {
    return true;
  }
  if (!gpu_image_pyramid_initialized_) {
    return false;
  }
  bool res = gpu_gradient_pyramid_.set(gpu_image_pyramid_, s, horizontal);
  if (res) {
    gpu_gradient_pyramid_initialized_ = true;
  }
  return res;
}

bool ImageContext::build_gpu_depth_pyramid(const ImageSource& source, cudaStream_t s, const ImageSource* mask_source) {
  if (!support_depth_) {
    return false;
  }
  std::lock_guard<std::mutex> guard(mutex_);
  if (gpu_depth_pyramid_initialized_) {
    return true;
  }
  if (source.data == nullptr) {
    TraceError("Invalid depth pointer");
    return false;
  }

  if (source.memory_type == ImageSource::Host) {
    if (source.type == ImageSource::Type::F32) {
      gpu_depth_.copy(cuda::ToGPU, reinterpret_cast<const float*>(source.data), s);
    } else if (source.type == ImageSource::Type::U16) {
      assert(meta_.pixel_scale_factor != std::nullopt);
      gpu_image_cast_.cast_depth(reinterpret_cast<uint16_t*>(source.data), *meta_.pixel_scale_factor, meta_.shape,
                                 gpu_depth_, s);
    } else {
      TraceError("Unsupported depth data type!");
      return false;
    }
  } else {
    if (source.type == ImageSource::Type::F32) {
      const size_t row_size_bytes = meta_.shape.width * sizeof(float);
      CUDA_CHECK(cudaMemcpy2DAsync((void*)gpu_depth_.ptr(), gpu_depth_.pitch(), (void*)source.data, source.pitch,
                                   row_size_bytes, meta_.shape.height, cudaMemcpyDeviceToDevice, s));
    } else if (source.type == ImageSource::Type::U16) {
      assert(meta_.pixel_scale_factor != std::nullopt);

      uint2 size = {(unsigned)meta_.shape.width, (unsigned)meta_.shape.height};
      CUDA_CHECK(cuda::cast_depth_u16(reinterpret_cast<uint16_t*>(source.data), source.pitch, *meta_.pixel_scale_factor,
                                      gpu_depth_.ptr(), gpu_depth_.pitch(), size, s));
    } else {
      TraceError("Unsupported depth data type!");
      return false;
    }
  }
  if (mask_source != nullptr) {
    if (mask_source->memory_type == ImageSource::Host) {
      gpu_image_cast_.burn_mask_depth(reinterpret_cast<uint8_t*>(mask_source->data), meta_.shape, gpu_depth_, s);
    } else {
      uint2 size = {(unsigned)meta_.shape.width, (unsigned)meta_.shape.height};
      CUDA_CHECK(cuda::burn_depth_mask(gpu_depth_.ptr(), gpu_depth_.pitch(),
                                       reinterpret_cast<uint8_t*>(mask_source->data), mask_source->pitch, size, s));
    }
  }

  gpu_depth_pyramid_.build(gpu_depth_, false, s);
  gpu_depth_pyramid_initialized_ = true;
  return true;
}

const cuda::GaussianGPUImagePyramid& ImageContext::gpu_image_pyramid() const {
  assert(gpu_image_pyramid_initialized_);
  return gpu_image_pyramid_;
}

const cuda::GPUGradientPyramid& ImageContext::gpu_gradient_pyramid() const {
  assert(gpu_gradient_pyramid_initialized_);
  return gpu_gradient_pyramid_;
}

ImageContext::DepthOptRef ImageContext::gpu_depth_pyramid() const {
  if (!support_depth_ || !gpu_depth_pyramid_initialized_) {
    return std::nullopt;
  }
  return std::cref(gpu_depth_pyramid_);
}

bool ImageContext::process_mask_gpu(const ImageSource& mask_source, ImageMatrix<uint8_t>& mask_resized,
                                    cudaStream_t s) {
  std::lock_guard<std::mutex> guard(mutex_);

  if (mask_source.data == nullptr) {
    TraceError("Invalid mask source");
    return false;
  }

  uint2 src_size = {static_cast<decltype(uint2::x)>(meta_.mask_shape.width),
                    static_cast<decltype(uint2::y)>(meta_.mask_shape.height)};
  uint2 dst_size = {static_cast<decltype(uint2::x)>(gpu_mask_.cols()),
                    static_cast<decltype(uint2::y)>(gpu_mask_.rows())};

  if (mask_source.memory_type == ImageSource::Host) {
    if (src_size.x == dst_size.x && src_size.y == dst_size.y) {
      mask_resized = mask_source.as<uint8_t>(meta_.mask_shape);
      return true;
    } else {
      if (gpu_mask_source_ == nullptr) {
        gpu_mask_source_ = std::make_unique<cuda::GPUImage8>(src_size.x, src_size.y);
      }
      const unsigned char* mask_source_ptr = static_cast<const unsigned char*>(mask_source.data);
      gpu_mask_source_->copy(cuda::GPUCopyDirection::ToGPU, mask_source_ptr, s);
      if (mask_source.image_encoding == ImageEncoding::MONO8) {
        CUDA_CHECK(cuda::resize_mask(gpu_mask_source_->ptr(), src_size, gpu_mask_source_->pitch(), gpu_mask_.ptr(),
                                     dst_size, gpu_mask_.pitch(), s));
        gpu_mask_.copy(cuda::GPUCopyDirection::ToCPU, mask_resized.data(), s);
        return true;
      } else {
        TraceError("Invalid mask source");
        return false;
      }
    }
  } else {
    if (src_size.x == dst_size.x && src_size.y == dst_size.y) {
      thread_local std::vector<uint8_t, cuda::HostAllocator<uint8_t>> cpu_input_mask;
      cpu_input_mask.resize(src_size.y * src_size.x);
      cudaMemcpy2DAsync((void*)cpu_input_mask.data(), src_size.x, mask_source.data, mask_source.pitch, src_size.x,
                        src_size.y, cudaMemcpyDeviceToHost, s);
      mask_resized = Eigen::Map<const ImageMatrix<uint8_t>>(cpu_input_mask.data(), src_size.y, src_size.x);
      return true;
    } else {
      if (mask_source.image_encoding == ImageEncoding::MONO8) {
        CUDA_CHECK(cuda::resize_mask(static_cast<const uint8_t*>(mask_source.data), src_size, mask_source.pitch,
                                     gpu_mask_.ptr(), dst_size, gpu_mask_.pitch(), s));
        gpu_mask_.copy(cuda::GPUCopyDirection::ToCPU, mask_resized.data(), s);
        return true;
      } else if (mask_source.image_encoding == ImageEncoding::RGB8) {
        TraceError("Invalid mask source");
        return false;
      }
    }
  }
  return true;
}

#endif

}  // namespace cuvslam::sof
