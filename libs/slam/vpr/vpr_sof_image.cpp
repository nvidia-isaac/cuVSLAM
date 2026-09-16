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

#include "slam/vpr/vpr_sof_image.h"

#include "common/log.h"

#ifdef USE_CUDA
#include "cuda_modules/cuda_helper.h"
#endif

#include "slam/vpr/vpr_image.h"

namespace cuvslam::slam::vpr {

VprImage MakeVprImageFromContext(const ImageContextPtr& context) {
  if (context == nullptr) {
    return VprImage{};
  }

#ifdef USE_CUDA
  // Ask the context which pipeline filled it: the CPU getters assert when the frame was built on
  // the GPU, and in an assertions-enabled build (this project keeps them on in RelWithDebInfo)
  // reaching them would abort the SLAM worker rather than return an empty image.
  if (context->has_gpu_image()) {
    const cuda::GPUImageT& gpu_image = context->gpu_image();
    if (gpu_image.ptr() == nullptr || gpu_image.rows() == 0 || gpu_image.cols() == 0) {
      return VprImage{};
    }
    // The tracker keeps the frame on the device as float luminance. One download per keyframe is
    // cheap next to the tracking work already done on that frame, and it keeps the VPR backends
    // free of any CUDA dependency of their own.
    const size_t width = gpu_image.cols();
    const size_t height = gpu_image.rows();
    std::vector<float> host(width * height);
    cuda::Stream stream{true};
    gpu_image.copy(cuda::ToCPU, host.data(), stream.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(stream.get_stream()));
    return MakeVprImage(host.data(), static_cast<int>(width), static_cast<int>(height), 0, VprPixelFormat::kFloat);
  }
#endif

  if (!context->has_cpu_image()) {
    return VprImage{};
  }
  const auto& [source, shape] = context->cpu_image_pyramid_u8()[0];
  if (source.data == nullptr || shape.width <= 0 || shape.height <= 0) {
    return VprImage{};
  }
  return MakeVprImage(source.data, shape.width, shape.height, 0, VprPixelFormat::kMono8);
}

VprImage MakeVprImageFromImages(const Images& images) {
  for (const auto& context : images) {
    VprImage image = MakeVprImageFromContext(context);
    if (!image.Empty()) {
      return image;
    }
  }
  return VprImage{};
}

}  // namespace cuvslam::slam::vpr
