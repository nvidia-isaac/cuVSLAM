
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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include "common/environment.h"
#include "common/image.h"
#include "common/include_gtest.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "cuda_modules/image_cast.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

namespace {
using namespace cuvslam;

using ProfilerDomain = cuvslam::profiler::DefaultProfiler::DomainHelper;

TEST(Cuda, CastImageRGB) {
  ProfilerDomain profiler("CAST_IMAGE_RGB_TEST");
  ImageShape image_shape = {1920, 1200, 0, ImageEncoding::RGB8};
  std::vector<uint8_t> image_data(image_shape.width * image_shape.height * 3);

  int k = 0;

  for (int y = 0; y < image_shape.height; y++) {
    for (int x = 0; x < image_shape.width; x++) {
      uint8_t r = k;
      uint8_t g = k + 1;
      uint8_t b = k + 2;
      k += 3;

      image_data[(y * image_shape.width + x) * 3] = r;
      image_data[(y * image_shape.width + x) * 3 + 1] = g;
      image_data[(y * image_shape.width + x) * 3 + 2] = b;
    }
  }

  cuda::ImageCast image_cast;
  cuda::GPUImageT gpu_image;
  cuda::Stream stream;
  ImageMatrixT output_grayscale_image(image_shape.height, image_shape.width);

  gpu_image.init(image_shape.width, image_shape.height);

  {
    TRACE_EVENT ev = profiler.trace_event("cast_rgb2gs");
    for (int i = 0; i < 100; i++) {
      ASSERT_TRUE(image_cast.cast(image_data.data(), image_shape, gpu_image, stream.get_stream()));
      gpu_image.copy(cuda::GPUCopyDirection::ToCPU, output_grayscale_image.data(), stream.get_stream());
      CUDA_CHECK(cudaStreamSynchronize(stream.get_stream()));
    }
  }

  k = 0;
  bool verify_gs = true;
  for (int y = 0; y < image_shape.height; ++y) {
    for (int x = 0; x < image_shape.width; ++x) {
      uint8_t r = k;
      uint8_t g = k + 1;
      uint8_t b = k + 2;
      k += 3;

      // NTSC formula Y = 0.299*r + 0.587*g + 0.114*b
      float expected_value = static_cast<float>(0.299 * static_cast<float>(r) + 0.587 * static_cast<float>(g) +
                                                0.114 * static_cast<float>(b));

      float actual_value = output_grayscale_image(y, x);

      // The kernel computes in float, so allow a few ulps relative to the value.
      if (std::abs(expected_value - actual_value) > 4 * epsilon() * std::max(1.f, expected_value)) {
        verify_gs = false;
        break;
      }
    }
  }
  ASSERT_TRUE(verify_gs);
}

TEST(Cuda, CastDepthU16) {
  const ImageShape image_shape = {640, 480, 0, ImageEncoding::MONO8};
  std::vector<uint16_t> depth_data(image_shape.width * image_shape.height);
  for (size_t i = 0; i < depth_data.size(); ++i) {
    depth_data[i] = static_cast<uint16_t>(i % 65536);
  }

  cuda::ImageCast image_cast;
  cuda::GPUImageT gpu_image;
  cuda::Stream stream;
  ImageMatrixT output_depth(image_shape.height, image_shape.width);
  gpu_image.init(image_shape.width, image_shape.height);

  for (const float scale : {1.f, 1000.f, 5000.f}) {
    ASSERT_TRUE(image_cast.cast_depth(depth_data.data(), scale, image_shape, gpu_image, stream.get_stream()));
    gpu_image.copy(cuda::GPUCopyDirection::ToCPU, output_depth.data(), stream.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(stream.get_stream()));

    for (int y = 0; y < image_shape.height; ++y) {
      for (int x = 0; x < image_shape.width; ++x) {
        const float expected = static_cast<float>(depth_data[y * image_shape.width + x]) / scale;
        ASSERT_NEAR(output_depth(y, x), expected, 4 * epsilon() * std::max(1.f, expected))
            << "scale=" << scale << " x=" << x << " y=" << y;
      }
    }
  }
}

}  // namespace
