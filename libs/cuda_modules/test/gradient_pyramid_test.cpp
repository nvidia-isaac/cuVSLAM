
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

#include <chrono>
#include <iostream>
#include <random>

#include "benchmark_utils.h"
#include "common/include_gtest.h"
#include "cuda_modules/gradient_pyramid.h"
#include "sof/gradient_pyramid.h"
#include "sof/sof_config.h"
namespace test {
using namespace cuvslam;

TEST(Cuda, GradientPyramid) {
  cuda::Stream s;

  std::mt19937 rng(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_int_distribution<int> dim(30, 1080);
  for (int k = 0; k < 20; k++) {
    size_t img_size = dim(rng);

    ImageMatrixT input = ImageMatrixT::Random(img_size, img_size) * 100;

    cuda::GPUImageT device_input(input);
    sof::ImagePyramidT image_pyramid = cuvslam::sof::ImagePyramidT(input);
    cuda::GaussianGPUImagePyramid cuda_image_pyramid(img_size, img_size);

    image_pyramid.buildPyramid();
    cuda_image_pyramid.build(device_input, false, s.get_stream());
    sof::Settings sof_settings;
    sof::GradientPyramidT gradient_pyramid;
    cuda::GPUGradientPyramid cuda_gradient_pyramid(img_size, img_size);

    ASSERT_TRUE(gradient_pyramid.set(image_pyramid, false));
    gradient_pyramid.forceNonLazyEvaluation(0);

    ASSERT_TRUE(cuda_gradient_pyramid.set(cuda_image_pyramid, s.get_stream(), false));
    cudaStreamSynchronize(s.get_stream());

    ASSERT_TRUE(gradient_pyramid.getLevelsCount() == cuda_gradient_pyramid.getLevelsCount());

    auto& x_pyramid = gradient_pyramid.gradX();
    const auto& y_pyramid = gradient_pyramid.gradY();

    auto& cuda_x_pyramid = cuda_gradient_pyramid.gradX();
    const auto& cuda_y_pyramid = cuda_gradient_pyramid.gradY();

    ASSERT_TRUE(static_cast<size_t>(x_pyramid.getLevelsCount()) == cuda_x_pyramid.getLevelsCount());
    ASSERT_TRUE(static_cast<size_t>(y_pyramid.getLevelsCount()) == cuda_y_pyramid.getLevelsCount());

    for (int i = 0; i < x_pyramid.getLevelsCount(); i++) {
      const auto& level = x_pyramid[i];
      cuda::GPUImageT& cuda_level = cuda_x_pyramid[i];
      ImageMatrixT level_cpu(cuda_level.rows(), cuda_level.cols());
      cuda_level.copy(cuda::GPUCopyDirection::ToCPU, level_cpu.data(), s.get_stream());
      cudaStreamSynchronize(s.get_stream());

      ASSERT_TRUE(static_cast<size_t>(level.rows()) == cuda_level.rows());
      ASSERT_TRUE(static_cast<size_t>(level.cols()) == cuda_level.cols());

      if (!level.isApprox(level_cpu, 0.01f)) {
        std::cout << "basic_output = " << std::endl << level.block(0, 0, 10, 10) << std::endl;
        std::cout << "cuda_output = " << std::endl << level_cpu.block(0, 0, 10, 10) << std::endl;
      }

      ASSERT_TRUE(level.isApprox(level_cpu, 0.01f));
    }
  }
}

TEST(Cuda, GradientPyramidSpeedUp) {
  size_t img_size = 1080;

  ImageMatrixT input = ImageMatrixT::Random(img_size, img_size) * 100;

  cuda::GPUImageT device_input(input);
  sof::ImagePyramidT image_pyramid = cuvslam::sof::ImagePyramidT(input);
  cuda::GaussianGPUImagePyramid cuda_image_pyramid(img_size, img_size);

  image_pyramid.buildPyramid();
  cuda::Stream s;
  cuda_image_pyramid.build(device_input, false, s.get_stream());
  cudaStreamSynchronize(s.get_stream());
  sof::Settings sof_settings;
  sof::GradientPyramidT gradient_pyramid;
  cuda::GPUGradientPyramid cuda_gradient_pyramid(img_size, img_size);

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    gradient_pyramid.set(image_pyramid);
    gradient_pyramid.forceNonLazyEvaluation(0);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    cuda_gradient_pyramid.set(cuda_image_pyramid, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

}  // namespace test
