
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
#include "cuda_modules/box_prefilter.h"
#include "cuda_modules/cuda_convolutor.h"
#include "sof/basic_convolutor.h"
#include "sof/box_blur.h"
namespace test {
using namespace cuvslam;

TEST(Cuda, convKernelGradDerivX) {
  cuvslam::cuda::CudaConvolutor cudaConvolutor;
  cuvslam::sof::BasicConvolutor basicConvolutor;
  cuvslam::cuda::Stream s;

  std::mt19937 rng(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_int_distribution<int> dim(15, 1920);
  for (int i = 0; i < 20; i++) {
    size_t width = dim(rng);
    size_t height = dim(rng);
    cuvslam::cuda::GPUImageT device_input(width, height);
    cuvslam::cuda::GPUImageT device_output(width, height);

    ImageMatrixT basic_output = ImageMatrixT::Zero(height, width);
    ImageMatrixT out = ImageMatrixT::Zero(height, width);

    ImageMatrixT input = ImageMatrixT::Random(height, width) * 100;

    device_input = input;
    cudaConvolutor.convKernelGradDerivX(device_input, device_output, s.get_stream());
    basicConvolutor.convKernelGradDerivX(input, basic_output);

    device_output.copy(cuvslam::cuda::GPUCopyDirection::ToCPU, out.data(), s.get_stream());
    cudaStreamSynchronize(s.get_stream());
    if (!basic_output.isApprox(out, 0.001f)) {
      std::cout << "width = " << width << ", height = " << height << std::endl;
      std::cout << "basic_output = " << std::endl << basic_output.block(0, 0, 10, 10) << std::endl;
      std::cout << "cuda_output = " << std::endl << out.block(0, 0, 10, 10) << std::endl;
    }
    ASSERT_TRUE(basic_output.isApprox(out, 0.001f));
  }
}

TEST(Cuda, convKernelGradDerivY) {
  cuvslam::cuda::CudaConvolutor cudaConvolutor;
  cuvslam::sof::BasicConvolutor basicConvolutor;
  cuvslam::cuda::Stream s;

  std::mt19937 rng(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_int_distribution<int> dim(15, 1920);
  for (int i = 0; i < 20; i++) {
    size_t width = dim(rng);
    size_t height = dim(rng);
    // std::cout << "wh = " << width << " " << height << std::endl;
    cuvslam::cuda::GPUImageT device_input(width, height);
    cuvslam::cuda::GPUImageT device_output(width, height);

    ImageMatrixT basic_output = ImageMatrixT::Zero(height, width);

    ImageMatrixT out = ImageMatrixT::Random(height, width);
    ImageMatrixT input = ImageMatrixT::Random(height, width) * 100;

    device_input = input;
    cudaConvolutor.convKernelGradDerivY(device_input, device_output, s.get_stream());
    basicConvolutor.convKernelGradDerivY(input, basic_output);

    device_output.copy(cuvslam::cuda::GPUCopyDirection::ToCPU, out.data(), s.get_stream());
    cudaStreamSynchronize(s.get_stream());

    if (!basic_output.isApprox(out, 0.001f)) {
      std::cout << "width = " << width << ", height = " << height << std::endl;
      std::cout << "basic_output = " << std::endl << basic_output << std::endl;
      std::cout << "cuda_output = " << std::endl << out << std::endl;
    }
    ASSERT_TRUE(basic_output.isApprox(out, 0.001f));
  }
}

TEST(Cuda, SpeedUpKernelGradDerivX) {
  cuvslam::cuda::CudaConvolutor cudaConvolutor;
  cuvslam::sof::BasicConvolutor basicConvolutor;

  size_t width = 1080;
  size_t height = 1080;

  cuvslam::cuda::GPUImageT device_input(width, height);
  cuvslam::cuda::GPUImageT device_output(width, height);

  ImageMatrixT input = ImageMatrixT::Random(height, width) * 100;
  ImageMatrixT basic_output = ImageMatrixT::Zero(height, width);

  device_input = input;

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    basicConvolutor.convKernelGradDerivX(input, basic_output);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  cuvslam::cuda::Stream s;
  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    cudaConvolutor.convKernelGradDerivX(device_input, device_output, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);

  // ASSERT_TRUE(basic_output.isApprox(device_output.map(), 0.01f));
}

TEST(Cuda, SpeedUpKernelGradDerivY) {
  cuvslam::cuda::CudaConvolutor cudaConvolutor;
  cuvslam::sof::BasicConvolutor basicConvolutor;

  size_t width = 1080;
  size_t height = 1080;

  ImageMatrixT input = ImageMatrixT::Random(height, width) * 100;

  cuvslam::cuda::GPUImageT device_input(input);
  cuvslam::cuda::GPUImageT device_output(width, height);

  ImageMatrixT basic_output = ImageMatrixT::Zero(height, width);

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    basicConvolutor.convKernelGradDerivY(input, basic_output);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  cuvslam::cuda::Stream s;
  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    cudaConvolutor.convKernelGradDerivY(device_input, device_output, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);

  // ASSERT_TRUE(basic_output.isApprox(device_output.map(), 0.01f));
}

#ifdef __GNUC__
TEST(Cuda, BoxBlur) {
  cuvslam::cuda::CudaBoxPrefilter gpu_prefilter;
  cuvslam::cuda::Stream s;

  std::mt19937 rng(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_int_distribution<int> dim(15, 1920);
  for (int i = 0; i < 20; i++) {
    size_t width = dim(rng);
    size_t height = dim(rng);
    cuvslam::cuda::GPUImageT device_input(width, height);
    cuvslam::cuda::GPUImageT device_output(width, height);

    ImageMatrix<uint8_t> basic_output = ImageMatrix<uint8_t>::Zero(height, width);
    ImageMatrixT out = ImageMatrixT::Zero(height, width);

    ImageMatrixT input = (ImageMatrix<uint8_t>::Random(height, width) * 100).cast<float>();

    device_input = input;
    gpu_prefilter.prefilter(device_input, device_output, s.get_stream());
    cudaStreamSynchronize(s.get_stream());

    cuvslam::sof::BoxBlur3(input.cast<uint8_t>(), basic_output.data());

    device_output.copy(cuvslam::cuda::GPUCopyDirection::ToCPU, out.data(), s.get_stream());
    cudaStreamSynchronize(s.get_stream());
    if (!basic_output.cast<float>().isApprox(out, 0.001f)) {
      std::cout << "width = " << width << ", height = " << height << std::endl;
      std::cout << "basic_output = " << std::endl << basic_output.cast<float>().block(0, 0, 10, 10) << std::endl;
      std::cout << "cuda_output = " << std::endl << out.block(0, 0, 10, 10) << std::endl;
    }
    ASSERT_TRUE(basic_output.cast<float>().isApprox(out, 0.001f));
  }
}
TEST(Cuda, SpeedUpBoxBlur) {
  cuvslam::cuda::CudaBoxPrefilter gpu_prefilter;
  cuvslam::cuda::Stream s;

  size_t width = 1080;
  size_t height = 1080;

  cuvslam::cuda::GPUImageT device_input(width, height);
  cuvslam::cuda::GPUImageT device_output(width, height);

  ImageMatrixT input = (ImageMatrix<uint8_t>::Random(height, width) * 100).cast<float>();
  ImageMatrix<uint8_t> input_u8 = input.cast<uint8_t>();
  ImageMatrix<uint8_t> basic_output = ImageMatrix<uint8_t>::Zero(height, width);

  device_input = input;

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    cuvslam::sof::BoxBlur3(input_u8, basic_output.data());
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    gpu_prefilter.prefilter(device_input, device_output, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);

  // ASSERT_TRUE(basic_output.isApprox(device_output.map(), 0.01f));
}
#endif  //__GNUC__

}  // namespace test
