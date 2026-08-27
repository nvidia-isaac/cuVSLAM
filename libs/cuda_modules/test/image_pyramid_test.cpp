
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

#include "benchmark_utils.h"
#include "common/include_gtest.h"
#include "cuda_modules/image_pyramid.h"
#include "sof/image_pyramid_float.h"
#include "sof/image_pyramid_u8.h"

namespace test {
using namespace cuvslam;

TEST(Cuda, ImageGaussianScaler) {
  sof::ImageGaussianScaler basic_scaler;
  cuda::ImageGaussianScaler cuda_scaler;

  size_t img_size = 1080;

  ImageMatrixT input = ImageMatrixT::Random(img_size, img_size) * 100;
  input.array() += 110;

  cuda::GPUImageT device_input(input);
  cuda::GPUImageT device_output((img_size + 1) / 2, (img_size + 1) / 2);
  ImageMatrixT cpu_output((img_size + 1) / 2, (img_size + 1) / 2);
  ImageMatrixT basic_output;

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    basic_scaler(input, basic_output);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  cuda::Stream s;
  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    cuda_scaler(device_input, device_output, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  std::cout << "Basic convolver time, nano_sec = " << duration_basic.count() / 100 << std::endl;
  std::cout << "Cuda convolver time, nano_sec = " << duration_cuda.count() / 100 << std::endl;
  float speedup = static_cast<float>(duration_basic.count()) / static_cast<float>(duration_cuda.count());
  std::cout << "Speedup, times = " << speedup << std::endl;
  ASSERT_TRUE(duration_basic >= duration_cuda);

  ASSERT_TRUE(static_cast<size_t>(basic_output.rows()) == device_output.rows());
  ASSERT_TRUE(static_cast<size_t>(basic_output.cols()) == device_output.cols());

  device_output.copy(cuda::GPUCopyDirection::ToCPU, cpu_output.data(), s.get_stream());
  cudaStreamSynchronize(s.get_stream());
  if (!basic_output.isApprox(cpu_output, 0.01f)) {
    std::cout << "input = " << std::endl << input << std::endl;
    std::cout << "basic_output = " << std::endl << basic_output << std::endl;
    std::cout << "cuda_output = " << std::endl << cpu_output << std::endl;
  }
  ASSERT_TRUE(basic_output.isApprox(cpu_output, 0.01f));
}

TEST(Cuda, ImagePyramid) {
  size_t img_size = 1080;

  ImageMatrixT input = ImageMatrixT::Random(img_size, img_size) * 100;
  input.array() += 110;

  cuda::GPUImageT device_input(input);

  sof::ImagePyramidT image_pyramid = sof::ImagePyramidT(input);
  cuda::GaussianGPUImagePyramid cuda_image_pyramid(img_size, img_size);

  image_pyramid.buildPyramid();
  cuda::Stream s;
  cuda_image_pyramid.build(device_input, false, s.get_stream());
  cudaStreamSynchronize(s.get_stream());
  ASSERT_TRUE(static_cast<size_t>(image_pyramid.getLevelsCount()) == cuda_image_pyramid.getLevelsCount());

  {
    for (int i = 0; i < image_pyramid.getLevelsCount(); i++) {
      auto level = image_pyramid[i];
      auto& cuda_level = cuda_image_pyramid[i];
      ImageMatrixT level_cpu(cuda_level.rows(), cuda_level.cols());
      cuda_level.copy(cuda::GPUCopyDirection::ToCPU, level_cpu.data(), s.get_stream());
      cudaStreamSynchronize(s.get_stream());
      if (!level.isApprox(level_cpu, 0.01f)) {
        std::cout << "basic_output = " << std::endl << level << std::endl;
        std::cout << "cuda_output = " << std::endl << level_cpu << std::endl;
      }
      ASSERT_TRUE(level.isApprox(level_cpu, 0.01f));
    }
  }
}

TEST(Cuda, ImagePyramidSpeedup) {
  size_t width = 1080;
  size_t height = 1080;

  ImageMatrixT input = ImageMatrixT::Random(height, width) * 100;

  cuda::GPUImageT device_input(input);

  sof::ImagePyramidT image_pyramid = sof::ImagePyramidT(input);
  cuda::GaussianGPUImagePyramid cuda_image_pyramid(width, height);

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    image_pyramid.buildPyramid();
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  cuda::Stream s;
  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    cuda_image_pyramid.build(device_input, false, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, ImagePyramidU8) {
  size_t img_size = 512;

  ImageMatrix<uint8_t> input = ImageMatrix<uint8_t>::Random(img_size, img_size) * 100;
  ImageMatrixT input_float = input.cast<float>();
  ImageSource source;
  source.type = ImageSource::Type::U8;
  source.data = input.data();

  ImageShape shape{static_cast<int>(img_size), static_cast<int>(img_size)};

  cuda::GPUImageT device_input(input_float);

  sof::ImagePyramidU8 image_pyramid;
  cuda::GaussianGPUImagePyramid cuda_image_pyramid(img_size, img_size);

  image_pyramid.build(source, shape, false);
  cuda::Stream s;
  cuda_image_pyramid.build(device_input, false, s.get_stream());
  cudaStreamSynchronize(s.get_stream());
  ASSERT_TRUE(static_cast<size_t>(image_pyramid.num_levels()) == cuda_image_pyramid.getLevelsCount());

  {
    for (int i = 0; i < image_pyramid.num_levels(); i++) {
      if (i > 3) {
        break;
      }
      auto level = image_pyramid[i];
      auto& cuda_level = cuda_image_pyramid[i];
      ImageMatrixT level_cpu(cuda_level.rows(), cuda_level.cols());
      ImageMatrix<uint8_t> level_cpu_u8 = level.first.as<uint8_t>(level.second);
      cuda_level.copy(cuda::GPUCopyDirection::ToCPU, level_cpu.data(), s.get_stream());
      cudaStreamSynchronize(s.get_stream());
      if (!level_cpu_u8.cast<float>().isApprox(level_cpu, 0.01f)) {
        std::cout << "i = " << std::endl << i << std::endl;
        std::cout << "basic_output = " << std::endl << level_cpu_u8.cast<float>().block<5, 5>(0, 0) << std::endl;
        std::cout << "cuda_output = " << std::endl << level_cpu.block<5, 5>(0, 0) << std::endl;
      }
      ASSERT_TRUE(level_cpu_u8.cast<float>().isApprox(level_cpu, 0.01f));
    }
  }
}

TEST(Cuda, ImagePyramidU8Prefilter) {
  size_t img_size = 512;

  ImageMatrix<uint8_t> input = ImageMatrix<uint8_t>::Random(img_size, img_size) * 100;
  ImageMatrixT input_float = input.cast<float>();
  ImageSource source;
  source.type = ImageSource::Type::U8;
  source.data = input.data();

  ImageShape shape{static_cast<int>(img_size), static_cast<int>(img_size)};

  cuda::GPUImageT device_input(input_float);

  sof::ImagePyramidU8 image_pyramid;
  cuda::GaussianGPUImagePyramid cuda_image_pyramid(img_size, img_size);

  image_pyramid.build(source, shape, true);
  cuda::Stream s;
  cuda_image_pyramid.build(device_input, true, s.get_stream());
  cudaStreamSynchronize(s.get_stream());
  ASSERT_TRUE(static_cast<size_t>(image_pyramid.num_levels()) == cuda_image_pyramid.getLevelsCount());

  {
    for (int i = 0; i < image_pyramid.num_levels(); i++) {
      if (i > 3) {
        break;
      }
      auto level = image_pyramid[i];
      auto& cuda_level = cuda_image_pyramid[i];
      ImageMatrixT level_cpu(cuda_level.rows(), cuda_level.cols());
      ImageMatrix<uint8_t> level_cpu_u8 = level.first.as<uint8_t>(level.second);
      cuda_level.copy(cuda::GPUCopyDirection::ToCPU, level_cpu.data(), s.get_stream());
      cudaStreamSynchronize(s.get_stream());
      if (!level_cpu_u8.cast<float>().isApprox(level_cpu, 0.01f)) {
        std::cout << "i = " << std::endl << i << std::endl;
        std::cout << "basic_output = " << std::endl << level_cpu_u8.cast<float>().block<5, 5>(0, 0) << std::endl;
        std::cout << "cuda_output = " << std::endl << level_cpu.block<5, 5>(0, 0) << std::endl;
      }
      ASSERT_TRUE(level_cpu_u8.cast<float>().isApprox(level_cpu, 0.01f));
    }
  }
}

TEST(Cuda, ImagePyramidU8Speedup) {
  size_t width = 1080;
  size_t height = 1080;

  ImageMatrix<uint8_t> input = ImageMatrix<uint8_t>::Random(height, width) * 100;
  ImageMatrixT input_float = input.cast<float>();

  ImageSource source;
  source.type = ImageSource::Type::U8;
  source.data = input.data();

  ImageShape shape{static_cast<int>(width), static_cast<int>(height)};

  cuda::GPUImageT device_input(input_float);

  sof::ImagePyramidU8 image_pyramid;
  cuda::GaussianGPUImagePyramid cuda_image_pyramid(width, height);
  cuda::Stream s;

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    image_pyramid.build(source, shape, false);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    cuda_image_pyramid.build(device_input, false, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, ImagePyramidU8PrefilterSpeedup) {
  size_t width = 1080;
  size_t height = 1080;

  ImageMatrix<uint8_t> input = ImageMatrix<uint8_t>::Random(height, width) * 100;
  ImageMatrixT input_float = input.cast<float>();

  ImageSource source;
  source.type = ImageSource::Type::U8;
  source.data = input.data();

  ImageShape shape{static_cast<int>(width), static_cast<int>(height)};

  cuda::GPUImageT device_input(input_float);

  sof::ImagePyramidU8 image_pyramid;
  cuda::GaussianGPUImagePyramid cuda_image_pyramid(width, height);
  cuda::Stream s;

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    image_pyramid.build(source, shape, true);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    cuda_image_pyramid.build(device_input, true, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

}  // namespace test
