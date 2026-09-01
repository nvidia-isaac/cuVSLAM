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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "common/include_gtest.h"

namespace test {

inline std::string FormatBenchmarkValue(double value) {
  std::ostringstream stream;
  stream << std::setprecision(10) << value;
  return stream.str();
}

inline void ReportSpeedBenchmark(std::chrono::nanoseconds cpu_duration, std::chrono::nanoseconds gpu_duration,
                                 std::size_t iterations) {
  ASSERT_GT(iterations, 0U);
  ASSERT_GT(cpu_duration.count(), 0);
  ASSERT_GT(gpu_duration.count(), 0);

  const auto cpu_ns_per_iteration = cpu_duration.count() / static_cast<std::int64_t>(iterations);
  const auto gpu_ns_per_iteration = gpu_duration.count() / static_cast<std::int64_t>(iterations);
  const double speedup = static_cast<double>(cpu_duration.count()) / static_cast<double>(gpu_duration.count());

  ::testing::Test::RecordProperty("iterations", std::to_string(iterations));
  ::testing::Test::RecordProperty("cpu_ns_per_iteration", std::to_string(cpu_ns_per_iteration));
  ::testing::Test::RecordProperty("gpu_ns_per_iteration", std::to_string(gpu_ns_per_iteration));
  ::testing::Test::RecordProperty("speedup", FormatBenchmarkValue(speedup));

  std::cout << "CPU time, ns/iteration = " << cpu_ns_per_iteration << std::endl;
  std::cout << "GPU time, ns/iteration = " << gpu_ns_per_iteration << std::endl;
  std::cout << "Speedup, times = " << speedup << std::endl;
}

}  // namespace test
