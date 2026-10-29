
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

#include <string>
#include <string_view>
#include <thread>

#include "common/include_gtest.h"
#include "profiler/profiler.h"

#ifdef __linux__
#include <pthread.h>

namespace test::profiler {

std::string NameInNewThread(std::string_view name) {
  std::string result;
  std::thread([&] {
    cuvslam::profiler::NameCurrentThread(name);
    char buffer[16] = {};
    pthread_getname_np(pthread_self(), buffer, sizeof(buffer));
    result = buffer;
  }).join();
  return result;
}

TEST(NameCurrentThread, SetsOsThreadName) { EXPECT_EQ(NameInNewThread("cuvslam:sba"), "cuvslam:sba"); }

TEST(NameCurrentThread, TruncatesToLinuxLimit) { EXPECT_EQ(NameInNewThread("cuvslam:0123456789"), "cuvslam:0123456"); }

}  // namespace test::profiler
#endif
