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

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <NvInferVersion.h>
#include <dlfcn.h>

#include "common/include_gtest.h"
#include "slam/vpr/anyloc_network.h"

namespace test::vpr {
using namespace cuvslam::slam::vpr;

TEST(TensorRtLoader, LoadsTensorRt) {
  // This executable links no TensorRT, so LoadTensorRt() has to find it
  const std::string library = "libnvinfer.so." + std::to_string(NV_TENSORRT_MAJOR);
  ASSERT_EQ(dlopen(library.c_str(), RTLD_LAZY | RTLD_NOLOAD), nullptr) << library << " is loaded already";
  EXPECT_NE(LoadTensorRt(), nullptr);
}

TEST(TensorRtLoaderDeathTest, ReportsAMissingTensorRt) {
  // The empty libnvinfer in CUVSLAM_NO_TENSORRT_DIR stands in for a missing one; only a new process reads the path
  const char* path = std::getenv("LD_LIBRARY_PATH");
  const std::string search = std::string(CUVSLAM_NO_TENSORRT_DIR) + (path && *path ? std::string(":") + path : "");
  setenv("LD_LIBRARY_PATH", search.c_str(), 1);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        try {
          LoadTensorRt();
        } catch (const std::runtime_error& error) {
          std::fputs(error.what(), stderr);
          std::exit(0);
        }
        std::exit(1);
      },
      ::testing::ExitedWithCode(0), "VPR AnyLoc needs TensorRT [0-9]+: ");
}

}  // namespace test::vpr
