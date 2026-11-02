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

#include "common/include_gtest.h"

#include "sba/sba_config.h"

namespace cuvslam::sba {
namespace {

TEST(SbaConfig, SelectModeOnCpu) {
  EXPECT_EQ(SelectMode(false, false), OriginalCPU);
  EXPECT_EQ(SelectMode(true, false), InertialCPU);
}

TEST(SbaConfig, SelectModeOnGpu) {
#ifdef USE_CUDA
  EXPECT_EQ(SelectMode(false, true), OriginalGPU);
  EXPECT_EQ(SelectMode(true, true), InertialGPU);
#else
  EXPECT_EQ(SelectMode(false, true), OriginalCPU);
  EXPECT_EQ(SelectMode(true, true), InertialCPU);
#endif
}

TEST(SbaConfig, SupportedModeKeepsCpuModes) {
  EXPECT_EQ(SupportedMode(Disabled), Disabled);
  EXPECT_EQ(SupportedMode(OriginalCPU), OriginalCPU);
  EXPECT_EQ(SupportedMode(InertialCPU), InertialCPU);
}

TEST(SbaConfig, SupportedModeMapsGpuModesForBuild) {
#ifdef USE_CUDA
  EXPECT_EQ(SupportedMode(OriginalGPU), OriginalGPU);
  EXPECT_EQ(SupportedMode(InertialGPU), InertialGPU);
#else
  EXPECT_EQ(SupportedMode(OriginalGPU), OriginalCPU);
  EXPECT_EQ(SupportedMode(InertialGPU), InertialCPU);
#endif
}

}  // namespace
}  // namespace cuvslam::sba
