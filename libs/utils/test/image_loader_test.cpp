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

#include "common/include_gtest.h"
#include "common/log.h"
#include "utils/image_loader.h"

namespace test {
using namespace cuvslam;

// A path is user input: logging it as a printf format would read %s and write through %n.
TEST(ImageLoader, LogsPathWithConversionsLiterally) {
  const Trace::Verbosity verbosity = Trace::GetVerbosity();
  Trace::SetVerbosity(Trace::Verbosity::Error);

  const std::string path = "/nonexistent/%s%s%s%n%n%08x.png";
  utils::ImageLoaderT loader;
  testing::internal::CaptureStdout();
  const bool loaded = loader.load(path);
  const std::string output = testing::internal::GetCapturedStdout();
  Trace::SetVerbosity(verbosity);

  EXPECT_FALSE(loaded);
  EXPECT_NE(output.find(path), std::string::npos) << output;
}

}  // namespace test
