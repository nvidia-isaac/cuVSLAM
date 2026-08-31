
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

#include <cstdlib>

#include "gflags/gflags.h"

#include "common/include_gtest.h"
#include "common/log.h"

#ifdef USE_RERUN
#include "rerun/config.hpp"
#endif

DEFINE_string(logger_filename, "", "Log filename");

namespace test {

// Seeds std::rand() for each test and prints a retry command on failure.
//
// std::rand() is re-seeded before each test with gtest's --gtest_random_seed
// (time-based when unset), making random data independent of execution order
// and --gtest_filter. gtest advances its own seed between iterations when
// --gtest_shuffle is set, so use it with --gtest_repeat to get varying data.
class GTestEventListener : public ::testing::EmptyTestEventListener {
  const char* binary_ = nullptr;

public:
  explicit GTestEventListener(const char* binary) : binary_(binary) {}

  void OnTestStart(const ::testing::TestInfo&) override {
    std::srand(::testing::UnitTest::GetInstance()->random_seed());
  }

  void OnTestEnd(const ::testing::TestInfo& info) override {
    if (info.result()->Failed()) {
      printf("\n[CUVSLAM_TEST_RETRY]\t%s --gtest_filter=%s.%s --gtest_random_seed=%d\n", binary_,
             info.test_suite_name(), info.name(), ::testing::UnitTest::GetInstance()->random_seed());
    }
  }
};

}  // namespace test

using namespace cuvslam;

int main(int argc, char** argv) {
#ifdef USE_RERUN
  // Unit tests must not require a viewer. RERUN=1 can override this default for interactive debugging.
  //
  // The Rerun SDK is linked statically into both test executables and libcuvslam.so, so each binary
  // has its own programmatic default. The environment variable is process-wide and disables every
  // copy before any RecordingStream is created.
  if (std::getenv("RERUN") == nullptr) {
#ifdef _WIN32
    _putenv_s("RERUN", "0");
#else
    setenv("RERUN", "0", 0);
#endif
  }
  rerun::set_default_enabled(false);
#endif

  ::testing::InitGoogleTest(&argc, argv);

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Trace::SetVerbosity(Trace::Verbosity::Error);
  if (!FLAGS_logger_filename.empty()) {
#ifdef CUVSLAM_LOG_ENABLE
    std::cout << "Create spd logger: " << FLAGS_logger_filename << std::endl;
    auto logger = log::CreateSpdlogLogger(FLAGS_logger_filename.c_str());
    log::SetLogger(logger);
#else  // !CUVSLAM_LOG_ENABLE
    std::cout << "No CUVSLAM_LOG_ENABLE definition. Flag -logger_filename will be ignored " << std::endl;
#endif
  }

  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  listeners.Append(new test::GTestEventListener(argv[0]));

  return RUN_ALL_TESTS();
}
