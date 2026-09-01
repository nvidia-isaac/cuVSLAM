
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

#include "track_edex_api2.h"

#include <sstream>

#include "common/include_gtest.h"
#include "gflags/gflags.h"

DEFINE_string(data_folder, ".", "Edex data folder");
DEFINE_string(odom_poses_file, ".", "Odom poses file");
DEFINE_string(slam_poses_file, ".", "Slam poses file");

using namespace cuvslam;

#if defined(ENFORCE_GPU)
static_assert(USE_CUDA);
auto use_cuda = testing::Values(true);
auto use_gpu_mem = testing::Bool();
#elif defined(USE_CUDA)
auto use_cuda = testing::Bool();
auto use_gpu_mem = testing::Bool();
#else
auto use_cuda = testing::Values(false);
auto use_gpu_mem = testing::Values(false);
#endif

struct VioApi2TestParams {
  using Tuple = std::tuple<Odometry::MulticameraMode, Odometry::OdometryMode, bool, bool, bool, bool, bool, bool, bool,
                           bool, bool>;

  VioApi2TestParams(Tuple t) {
    cfg.multicam_mode = std::get<0>(t);
    cfg.odometry_mode = std::get<1>(t);
    cfg.use_gpu = std::get<2>(t);
    test.use_gpu_mem = std::get<3>(t);
    cfg.use_motion_model = std::get<4>(t);
    cfg.use_denoising = std::get<5>(t);
    cfg.rectified_stereo_camera = std::get<6>(t);
    cfg.enable_observations_export = std::get<7>(t);
    cfg.enable_landmarks_export = std::get<8>(t);
    cfg.enable_final_landmarks_export = std::get<9>(t);
    test.enable_slam = std::get<10>(t);
  }

  friend std::ostream& operator<<(std::ostream& stream, const VioApi2TestParams& params) {
    stream << static_cast<uint32_t>(params.cfg.multicam_mode);
    stream << static_cast<uint32_t>(params.cfg.odometry_mode);
    stream << params.cfg.use_gpu;
    stream << params.test.use_gpu_mem;
    stream << params.cfg.use_motion_model;
    stream << params.cfg.use_denoising;
    stream << params.cfg.rectified_stereo_camera;
    stream << params.cfg.enable_observations_export;
    stream << params.cfg.enable_landmarks_export;
    stream << params.cfg.enable_final_landmarks_export;
    stream << params.test.enable_slam;

    return stream;
  }

  std::string info() const {
    std::ostringstream ss;
    ss << "Multicam mode=" << static_cast<uint32_t>(cfg.multicam_mode);
    ss << ", Odom mode=" << static_cast<uint32_t>(cfg.odometry_mode);
    ss << ", GPU=" << cfg.use_gpu;
    ss << ", GPU mem=" << test.use_gpu_mem;
    ss << ", Motion=" << cfg.use_motion_model;
    ss << ", Denoise=" << cfg.use_denoising;
    ss << ", Rectified=" << cfg.rectified_stereo_camera;
    ss << ", ObsExp=" << cfg.enable_observations_export;
    ss << ", LmsExp=" << cfg.enable_landmarks_export;
    ss << ", FinalLmsExp=" << cfg.enable_final_landmarks_export;
    ss << ", SLAM=" << test.enable_slam;
    return ss.str();
  }

  Odometry::Config cfg;
  TestingSettings test;
};

class VioApi2Test : public testing::TestWithParam<VioApi2TestParams> {};

TEST_P(VioApi2Test, TrackEdex) {
  auto params = GetParam();
  std::cout << "Params: " << params.info() << std::endl;
  params.test.data_folder = FLAGS_data_folder;
  params.test.odom_poses_file = FLAGS_odom_poses_file;
  params.test.slam_poses_file = FLAGS_slam_poses_file;
  ASSERT_TRUE(DoesEdexExist(params.test.data_folder));
  if (params.cfg.odometry_mode == Odometry::OdometryMode::Inertial && !DoesEdexHaveImu(params.test.data_folder)) {
    GTEST_SKIP() << "IMU fusion is enabled, but dataset has no IMU data.";
  }
  if (params.test.enable_slam && !params.cfg.enable_observations_export && !params.cfg.enable_landmarks_export &&
      !params.cfg.enable_final_landmarks_export) {
    ASSERT_FALSE(TrackEdexApi2(params.test, params.cfg));
    // GTEST_SKIP() << "SLAM is enabled, but no export of observations, landmarks or final landmarks is enabled.";
    return;
  }

  ASSERT_TRUE(TrackEdexApi2(params.test, params.cfg));
  // TODO(vikuznetsov): implement some sanity check for tracking results
}

INSTANTIATE_TEST_SUITE_P(CppApiConfigs, VioApi2Test,
                         testing::ConvertGenerator<VioApi2TestParams::Tuple>(testing::Combine(
                             testing::Values(Odometry::MulticameraMode::Performance,
                                             Odometry::MulticameraMode::Precision, Odometry::MulticameraMode::Moderate),
                             testing::Values(Odometry::OdometryMode::Multicamera, Odometry::OdometryMode::Inertial,
                                             Odometry::OdometryMode::RGBD, Odometry::OdometryMode::Mono),
                             use_cuda, use_gpu_mem, testing::Bool(), testing::Bool(), testing::Bool(), testing::Bool(),
                             testing::Bool(), testing::Bool(), testing::Bool())),
                         [](const testing::TestParamInfo<VioApi2TestParams>& info) -> std::string {
                           std::ostringstream ss;
                           ss << info.param;
                           return ss.str();
                         });

// Tests that need no dataset live in libs/cuvslam/test, where ctest runs them. This file keeps only
// the edex replay suite, which requires --data_folder.
