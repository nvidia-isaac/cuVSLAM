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

#include "cuvslam/cuvslam_params.h"

#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "params/registry.h"

namespace cuvslam::params {
namespace {

/// Registers the public configs the way a tool or the tracker does.
class ConfigParamsTest : public ::testing::Test {
protected:
  void SetUp() override {
    registry.Add("odometry", odom_cfg);
    registry.Add("odometry.rgbd", odom_cfg.rgbd_settings);
    registry.Add("odometry.multisensor", odom_cfg.multisensor_settings);
    registry.Add("slam", slam_cfg);
  }

  Odometry::Config odom_cfg;
  Slam::Config slam_cfg;
  Registry registry;
};

TEST_F(ConfigParamsTest, ReachesStringFields) {
  // The YAML loader this replaced warned and gave up on both of these: Config holds string_view
  // and the loader had nowhere to keep the text. The registry owns that storage, so they work.
  registry.Set("odometry.debug_dump_directory", "/tmp/dump", Source::File);
  registry.Set("slam.map_cache_path", "/tmp/map.lmdb", Source::File);

  EXPECT_EQ(odom_cfg.debug_dump_directory, "/tmp/dump");
  EXPECT_EQ(slam_cfg.map_cache_path, "/tmp/map.lmdb");
}

TEST_F(ConfigParamsTest, StringFieldsOutliveTheCallersArgument) {
  {
    const std::string temporary = "/tmp/scoped-path";
    registry.Set("odometry.debug_dump_directory", temporary, Source::Api);
  }
  // Many more assignments, to make a reallocating container give up a dangling view if the
  // registry kept one.
  for (int i = 0; i < 256; ++i) {
    registry.Set("slam.map_cache_path", "/tmp/path-" + std::to_string(i), Source::Api);
  }
  EXPECT_EQ(odom_cfg.debug_dump_directory, "/tmp/scoped-path");
  EXPECT_EQ(slam_cfg.map_cache_path, "/tmp/path-255");
}

TEST_F(ConfigParamsTest, SetsMultisensorDepthCameraList) {
  registry.Set("odometry.multisensor.depth_camera_ids", "0,2,3", Source::File);
  EXPECT_EQ(odom_cfg.multisensor_settings.depth_camera_ids, std::vector<int32_t>({0, 2, 3}));

  registry.Set("odometry.multisensor.depth_camera_ids", "", Source::File);
  EXPECT_TRUE(odom_cfg.multisensor_settings.depth_camera_ids.empty());
}

TEST_F(ConfigParamsTest, NegativeDepthCameraIdStaysLegalBecauseItMeansUnset) {
  registry.Set("odometry.rgbd.depth_camera_id", "-1", Source::File);
  EXPECT_EQ(odom_cfg.rgbd_settings.depth_camera_id, -1);
}

TEST_F(ConfigParamsTest, TwoSettingsGroupsSharingAFieldNameStayDistinct) {
  // Both depth settings structs have depth_scale_factor and enable_depth_stereo_tracking.
  registry.Set("odometry.rgbd.depth_scale_factor", "5000", Source::File);
  registry.Set("odometry.multisensor.depth_scale_factor", "1000", Source::File);

  EXPECT_EQ(odom_cfg.rgbd_settings.depth_scale_factor, 5000.f);
  EXPECT_EQ(odom_cfg.multisensor_settings.depth_scale_factor, 1000.f);
  // The shared bare name is ambiguous and must not silently pick one.
  EXPECT_THROW(registry.Resolve("depth_scale_factor"), std::invalid_argument);
}

}  // namespace
}  // namespace cuvslam::params
