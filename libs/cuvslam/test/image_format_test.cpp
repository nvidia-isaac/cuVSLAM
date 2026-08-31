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

#include <memory>
#include <vector>

#include "common/include_gtest.h"
#include "cuvslam/cuvslam2.h"

class TestImageFormat : public testing::Test {
protected:
  void SetUp() override {
    // Create a simple camera rig for testing
    cuvslam::Camera camera;
    camera.size = {640, 480};
    camera.focal = {320.0f, 320.0f};
    camera.principal = {320.0f, 240.0f};
    rig.cameras.push_back(camera);
    cfg = cuvslam::Odometry::Config{};
    odometry = std::make_unique<cuvslam::Odometry>(rig, cfg);
    timestamp = 1000;
  }

  cuvslam::Rig rig;
  cuvslam::Odometry::Config cfg;
  std::unique_ptr<cuvslam::Odometry> odometry;
  int64_t timestamp;
};

TEST_F(TestImageFormat, ValidMonoImage) {
  // Valid mono image
  std::vector<uint8_t> valid_mono(480 * 640, 0);
  cuvslam::Image img;
  img.timestamp_ns = timestamp;
  img.camera_index = 0;
  img.width = 640;
  img.height = 480;
  img.pixels = valid_mono.data();
  img.encoding = cuvslam::Image::Encoding::MONO;
  img.data_type = cuvslam::Image::DataType::UINT8;
  img.is_gpu_mem = false;
  img.pitch = 640;

  auto result = odometry->Track({img});
  EXPECT_TRUE(result.world_from_rig.has_value());
}

TEST_F(TestImageFormat, ValidRGBImage) {
  // Valid RGB image
  std::vector<uint8_t> valid_rgb(480 * 640 * 3, 0);
  cuvslam::Image img;
  img.timestamp_ns = timestamp;
  img.camera_index = 0;
  img.width = 640;
  img.height = 480;
  img.pixels = valid_rgb.data();
  img.encoding = cuvslam::Image::Encoding::RGB;
  img.data_type = cuvslam::Image::DataType::UINT8;
  img.is_gpu_mem = false;
  img.pitch = 640 * 3;

  auto result = odometry->Track({img});
  EXPECT_TRUE(result.world_from_rig.has_value());
}

TEST_F(TestImageFormat, InvalidDtype) {
  // Test invalid data type (float32 instead of uint8)
  std::vector<float> invalid_dtype(480 * 640, 0.0f);
  cuvslam::Image img;
  img.timestamp_ns = timestamp;
  img.camera_index = 0;
  img.width = 640;
  img.height = 480;
  img.pixels = invalid_dtype.data();
  img.encoding = cuvslam::Image::Encoding::MONO;
  img.data_type = cuvslam::Image::DataType::FLOAT32;
  img.is_gpu_mem = false;
  img.pitch = 640 * sizeof(float);

  EXPECT_THROW(odometry->Track({img}), std::invalid_argument);
}
