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

#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/include_gtest.h"
#include "cuvslam/cuvslam2.h"

#ifdef USE_CUDA
#include "cuda_modules/cuda_helper.h"
#endif

namespace {

using cuvslam::Image;
using cuvslam::Odometry;

constexpr int32_t kWidth = 640;
constexpr int32_t kHeight = 480;
constexpr int64_t kFramePeriodNs = 33'333'333;
constexpr int32_t kMaxShift = 64;
constexpr uint16_t kDepthMm = 2000;

cuvslam::Rig MakeMonoRig() {
  cuvslam::Camera camera;
  camera.size = {kWidth, kHeight};
  camera.focal = {320.f, 320.f};
  camera.principal = {kWidth / 2.f, kHeight / 2.f};
  cuvslam::Rig rig;
  rig.cameras.push_back(camera);
  return rig;
}

// Blocky random texture, wider than the image so that frames can slide over it.
std::vector<uint8_t> MakeTexture() {
  constexpr int32_t kBlock = 8;
  const int32_t width = kWidth + kMaxShift;
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> value(0, 255);
  std::vector<uint8_t> blocks((width / kBlock + 1) * (kHeight / kBlock + 1));
  for (auto& b : blocks) {
    b = static_cast<uint8_t>(value(rng));
  }
  std::vector<uint8_t> texture(width * kHeight);
  for (int32_t y = 0; y < kHeight; ++y) {
    for (int32_t x = 0; x < width; ++x) {
      texture[y * width + x] = blocks[(y / kBlock) * (width / kBlock + 1) + x / kBlock];
    }
  }
  return texture;
}

std::vector<uint8_t> MakeFrame(const std::vector<uint8_t>& texture, int32_t shift) {
  const int32_t width = kWidth + kMaxShift;
  std::vector<uint8_t> frame(kWidth * kHeight);
  for (int32_t y = 0; y < kHeight; ++y) {
    for (int32_t x = 0; x < kWidth; ++x) {
      frame[y * kWidth + x] = texture[y * width + x + shift];
    }
  }
  return frame;
}

// Excludes a block that covers rows [1/4, 3/4) and columns [1/4, 3/4) of the mask.
std::vector<uint8_t> MakeCenterMask(int32_t width, int32_t height) {
  std::vector<uint8_t> mask(width * height, 0);
  for (int32_t y = height / 4; y < 3 * height / 4; ++y) {
    for (int32_t x = width / 4; x < 3 * width / 4; ++x) {
      mask[y * width + x] = 1;
    }
  }
  return mask;
}

Image MakeImage(const void* pixels, int32_t width, int32_t height, int32_t bytes_per_pixel, bool is_gpu_mem,
                int64_t timestamp_ns) {
  Image img;
  img.pixels = pixels;
  img.width = width;
  img.height = height;
  img.pitch = width * bytes_per_pixel;
  img.encoding = Image::Encoding::MONO;
  img.data_type = bytes_per_pixel == 2 ? Image::DataType::UINT16 : Image::DataType::UINT8;
  img.is_gpu_mem = is_gpu_mem;
  img.timestamp_ns = timestamp_ns;
  img.camera_index = 0;
  return img;
}

Odometry::Config MakeConfig(Odometry::OdometryMode mode) {
  Odometry::Config cfg;
  cfg.odometry_mode = mode;
  cfg.async_sba = false;
  cfg.enable_observations_export = true;
  if (mode == Odometry::OdometryMode::RGBD) {
    cfg.rgbd_settings.depth_scale_factor = 1000.f;
    cfg.rgbd_settings.depth_camera_id = 0;
  }
  if (mode == Odometry::OdometryMode::Multisensor) {
    cfg.multisensor_settings.depth_camera_ids = {0};
    cfg.multisensor_settings.depth_scale_factor = 1000.f;
  }
  return cfg;
}

// Keeps a host buffer and, when requested, its device copy, so the same data can be passed in either memory space.
class Buffer {
public:
  template <typename T>
  Buffer(const std::vector<T>& host, [[maybe_unused]] bool on_gpu)
      : host_(reinterpret_cast<const uint8_t*>(host.data())) {
#ifdef USE_CUDA
    if (on_gpu) {
      const size_t bytes = host.size() * sizeof(T);
      device_ = std::make_unique<cuvslam::cuda::GPUOnlyArray<uint8_t>>(bytes);
      CUDA_CHECK(cudaMemcpy(device_->ptr(), host.data(), bytes, cudaMemcpyHostToDevice));
    }
#endif
  }
  const void* ptr() const {
#ifdef USE_CUDA
    if (device_) {
      return device_->ptr();
    }
#endif
    return host_;
  }

private:
  const uint8_t* host_;
#ifdef USE_CUDA
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<uint8_t>> device_;
#endif
};

struct RunResult {
  std::vector<size_t> observations_per_frame;
  std::vector<bool> valid_per_frame;
};

// Tracks a sliding texture with a constant mask; an empty mask means no mask is passed.
RunResult TrackSequence(Odometry::OdometryMode mode, const std::vector<uint8_t>& mask, int32_t mask_width,
                        int32_t mask_height, bool on_gpu, int num_frames) {
  Odometry odometry(MakeMonoRig(), MakeConfig(mode));
  const auto texture = MakeTexture();
  const std::vector<uint16_t> depth(kWidth * kHeight, kDepthMm);
  const Buffer depth_buffer(depth, on_gpu);
  const bool uses_depth = mode == Odometry::OdometryMode::RGBD || mode == Odometry::OdometryMode::Multisensor;
  std::unique_ptr<Buffer> mask_buffer;
  if (!mask.empty()) {
    mask_buffer = std::make_unique<Buffer>(mask, on_gpu);
  }

  RunResult result;
  for (int frame = 0; frame < num_frames; ++frame) {
    const int64_t timestamp = (frame + 1) * kFramePeriodNs;
    const auto pixels = MakeFrame(texture, frame);
    const Buffer image_buffer(pixels, on_gpu);
    Odometry::ImageSet masks;
    if (mask_buffer) {
      masks.push_back(MakeImage(mask_buffer->ptr(), mask_width, mask_height, 1, on_gpu, timestamp));
    }
    Odometry::ImageSet depths;
    if (uses_depth) {
      depths.push_back(MakeImage(depth_buffer.ptr(), kWidth, kHeight, 2, on_gpu, timestamp));
    }
    const auto pose =
        odometry.Track({MakeImage(image_buffer.ptr(), kWidth, kHeight, 1, on_gpu, timestamp)}, masks, depths);
    result.valid_per_frame.push_back(pose.world_from_rig.has_value());
    result.observations_per_frame.push_back(odometry.GetLastObservations(0).size());
  }
  return result;
}

Image ValidHostMask(const std::vector<uint8_t>& pixels) {
  return MakeImage(pixels.data(), kWidth, kHeight, 1, false, kFramePeriodNs);
}

}  // namespace

TEST(Mask, RejectsInvalidMasks) {
  const std::vector<uint8_t> pixels(kWidth * kHeight, 0);
  const Image image = MakeImage(pixels.data(), kWidth, kHeight, 1, false, kFramePeriodNs);
  const std::vector<uint8_t> mask_pixels(kWidth * kHeight, 0);

  Image no_buffer = ValidHostMask(mask_pixels);
  no_buffer.pixels = nullptr;
  Image wrong_type = ValidHostMask(mask_pixels);
  wrong_type.data_type = Image::DataType::UINT16;
  Image wrong_encoding = ValidHostMask(mask_pixels);
  wrong_encoding.encoding = Image::Encoding::RGB;
  Image empty = ValidHostMask(mask_pixels);
  empty.width = 0;
  Image unknown_camera = ValidHostMask(mask_pixels);
  unknown_camera.camera_index = 1;

  for (const Image& mask : {no_buffer, wrong_type, wrong_encoding, empty, unknown_camera}) {
    Odometry odometry(MakeMonoRig(), MakeConfig(Odometry::OdometryMode::Mono));
    EXPECT_THROW(odometry.Track({image}, {mask}), std::invalid_argument);
  }

  Odometry odometry(MakeMonoRig(), MakeConfig(Odometry::OdometryMode::Mono));
  EXPECT_THROW(odometry.Track({image}, {ValidHostMask(mask_pixels), ValidHostMask(mask_pixels)}),
               std::invalid_argument);
}

#ifdef USE_CUDA

// Depth is masked as well as features, so a low-resolution mask must act exactly like its nearest-neighbor upscale.
TEST(Mask, RgbdLowResolutionMaskMatchesUpscaled) {
  const auto half = MakeCenterMask(kWidth / 2, kHeight / 2);
  const auto full = MakeCenterMask(kWidth, kHeight);
  for (bool on_gpu : {false, true}) {
    SCOPED_TRACE(on_gpu ? "device memory" : "host memory");
    const auto expected = TrackSequence(Odometry::OdometryMode::RGBD, full, kWidth, kHeight, on_gpu, 10);
    const auto actual = TrackSequence(Odometry::OdometryMode::RGBD, half, kWidth / 2, kHeight / 2, on_gpu, 10);
    EXPECT_GT(expected.observations_per_frame.back(), 0u);
    EXPECT_EQ(actual.observations_per_frame, expected.observations_per_frame);
    EXPECT_EQ(actual.valid_per_frame, expected.valid_per_frame);
  }
}

// A device mask has to apply to the frame it was passed with, starting with the first one.
TEST(Mask, DeviceMaskAppliesToCurrentFrame) {
  const std::vector<uint8_t> exclude_all(kWidth * kHeight, 1);
  const auto unmasked = TrackSequence(Odometry::OdometryMode::Mono, {}, kWidth, kHeight, true, 1);
  ASSERT_GT(unmasked.observations_per_frame[0], 0u);

  const auto masked = TrackSequence(Odometry::OdometryMode::Mono, exclude_all, kWidth, kHeight, true, 3);
  for (size_t frame = 0; frame < masked.observations_per_frame.size(); ++frame) {
    EXPECT_EQ(masked.observations_per_frame[frame], 0u) << "frame " << frame;
  }
}

#endif

#ifdef USE_CUNLS

// With every pixel excluded there is nothing to solve for; tracking has to fail softly rather than throw.
TEST(Mask, MultisensorExcludeAllDoesNotThrow) {
  const std::vector<uint8_t> exclude_all(kWidth * kHeight, 1);
  for (bool on_gpu : {false, true}) {
    SCOPED_TRACE(on_gpu ? "device memory" : "host memory");
    EXPECT_NO_THROW(TrackSequence(Odometry::OdometryMode::Multisensor, exclude_all, kWidth, kHeight, on_gpu, 5));
  }
}

#endif
