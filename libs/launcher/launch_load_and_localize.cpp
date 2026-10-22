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

#include "common/json_to_eigen.h"
#include "launcher/base_launcher.h"
#include "utils/image_loader.h"

#include <algorithm>

#include "gflags/gflags.h"

DEFINE_int32(slam_simulate_slow_map_load, 0, "Delay in ms after map loading.");
DEFINE_string(slam_input_database, "", "SLAM: input database for localization");
DEFINE_string(slam_localize_image, "", "Image path. If empty use current frame image");
DEFINE_string(slam_localize_guess_translation, "",
              "Guess pose (translation) for localization. Format: \"[float, float, float]\"");

namespace {
using namespace cuvslam;

Isometry3T ReadGuessPoseFromGFlags() {
  Isometry3T guess_pose = Isometry3T::Identity();
  Vector3T translation(0, 0, 0);
  ReadEigenFromString(FLAGS_slam_localize_guess_translation, translation);
  guess_pose.translate(translation);
  return guess_pose;
}

Slam::LocalizationSettings GetDefaultLocalizationSettings() {
  Slam::LocalizationSettings settings{};
  settings.horizontal_search_radius = 1.5f;  // meters
  settings.vertical_search_radius = 0.5;     ///< vertical search radius in meters
  settings.horizontal_step = 0.5f;           ///< horizontal step in meters
  settings.vertical_step = 0.25f;            ///< vertical step in meters
  settings.angular_step_rads = 2 * PI / 36;  ///< angular step around vertical axis in radians
  return settings;
}

std::shared_ptr<sof::ImageContext> LoadImage(const std::string& filename, bool to_gpu) {
  utils::ImageLoaderT image_loader;

  if (!image_loader.load(filename)) {
    return nullptr;
  }
  const ImageMatrix<uint8_t>& image_matrix = image_loader.getImage();
  ImageSource image_source{};
  image_source.type = ImageSource::U8;
  image_source.memory_type = ImageSource::Host;
  image_source.data = const_cast<uint8_t*>(image_matrix.data());
  image_source.pitch = 0;  // ignored for cpu images
  image_source.image_encoding = MONO8;

  ImageShape shape{};
  shape.height = static_cast<int>(image_matrix.rows());
  shape.width = static_cast<int>(image_matrix.cols());

  auto image_ctx = std::make_shared<sof::ImageContext>(shape, to_gpu, false);

  ImageMeta meta;
  meta.shape = shape;
  meta.frame_id = -1;
  meta.camera_index = 0;
  meta.timestamp = -1;
  image_ctx->set_image_meta(meta);
  if (to_gpu) {
#ifdef USE_CUDA
    cuda::Stream s{true};
    image_ctx->build_gpu_image_pyramid(image_source, false, s.get_stream());
    image_ctx->build_gpu_gradient_pyramid(false, s.get_stream());
#endif
  } else {
    image_ctx->build_cpu_image_pyramid(image_source, false);
    image_ctx->build_cpu_gradient_pyramid(false);
  }
  return image_ctx;
}
}  // namespace

namespace cuvslam::launcher {
void LaunchLoadMapAndLocalize(slam::AsyncSlam& slam, int64_t timestamp_ns, const sof::Images& current_images,
                              bool use_gpu) {
  const Isometry3T guess_pose = ReadGuessPoseFromGFlags();

  const bool user_provide_image = !FLAGS_slam_localize_image.empty();  // user provide image path for localization
  sof::Images user_provided_images(std::max<size_t>(current_images.size(), 1), nullptr);
  if (user_provide_image) {
    user_provided_images[0] = LoadImage(FLAGS_slam_localize_image, use_gpu);
    if (!user_provided_images[0]) {
      throw std::runtime_error("Can't load image for localization");
    }
  }

  const Slam::LocalizeStartCB start_cb = [] {
    if (FLAGS_slam_simulate_slow_map_load > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(FLAGS_slam_simulate_slow_map_load));
    }
  };

  slam.LocalizeInMap(FLAGS_slam_input_database, timestamp_ns, guess_pose,
                     user_provide_image ? user_provided_images : current_images, GetDefaultLocalizationSettings(),
                     /*attach=*/false, start_cb, nullptr);
}
}  // namespace cuvslam::launcher
