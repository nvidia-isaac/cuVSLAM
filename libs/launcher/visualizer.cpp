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

#include <algorithm>
#include <unordered_map>

#include <rerun/archetypes/depth_image.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/points3d.hpp>

#include "common/coordinate_system.h"
#include "common/log.h"
#include "launcher/visualizer.h"
#include "visualizer/visualizer.hpp"

namespace cuvslam::launcher {

void setupFrameTimeline(const Metas& meta) {
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  size_t frame_id = meta.at(0).frame_id;
  int64_t timestamp = meta.at(0).timestamp;
  visualizer.setupTimeline(frame_id, timestamp);

  // Set coordinate system (right-hand, Y-down, Z-forward)
  visualizer.getRecordingStream().log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN);
}

void logCameraImages(const Metas& meta, const Sources& images, const std::vector<CameraId>& cam_ids,
                     const std::vector<std::string>& viewport_names) {
  // Check if the number of cameras and viewport names match
  if (cam_ids.size() != viewport_names.size()) {
    TraceError("logCameraImages: The number of cameras and viewport names do not match");
    return;
  }
  // Log images for all cameras specified in cam_ids
  auto& visualizer = visualizer::RerunVisualizer::getInstance();

  for (size_t cam_index = 0; cam_index < cam_ids.size(); cam_index++) {
    const CameraId cam_id = cam_ids[cam_index];

    if (cam_id >= images.size() || cam_id >= meta.size() || images[cam_id].data == nullptr) {
      continue;
    }
    const auto& image = images[cam_id];
    const auto& meta_data = meta[cam_id];

    // Log the image
    visualizer.getRecordingStream().log(
        viewport_names.at(cam_index),
        rerun::Image(static_cast<const uint8_t*>(image.data),
                     {static_cast<uint32_t>(meta_data.shape.width), static_cast<uint32_t>(meta_data.shape.height)},
                     rerun::datatypes::ColorModel::L));
  }
}

void logDepthImage(const ImageSource& depth_source, const ImageMeta& meta, const std::string& viewport_name,
                   float depth_meter, float min_depth, float max_depth) {
  if (depth_source.data == nullptr) {
    TraceWarning("logDepthImage: Null depth data for viewport %s", viewport_name.c_str());
    return;
  }

  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  auto& recording = visualizer.getRecordingStream();

  const uint32_t width = static_cast<uint32_t>(meta.shape.width);
  const uint32_t height = static_cast<uint32_t>(meta.shape.height);

  if (depth_source.type == ImageSource::F32) {
    // Float32 depth image (values typically in meters)
    recording.log(viewport_name, rerun::DepthImage(static_cast<const float*>(depth_source.data), {width, height})
                                     .with_meter(depth_meter)
                                     .with_colormap(rerun::components::Colormap::Turbo)
                                     .with_depth_range(rerun::components::ValueRange(
                                         {static_cast<double>(min_depth), static_cast<double>(max_depth)})));
  } else if (depth_source.type == ImageSource::U16) {
    // Uint16 depth image (common for depth cameras like RealSense)
    recording.log(viewport_name, rerun::DepthImage(static_cast<const uint16_t*>(depth_source.data), {width, height})
                                     .with_meter(depth_meter)
                                     .with_colormap(rerun::components::Colormap::Turbo)
                                     .with_depth_range(rerun::components::ValueRange(
                                         {static_cast<double>(min_depth), static_cast<double>(max_depth)})));
  } else {
    TraceWarning("logDepthImage: Unsupported depth image type for viewport %s", viewport_name.c_str());
  }
}

void clearDepthImage(const std::string& viewport_name) {
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  visualizer.getRecordingStream().log(viewport_name, rerun::Clear());
}

}  // namespace cuvslam::launcher
