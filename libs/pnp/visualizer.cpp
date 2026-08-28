

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

#include "pnp/visualizer.h"

#include <memory>
#include <string>
#include <vector>

#include "camera/camera.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/log.h"
#include "common/vector_3t.h"
#include "epipolar/camera_selection.h"

#include "common/rerun.h"
#include "visualizer/visualizer.hpp"

namespace cuvslam::pnp {
void clearViewport(const std::string& viewport_name) {
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  visualizer.clearViewport(viewport_name);
}

void logLandmarks(const std::vector<std::reference_wrapper<const Vector3T>>& landmarks,
                  const Isometry3T& camera_from_world, const camera::ICameraModel* camera_model,
                  const std::string& viewport_name, const Color& color) {
  if (landmarks.empty()) {
    return;
  }
  if (camera_model == nullptr) {
    TraceWarning("RerunVisualizer: Camera intrinsics unavailable for viewport %s", viewport_name.c_str());
    return;
  }

  // Convert landmarks to 2D UV coordinates
  thread_local std::vector<rerun::Position2D> points;

  points.clear();
  points.reserve(landmarks.size());

  for (const auto& landmark_ref : landmarks) {
    const Vector3T& landmark_world = landmark_ref.get();

    // Transform landmark from world frame to camera frame
    Vector3T landmark_camera = camera_from_world * landmark_world;

    if (!epipolar::IsDepthInFront(landmark_camera.z())) {
      continue;
    }

    Vector2T normalized_coords(landmark_camera.x() / landmark_camera.z(), landmark_camera.y() / landmark_camera.z());

    // Convert normalized coordinates to pixel UV coordinates using camera intrinsics
    Vector2T uv_coords;
    if (camera_model->denormalizePoint(normalized_coords, uv_coords)) {
      // Add the 2D UV position
      points.emplace_back(uv_coords.x(), uv_coords.y());
    } else {
      TraceWarning("RerunVisualizer: Failed to project landmark to UV coordinates");
    }
  }

  // Log the 2D landmark projections to Rerun
  if (!points.empty()) {
    auto& visualizer = visualizer::RerunVisualizer::getInstance();
    visualizer.getRecordingStream().log(viewport_name,
                                        rerun::Points2D(points).with_colors(color).with_radii(3.0f).with_draw_order(
                                            10.0f));  // Slightly larger radius for 2D points
  } else {
    TraceWarning("RerunVisualizer: No landmarks to log for viewport %s", viewport_name.c_str());
  }
}

void logObservations(const std::vector<std::reference_wrapper<const camera::Observation>>& observations,
                     const camera::Rig& rig, const std::string& viewport_name, const Color& color) {
  // Log observations for camera 0 in the same view as the image
  thread_local std::vector<rerun::Position2D> points;

  points.clear();
  points.reserve(observations.size());

  for (const auto& obs : observations) {
    // Convert from normalized xy coordinates to pixel uv coordinates
    Vector2T uv;
    if (obs.get().cam_id < rig.num_cameras && rig.intrinsics[obs.get().cam_id]) {
      const camera::ICameraModel& camera = *rig.intrinsics[obs.get().cam_id];
      if (camera.denormalizePoint(obs.get().xy, uv)) {
        points.emplace_back(uv.x(), uv.y());
      } else {
        TraceError("RerunVisualizer: Failed to denormalize point for observation %d", obs.get().id);
      }
    } else {
      TraceWarning("RerunVisualizer: Camera intrinsics not available for observation %d", obs.get().id);
      continue;
    }
  }

  if (!points.empty()) {
    auto& visualizer = visualizer::RerunVisualizer::getInstance();
    visualizer.getRecordingStream().log(viewport_name,
                                        rerun::Points2D(points).with_colors(color).with_radii(2.0f).with_draw_order(
                                            100.0f));  // High draw order to render on top
  } else {
    TraceWarning("RerunVisualizer: No observations to log for viewport %s", viewport_name.c_str());
  }
}
}  // namespace cuvslam::pnp
