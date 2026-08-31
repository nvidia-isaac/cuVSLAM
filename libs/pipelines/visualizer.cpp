

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

#include "pipelines/visualizer.h"

#include <unordered_map>

#include "epipolar/camera_selection.h"

#include <rerun/archetypes/arrows3d.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/mesh3d.hpp>
#include <rerun/archetypes/points3d.hpp>

namespace cuvslam::pipelines {

namespace {
// Storage for accumulated trajectory positions by type
// TODO[Zheng]: change this to a functor s.t. it is thread-safe.
//
// Currently this variable is only used in function logTrajector and clearTrajectory.
// This two functions are only used in the main thread for RGBD pipeline and base launcher.
// So it is ok to define it as a thread_local variable.
//
// Limitation: This variable is not thread-safe. So the related functions should not be called in multiple threads or
// mulitple instances of cuvslam.
thread_local std::unordered_map<TrajectoryType, std::vector<rerun::Position3D>> trajectory_positions_map;

std::vector<rerun::Position3D>& getTrajectoryPositions(TrajectoryType type) { return trajectory_positions_map[type]; }
}  // namespace

void logObservations(const std::vector<camera::Observation>& observations, const camera::Rig& rig,
                     const std::string& viewport_name, const Color& color) {
  // Log observations for camera 0 in the same view as the image
  thread_local std::vector<rerun::Position2D> points;

  points.clear();
  points.reserve(observations.size());

  for (const auto& obs : observations) {
    // Convert from normalized xy coordinates to pixel uv coordinates
    Vector2T uv;
    if (obs.cam_id < rig.num_cameras && rig.intrinsics[obs.cam_id]) {
      const camera::ICameraModel& camera = *rig.intrinsics[obs.cam_id];
      if (camera.denormalizePoint(obs.xy, uv)) {
        points.emplace_back(uv.x(), uv.y());
      } else {
        TraceError("RerunVisualizer: Failed to denormalize point for observation %d", obs.id);
      }
    } else {
      TraceWarning("RerunVisualizer: Camera intrinsics not available for observation %d", obs.id);
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

void logLandmarks(const std::unordered_map<TrackId, Vector3T>& landmarks, const Isometry3T& camera_from_world,
                  const camera::ICameraModel& camera_model, const std::string& viewport_name, const Color& color) {
  if (landmarks.empty()) {
    return;
  }

  // Convert landmarks to 2D UV coordinates
  thread_local std::vector<rerun::Position2D> points;

  points.clear();
  points.reserve(landmarks.size());

  for (const auto& [track_id, landmark_world] : landmarks) {
    // Transform landmark from world frame to camera frame
    Vector3T landmark_camera = camera_from_world * landmark_world;

    // OpenCV camera: +Z forward; skip anything behind the camera or inside the near plane.
    if (!epipolar::IsDepthInFront(landmark_camera.z())) {
      continue;
    }

    Vector2T normalized_coords(landmark_camera.x() / landmark_camera.z(), landmark_camera.y() / landmark_camera.z());

    // Convert normalized coordinates to pixel UV coordinates using camera intrinsics
    Vector2T uv_coords;
    if (camera_model.denormalizePoint(normalized_coords, uv_coords)) {
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

void logLandmarks(const std::vector<pipelines::Landmark>& landmarks, const Isometry3T& camera_from_world,
                  const camera::ICameraModel& camera_model, const std::string& viewport_name, const Color& color) {
  if (landmarks.empty()) {
    return;
  }

  // Convert landmarks to 2D UV coordinates
  thread_local std::vector<rerun::Position2D> points;

  points.clear();
  points.reserve(landmarks.size());

  for (const auto& landmark : landmarks) {
    const Vector3T& landmark_world = landmark.point_w;

    // Transform landmark from world frame to camera frame
    Vector3T landmark_camera = camera_from_world * landmark_world;

    // OpenCV camera: +Z forward; skip anything behind the camera or inside the near plane.
    if (!epipolar::IsDepthInFront(landmark_camera.z())) {
      continue;
    }

    Vector2T normalized_coords(landmark_camera.x() / landmark_camera.z(), landmark_camera.y() / landmark_camera.z());

    // Convert normalized coordinates to pixel UV coordinates using camera intrinsics
    Vector2T uv_coords;
    if (camera_model.denormalizePoint(normalized_coords, uv_coords)) {
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

void logLandmarks3D(const std::unordered_map<TrackId, Vector3T>& landmarks, const std::string& viewport_name,
                    const Color& color, float point_radius) {
  if (landmarks.empty()) {
    return;
  }

  thread_local std::vector<rerun::Position3D> points;

  points.clear();
  points.reserve(landmarks.size());

  for (const auto& [track_id, landmark_world] : landmarks) {
    points.emplace_back(landmark_world.x(), landmark_world.y(), landmark_world.z());
  }

  if (!points.empty()) {
    auto& visualizer = visualizer::RerunVisualizer::getInstance();
    visualizer.getRecordingStream().log(viewport_name,
                                        rerun::Points3D(points).with_colors(color).with_radii(point_radius));
  }
}

void logLandmarks3D(const std::vector<pipelines::Landmark>& landmarks, const std::string& viewport_name,
                    const Color& color, float point_radius) {
  if (landmarks.empty()) {
    return;
  }

  thread_local std::vector<rerun::Position3D> points;

  points.clear();
  points.reserve(landmarks.size());

  for (const auto& landmark : landmarks) {
    const Vector3T& pos = landmark.point_w;
    points.emplace_back(pos.x(), pos.y(), pos.z());
  }

  if (!points.empty()) {
    auto& visualizer = visualizer::RerunVisualizer::getInstance();
    visualizer.getRecordingStream().log(viewport_name,
                                        rerun::Points3D(points).with_colors(color).with_radii(point_radius));
  }
}

void clearLandmarks3D(const std::string& viewport_name) {
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  visualizer.getRecordingStream().log(viewport_name, rerun::Clear());
}

void logTrajectory(const Isometry3T& rig_from_world, const std::string& viewport_name, const Color& color,
                   TrajectoryType trajectory_type, bool show_axes, float axis_length) {
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  auto& recording = visualizer.getRecordingStream();

  // Compute world_from_rig to get camera position in world frame
  Isometry3T world_from_rig = rig_from_world.inverse();

  // Extract camera position in world frame
  Vector3T position = world_from_rig.translation();
  rerun::Position3D current_pos{position.x(), position.y(), position.z()};

  // Add to accumulated trajectory for the specified type
  auto& trajectory_positions = getTrajectoryPositions(trajectory_type);
  trajectory_positions.push_back(current_pos);

  // Log current camera position as a point
  recording.log(viewport_name + "/position",
                rerun::Points3D(current_pos).with_colors(Color(0, 255, 0)).with_radii({0.001f}));

  // Log trajectory as a line strip (if we have at least 2 points)
  if (trajectory_positions.size() >= 2) {
    recording.log(
        viewport_name + "/path",
        rerun::LineStrips3D(rerun::components::LineStrip3D(trajectory_positions)).with_colors(color).with_radii(0.5f));
  }

  // Optionally show camera orientation axes
  if (show_axes) {
    // Extract rotation matrix from world_from_rig
    Matrix3T R = world_from_rig.linear();

    // Camera axes in world frame (scaled by axis_length)
    // X-axis (red), Y-axis (green), Z-axis (blue)
    Vector3T x_axis = R.col(0) * axis_length;
    Vector3T y_axis = R.col(1) * axis_length;
    Vector3T z_axis = R.col(2) * axis_length;

    std::vector<rerun::Position3D> origins = {current_pos, current_pos, current_pos};

    std::vector<rerun::Vector3D> vectors = {{x_axis.x(), x_axis.y(), x_axis.z()},
                                            {y_axis.x(), y_axis.y(), y_axis.z()},
                                            {z_axis.x(), z_axis.y(), z_axis.z()}};

    std::vector<rerun::Color> axis_colors = {
        {255, 0, 0},  // Red for X
        {0, 255, 0},  // Green for Y
        {0, 0, 255}   // Blue for Z
    };

    recording.log(viewport_name + "/axes",
                  rerun::Arrows3D::from_vectors(vectors).with_origins(origins).with_colors(axis_colors));
  }
}

void clearTrajectory(const std::string& viewport_name, TrajectoryType trajectory_type) {
  auto& trajectory_positions = getTrajectoryPositions(trajectory_type);
  trajectory_positions.clear();

  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  auto& recording = visualizer.getRecordingStream();

  recording.log(viewport_name + "/position", rerun::Clear());
  recording.log(viewport_name + "/path", rerun::Clear());
  recording.log(viewport_name + "/axes", rerun::Clear());
}

void logCameraFrustums(const camera::Rig& rig, const Isometry3T& world_from_rig, const std::string& viewport_name,
                       float frustum_depth) {
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  auto& recording = visualizer.getRecordingStream();

  const rerun::Color kFrustumColors[] = {
      {0, 200, 255}, {255, 100, 0}, {0, 255, 100}, {255, 0, 200},
      {200, 200, 0}, {0, 100, 255}, {255, 200, 0}, {100, 255, 200},
  };
  constexpr int kNumColors = sizeof(kFrustumColors) / sizeof(kFrustumColors[0]);

  for (int32_t cam_id = 0; cam_id < rig.num_cameras; ++cam_id) {
    const auto* model = rig.intrinsics[cam_id];
    if (!model) continue;

    const Vector2T res = model->getResolution();
    const Vector2T focal = model->getFocal();
    const Vector2T principal = model->getPrincipal();

    // world_from_cam = world_from_rig * (camera_from_rig)^{-1}
    Isometry3T world_from_cam = world_from_rig * rig.camera_from_rig[cam_id].inverse();
    Vector3T origin_w = world_from_cam.translation();

    // 4 image corners in pixel coords → normalized → 3D ray at frustum_depth
    const float corners_uv[4][2] = {{0.f, 0.f}, {res.x(), 0.f}, {res.x(), res.y()}, {0.f, res.y()}};

    std::vector<rerun::Position3D> corners_w;
    corners_w.reserve(4);
    for (const auto& uv : corners_uv) {
      float nx = (uv[0] - principal.x()) / focal.x();
      float ny = (uv[1] - principal.y()) / focal.y();
      Vector3T pt_cam(nx * frustum_depth, ny * frustum_depth, frustum_depth);
      Vector3T pt_w = world_from_cam * pt_cam;
      corners_w.emplace_back(pt_w.x(), pt_w.y(), pt_w.z());
    }

    rerun::Position3D origin_pos{origin_w.x(), origin_w.y(), origin_w.z()};
    const rerun::Color& color = kFrustumColors[cam_id % kNumColors];
    const std::string base = viewport_name + "/cam_" + std::to_string(cam_id);

    // 4 edges from origin to each corner
    std::vector<std::vector<rerun::Position3D>> edges;
    edges.reserve(8);
    for (int i = 0; i < 4; ++i) {
      edges.push_back({origin_pos, corners_w[i]});
    }
    // 4 edges forming the near-plane rectangle
    for (int i = 0; i < 4; ++i) {
      edges.push_back({corners_w[i], corners_w[(i + 1) % 4]});
    }

    recording.log(base, rerun::LineStrips3D(edges).with_colors(color).with_radii(0.003f));
  }
}

void logVector3D(const Vector3T& vec, const Vector3T& origin, const std::string& viewport_name, const Color& color,
                 float scale) {
  rerun::Position3D rerun_origin{origin.x(), origin.y(), origin.z()};
  rerun::Vector3D rerun_vec{vec.x() * scale, vec.y() * scale, vec.z() * scale};
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  visualizer.getRecordingStream().log(
      viewport_name, rerun::Arrows3D::from_vectors({rerun_vec}).with_origins({rerun_origin}).with_colors(color));
}

#ifdef USE_CUNLS
namespace {
const rerun::Color kPlaneColors[] = {
    {230, 25, 75, 80},  {60, 180, 75, 80},  {255, 225, 25, 80}, {0, 130, 200, 80},  {245, 130, 48, 80},
    {145, 30, 180, 80}, {70, 240, 240, 80}, {240, 50, 230, 80}, {210, 245, 60, 80}, {250, 190, 212, 80},
};
const int kNumPlaneColors = sizeof(kPlaneColors) / sizeof(kPlaneColors[0]);

rerun::Color opaque(rerun::Color c) { return {c.r(), c.g(), c.b(), 255}; }
}  // namespace

void logPlanes(const std::vector<map::Plane>& planes, const std::string& viewport_name) {
  if (planes.empty()) return;

  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  auto& recording = visualizer.getRecordingStream();

  for (size_t i = 0; i < planes.size(); ++i) {
    const auto& plane = planes[i];
    const std::string base = viewport_name + "/plane_" + std::to_string(i);
    const rerun::Color color = kPlaneColors[i % kNumPlaneColors];

    // Convex hull mesh (fan triangulation from centroid)
    if (plane.convex_hull.size() >= 3) {
      std::vector<rerun::Position3D> positions;
      positions.reserve(plane.convex_hull.size() + 1);
      positions.emplace_back(plane.centroid.x(), plane.centroid.y(), plane.centroid.z());
      for (const auto& v : plane.convex_hull) {
        positions.emplace_back(v.x(), v.y(), v.z());
      }

      std::vector<rerun::TriangleIndices> triangles;
      const uint32_t n_hull = static_cast<uint32_t>(plane.convex_hull.size());
      triangles.reserve(n_hull);
      for (uint32_t j = 0; j < n_hull; ++j) {
        triangles.push_back({0, j + 1, ((j + 1) % n_hull) + 1});
      }

      std::vector<rerun::Color> vert_colors(positions.size(), color);
      recording.log(base + "/mesh",
                    rerun::Mesh3D(positions).with_triangle_indices(triangles).with_vertex_colors(vert_colors));

      // Hull outline
      std::vector<rerun::Position3D> outline;
      outline.reserve(plane.convex_hull.size() + 1);
      for (const auto& v : plane.convex_hull) {
        outline.emplace_back(v.x(), v.y(), v.z());
      }
      outline.push_back(outline.front());
      recording.log(base + "/outline",
                    rerun::LineStrips3D(rerun::components::LineStrip3D(outline)).with_colors(opaque(color)));
    }

    // Normal arrow
    {
      const float arrow_len = 0.3f;
      rerun::Position3D origin{plane.centroid.x(), plane.centroid.y(), plane.centroid.z()};
      rerun::Vector3D vec{plane.normal.x() * arrow_len, plane.normal.y() * arrow_len, plane.normal.z() * arrow_len};
      recording.log(base + "/normal",
                    rerun::Arrows3D::from_vectors({vec}).with_origins({origin}).with_colors(opaque(color)));
    }

    // Centroid point
    {
      rerun::Position3D pt{plane.centroid.x(), plane.centroid.y(), plane.centroid.z()};
      recording.log(base + "/centroid", rerun::Points3D(pt).with_colors(opaque(color)).with_radii(0.02f));
    }
  }
}

void clearPlanes(const std::string& viewport_name) {
  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  visualizer.getRecordingStream().log(viewport_name, rerun::Clear(true));
}

void logPoints3D(const std::vector<Vector3T>& points, const std::string& viewport_name, const Color& color,
                 float point_radius) {
  if (points.empty()) {
    return;
  }

  thread_local std::vector<rerun::Position3D> rerun_points;
  rerun_points.clear();
  rerun_points.reserve(points.size());

  for (const auto& p : points) {
    rerun_points.emplace_back(p.x(), p.y(), p.z());
  }

  auto& visualizer = visualizer::RerunVisualizer::getInstance();
  visualizer.getRecordingStream().log(viewport_name,
                                      rerun::Points3D(rerun_points).with_colors(color).with_radii(point_radius));
}

#endif  // USE_CUNLS

}  // namespace cuvslam::pipelines
