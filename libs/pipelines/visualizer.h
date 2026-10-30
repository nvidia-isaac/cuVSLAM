
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

#pragma once

#ifdef USE_RERUN
#include <unordered_map>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/log.h"
#include "common/rerun.h"
#include "common/track_id.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "pipelines/track.h"
#ifdef USE_CUNLS
#include "map/plane_map.h"
#endif

#include "visualizer/visualizer.hpp"

namespace cuvslam::pipelines {
// Camera 0 entities: the pose lives on kCamera0Entity; image and depth are pinhole children of it.
inline constexpr const char* kCamera0Entity = "world/camera_0";
inline constexpr const char* kCamera0ImageEntity = "world/camera_0/images";
inline constexpr const char* kCamera0DepthEntity = "world/camera_0/depth";

void logObservations(const std::vector<camera::Observation>& observations, const camera::Rig& rig,
                     const std::string& viewport_name, const Color& color);

void logLandmarks(const std::unordered_map<TrackId, Vector3T>& landmarks, const Isometry3T& camera_from_world,
                  const camera::ICameraModel& camera_model, const std::string& viewport_name, const Color& color);

void logLandmarks(const std::vector<pipelines::Landmark>& landmarks, const Isometry3T& camera_from_world,
                  const camera::ICameraModel& camera_model, const std::string& viewport_name, const Color& color);

/**
 * Log landmarks as 3D points in world frame
 * @param landmarks Map of track IDs to 3D positions in world frame
 * @param viewport_name Rerun viewport name for the 3D landmarks
 * @param color Color for the landmark points
 * @param point_radius Radius of the landmark points
 */
void logLandmarks3D(const std::unordered_map<TrackId, Vector3T>& landmarks, const std::string& viewport_name,
                    const Color& color, float point_radius = 0.01f);

/**
 * Log landmarks as 3D points in world frame (vector version)
 * @param landmarks Vector of landmarks with 3D positions in world frame
 * @param viewport_name Rerun viewport name for the 3D landmarks
 * @param color Color for the landmark points
 * @param point_radius Radius of the landmark points
 */
void logLandmarks3D(const std::vector<pipelines::Landmark>& landmarks, const std::string& viewport_name,
                    const Color& color, float point_radius = 0.01f);

/**
 * Clear 3D landmarks visualization
 * @param viewport_name Rerun viewport name to clear
 */
void clearLandmarks3D(const std::string& viewport_name);

/**
 * Log camera trajectory in world frame.
 * Limitation: Currently this function can only be called in a single thread (main thread for a pipeline and base
 * launcher).
 * @param rig_from_world The current rig pose (world to rig transform)
 * @param viewport_name Rerun viewport name for the trajectory
 * @param color Color for the trajectory points/line
 * @param trajectory_type Type of trajectory (VO, SBA, or GT)
 * @param show_axes If true, show camera orientation axes
 * @param axis_length Length of the camera axes arrows
 */
void logTrajectory(const Isometry3T& rig_from_world, const std::string& viewport_name, const Color& color,
                   TrajectoryType trajectory_type = TrajectoryType::VO, bool show_axes = true,
                   float axis_length = 0.1f);

/**
 * Clear the accumulated trajectory
 * Limitation: Currently this function can only be called in a single thread (main thread for a pipeline and base
 * launcher).
 * @param viewport_name Rerun viewport name to clear
 * @param trajectory_type Type of trajectory to clear (VO, SBA, or GT)
 */
void clearTrajectory(const std::string& viewport_name, TrajectoryType trajectory_type);

/**
 * Log camera frustum wireframes for all cameras in the rig.
 * Each camera gets a distinct color. The frustum is drawn as 8 line segments:
 * 4 edges from the camera origin to the near-plane corners, and 4 edges
 * forming the near-plane rectangle.
 * @param rig The camera rig with intrinsics and extrinsics
 * @param world_from_rig Transform from rig frame to world frame
 * @param viewport_name Rerun viewport name prefix (e.g. "world/frustums")
 * @param frustum_depth How far (in meters) the frustum extends along the optical axis
 */
void logCameraFrustums(const camera::Rig& rig, const Isometry3T& world_from_rig, const std::string& viewport_name,
                       float frustum_depth = 0.3f);

/**
 * Log a static pinhole model so 2D content under entity_path (image, observations) is also shown in the 3D view.
 * Lens distortion is not modeled, so for distorted cameras the 3D placement of the image is approximate.
 * @param camera_model Camera intrinsics (focal, principal point, resolution)
 * @param entity_path Rerun entity of the camera image (e.g. "world/camera_0/images")
 * @param image_plane_distance Distance in meters at which the image plane is drawn in 3D
 */
void logCameraPinhole(const camera::ICameraModel& camera_model, const std::string& entity_path,
                      float image_plane_distance = 0.3f);

/**
 * Log the camera pose as the transform of entity_path relative to its parent ("world").
 * @param world_from_camera Camera pose in world frame
 * @param entity_path Rerun entity of the camera (e.g. "world/camera_0")
 */
void logCameraPose(const Isometry3T& world_from_camera, const std::string& entity_path);

/**
 * Log a 3D vector as an arrow anchored at a given world-frame origin.
 * Useful for visualizing per-frame quantities like IMU gravity and velocity.
 * @param vec    The 3D vector (will be drawn at its true magnitude unless `scale` is provided).
 * @param origin World-frame origin of the arrow.
 * @param viewport_name Rerun viewport name (e.g. "world/imu/gravity").
 * @param color  Color of the arrow.
 * @param scale  Optional length multiplier; defaults to 1.0 (vector drawn at its true magnitude).
 */
void logVector3D(const Vector3T& vec, const Vector3T& origin, const std::string& viewport_name, const Color& color,
                 float scale = 1.0f);

#ifdef USE_CUNLS
void logPlanes(const std::vector<map::Plane>& planes, const std::string& viewport_name);
void clearPlanes(const std::string& viewport_name);

void logPoints3D(const std::vector<Vector3T>& points, const std::string& viewport_name, const Color& color,
                 float point_radius = 0.01f);
#endif

}  // namespace cuvslam::pipelines
#endif
