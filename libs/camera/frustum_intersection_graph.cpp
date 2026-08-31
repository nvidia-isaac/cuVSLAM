
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

#include "camera/frustum_intersection_graph.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "common/log.h"
#include "common/vector_3t.h"
#include "epipolar/camera_selection.h"

namespace cuvslam::camera {

MulticameraMode ParseMulticameraMode(const std::string& mode, MulticameraMode default_mode) {
  const std::unordered_map<std::string, MulticameraMode> modes{
      {"performance", MulticameraMode::Performance},
      {"precision", MulticameraMode::Precision},
      {"moderate", MulticameraMode::Moderate},
      {"manual", MulticameraMode::Manual},
  };
  auto it = modes.find(mode);
  if (it != modes.end()) {
    return it->second;
  }
  TraceError("Cannot find '%s' multicam mode, defaulting to %d", mode.c_str(), default_mode);
  return default_mode;
}

namespace {

using CameraGraphNode = FrustumIntersectionGraph::CameraGraphNode;

std::vector<CameraGraphNode> BuildCameraGraph(const camera::Rig& rig) {
  std::vector<CameraGraphNode> camera_config_graph;
  camera_config_graph.resize(rig.num_cameras);
  int total_number_of_points = 1000;
  // UV is back-projected to the minimal and maximal depth d_min, d_max.
  // These values should establish the effective depth range for our datasets and HAWK, Realsense stereo pairs.
  float d_min = 2;
  float d_max = 4;
  float intersected_num_points_ratio_threshold = 0.5;
  int side_number_of_points = static_cast<int>(sqrt(total_number_of_points));
  for (int32_t cam_id_i = 0; cam_id_i < rig.num_cameras; cam_id_i++) {
    for (int32_t cam_id_j = cam_id_i + 1; cam_id_j < rig.num_cameras; cam_id_j++) {
      int intersected_points = 0;
      int step_x = rig.intrinsics[cam_id_i]->getResolution().x() / (side_number_of_points + 2);
      int step_y = rig.intrinsics[cam_id_i]->getResolution().y() / (side_number_of_points + 2);
      for (int32_t point_id_i = 0; point_id_i < side_number_of_points; point_id_i++) {
        for (int32_t point_id_j = 0; point_id_j < side_number_of_points; point_id_j++) {
          // uv point 2D in camera i plane
          Vector2T uv;
          uv[0] = static_cast<float>((point_id_i + 1) * step_x);
          uv[1] = static_cast<float>((point_id_j + 1) * step_y);

          // back-projected 3D point in camera frame
          // xy point 2D
          Vector2T xy;
          if (!rig.intrinsics[cam_id_i]->normalizePoint(uv, xy)) {
            continue;
          };
          // 3d point
          const Vector3T point_d_min_i(xy.x() * d_min, xy.y() * d_min, d_min);
          const Vector3T point_d_max_i(xy.x() * d_max, xy.y() * d_max, d_max);

          // project 3d point to camera j plane
          Isometry3T T_i_j = rig.camera_from_rig[cam_id_i] * rig.camera_from_rig[cam_id_j].inverse();
          Vector3T point_d_min_j = T_i_j * point_d_min_i;
          Vector3T point_d_max_j = T_i_j * point_d_max_i;
          if (!epipolar::IsDepthInFront(point_d_min_j.z()) || !epipolar::IsDepthInFront(point_d_max_j.z())) {
            continue;
          }
          Vector2T xy_d_min_j = point_d_min_j.topRows(2) / point_d_min_j.z();
          Vector2T xy_d_max_j = point_d_max_j.topRows(2) / point_d_max_j.z();
          Vector2T uv_pix_d_min_j;
          Vector2T uv_pix_d_max_j;
          if (!rig.intrinsics[cam_id_j]->denormalizePoint(xy_d_min_j, uv_pix_d_min_j)) {
            continue;
          };
          if (!rig.intrinsics[cam_id_j]->denormalizePoint(xy_d_max_j, uv_pix_d_max_j)) {
            continue;
          };

          // confirm that these points are in camera j frustum
          auto res_cam_j = rig.intrinsics[cam_id_j]->getResolution();
          if (uv_pix_d_min_j.x() > 0 && uv_pix_d_min_j.x() < res_cam_j.x() && uv_pix_d_min_j.y() > 0 &&
              uv_pix_d_min_j.y() < res_cam_j.y() && uv_pix_d_max_j.x() > 0 && uv_pix_d_max_j.x() < res_cam_j.x() &&
              uv_pix_d_max_j.y() > 0 && uv_pix_d_max_j.y() < res_cam_j.y()) {
            intersected_points++;
          }
        }
      }

      float intersected_num_points_ratio =
          static_cast<float>(intersected_points) / static_cast<float>(total_number_of_points);

      if (intersected_num_points_ratio > intersected_num_points_ratio_threshold) {
        {
          camera_config_graph[cam_id_i].stereo_camera_pairs.emplace_back(cam_id_j, intersected_num_points_ratio);
          camera_config_graph[cam_id_i].degree_of_vertex++;
        }

        {
          camera_config_graph[cam_id_j].stereo_camera_pairs.emplace_back(cam_id_i, intersected_num_points_ratio);
          camera_config_graph[cam_id_j].degree_of_vertex++;
        }
      }
    }
  }
  return camera_config_graph;
}

}  // namespace

void FrustumIntersectionGraph::set_precision_mode(const std::vector<CameraGraphNode>& graph) {
  primary_cameras_.clear();
  primary_cameras_.reserve(graph.size());
  secondary_from_primary_.clear();
  for (size_t cam_id = 0; cam_id < graph.size(); cam_id++) {
    primary_cameras_.push_back(cam_id);

    std::vector<CameraId>& sec_cam_ids = secondary_from_primary_[cam_id];

    const auto& connected_cams = graph[cam_id].stereo_camera_pairs;
    for (const auto& [sec_id, fir] : connected_cams) {
      sec_cam_ids.push_back(sec_id);
    }
  }
}

void FrustumIntersectionGraph::set_performance_mode(const std::vector<CameraGraphNode>& graph) {
  set_moderate_mode(graph);

  std::unordered_map<CameraId, std::vector<CameraId>> primary_from_secondary;
  for (const auto& [prim_cam, sec_cams] : secondary_from_primary_) {
    for (const auto& sec_cam : sec_cams) {
      primary_from_secondary[sec_cam].push_back(prim_cam);
    }
  }

  for (const auto& [sec_cam, prim_cams] : primary_from_secondary) {
    CameraId max_primary_cam;
    {
      std::vector<float> firs;

      const auto& connected_cams = graph[sec_cam].stereo_camera_pairs;
      for (CameraId prim_cam : prim_cams) {
        auto it = std::find_if(connected_cams.begin(), connected_cams.end(),
                               [prim_cam](const CameraGraphNode::ConnectedCam& cam) { return cam.id == prim_cam; });

        firs.push_back(it->frustrim_intersection_ratio);
      }

      auto max_it = std::max_element(firs.begin(), firs.end());
      size_t argmax = std::distance(firs.begin(), max_it);

      max_primary_cam = prim_cams[argmax];
    }

    for (CameraId cam_id : prim_cams) {
      if (cam_id == max_primary_cam) {
        continue;
      }
      auto& sec_cams = secondary_from_primary_[cam_id];
      auto it = std::remove(sec_cams.begin(), sec_cams.end(), sec_cam);
      sec_cams.erase(it, sec_cams.end());
    }
  }
}

void FrustumIntersectionGraph::set_moderate_mode(const std::vector<CameraGraphNode>& graph) {
  std::unordered_set<CameraId> visited_nodes;

  std::queue<CameraId> camera_queue;
  {
    std::vector<std::pair<CameraId, CameraGraphNode>> cameras;
    for (size_t cam_id = 0; cam_id < graph.size(); cam_id++) {
      cameras.push_back({cam_id, graph[cam_id]});
    }

    std::sort(cameras.begin(), cameras.end(), [](const auto& pair_lhs, const auto& pair_rhs) {
      size_t deg_lhs = pair_lhs.second.degree_of_vertex;
      size_t deg_rhs = pair_rhs.second.degree_of_vertex;

      CameraId id_lhs = pair_lhs.first;
      CameraId id_rhs = pair_rhs.first;
      return std::tie(deg_lhs, id_lhs) < std::tie(deg_rhs, id_rhs);  // first camera must have the largest degree
    });

    for (const auto& [cam_id, node] : cameras) {
      camera_queue.push(cam_id);
    }
  }

  primary_cameras_.clear();
  secondary_from_primary_.clear();

  while (!camera_queue.empty()) {
    CameraId cam_id = camera_queue.front();
    camera_queue.pop();

    if (visited_nodes.find(cam_id) != visited_nodes.end()) {
      continue;
    }

    primary_cameras_.push_back(cam_id);
    visited_nodes.insert(cam_id);

    auto& sec_cams = secondary_from_primary_[cam_id];
    for (const auto& [sec_cam_id, fir] : graph.at(cam_id).stereo_camera_pairs) {
      sec_cams.push_back(sec_cam_id);
      visited_nodes.insert(sec_cam_id);
    }
  }
}

void FrustumIntersectionGraph::set_manual_mode(const std::vector<CameraGraphNode>& graph,
                                               const MulticamManualSetup& manual_setup) {
  if (manual_setup.size() != graph.size()) {
    throw std::runtime_error("Multicamera manual setup has " + std::to_string(manual_setup.size()) +
                             " cams, must have " + std::to_string(graph.size()));
  }
  primary_cameras_.clear();
  primary_cameras_.reserve(manual_setup.size());
  secondary_from_primary_.clear();

  for (size_t cam_id = 0; cam_id < manual_setup.size(); cam_id++) {
    if (manual_setup[cam_id].empty()) {
      continue;
    }
    const auto& connected_cams = graph[cam_id].stereo_camera_pairs;
    for (auto sec_id : manual_setup[cam_id]) {
      if (sec_id == cam_id || sec_id >= manual_setup.size()) {
        throw std::runtime_error("Wrong secondary camera index " + std::to_string(sec_id) + " at primary camera " +
                                 std::to_string(cam_id));
      }
      auto it = std::find_if(connected_cams.begin(), connected_cams.end(),
                             [sec_id](const CameraGraphNode::ConnectedCam& cam) { return cam.id == sec_id; });
      if (it == connected_cams.end()) {
        TraceWarning("Cameras %d and %d frustum intersection is poor.", cam_id, sec_id);
      }
    }
    primary_cameras_.push_back(cam_id);
    secondary_from_primary_[cam_id] = manual_setup[cam_id];
  }
}

FrustumIntersectionGraph::FrustumIntersectionGraph(const camera::Rig& rig, const FigSettings& settings)
    : FrustumIntersectionGraph(BuildCameraGraph(rig), settings) {}

FrustumIntersectionGraph::FrustumIntersectionGraph(const std::vector<CameraGraphNode>& graph,
                                                   const FigSettings& settings)
    : depth_ids_(settings.depth_ids) {
  TraceErrorIf((settings.mode == MulticameraMode::Manual) == settings.manual_setup.empty(),
               "manual_setup should be provided if mode is manual");

  switch (settings.mode) {
    case MulticameraMode::Precision:
      set_precision_mode(graph);
      break;

    case MulticameraMode::Performance:
      set_performance_mode(graph);
      break;

    case MulticameraMode::Manual:
      set_manual_mode(graph, settings.manual_setup);
      break;

    case MulticameraMode::Moderate:
    default:
      set_moderate_mode(graph);
      break;
  }

  const std::vector<CameraId>& depth_ids = settings.depth_ids;
  if (!settings.allow_stereo_track_for_depth && !depth_ids.empty()) {
    for (auto& [_, sec_cams] : secondary_from_primary_) {
      std::vector<CameraId> cams_to_remove;
      for (CameraId sec_cam : sec_cams) {
        auto it2 = std::find(depth_ids.begin(), depth_ids.end(), sec_cam);
        if (it2 != depth_ids.end()) {
          cams_to_remove.push_back(sec_cam);
        }
      }
      for (CameraId cam_id : cams_to_remove) {
        sec_cams.erase(std::find(sec_cams.begin(), sec_cams.end(), cam_id));
      }
    }
  }
  {
    // remove primary cameras that have no stereo pairs
    std::vector<CameraId> cams_to_remove;
    const std::vector<CameraId>& prim_cams = primary_cameras();
    for (CameraId cam_id : prim_cams) {
      auto it = secondary_from_primary_.find(cam_id);
      if (it == secondary_from_primary_.end()) {
        cams_to_remove.push_back(cam_id);
        continue;
      }

      const std::vector<CameraId>& sec_cams = it->second;
      if (sec_cams.empty()) {
        cams_to_remove.push_back(cam_id);
      }
    }

    for (CameraId cam_id : cams_to_remove) {
      primary_cameras_.erase(std::find(primary_cameras_.begin(), primary_cameras_.end(), cam_id));
    }
  }

  for (CameraId id : depth_ids) {
    auto it = std::find(primary_cameras_.begin(), primary_cameras_.end(), id);
    if (it != primary_cameras_.end()) {
      assert(!secondary_from_primary_.at(id).empty());
      continue;
    }
    primary_cameras_.push_back(id);
    secondary_from_primary_[id] = {};
  }
}

bool FrustumIntersectionGraph::is_valid() const {
  const std::vector<CameraId>& prim_cams = primary_cameras();
  if (prim_cams.empty()) {
    return false;
  }
  // TODO: global refactoring, since section below is not working anymore
  bool valid = true;
  for (CameraId cam_id : prim_cams) {
    auto it = secondary_from_primary_.find(cam_id);
    if (it == secondary_from_primary_.end()) {
      TraceWarning("No container with secondary cameras for camera %d", (int)cam_id);
      valid = false;
      break;
    }
  }

  return valid;
}

const std::vector<CameraId>& FrustumIntersectionGraph::primary_cameras() const { return primary_cameras_; }
const std::vector<CameraId>& FrustumIntersectionGraph::secondary_cameras(CameraId primary_camera) const {
  return secondary_from_primary_.at(primary_camera);
}

}  // namespace cuvslam::camera
