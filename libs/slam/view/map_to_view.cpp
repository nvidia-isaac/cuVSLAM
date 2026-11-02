

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

#include "slam/view/map_to_view.h"

namespace cuvslam::slam {
void PublishAllLandmarksToView(const Map& map, int64_t timestamp_ns, ViewLandmarks& view) {
  const size_t view_capacity = view.landmarks.capacity();
  if (view_capacity == 0) {
    return;
  }
  const auto& landmarks_spatial_index = map.GetLandmarksSpatialIndex();
  const PoseGraphHypothesis& pose_graph_hypothesis = map.GetPoseGraphHypothesis();
  // index multiple of each_div
  // example: log2(~1million / 1024) = 10 => 10 times less we can publish
  const float times = log2(landmarks_spatial_index->LandmarksCount() / static_cast<float>(view_capacity));
  const int pow = static_cast<int>(ceil(times));  // example == 10
  bool sparse_publish = false;
  int each_div = 1;
  if (pow > 0) {
    sparse_publish = true;
    each_div = 1 << pow;  // example each_div = 1024
  }

  int index = 0;
  landmarks_spatial_index->Query([&](LandmarkId id) -> bool {
    if (view.landmarks.size() >= view_capacity) {
      return false;  // stop query loop - no need more landmarks
    }
    if (sparse_publish) {
      if (index % each_div != 0) {
        ++index;
        return true;  // skip it and continue to the next landmark
      }
    }
    ++index;
    // Not kept in memory, since the map in the database can be bigger than memory.
    Vector3T xyz;
    if (landmarks_spatial_index->PeekLandmarkOrStagedCoords(id, pose_graph_hypothesis, xyz)) {
      view.landmarks.push_back({id, 1, ToArray<float, 3>(xyz)});
    }
    return true;  // continue to the next landmark
  });
  view.timestamp_ns = timestamp_ns;
}

void PublishLandmarksToView(const Map& map, int64_t timestamp_ns,
                            const std::unordered_map<LandmarkId, uint32_t>& landmarks, uint32_t max_landmarks,
                            ViewLandmarks& view) {
  const auto& landmarks_spatial_index = map.GetLandmarksSpatialIndex();
  const PoseGraphHypothesis& pose_graph_hypothesis = map.GetPoseGraphHypothesis();

  for (const auto& landmark : landmarks) {
    if (view.landmarks.size() >= view.landmarks.capacity()) {
      break;
    }
    float w = landmark.second / static_cast<float>(max_landmarks);
    w = std::max(w, 0.1f);
    Vector3T xyz = landmarks_spatial_index->GetLandmarkOrStagedCoords(landmark.first, pose_graph_hypothesis);
    view.landmarks.push_back({static_cast<uint64_t>(landmark.first), w, ToArray<float, 3>(xyz)});
  }
  view.timestamp_ns = timestamp_ns;
}

void PublishPoseGraphToView(const Map& map, int64_t timestamp_ns, ViewPoseGraph& view) {
  const PoseGraph& pg = map.GetPoseGraph();
  const PoseGraphHypothesis& pgh = map.GetPoseGraphHypothesis();

  view.nodes.clear();
  view.edges.clear();
  pg.QueryKeyframes([&](KeyFrameId keyframe_id) {
    const Isometry3T* pose_ptr = pgh.GetKeyframePose(keyframe_id);
    if (pose_ptr == nullptr) {
      return true;
    }
    if (view.nodes.size() >= view.nodes.capacity()) {
      return false;
    }
    ViewPoseGraphNode& dst = view.nodes.emplace_back();
    dst.id = keyframe_id;
    dst.node_pose.set(*pose_ptr);
    return true;
  });
  pg.QueryEdges(
      [&](KeyFrameId from_keyframe, KeyFrameId to_keyframe, const Isometry3T& transform, const Matrix6T& covariance) {
        if (view.edges.size() >= view.edges.capacity()) {
          return false;
        }
        ViewPoseGraphEdge& dst = view.edges.emplace_back();
        dst.node_from = from_keyframe;
        dst.node_to = to_keyframe;
        dst.transform.set(transform);
        dst.covariance = covariance;
        return true;
      });
  view.timestamp_ns = timestamp_ns;
}

void PublishLoopClosureToView(const Map& map, const std::vector<LandmarkInSolver>& landmarks, ViewLandmarks& view) {
  const auto spatial_index = map.GetLandmarksSpatialIndex();
  const PoseGraphHypothesis& pose_graph_hypothesis = map.GetPoseGraphHypothesis();

  view.landmarks.clear();
  for (auto& landmark : landmarks) {
    if (view.landmarks.size() >= view.landmarks.capacity()) {
      break;
    }
    const Vector3T xyz = spatial_index->GetLandmarkOrStagedCoords(landmark.id, pose_graph_hypothesis);
    ViewLandmark dst{landmark.id, 1, ToArray<float, 3>(xyz)};
    view.landmarks.push_back(dst);
  }
}

}  // namespace cuvslam::slam
