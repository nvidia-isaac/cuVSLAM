
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

#include "slam/map/pose_graph/pose_graph.h"

#include "common/log_types.h"
#include "math/twist.h"

namespace {
using namespace cuvslam;

// Blob format compatibility: same size/layout as previous Gravity
struct Gravity {
  [[maybe_unused]] bool unused1 = false;
  Vector3T unused2 = Vector3T::Zero();
};

}  // namespace
namespace cuvslam::slam {

void PoseGraphKeyFrame::AddLandmark(LandmarkId landmark_id) {
  const auto it = std::lower_bound(landmarks.begin(), landmarks.end(), landmark_id);
  if (it != landmarks.end() && *it == landmark_id) {
    return;  // already in list
  }
  landmarks.insert(it, landmark_id);
}
void PoseGraphKeyFrame::RemoveLandmark(LandmarkId landmark_id) {
  const auto it = std::lower_bound(landmarks.begin(), landmarks.end(), landmark_id);
  if (it == landmarks.end() || *it != landmark_id) {
    return;  // not in list
  }
  landmarks.erase(it);
}

void PoseGraphKeyFrame::AddKeyframeTransform(const Isometry3T& me, const Isometry3T& other) {
  const Isometry3T from_to = me.inverse() * other;
  merged_keyframe_transforms.push_back(from_to);
}

float PoseGraphEdgeStat::Weight() const { return static_cast<float>(tracks3d_number); }

PoseGraph::PoseGraph(size_t max_keyframes_to_optimize) : max_keyframes_to_optimize_(max_keyframes_to_optimize) {}

PoseGraph::~PoseGraph() { SlamStdout("Destroyed PoseGraph instance. "); }

bool PoseGraph::PutToDatabase(ISlamDatabase* database) const {
  // copy all to DB
  return database->SetSingleton(SlamDatabaseSingleton::PoseGraph, 0,
                                [&](BlobWriter& blob_writer) { return ToBlob(blob_writer); });
}
bool PoseGraph::GetFromDatabase(const ISlamDatabase* database) {
  Clear();
  return database->GetSingleton(SlamDatabaseSingleton::PoseGraph,
                                [&](const BlobReader& blob_reader) { return FromBlob(blob_reader); });
}

void PoseGraph::Clear() {
  keyframes_.clear();
  edges_.clear();
  edges_from_to_.clear();
  edges_from_.clear();
  edges_to_.clear();
  head_keyframe_id_ = InvalidKeyFrameId;

  next_edge_auto_id_ = 0;
}

const PoseGraphKeyFrame& PoseGraph::GetKeyframe(KeyFrameId id) const { return keyframes_.at(id); }

bool PoseGraph::GetHeadKeyframe(KeyFrameId& head_keyframe_id) const {
  head_keyframe_id = head_keyframe_id_;
  return head_keyframe_id_ != InvalidKeyFrameId;
}

bool PoseGraph::SelectHeadKeyframe(KeyFrameId head_keyframe_id, int64_t timestamp_ns) {
  auto it = keyframes_.find(head_keyframe_id);
  if (it == keyframes_.end()) {
    return false;
  }
  it->second.keyframe_info.timestamp_ns = timestamp_ns;
  head_keyframe_id_ = head_keyframe_id;
  return true;
}

size_t PoseGraph::GetKeyframeCount() const { return keyframes_.size(); }

KeyFrameId PoseGraph::AddKeyframe(const PoseGraphHypothesis& pgh, const Isometry3T* pose_rel,
                                  const Matrix6T* head_pose_covariance, const std::string& frame_information,
                                  const PoseGraphKeyframeInfo& extra_keyframe_info, const PoseGraphEdgeStat* stat) {
  TRACE_EVENT ev = profiler_domain_.trace_event("AddKeyframe", profiler_color_);
  KeyFrameId from_keyframe;
  const bool has_head = GetHeadKeyframe(from_keyframe);
  auto keyframe_id = next_keyframe_auto_id_++;

  if (keyframes_.find(keyframe_id) != keyframes_.end()) {
    SlamStderr("Failed to add keyframe to pose graph with auto_id.\n");
    return InvalidKeyFrameId;
  }

  auto& keyFrame = keyframes_[keyframe_id];
  keyFrame.frame_information = frame_information;
  keyFrame.keyframe_info = extra_keyframe_info;
  head_keyframe_id_ = keyframe_id;

  // add edge
  if (has_head && pose_rel && head_pose_covariance) {
    AddEdge(pgh, from_keyframe, keyframe_id, *pose_rel, *head_pose_covariance, stat);
  }

  // latest_keyframes_. For proper GetSmallestVarianceEdgeId()
  latest_keyframes_.push_back(keyframe_id);
  while (latest_keyframes_.size() > max_latest_keyframes_) {
    latest_keyframes_.pop_front();
  }

  log::Message<LogFrames>(log::kInfo, "AddKeyframe() = %zd", keyframe_id);
  return keyframe_id;
}

void PoseGraph::RemoveKeyframe(KeyFrameId keyframe_id) {
  TRACE_EVENT ev = profiler_domain_.trace_event("RemoveKeyframe", profiler_color_);
  // remove all edges with keyframe_id
  std::vector<std::pair<KeyFrameId, KeyFrameId>> edge_pairs;
  QueryKeyframeEdges(keyframe_id,
                     [&](KeyFrameId from_keyframe, KeyFrameId to_keyframe, const Isometry3T&, const Matrix6T&) {
                       edge_pairs.push_back({from_keyframe, to_keyframe});
                     });

  for (const auto& edge_pair : edge_pairs) {
    const KeyFrameId from_keyframe = edge_pair.first;
    const KeyFrameId to_keyframe = edge_pair.second;
    RemoveEdge(from_keyframe, to_keyframe);
  }

  keyframes_.erase(keyframe_id);
}

// add landmark relation
bool PoseGraph::AddLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id) {
  const auto it = keyframes_.find(keyframe_id);
  if (it == keyframes_.end()) {
    return false;
  }
  it->second.AddLandmark(landmark_id);
  return true;
}

void PoseGraph::RemoveLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id) {
  const auto it = keyframes_.find(keyframe_id);
  if (it == keyframes_.end()) {
    return;
  }
  auto& keyframe = it->second;
  keyframe.RemoveLandmark(landmark_id);
}

EdgeId PoseGraph::AddEdge(const PoseGraphHypothesis& pgh, KeyFrameId from_keyframe, KeyFrameId to_keyframe,
                          const Isometry3T& from_to, const Matrix6T& from_to_covariance,
                          const PoseGraphEdgeStat* stat) {
  TRACE_EVENT ev = profiler_domain_.trace_event("AddEdge", profiler_color_);
  (void)pgh;
  if (from_keyframe == to_keyframe) {
    SlamStderr("Failed to add edge to pose graph (%zd->%zd).", from_keyframe, to_keyframe);
    return InvalidEdgeId;
  }
  if (keyframes_.find(from_keyframe) == keyframes_.end()) {
    SlamStderr("Failed to add edge to pose graph (%zd->%zd): %zd does not exist.", from_keyframe, to_keyframe,
               from_keyframe);
    return InvalidEdgeId;
  }
  if (keyframes_.find(to_keyframe) == keyframes_.end()) {
    SlamStderr("Failed to add edge to pose graph (%zd->%zd): %zd does not exist.", from_keyframe, to_keyframe,
               to_keyframe);
    return InvalidEdgeId;
  }

  EdgeId id;
  const auto it_edges_from_to = edges_from_to_.find(std::pair(from_keyframe, to_keyframe));
  if (it_edges_from_to == edges_from_to_.end()) {
    // Add new edge
    id = next_edge_auto_id_++;

    edges_from_to_[std::pair(from_keyframe, to_keyframe)] = id;

    auto& from_list = edges_from_[to_keyframe];
    from_list.push_back(from_keyframe);

    auto& to_list = edges_to_[from_keyframe];
    to_list.push_back(to_keyframe);
  } else {
    // update exists edge
    id = it_edges_from_to->second;
  }

  auto& edge = edges_[id];
  edge.from_keyframe = from_keyframe;
  edge.to_keyframe = to_keyframe;
  edge.from_to = from_to;
  edge.from_to_covariance = from_to_covariance;
  if (stat) {
    edge.statistic = *stat;
  }
  return id;
}

// Remove edge
void PoseGraph::RemoveEdge(KeyFrameId from_keyframe, KeyFrameId to_keyframe) {
  TRACE_EVENT ev = profiler_domain_.trace_event("RemoveEdge", profiler_color_);
  const auto it_edges = edges_from_to_.find(std::pair(from_keyframe, to_keyframe));
  if (it_edges != edges_from_to_.end()) {
    const EdgeId edge_id = it_edges->second;
    edges_.erase(edge_id);

    edges_from_to_.erase(it_edges);
  }

  const auto it_from = edges_from_.find(to_keyframe);
  if (it_from != edges_from_.end()) {
    std::list<KeyFrameId>& list = it_from->second;
    list.remove(from_keyframe);
  }

  const auto it_to = edges_to_.find(from_keyframe);
  if (it_to != edges_to_.end()) {
    std::list<KeyFrameId>& list = it_to->second;
    list.remove(to_keyframe);
  }
}

//
const PoseGraphEdgeStat* PoseGraph::GetEdgeStatistic(KeyFrameId from_keyframe, KeyFrameId to_keyframe) const {
  const std::pair key(from_keyframe, to_keyframe);
  const auto it = edges_from_to_.find(key);
  if (it == edges_from_to_.end()) {
    return nullptr;
  }
  auto& edge = edges_.at(it->second);
  return &edge.statistic;
}

void PoseGraph::RegisterRemoveNodeCB(const RemoveNodeCB& cb) { remove_node_cb_ = cb; }

bool PoseGraph::ToBlob(BlobWriter& blob) const {
  auto keyframe_to_blob = [&](KeyFrameId keyframe_id, const PoseGraphKeyFrame& keyframe) {
    blob.write(keyframe_version_);
    blob.write(keyframe_id);
    blob.write_std(keyframe.landmarks);
    blob.write_std(keyframe.merged_keyframe_transforms);
    blob.write_str(keyframe.frame_information);
    {
      // keyframe_info
      blob.write(keyframe.keyframe_info.current_version);
      blob.write(keyframe.keyframe_info.timestamp_ns);

      // This is not mandatory, but we maintain this to save compatibility with
      // previously created maps and avoid wrong keyframe version in map
      const Gravity dummy_gravity;
      blob.write(dummy_gravity);
    }
  };
  auto edge_to_blob = [&](EdgeId edge_id, const Edge& edge) {
    blob.write(edge_version_);
    blob.write(edge_id);
    blob.write(edge.from_keyframe);
    blob.write(edge.to_keyframe);
    blob.write_eigen(edge.from_to);
    blob.write_eigen(edge.from_to_covariance);

    // edge statistic
    blob.write(edge.statistic.tracks3d_number);
    blob.write(edge.statistic.square_reprojection_errors);
  };

  blob.write_str(format_and_version_);

  blob.write(next_keyframe_auto_id_);
  blob.write(next_edge_auto_id_);
  blob.write(head_keyframe_id_);

  blob.write(keyframes_.size());
  for (auto& it : keyframes_) {
    keyframe_to_blob(it.first, it.second);
  }
  blob.write(edges_.size());
  for (auto& it : edges_) {
    edge_to_blob(it.first, it.second);
  }
  return true;
}
bool PoseGraph::FromBlob(const BlobReader& blob) {
  auto keyframe_from_blob = [&](KeyFrameId& keyframe_id, PoseGraphKeyFrame& keyframe) {
    uint32_t keyframe_version;
    if (!blob.read(keyframe_version)) {
      SlamStderr("Can't read keyframe version in pose graph.\n");
      return false;
    }
    if (keyframe_version != keyframe_version_) {
      SlamStderr("Wrong keyframe version in pose graph: %d!=%d.\n", keyframe_version, keyframe_version_);
      return false;
    }
    bool res = blob.read(keyframe_id) && blob.read_std(keyframe.landmarks) &&
               blob.read_std(keyframe.merged_keyframe_transforms) && blob.read_str(keyframe.frame_information);
    // keyframe_info
    uint32_t ki_version;
    res &= blob.read(ki_version);
    if (!res) {
      SlamStderr("Can't read keyframe_info version in pose graph.\n");
      return false;
    }
    if (ki_version != keyframe.keyframe_info.current_version) {
      SlamStderr("Wrong keyframe_info version in pose graph: %d!=%d.\n", ki_version,
                 keyframe.keyframe_info.current_version);
      return false;
    }
    res &= blob.read(keyframe.keyframe_info.timestamp_ns);

    // This is not mandatory, but we maintain this to save compatibility with
    // previously created maps and avoid wrong keyframe version in map
    Gravity dummy_gravity;
    res &= blob.read(dummy_gravity);
    return res;
  };
  auto edge_from_blob = [&](EdgeId& edge_id, Edge& edge) {
    uint32_t edge_version;

    if (!blob.read(edge_version)) {
      SlamStderr("Can't read edge version in pose graph.\n");
      return false;
    }
    if (edge_version != edge_version_) {
      SlamStderr("Wrong edge version in pose graph: %d!=%d.\n", edge_version, edge_version_);
      return false;
    }

    return blob.read(edge_id) && blob.read(edge.from_keyframe) && blob.read(edge.to_keyframe) &&
           blob.read_eigen(edge.from_to) && blob.read_eigen(edge.from_to_covariance) &&
           // edge statistic
           blob.read(edge.statistic.tracks3d_number) && blob.read(edge.statistic.square_reprojection_errors);
  };

  std::string format_and_version;
  if (!blob.read_str(format_and_version) || format_and_version != format_and_version_) {
    SlamStderr("Failed to read PoseGraph: wrong version %s.\n", format_and_version.c_str());
    SlamStderr("Current is %s.\n", format_and_version_.c_str());
    return false;
  }

  blob.read(next_keyframe_auto_id_);
  blob.read(next_edge_auto_id_);
  blob.read(head_keyframe_id_);

  size_t sz = 0;
  blob.read(sz);
  for (size_t i = 0; i < sz; i++) {
    KeyFrameId keyframe_id;
    PoseGraphKeyFrame keyframe;
    if (!keyframe_from_blob(keyframe_id, keyframe)) {
      return false;
    }
    keyframes_[keyframe_id] = keyframe;
  }

  blob.read(sz);
  for (size_t i = 0; i < sz; i++) {
    EdgeId edge_id;
    Edge edge;
    if (!edge_from_blob(edge_id, edge)) {
      return false;
    }
    edges_[edge_id] = edge;

    edges_to_[edge.from_keyframe].push_back(edge.to_keyframe);
    edges_from_[edge.to_keyframe].push_back(edge.from_keyframe);
    std::pair key(edge.from_keyframe, edge.to_keyframe);
    edges_from_to_[key] = edge_id;
  }
  return true;
}

void PoseGraph::QueryKeyframes(const std::function<void(KeyFrameId)>& lambda) const {
  for (auto& it : keyframes_) {
    lambda(it.first);
  }
}

void PoseGraph::QueryKeyframeEdges(KeyFrameId keyframe_id, const QueryKeyframeEdgesLambda& func) const {
  // edges_from_
  {
    const auto it = edges_from_.find(keyframe_id);
    if (it != edges_from_.end()) {
      auto& list = it->second;
      for (KeyFrameId from_id : list) {
        auto it_edges_from_to = edges_from_to_.find(std::pair(from_id, keyframe_id));
        if (it_edges_from_to == edges_from_to_.end()) {
          continue;
        }
        EdgeId edge_id = it_edges_from_to->second;
        auto it_edge = edges_.find(edge_id);
        if (it_edge == edges_.end()) {
          continue;
        }
        auto& edge = it_edge->second;
        func(edge.from_keyframe, edge.to_keyframe, edge.from_to, edge.from_to_covariance);
      }
    }
  }
  // edges_to_
  {
    const auto it = edges_to_.find(keyframe_id);
    if (it != edges_to_.end()) {
      auto& list = it->second;
      for (KeyFrameId to_id : list) {
        auto it_edges_from_to = edges_from_to_.find(std::pair(keyframe_id, to_id));
        if (it_edges_from_to == edges_from_to_.end()) {
          continue;
        }
        EdgeId edge_id = it_edges_from_to->second;
        auto it_edge = edges_.find(edge_id);
        if (it_edge == edges_.end()) {
          continue;
        }
        auto& edge = it_edge->second;
        func(edge.from_keyframe, edge.to_keyframe, edge.from_to, edge.from_to_covariance);
      }
    }
  }
}

void PoseGraph::QueryEdges(
    const std::function<bool(KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&)>& func) const {
  for (const auto& x : edges_) {
    const Edge& edge = x.second;
    if (!func(edge.from_keyframe, edge.to_keyframe, edge.from_to, edge.from_to_covariance)) {
      break;
    }
  }
}

size_t PoseGraph::QueryKeyframeLandmarks(KeyFrameId keyframe_id, const std::function<bool(const LandmarkId&)>& func) {
  const auto it = keyframes_.find(keyframe_id);
  if (it == keyframes_.end()) {
    return 0;
  }
  const auto& kf = it->second;
  for (auto& it1 : kf.landmarks) {
    if (!func(it1)) {
      return kf.landmarks.size();
    }
  }
  return kf.landmarks.size();
}

void PoseGraph::QueryKeyframePoses(const std::function<void(const KeyFrameId&, const Isometry3T&)>& func) const {
  for (auto& it : keyframes_) {
    const KeyFrameId& keyframe_id = it.first;
    auto& keyframe = it.second;

    func(keyframe_id, Isometry3T::Identity());
    for (const auto& store_pose : keyframe.merged_keyframe_transforms) {
      Isometry3T pose(store_pose);
      func(keyframe_id, pose);
    }
  }
}

}  // namespace cuvslam::slam
