
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

#include <list>
#include <map>
#include <string>
#include <vector>

#include "common/isometry.h"
#include "common/vector_3t.h"
#include "math/pgo.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"
#include "slam/common/slam_common.h"
#include "slam/map/database/slam_database.h"
#include "slam/map/pose_graph/pose_graph_hypothesis.h"
#include "slam/map/pose_graph/slam_posegraph_keyframe_info.h"

namespace cuvslam::slam {

using EdgeId = size_t;
constexpr EdgeId InvalidEdgeId = ~static_cast<EdgeId>(0);

struct PoseGraphKeyFrame {
  PoseGraphKeyframeInfo keyframe_info;  // timestamp and version
  std::vector<LandmarkId> landmarks;
  std::string frame_information;  // use for test dumps
  // to keep spatial volume for landmarks
  std::vector<Isometry3T> merged_keyframe_transforms;

  void AddLandmark(LandmarkId landmark_id);
  void RemoveLandmark(LandmarkId landmark_id);

  // to keep transforms of merged keyframes
  void AddKeyframeTransform(const Isometry3T& me, const Isometry3T& other);
};

struct PoseGraphEdgeStat {
  size_t tracks3d_number = 0;
  float square_reprojection_errors = 0;

  [[nodiscard]] float Weight() const;
};

class PoseGraph {
public:
  explicit PoseGraph(size_t max_keyframes_to_optimize = 300);
  ~PoseGraph();

  PoseGraph(const PoseGraph&) = delete;
  PoseGraph& operator=(const PoseGraph&) = delete;

  bool PutToDatabase(ISlamDatabase* database) const;
  bool GetFromDatabase(const ISlamDatabase* database);

  void Clear();

  const PoseGraphKeyFrame& GetKeyframe(KeyFrameId id) const;

  // Asc the class PoseGraphHypothesis for pose of the keyframe
  // const Isometry3T* PoseGraphHypothesis::GetKeyframePose(KeyFrameId keyframe)

  // head_keyframe will contain last added keyframe or InvalidKeyFrameId
  bool GetHeadKeyframe(KeyFrameId& head_keyframe_id) const;
  [[nodiscard]] bool SelectHeadKeyframe(KeyFrameId head_keyframe_id, int64_t timestamp_ns);

  size_t GetKeyframeCount() const;

  /*
   * Build pose-graph
   */
  // Add keyframe
  KeyFrameId AddKeyframe(const PoseGraphHypothesis& pgh, const Isometry3T* pose_rel,
                         const Matrix6T* head_pose_covariance, const std::string& frame_information,
                         const PoseGraphKeyframeInfo& extra_keyframe_info, const PoseGraphEdgeStat* stat = nullptr);

  void RemoveKeyframe(KeyFrameId keyframe_id);
  // add landmark-keyframe relation
  bool AddLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id);
  // remove landmark-keyframe relation
  void RemoveLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id);

  // Add Edge
  EdgeId AddEdge(const PoseGraphHypothesis& pgh, KeyFrameId from_keyframe, KeyFrameId to_keyframe,
                 const Isometry3T& from_to, const Matrix6T& from_to_covariance,
                 const PoseGraphEdgeStat* stat = nullptr);

  // Optimize pose graph
  // Runs BFS from the head (tail) keyframe, collects up to max_keyframes_to_optimize_ keyframes,
  // and optimizes that subgraph. Boundary nodes (connected to the subgraph but outside it) are
  // pinned as fixed constraints; when the entire graph fits in the window, the oldest keyframe
  // (smallest ID) is pinned as the fallback reference.
  // OUT: vo_to_head - transform for last keyframe
  bool Optimize(const PoseGraphHypothesis& pose_graph_hypothesis_src, PoseGraphHypothesis& pose_graph_hypothesis_dst,
                bool planar_constraint, Isometry3T& vo_to_head) const;

  /*
   * Query pose-graph
   */
  const PoseGraphEdgeStat* GetEdgeStatistic(KeyFrameId from_keyframe, KeyFrameId to_keyframe) const;

  using RemoveNodeCB =
      std::function<void(KeyFrameId keyframe_id, KeyFrameId instead_keyframe_id, const Isometry3T& to_instead)>;
  void RegisterRemoveNodeCB(const RemoveNodeCB& cb);

  // QueryKeyframes([&](KeyFrameId keyframe_id){});
  void QueryKeyframes(const std::function<void(KeyFrameId)>& lambda) const;

  using QueryKeyframeEdgesLambda = std::function<void(KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&)>;
  // Usage:
  // QueryKeyframeEdges(keyframe_id, [&](KeyFrameId from_keyframe, KeyFrameId to_keyframe, const Isometry3T& from_to,
  // const Matrix6T& from_to_covariance){});
  void QueryKeyframeEdges(KeyFrameId keyframe_id, const QueryKeyframeEdgesLambda& func) const;

  // Usage:
  // QueryKeyframeEdges([&](KeyFrameId from_keyframe, KeyFrameId to_keyframe){});
  void QueryEdges(const std::function<bool(KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&)>& func) const;

  // Usage:
  // QueryKeyframeLandmarks(keyframe_id, [&](const LandmarkId& landmark_id){});
  size_t QueryKeyframeLandmarks(KeyFrameId keyframe_id, const std::function<bool(const LandmarkId&)>& func);

  // Usage:
  // QueryKeyframePoses(keyframe_id, [&](const KeyFrameId& keyframe_id, const Isometry3T& pose){});
  void QueryKeyframePoses(const std::function<void(const KeyFrameId&, const Isometry3T&)>& func) const;

  /*
   * Reduce pose-graph
   */
  // Find most reliable pose-graph edge.
  // Return InvalidEdgeId if not found
  EdgeId GetSmallestVarianceEdgeId() const;

  typedef void OnUpdateLandmarkRelation(KeyFrameId old, KeyFrameId neo, const std::vector<LandmarkId>& landmarks,
                                        const Isometry3T& transform_old_to_new);
  // merge keyframes of this edge
  bool ReduceSingleEdge(EdgeId edge_id, const PoseGraphHypothesis& pose_graph_hypothesis,
                        const std::function<OnUpdateLandmarkRelation>& func_update_landmark_relation);

private:
  struct Edge {
    KeyFrameId from_keyframe;
    KeyFrameId to_keyframe;
    Isometry3T from_to;
    Matrix6T from_to_covariance;
    PoseGraphEdgeStat statistic;
  };

  struct OptimizeEdgeInfo {
    KeyFrameId from_id;
    KeyFrameId to_id;
    Isometry3T from_to;
    Matrix6T from_to_covariance;
  };
  const std::string format_and_version_ = "PoseGraph v0.00";
  const uint32_t keyframe_version_ = 2;
  const uint32_t edge_version_ = 0;

  profiler::SLAMProfiler::DomainHelper profiler_domain_ = profiler::SLAMProfiler::DomainHelper("SLAM");
  uint32_t profiler_color_ = 0x0000FF;

  // Keyframes
  KeyFrameId next_keyframe_auto_id_ = 0;
  std::map<KeyFrameId, PoseGraphKeyFrame> keyframes_;
  RemoveNodeCB remove_node_cb_;

  EdgeId next_edge_auto_id_ = 0;
  std::map<EdgeId, Edge> edges_;

  // nodes "before" key
  std::map<KeyFrameId, std::list<KeyFrameId>> edges_from_;
  // nodes "after" key
  std::map<KeyFrameId, std::list<KeyFrameId>> edges_to_;
  // from-to
  std::map<std::pair<KeyFrameId, KeyFrameId>, EdgeId> edges_from_to_;

  // tail
  KeyFrameId head_keyframe_id_ = InvalidKeyFrameId;

  size_t max_keyframes_to_optimize_;

  // latest keyframes to guard in FindHardestEdge()
  size_t max_latest_keyframes_ = 20;
  std::list<KeyFrameId> latest_keyframes_;

  math::PGO pgo;

  // cached memory
  mutable std::vector<KeyFrameId> keyframes_to_optimize_;
  mutable std::vector<EdgeId> edges_to_optimize_;
  mutable std::vector<KeyFrameId> constrained_keyframes_;
  mutable math::PGOInput inputs_;
  // scratch buffers for BFS: neighbor collection and sorted merge output
  mutable std::vector<KeyFrameId> bfs_visited_;
  mutable std::vector<KeyFrameId> bfs_merge_buffer_;

  bool OptimizeSubgraph(const std::vector<KeyFrameId>& keyframes_to_optimize,
                        const std::vector<EdgeId>& edges_to_optimize,
                        const std::vector<KeyFrameId>& constraint_keyframes,
                        const PoseGraphHypothesis& pose_graph_hypothesis_src, bool planar_constraint,
                        PoseGraphHypothesis& pose_graph_hypothesis_dst, Isometry3T& vo_to_head) const;
  bool ToBlob(BlobWriter& blob) const;
  bool FromBlob(const BlobReader& blob);

  // Remove edge
  void RemoveEdge(KeyFrameId from_keyframe, KeyFrameId to_keyframe);
};

}  // namespace cuvslam::slam
