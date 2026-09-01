
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
#include <exception>

#include "common/log_types.h"
#include "common/stopwatch.h"
#include "math/twist.h"
#include "profiler/profiler.h"

#include "slam/map/pose_graph/pose_graph.h"

namespace cuvslam::slam {
using namespace cuvslam::math;

bool PoseGraph::OptimizeSubgraph(const std::vector<KeyFrameId>& keyframes_to_optimize,
                                 const std::vector<EdgeId>& edges_to_optimize,
                                 const std::vector<KeyFrameId>& constraint_keyframes,
                                 const PoseGraphHypothesis& pose_graph_hypothesis_src, bool planar_constraint,
                                 PoseGraphHypothesis& pose_graph_hypothesis_dst, Isometry3T& vo_to_head) const {
  TRACE_EVENT te_pgo = profiler_domain_.trace_event("PGO", profiler_color_);
  vo_to_head = Isometry3T::Identity();

  pose_graph_hypothesis_src.CopyTo(pose_graph_hypothesis_dst);
  Stopwatch sw_full;
  StopwatchScope ssw_full(sw_full);

  if (edges_to_optimize.empty()) {
    return false;
  }
  if (keyframes_to_optimize.empty()) {
    return false;
  }

  // fill remap & initial
  TRACE_EVENT te_fill = profiler_domain_.trace_event("Fill graph", profiler_color_);

  inputs_.clear();

  inputs_.poses.reserve(keyframes_to_optimize.size());
  std::unordered_map<int, KeyFrameId> pose_to_kf;
  std::unordered_map<KeyFrameId, int> kf_to_pose;

  for (KeyFrameId keyframe_id : keyframes_to_optimize) {
    const auto keyframe_pose = pose_graph_hypothesis_src.GetKeyframePose(keyframe_id);
    if (!keyframe_pose) {
      return false;
    }
    // copy existing keyframes
    pose_graph_hypothesis_dst.SetKeyframePose(keyframe_id, *keyframe_pose);

    // don't add standalone nodes
    int edges_count = 0;
    QueryKeyframeEdges(keyframe_id, [&](KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&) { edges_count++; });
    if (edges_count == 0) {
      continue;
    }

    auto m = *keyframe_pose;

    pose_to_kf[inputs_.poses.size()] = keyframe_id;
    kf_to_pose[keyframe_id] = inputs_.poses.size();
    inputs_.poses.push_back(m);
  }

  if (planar_constraint) {
    inputs_.use_planar_constraint = true;
    inputs_.plane_normal = {0, 1.f, 0, 0};  // y-axis is the normal for the plane
    inputs_.planar_weight = 1e3;
  }

  // constraint first node
  for (KeyFrameId constrainedKey : constraint_keyframes) {
    // check if constrainedKey exists
    auto it_initial = kf_to_pose.find(constrainedKey);
    if (it_initial == kf_to_pose.end()) {
      SlamStderr("Initial pose for constrained key %lu not found.\n", constrainedKey);
    } else {
      inputs_.constrained_pose_ids.insert(it_initial->second);
    }
  }

  inputs_.robustifier = 0.5f;

  inputs_.deltas.reserve(edges_to_optimize.size());
  for (EdgeId edge_id : edges_to_optimize) {
    auto& edge = edges_.at(edge_id);
    auto& m = edge.from_to;
    auto& cov = edge.from_to_covariance;

    if (kf_to_pose.find(edge.from_keyframe) == kf_to_pose.end()) {
      SlamStderr("Initial pose for keyframe %zd not found.\n", edge.from_keyframe);
      continue;
    }
    if (kf_to_pose.find(edge.to_keyframe) == kf_to_pose.end()) {
      SlamStderr("Initial pose for keyframe %zd not found.\n", edge.to_keyframe);
      continue;
    }

    const int p1id = kf_to_pose[edge.from_keyframe];
    const int p2id = kf_to_pose[edge.to_keyframe];

    inputs_.deltas.push_back({p1id, p2id, m, cov.ldlt().solve(Matrix6T::Identity())});
  }
  te_fill.Pop();

  try {
    TRACE_EVENT te_opt = profiler_domain_.trace_event("optimizer.optimize()", profiler_color_);
    if (!pgo.run(inputs_, 10)) {
      TraceError("PoseGraph optimization failed.");
      return false;
    }
    te_opt.Pop();
  } catch (const std::exception& e) {
    TraceError("PoseGraph optimization crashed: %s\n", e.what());
    return false;
  } catch (...) {
    TraceError("PoseGraph optimization crashed (unknown exception).");
    return false;
  }

  TRACE_EVENT te_store = profiler_domain_.trace_event("store result", profiler_color_);
  KeyFrameId head_keyframe;
  GetHeadKeyframe(head_keyframe);

  for (size_t i = 0; i < inputs_.poses.size(); i++) {
    const Isometry3T& m = inputs_.poses[i];
    const auto keyframe_id = pose_to_kf[i];

    const auto keyframe_pose = pose_graph_hypothesis_src.GetKeyframePose(keyframe_id);
    if (!keyframe_pose) {
      return false;
    }

    if (head_keyframe == keyframe_id) {
      // VO correction: new_keyframe_pose = keyframe_pose * vo_to_head

      /*/
      Isometry3T vo_to_head_draft = keyframe_pose->inverse() * m;
      Matrix3T mat_rotation, mat_scaling;
      vo_to_head_draft.computeRotationScaling(&mat_rotation, &mat_scaling);

      vo_to_head.translate(vo_to_head_draft.translation());
      vo_to_head.rotate(mat_rotation);
      //*/

      vo_to_head = keyframe_pose->inverse() * m;
      RemoveScaleFromTransform(vo_to_head);
    }
    // update pose in hypothesis
    if (&pose_graph_hypothesis_dst != &pose_graph_hypothesis_src) {
      pose_graph_hypothesis_dst.SetKeyframePose(keyframe_id, m);
    }
  }
  te_store.Pop();
  ssw_full.Stop();

  return true;
}

// select which nodes/edges to optimize with what constraint and run optimization
bool PoseGraph::Optimize(const PoseGraphHypothesis& pose_graph_hypothesis_src,
                         PoseGraphHypothesis& pose_graph_hypothesis_dst, bool planar_constraint,
                         Isometry3T& vo_to_head) const {
  keyframes_to_optimize_.clear();
  edges_to_optimize_.clear();
  constrained_keyframes_.clear();
  bfs_visited_.clear();
  bfs_merge_buffer_.clear();

  KeyFrameId start_keyframe;
  if (!GetHeadKeyframe(start_keyframe)) {
    return false;
  }

  // Level-based BFS — O(K*d*log K) worst case, no hash collisions.
  // K = max_keyframes_to_optimize_, d = avg node degree.
  // keyframes_to_optimize_: sorted visited set, grown each level via merge.
  // edges_to_optimize_: nodes added in the previous level, to be expanded next (sorted).
  // bfs_visited_: raw neighbor scratch buffer for the current level.
  // bfs_merge_buffer_: merge output buffer, swapped in and cleared each level.
  if (max_keyframes_to_optimize_ == 0) {
    return false;
  }
  keyframes_to_optimize_.reserve(max_keyframes_to_optimize_);
  edges_to_optimize_.reserve(max_keyframes_to_optimize_);
  keyframes_to_optimize_.push_back(start_keyframe);
  edges_to_optimize_.push_back(start_keyframe);

  while (!edges_to_optimize_.empty() && keyframes_to_optimize_.size() < max_keyframes_to_optimize_) {
    // Collect all neighbors of nodes added in the previous level into bfs_visited_
    bfs_visited_.clear();
    for (const KeyFrameId frontier_node : edges_to_optimize_) {
      auto collect_neighbor = [&](KeyFrameId from, KeyFrameId to, const Isometry3T&, const Matrix6T&) {
        bfs_visited_.push_back((from == frontier_node) ? to : from);
      };
      QueryKeyframeEdges(frontier_node, collect_neighbor);
    }

    // Sort + dedup raw neighbors — O(L*d*log(L*d)), L = nodes expanded this level, d = avg degree
    std::sort(bfs_visited_.begin(), bfs_visited_.end());
    bfs_visited_.erase(std::unique(bfs_visited_.begin(), bfs_visited_.end()), bfs_visited_.end());

    // New level = unvisited neighbors (set_difference of two sorted ranges) — O(K)
    edges_to_optimize_.clear();
    std::set_difference(bfs_visited_.begin(), bfs_visited_.end(), keyframes_to_optimize_.begin(),
                        keyframes_to_optimize_.end(), std::back_inserter(edges_to_optimize_));

    // Trim to remaining capacity (saturating to avoid size_t underflow)
    const size_t remaining = max_keyframes_to_optimize_ > keyframes_to_optimize_.size()
                                 ? max_keyframes_to_optimize_ - keyframes_to_optimize_.size()
                                 : 0;
    if (edges_to_optimize_.size() > remaining) {
      edges_to_optimize_.resize(remaining);
    }
    if (edges_to_optimize_.empty()) {
      break;
    }

    // Merge sorted visited set + sorted new level into visited set — O(K)
    std::merge(keyframes_to_optimize_.begin(), keyframes_to_optimize_.end(), edges_to_optimize_.begin(),
               edges_to_optimize_.end(), std::back_inserter(bfs_merge_buffer_));
    std::swap(keyframes_to_optimize_, bfs_merge_buffer_);
    bfs_merge_buffer_.clear();
  }

  // keyframes_to_optimize_[0..bfs_count) is the sorted BFS subgraph
  const size_t bfs_count = keyframes_to_optimize_.size();
  edges_to_optimize_.clear();

  auto in_bfs = [&](KeyFrameId kf) {
    return std::binary_search(keyframes_to_optimize_.begin(), keyframes_to_optimize_.begin() + bfs_count, kf);
  };

  // Collect subgraph edges; append boundary nodes to constrained_keyframes_ (dedup deferred)
  for (size_t i = 0; i < bfs_count; i++) {
    const KeyFrameId current = keyframes_to_optimize_[i];
    auto collect_edge = [&](KeyFrameId from, KeyFrameId to, const Isometry3T&, const Matrix6T&) {
      const bool from_in = in_bfs(from);
      const bool to_in = in_bfs(to);
      if (!from_in && !to_in) {
        return;
      }
      if (from_in && to_in && from != current) {
        return;  // canonical direction: skip internal edges from the to-side
      }
      edges_to_optimize_.push_back(edges_from_to_.at({from, to}));
      if (from_in && to_in) {
        return;
      }
      constrained_keyframes_.push_back(from_in ? to : from);  // boundary node, dedup below
    };
    QueryKeyframeEdges(current, collect_edge);
  }

  // Dedup constrained_keyframes_ — O(B*log B), B = number of boundary nodes
  std::sort(constrained_keyframes_.begin(), constrained_keyframes_.end());
  constrained_keyframes_.erase(std::unique(constrained_keyframes_.begin(), constrained_keyframes_.end()),
                               constrained_keyframes_.end());

  // When the whole graph fits in the subgraph there are no boundary nodes; pin the oldest
  // keyframe (min ID, index 0 in sorted keyframes_to_optimize_) as the fixed reference.
  if (constrained_keyframes_.empty() && bfs_count > 0) {
    constrained_keyframes_.push_back(keyframes_to_optimize_[0]);
  }

  // Append constraint nodes to keyframes_to_optimize_ for OptimizeSubgraph.
  // Boundary nodes are outside the BFS subgraph and must be appended.
  // The fallback constraint (oldest BFS node) is already in the subgraph — skip it.
  for (const KeyFrameId kf : constrained_keyframes_) {
    if (!std::binary_search(keyframes_to_optimize_.begin(), keyframes_to_optimize_.begin() + bfs_count, kf)) {
      keyframes_to_optimize_.push_back(kf);
    }
  }

  return OptimizeSubgraph(keyframes_to_optimize_, edges_to_optimize_, constrained_keyframes_, pose_graph_hypothesis_src,
                          planar_constraint, pose_graph_hypothesis_dst, vo_to_head);
}

}  // namespace cuvslam::slam
