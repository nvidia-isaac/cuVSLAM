
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

#include "Eigen/Geometry"

#include "common/include_gtest.h"
#include "common/vector_3t.h"
#include "slam/map/pose_graph/pose_graph.h"
#include "slam/map/pose_graph/pose_graph_hypothesis.h"

namespace test::slam {

namespace {

using namespace cuvslam;
using namespace cuvslam::slam;

Isometry3T PoseX(float x) {
  Isometry3T p = Isometry3T::Identity();
  p.translation().x() = x;
  return p;
}

Matrix6T UniformCov(float diag) { return Matrix6T::Identity() * diag; }

// Builds a linear chain of n keyframes, each separated by step_x in X.
// Returns keyframe IDs in insertion order.
std::vector<KeyFrameId> BuildChain(PoseGraph& pg, PoseGraphHypothesis& hyp, int n, float step_x) {
  const Matrix6T cov = UniformCov(1.0f);
  const Isometry3T step = PoseX(step_x);
  std::vector<KeyFrameId> ids;
  for (int i = 0; i < n; ++i) {
    const Isometry3T* rel = (i > 0) ? &step : nullptr;
    const Matrix6T* cov_ptr = (i > 0) ? &cov : nullptr;
    const KeyFrameId id = pg.AddKeyframe(hyp, rel, cov_ptr, "", {});
    hyp.SetKeyframePose(id, PoseX(i * step_x));
    ids.push_back(id);
  }
  return ids;
}

}  // namespace

// No keyframes added — nothing to optimize.
TEST(PoseGraph, Optimize_NoKeyframes) {
  PoseGraph pg;
  PoseGraphHypothesis src, dst;
  Isometry3T vo_to_head;
  EXPECT_FALSE(pg.Optimize(src, dst, false, vo_to_head));
}

// Keyframes exist but no edges — optimizer has nothing to work with.
TEST(PoseGraph, Optimize_NoEdges) {
  PoseGraph pg;
  PoseGraphHypothesis src, dst;
  const KeyFrameId kf = pg.AddKeyframe(src, nullptr, nullptr, "", {});
  src.SetKeyframePose(kf, Isometry3T::Identity());
  Isometry3T vo_to_head;
  EXPECT_FALSE(pg.Optimize(src, dst, false, vo_to_head));
}

// Fully consistent chain — no loop closure, optimizer makes no correction.
TEST(PoseGraph, Optimize_ConsistentChain_VoToHeadIsIdentity) {
  PoseGraph pg;
  PoseGraphHypothesis src, dst;
  BuildChain(pg, src, 6, 1.0f);

  Isometry3T vo_to_head;
  ASSERT_TRUE(pg.Optimize(src, dst, false, vo_to_head));
  EXPECT_NEAR(vo_to_head.translation().norm(), 0.0f, 1e-3f);
}

// Chain with a loop closure that closes accumulated drift.
// Optimizer should apply a non-trivial correction to the head node.
TEST(PoseGraph, Optimize_LoopClosure_CorrectionApplied) {
  constexpr int kN = 8;
  constexpr float kStep = 1.0f;

  PoseGraph pg;
  PoseGraphHypothesis src, dst;
  const auto ids = BuildChain(pg, src, kN, kStep);

  // Inconsistent loop closure: 0.2-unit short of a perfect closure.
  // Normalized residual = sqrt(info) * 0.2 = 0.2 < robustifier 0.5 — stays in linear regime.
  const Isometry3T lc_rel = PoseX(-(kN - 1) * kStep + 0.2f);
  pg.AddEdge(src, ids[kN - 1], ids[0], lc_rel, UniformCov(1.0f));

  Isometry3T vo_to_head;
  ASSERT_TRUE(pg.Optimize(src, dst, false, vo_to_head));
  EXPECT_GT(vo_to_head.translation().norm(), 0.05f);
}

// With max_keyframes_to_optimize < N, BFS from the head reaches only a local subgraph.
// The correction applied to the head differs from a full-graph optimization.
TEST(PoseGraph, Optimize_BFSLimitsSubgraph) {
  constexpr int kN = 10;
  constexpr float kStep = 1.0f;

  auto run = [&](size_t max_k) {
    PoseGraph pg(max_k);
    PoseGraphHypothesis src, dst;
    const auto ids = BuildChain(pg, src, kN, kStep);
    const Isometry3T lc_rel = PoseX(-(kN - 1) * kStep + 0.2f);
    pg.AddEdge(src, ids[kN - 1], ids[0], lc_rel, UniformCov(1.0f));
    Isometry3T vo_to_head;
    EXPECT_TRUE(pg.Optimize(src, dst, false, vo_to_head));
    return vo_to_head.translation().x();
  };

  const float correction_full = run(kN);  // all nodes in subgraph
  const float correction_local = run(4);  // only 4 nodes near head

  // Both optimizations see the loop closure (head is always in subgraph)
  EXPECT_GT(std::abs(correction_full), 0.01f);
  EXPECT_GT(std::abs(correction_local), 0.01f);
  // Distributing the error over different subgraph sizes yields different corrections
  EXPECT_GT(std::abs(correction_full - correction_local), 0.01f);
}

TEST(SlamTest, PGO) {
  Eigen::AlignedBox3f observer_box;
  cuvslam::Vector3T v1(1, 2, 3);
  cuvslam::Vector3T v2(-1, -2, -3);
  observer_box.extend(v1);
  observer_box.extend(v2);

  auto scale = 1.2f;

  cuvslam::Vector3T center = observer_box.center();
  // pr
  double radius2 = observer_box.diagonal().squaredNorm() * (scale * scale);
  printf("%f, %f", center.norm(), radius2);
}

}  // namespace test::slam
