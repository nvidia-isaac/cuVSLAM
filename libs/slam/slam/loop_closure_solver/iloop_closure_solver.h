
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

#include "camera/rig.h"
#include "common/vector_3t.h"

#include "slam/map/spatial_index/lsi_grid.h"
#include "slam/vpr/vpr_map.h"

namespace cuvslam::slam {

enum class RansacType { kNone, kFundamental, kPnP };
enum class LoopClosureSolverType { kDummy, kSimple, kSimplePoint, kTwoStepsEasy, kVpr };

struct LoopClosureTask {
  std::shared_ptr<PoseGraphHypothesis> pose_graph_hypothesis;
  Isometry3T guess_world_from_rig;
  KeyFrameId pose_graph_head = InvalidKeyFrameId;
  Images current_images;
  // Places that look like current_images, best first, leaving out the recent past of the trajectory, which always
  // looks most like the current frame. Populated by DetectLoopClosure() only when VPR is enabled and the active
  // solver's WantsVprCandidates() is true; ignored by every other solver.
  std::vector<vpr::VprPlace> vpr_candidates;
};

struct LandmarkInSolver {
  LandmarkId id;
  Vector2T uv_norm;
};

class ILoopClosureSolver {
public:
  virtual ~ILoopClosureSolver() = default;

  using DiscardLandmarkCB = std::function<void(LandmarkId, LandmarkProbe)>;
  using KeyframeInSightCB = std::function<void(KeyFrameId)>;

  // True when Solve() reads LoopClosureTask::vpr_candidates, so the caller only pays for building the
  // query image and running place recognition when it is actually needed.
  virtual bool WantsVprCandidates() const { return false; }

  virtual bool Solve(const LoopClosureTask& task, const LSIGrid& landmarks_spatial_index,
                     const IFeatureDescriptorOps* feature_descriptor_ops, Isometry3T& pose, Matrix6T& pose_covariance,
                     std::vector<LandmarkInSolver>* landmarks, DiscardLandmarkCB* discard_landmark_cb,
                     KeyframeInSightCB* keyframe_in_sight_cb) const = 0;
};
using ILoopClosureSolverPtr = ILoopClosureSolver*;

// Dummy LC
ILoopClosureSolverPtr CreateLoopClosureSolverDummy();
// simple LC
ILoopClosureSolverPtr CreateLoopClosureSolverSimple(const camera::Rig& rig, RansacType ransac_type, bool randomized,
                                                    LSIGrid::FetchStrategy fetch_strategy);
// 2-steps LC with filtering by ransac/full and area
ILoopClosureSolverPtr CreateLoopClosureSolverTwoSteps(const camera::Rig& rig, RansacType ransac_type, bool randomized);
// 2-steps LC with easy filtering
ILoopClosureSolverPtr CreateLoopClosureSolverTwoStepsEasy(const camera::Rig& rig, RansacType ransac_type,
                                                          bool randomized);
// VPR-seeded LC: verifies visual place recognition candidates instead of searching around a guess pose
ILoopClosureSolverPtr CreateLoopClosureSolverVpr(const camera::Rig& rig, RansacType ransac_type, bool randomized);

ILoopClosureSolver* CreateLoopClosureSolver(LoopClosureSolverType solver_type, RansacType ransac_type, bool randomized,
                                            const camera::Rig& rig);

}  // namespace cuvslam::slam
