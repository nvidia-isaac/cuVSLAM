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
#include <memory>

#include "camera/rig.h"

#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"

namespace cuvslam::slam {

// Verifies each visual place recognition candidate (LoopClosureTask::vpr_candidates, best first) by handing a
// verifier the task re-centered on that candidate's pose, and returns on the first one that verifies. VPR is the
// sole candidate source in this mode: an empty candidate list returns false immediately, with no fallback to a
// search around the guess pose.
//
// Verification around a candidate's pose only shows that the current frame fits the landmarks there, so a candidate
// that verifies must also pass two checks against the current pose estimate (LoopClosureTask::guess_world_from_rig):
// - Accepting it may not move the estimate by more than kMaxCorrectionM or kMaxCorrectionDeg. On a road that repeats
//   itself the frame fits a look-alike place too, and a loop closure to it tears the map apart. The default solver
//   never needs this bound, because it only ever searches around the estimate.
// - When verification started from the estimate succeeds as well, the two results have to agree within
//   kMaxDisagreementM and kMaxDisagreementDeg. Started from a candidate's pose, verification can settle on the
//   candidate's own node a few meters away instead of on the camera; started from the estimate it cannot, so two
//   results that disagree mean the candidate's is wrong. It runs once per Solve(), the first time a candidate
//   verifies. When it fails, the drift is too large for a search around the estimate, which is the loop this mode is
//   for, and the candidate stands.
//
// Only the attempt that is accepted reports landmarks, discarded landmarks and keyframes in sight. A rejected
// candidate is usually a place the camera is not at, so what its attempt failed to find says nothing about the
// landmarks there, and counting it would lower their quality for the next landmark reduction.
class LoopClosureSolverVpr : public ILoopClosureSolver {
public:
  // Largest correction of the current pose estimate a loop closure may make; see the class comment.
  static constexpr float kMaxCorrectionM = 10.f;
  static constexpr float kMaxCorrectionDeg = 10.f;
  // Largest difference between the verifications from a candidate and from the estimate; see the class comment.
  static constexpr float kMaxDisagreementM = 1.f;
  static constexpr float kMaxDisagreementDeg = 2.f;

  // Verifies candidates the way the default solver verifies its own guess, with TwoStepsEasy.
  LoopClosureSolverVpr(const camera::Rig& rig, RansacType ransac_type, bool randomized);
  // Verifies candidates with `verifier`, which must not be null.
  explicit LoopClosureSolverVpr(std::unique_ptr<ILoopClosureSolver> verifier);
  ~LoopClosureSolverVpr() override;

  bool WantsVprCandidates() const override { return true; }

  bool Solve(const LoopClosureTask& task, const LSIGrid& landmarks_spatial_index,
             const IFeatureDescriptorOps* feature_descriptor_ops, Isometry3T& pose, Matrix6T& pose_covariance,
             std::vector<LandmarkInSolver>* landmarks, DiscardLandmarkCB* discard_landmark_cb,
             KeyframeInSightCB* keyframe_in_sight_cb) const override;

private:
  std::unique_ptr<ILoopClosureSolver> verifier_;
};

}  // namespace cuvslam::slam
