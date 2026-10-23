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

// Verifies VPR candidates best first and accepts the first that passes the checks below; no guess-pose fallback.
// Only the accepted attempt reports landmarks: a rejected candidate says nothing about the landmarks there.
class LoopClosureSolverVpr : public ILoopClosureSolver {
public:
  // A candidate may not move the current pose estimate more than this, which rejects look-alike places.
  static constexpr float kMaxCorrectionM = 10.f;
  static constexpr float kMaxCorrectionDeg = 10.f;
  // If verification from the estimate also succeeds, the candidate's result must agree with it this closely.
  static constexpr float kMaxDisagreementM = 1.f;
  static constexpr float kMaxDisagreementDeg = 2.f;

  // Verifies with TwoStepsEasy, like the LSI solver.
  LoopClosureSolverVpr(const camera::Rig& rig, RansacType ransac_type, bool randomized);
  // `verifier` must not be null.
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
