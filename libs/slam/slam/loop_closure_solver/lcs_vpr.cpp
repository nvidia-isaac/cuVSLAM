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

#include "slam/slam/loop_closure_solver/lcs_vpr.h"

#include <utility>

#include "slam/slam/loop_closure_solver/lcs_two_steps_easy.h"

namespace cuvslam::slam {

namespace {

constexpr float kRadiansPerDegree = 3.14159265358979f / 180.f;

// True when `b` is no more than `max_m` meters and `max_deg` degrees away from `a`.
bool IsWithin(const Isometry3T& a, const Isometry3T& b, float max_m, float max_deg) {
  const Isometry3T difference = a.inverse() * b;
  return difference.translation().norm() <= max_m &&
         Eigen::AngleAxis<float>(difference.linear()).angle() <= max_deg * kRadiansPerDegree;
}

}  // namespace

ILoopClosureSolverPtr CreateLoopClosureSolverVpr(const camera::Rig& rig, RansacType ransac_type, bool randomized) {
  return new LoopClosureSolverVpr(rig, ransac_type, randomized);
}

LoopClosureSolverVpr::LoopClosureSolverVpr(const camera::Rig& rig, RansacType ransac_type, bool randomized)
    : LoopClosureSolverVpr(std::make_unique<LoopClosureSolverTwoStepsEasy>(rig, ransac_type, randomized)) {}

LoopClosureSolverVpr::LoopClosureSolverVpr(std::unique_ptr<ILoopClosureSolver> verifier)
    : verifier_(std::move(verifier)) {}

LoopClosureSolverVpr::~LoopClosureSolverVpr() = default;

bool LoopClosureSolverVpr::Solve(const LoopClosureTask& task, const LSIGrid& landmarks_spatial_index,
                                 const IFeatureDescriptorOps* feature_descriptor_ops, Isometry3T& pose,
                                 Matrix6T& pose_covariance, std::vector<LandmarkInSolver>* landmarks,
                                 DiscardLandmarkCB* discard_landmark_cb,
                                 KeyframeInSightCB* keyframe_in_sight_cb) const {
  std::vector<std::pair<LandmarkId, LandmarkProbe>> discarded;
  std::vector<KeyFrameId> in_sight;
  DiscardLandmarkCB collect_discarded = [&discarded](LandmarkId id, LandmarkProbe probe) {
    discarded.emplace_back(id, probe);
  };
  KeyframeInSightCB collect_in_sight = [&in_sight](KeyFrameId id) { in_sight.push_back(id); };

  bool estimate_tried = false;
  bool estimate_verified = false;
  Isometry3T pose_from_estimate = Isometry3T::Identity();

  LoopClosureTask candidate_task = task;
  candidate_task.vpr_candidates.clear();
  for (const vpr::VprPlace& candidate : task.vpr_candidates) {
    if (landmarks) {
      landmarks->clear();
    }
    discarded.clear();
    in_sight.clear();
    candidate_task.guess_world_from_rig = candidate.pose;
    if (!verifier_->Solve(candidate_task, landmarks_spatial_index, feature_descriptor_ops, pose, pose_covariance,
                          landmarks, discard_landmark_cb ? &collect_discarded : nullptr,
                          keyframe_in_sight_cb ? &collect_in_sight : nullptr)) {
      continue;
    }
    if (!IsWithin(task.guess_world_from_rig, pose, kMaxCorrectionM, kMaxCorrectionDeg)) {
      continue;
    }
    if (!estimate_tried) {
      estimate_tried = true;
      LoopClosureTask estimate_task = candidate_task;
      estimate_task.guess_world_from_rig = task.guess_world_from_rig;
      std::vector<LandmarkInSolver> estimate_landmarks;
      Matrix6T estimate_covariance;
      estimate_verified =
          verifier_->Solve(estimate_task, landmarks_spatial_index, feature_descriptor_ops, pose_from_estimate,
                           estimate_covariance, &estimate_landmarks, nullptr, nullptr);
    }
    if (estimate_verified && !IsWithin(pose_from_estimate, pose, kMaxDisagreementM, kMaxDisagreementDeg)) {
      continue;
    }
    for (const auto& [id, probe] : discarded) {
      (*discard_landmark_cb)(id, probe);
    }
    for (const KeyFrameId id : in_sight) {
      (*keyframe_in_sight_cb)(id);
    }
    return true;
  }
  if (landmarks) {
    landmarks->clear();
  }
  return false;
}

}  // namespace cuvslam::slam
