
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

#include <cstdint>
#include <functional>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "params/params.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

namespace cuvslam::pnp {

struct PNPSettings {
  PNPSettings() = default;

  static PNPSettings LCSettings();
  static PNPSettings SLAMRansacSettings();
  static PNPSettings InertialSettings();

  bool recalculate_cov = true;
  float lambda = 1e-3;

  float huber = 2e-2;
  int32_t max_iteration = 13;

  bool filter_new_observations = true;
  int32_t max_obs_per_camera = 270;

  float point_z_thresh = 0.01f;

  bool verbose = false;

  int32_t min_observations = 13;

  float cost_thresh = 0.6;
};

class PNPSolver {
public:
  explicit PNPSolver(const camera::Rig& rig);

  // IN:  rig_from_world  - used as start guess
  // OUT: rig_from_world  - if success updated pose otherwise it has unpredicted behavior
  //      static_info_exp - information matrix for the static pose in exponential mapping form in
  //                        the world coordinate system
  bool solve(Isometry3T& rig_from_world, Matrix6T& static_info_exp,
             const std::vector<camera::Observation>& observations,
             const std::unordered_map<TrackId, Vector3T>& landmarks, const PNPSettings& settings) const;

private:
  float evaluate_cost(const Isometry3T& rig_from_world, const PNPSettings& settings) const;

  void build_hessian(const Isometry3T& rig_from_world, Matrix6T& H, Vector6T& rhs, const PNPSettings& settings) const;

  void build_camera_from_world(const Isometry3T& rig_from_world) const;

  using ObservationRef = std::reference_wrapper<const camera::Observation>;

  camera::Rig rig_;

  mutable std::vector<std::reference_wrapper<const Vector3T>> landmark_for_observation_;
  mutable std::vector<std::reference_wrapper<const camera::Observation>> observations_;
  mutable std::vector<std::vector<ObservationRef>> obs_per_camera_;
  mutable std::vector<Isometry3T> cam_from_w_;
  profiler::PnPProfiler::DomainHelper profiler_domain_ = profiler::PnPProfiler::DomainHelper("PNPSolver");
};

}  // namespace cuvslam::pnp

// Tunable fields of PNPSettings. Registered once per solver instance, so the same list serves both
// `vo_pnp.*` and the inertial-mode stereo fallback `inertial_stereo_pnp.*`. `verbose` stays out: it
// is a debugging aid, not a tuning knob.
CUVSLAM_PARAMS_BEGIN(cuvslam::pnp::PNPSettings)
CUVSLAM_PARAM_BOUNDED(lambda, "Levenberg-Marquardt damping factor", NonNegative())
CUVSLAM_PARAM_BOUNDED(huber, "Huber robustifier scale for reprojection residuals", NonNegative())
CUVSLAM_PARAM_BOUNDED(max_iteration, "Maximum LM solver iterations", NonNegative())
CUVSLAM_PARAM(recalculate_cov, "Recompute the covariance matrix after a successful solve")
CUVSLAM_PARAM(filter_new_observations, "Keep the oldest max_obs_per_camera tracks instead of all of them")
CUVSLAM_PARAM_BOUNDED(max_obs_per_camera, "Maximum observations fed to the solver per camera", NonNegative())
CUVSLAM_PARAM_BOUNDED(point_z_thresh, "Minimum landmark depth in the camera frame, meters", NonNegative())
CUVSLAM_PARAM_BOUNDED(min_observations, "Observations required to attempt a solve", NonNegative())
CUVSLAM_PARAM_BOUNDED(cost_thresh,
                      "Absolute convergence threshold; a net improvement over the initial cost also passes",
                      NonNegative())
CUVSLAM_PARAMS_END()
