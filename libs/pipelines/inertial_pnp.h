
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
#include <map>
#include <random>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/imu_calibration.h"
#include "common/isometry.h"
#include "imu/imu_sba_problem.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "pipelines/track.h"

#include "params/params.h"

namespace cuvslam::pipelines {

struct InertialPnPSettings {
  float robustifier_scale = 0.4f;
  int32_t max_iteration = 20;
  int32_t min_observations = 25;
};

}  // namespace cuvslam::pipelines

// Tunable fields of InertialPnPSettings, addressed as `imu_pnp.<name>`.
CUVSLAM_PARAMS_BEGIN(cuvslam::pipelines::InertialPnPSettings)
CUVSLAM_PARAM_BOUNDED(robustifier_scale, "Robustifier scale for inertial PnP reprojection residuals", NonNegative())
CUVSLAM_PARAM_BOUNDED(max_iteration, "Maximum inertial PnP solver iterations", NonNegative())
CUVSLAM_PARAM_BOUNDED(min_observations, "Observations required to attempt inertial PnP", NonNegative())
CUVSLAM_PARAMS_END()

namespace cuvslam::pipelines {

class InertialPnP {
public:
  // OUT: pose, prev_pose - updated poses if success, otherwise it's guaranteed to be unchanged
  bool Solve(const imu::ImuCalibration& calib, const std::unordered_map<TrackId, Track>& tracks3d,
             const std::vector<camera::Observation>& observations, const camera::Rig& rig, const Vector3T& gravity_w,
             const InertialPnPSettings& settings, sba_imu::Pose& prev_pose,  // non-const because of velocities updates
             sba_imu::Pose& curr_pose) const;

private:
  profiler::PnPProfiler::DomainHelper profiler_domain_ = profiler::PnPProfiler::DomainHelper("Inertial PnP");
};

}  // namespace cuvslam::pipelines
