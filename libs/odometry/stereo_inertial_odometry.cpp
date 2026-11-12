
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

#include "odometry/stereo_inertial_odometry.h"

#include "common/imu_measurement.h"
#include "math/twist.h"

namespace cuvslam::odom {

StereoInertialOdometry::StereoInertialOdometry(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fig,
                                               const Settings& svo_settings, bool use_gpu, bool debug_imu_mode,
                                               bool disable_fusion_except_gravity)
    : MultiVisualOdometryBase(rig, fig, svo_settings, use_gpu),
      calib_(svo_settings.imu_calibration),
      solver_(map_, rig, svo_settings.sba_settings, svo_settings.imu_calibration, debug_imu_mode,
              disable_fusion_except_gravity) {
  solver_.set_verbose(svo_settings.verbose);
}

bool StereoInertialOdometry::do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction) {
  Isometry3T update;
  if (solver_.predict_pose(update)) {
    sof_prediction = update;
    return true;
  } else {
    const int64_t prev_abs_timestamp = prediction_model_.last_timestamp_ns();
    if (predictor->predict(prev_abs_timestamp, timestamp, update)) {
      sof_prediction = update * sof_prediction;
      return true;
    }
  }
  return false;
}

void StereoInertialOdometry::add_imu_measurement(const imu::ImuMeasurement& m) { solver_.add_imu_measurement(m); }

std::optional<Vector3T> StereoInertialOdometry::get_gravity() const { return solver_.get_gravity(); }

std::optional<pipelines::SolverSfMInertial::ImuState> StereoInertialOdometry::GetImuState() const {
  return solver_.GetImuState();
}

bool StereoInertialOdometry::can_track_visual_blackout() const { return solver_.get_gravity().has_value(); }

}  // namespace cuvslam::odom
