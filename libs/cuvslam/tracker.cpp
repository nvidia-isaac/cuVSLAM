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

#include "cuvslam/cuvslam2.h"

#include <memory>
#include <stdexcept>

namespace cuvslam {

namespace {

Odometry::Config TrackerOdometryConfig(const Odometry::Config& odometry_config, const Slam::Config* slam_config) {
  if (slam_config != nullptr && slam_config->gt_align_mode) {
    throw std::invalid_argument{"Tracker does not support gt_align_mode; use standalone Odometry and Slam instances."};
  }
  Odometry::Config odometry = odometry_config;
  if (slam_config != nullptr) {
    odometry.enable_observations_export = true;
    odometry.enable_landmarks_export = true;
  }
  return odometry;
}

}  // namespace

Tracker::Tracker(const Rig& rig, const Odometry::Config& odometry_config, const Slam::Config* slam_config)
    : odometry_{rig, TrackerOdometryConfig(odometry_config, slam_config)} {
  if (slam_config != nullptr) {
    slam_ = std::make_unique<Slam>(rig, odometry_.GetPrimaryCameras(), *slam_config);
  }
}

Tracker::TrackResult Tracker::Track(const ImageSet& images, const ImageSet& masks, const ImageSet& depths) {
  TrackResult result;
  result.odometry = odometry_.Track(images, masks, depths);

  // Odometry state is only meaningful once odometry has produced a pose, so SLAM stays untouched
  // on a lost frame and keeps its previous pose.
  if (slam_ && result.odometry.world_from_rig.has_value()) {
    Odometry::State state;
    odometry_.GetState(state);
    slam_->Track(state);
    result.slam = slam_->GetPose();
  }

  return result;
}

void Tracker::RegisterImuMeasurement(uint32_t sensor_index, const ImuMeasurement& imu) {
  odometry_.RegisterImuMeasurement(sensor_index, imu);
}

bool Tracker::IsSlamEnabled() const { return slam_ != nullptr; }

const Odometry& Tracker::GetOdometry() const { return odometry_; }

Slam& Tracker::GetSlam() {
  if (!slam_) {
    throw std::logic_error{"SLAM is not enabled"};
  }
  return *slam_;
}

const Slam& Tracker::GetSlam() const {
  if (!slam_) {
    throw std::logic_error{"SLAM is not enabled"};
  }
  return *slam_;
}

}  // namespace cuvslam
