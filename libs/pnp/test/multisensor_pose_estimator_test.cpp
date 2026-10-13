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

#include <random>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

#include "camera/camera.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "math/twist.h"

#include "pnp/multisensor_pose_estimator.h"

namespace test::pnp {

using namespace cuvslam;

namespace {

// Synthetic pinhole camera: 640×480, focal=320, principal=(320,240)
const Vector2T kResolution{640.f, 480.f};
const Vector2T kFocal{320.f, 320.f};
const Vector2T kPrincipal{320.f, 240.f};

camera::PinholeCameraModel MakePinhole() { return camera::PinholeCameraModel(kResolution, kFocal, kPrincipal); }

camera::Rig MakeRig(const camera::ICameraModel& cam) {
  camera::Rig rig;
  rig.num_cameras = 1;
  rig.camera_from_rig[0] = Isometry3T::Identity();
  rig.intrinsics[0] = &cam;
  return rig;
}

// cuVSLAM uses the OpenCV convention (x-right, y-down, z-forward), so visible landmarks
// have z > 0 in the camera frame. The cunls PnP factor enforces this with a z_threshold
// check; landmarks with non-positive z would be rejected (zero residual + zero Jacobian).
std::unordered_map<TrackId, Vector3T> GenerateLandmarks(const Isometry3T& world_from_cam, size_t count,
                                                        std::mt19937& rng) {
  std::unordered_map<TrackId, Vector3T> landmarks;
  std::uniform_real_distribution<float> xy_dist(-1.5f, 1.5f);
  std::uniform_real_distribution<float> z_dist(1.f, 5.f);

  for (size_t i = 0; i < count; i++) {
    Vector3T p_cam(xy_dist(rng), xy_dist(rng), z_dist(rng));
    landmarks[static_cast<TrackId>(i)] = world_from_cam * p_cam;
  }
  return landmarks;
}

std::vector<camera::Observation> ProjectLandmarks(const std::unordered_map<TrackId, Vector3T>& landmarks,
                                                  const Isometry3T& cam_from_world, CameraId cam_id = CameraId{0}) {
  std::vector<camera::Observation> observations;
  for (const auto& [id, lm_world] : landmarks) {
    Vector3T p_cam = cam_from_world * lm_world;
    // Skip points behind / on the camera plane (cuVSLAM uses positive-z-forward).
    if (p_cam.z() < 0.01f) {
      continue;
    }

    float inv_z = 1.f / p_cam.z();
    Vector2T xy(p_cam.x() * inv_z, p_cam.y() * inv_z);
    Matrix2T info = Matrix2T::Identity();
    observations.emplace_back(cam_id, id, xy, info);
  }
  return observations;
}

}  // namespace

// Recover a known pose from synthetic reprojection data (no depth).
TEST(MultisensorPoseEstimatorTest, ReprojectionConvergence) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  std::mt19937 rng(42);

  Isometry3T gt_cam_from_world = Isometry3T::Identity();
  gt_cam_from_world.translation() = Vector3T(0.f, 0.f, 0.f);

  auto landmarks = GenerateLandmarks(gt_cam_from_world.inverse(), 100, rng);
  auto observations = ProjectLandmarks(landmarks, gt_cam_from_world);
  ASSERT_GT(observations.size(), 10u);

  // Perturbed initial guess
  Vector6T perturbation;
  perturbation << 0.02f, -0.01f, 0.015f, 0.05f, -0.03f, 0.04f;
  Isometry3T delta;
  math::Exp(delta, perturbation);
  Isometry3T init_rig_from_world = gt_cam_from_world * delta;

  cuvslam::pnp::MultisensorSolverSettings settings;
  settings.max_iteration = 30;
  cuvslam::pnp::MultisensorPoseEstimator estimator(rig, settings);

  Isometry3T rig_from_world = init_rig_from_world;
  Matrix6T info;
  bool ok = estimator.solve(rig_from_world, info, observations, landmarks);
  ASSERT_TRUE(ok);

  Vector6T error;
  Isometry3T diff = gt_cam_from_world.inverse() * rig_from_world;
  math::Log(error, diff);
  float angular_err = error.head<3>().norm();
  float trans_err = error.tail<3>().norm();

  EXPECT_LT(angular_err, 0.01f) << "Angular error too large: " << angular_err;
  EXPECT_LT(trans_err, 0.02f) << "Translation error too large: " << trans_err;
}

// Verify solver runs without crashing on empty input.
TEST(MultisensorPoseEstimatorTest, EmptyObservations) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  cuvslam::pnp::MultisensorPoseEstimator estimator(rig);

  Isometry3T pose = Isometry3T::Identity();
  Matrix6T info;
  std::unordered_map<TrackId, Vector3T> landmarks;
  std::vector<camera::Observation> obs;

  bool ok = estimator.solve(pose, info, obs, landmarks);
  EXPECT_FALSE(ok);
}

// An inertial prior without preintegration contributes no factor, so with no observations the
// problem is empty. The solver has to report failure instead of handing cuNLS an empty problem.
TEST(MultisensorPoseEstimatorTest, InertialPriorWithoutResidualsFails) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  cuvslam::pnp::MultisensorPoseEstimator estimator(rig);

  Isometry3T pose = Isometry3T::Identity();
  Matrix6T info = Matrix6T::Identity();
  std::unordered_map<TrackId, Vector3T> landmarks;
  std::vector<camera::Observation> obs;
  cuvslam::pnp::InertialPriorInput imu_in;

  bool ok = true;
  EXPECT_NO_THROW(ok = estimator.solve(pose, info, obs, landmarks, {}, {}, {}, imu_in));
  EXPECT_FALSE(ok);
  EXPECT_TRUE(info.isZero());
}

// Multi-camera rig: verify that per-observation camera_from_rig extrinsics
// are correctly accounted for in the PnP factor.
TEST(MultisensorPoseEstimatorTest, MultiCameraReprojectionConvergence) {
  auto cam0 = MakePinhole();
  auto cam1 = MakePinhole();

  camera::Rig rig;
  rig.num_cameras = 2;
  rig.camera_from_rig[0] = Isometry3T::Identity();

  // Camera 1 is translated 0.5m to the right and rotated ~5 deg around Y.
  Vector6T extrinsic_twist;
  extrinsic_twist << 0.f, 0.09f, 0.f, 0.5f, 0.f, 0.f;
  Isometry3T cam1_from_rig;
  math::Exp(cam1_from_rig, extrinsic_twist);
  rig.camera_from_rig[1] = cam1_from_rig;

  rig.intrinsics[0] = &cam0;
  rig.intrinsics[1] = &cam1;

  std::mt19937 rng(123);

  Isometry3T gt_rig_from_world = Isometry3T::Identity();
  auto landmarks = GenerateLandmarks(gt_rig_from_world.inverse(), 150, rng);

  Isometry3T cam0_from_world = rig.camera_from_rig[0] * gt_rig_from_world;
  Isometry3T cam1_from_world = rig.camera_from_rig[1] * gt_rig_from_world;

  auto obs0 = ProjectLandmarks(landmarks, cam0_from_world, CameraId{0});
  auto obs1 = ProjectLandmarks(landmarks, cam1_from_world, CameraId{1});
  std::vector<camera::Observation> all_obs;
  all_obs.insert(all_obs.end(), obs0.begin(), obs0.end());
  all_obs.insert(all_obs.end(), obs1.begin(), obs1.end());
  ASSERT_GT(all_obs.size(), 20u);

  Vector6T perturbation;
  perturbation << 0.02f, -0.01f, 0.015f, 0.05f, -0.03f, 0.04f;
  Isometry3T delta;
  math::Exp(delta, perturbation);
  Isometry3T init_rig_from_world = gt_rig_from_world * delta;

  cuvslam::pnp::MultisensorSolverSettings settings;
  settings.max_iteration = 30;
  cuvslam::pnp::MultisensorPoseEstimator estimator(rig, settings);

  Isometry3T rig_from_world = init_rig_from_world;
  Matrix6T info;
  bool ok = estimator.solve(rig_from_world, info, all_obs, landmarks);
  ASSERT_TRUE(ok);

  Vector6T error;
  Isometry3T diff = gt_rig_from_world.inverse() * rig_from_world;
  math::Log(error, diff);
  float angular_err = error.head<3>().norm();
  float trans_err = error.tail<3>().norm();

  EXPECT_LT(angular_err, 0.01f) << "Angular error too large: " << angular_err;
  EXPECT_LT(trans_err, 0.02f) << "Translation error too large: " << trans_err;
}

}  // namespace test::pnp
