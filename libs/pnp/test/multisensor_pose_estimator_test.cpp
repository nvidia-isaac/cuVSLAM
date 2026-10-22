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

#include <array>
#include <iterator>
#include <optional>
#include <random>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

#include "camera/camera.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "imu/imu_preintegration.h"
#include "imu/imu_sba_problem.h"
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

// Recover a pose far from the world origin, with non-identity camera_from_rig. With an
// identity ground truth rig_from_world and world_from_rig coincide, so only a pose like this
// one checks that the estimator hands cuNLS the pose convention its factors expect.
TEST(MultisensorPoseEstimatorTest, ReprojectionConvergenceAwayFromOrigin) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);
  Vector6T extrinsic_twist;
  extrinsic_twist << 0.05f, -0.1f, 0.02f, 0.1f, 0.05f, -0.02f;
  math::Exp(rig.camera_from_rig[0], extrinsic_twist);

  Vector6T gt_twist;
  gt_twist << 0.3f, -0.5f, 0.2f, 3.f, -2.f, 5.f;
  Isometry3T gt_rig_from_world;
  math::Exp(gt_rig_from_world, gt_twist);
  const Isometry3T gt_cam_from_world = rig.camera_from_rig[0] * gt_rig_from_world;

  std::mt19937 rng(7);
  auto landmarks = GenerateLandmarks(gt_cam_from_world.inverse(), 100, rng);
  auto observations = ProjectLandmarks(landmarks, gt_cam_from_world);
  ASSERT_GT(observations.size(), 10u);

  Vector6T perturbation;
  perturbation << 0.02f, -0.01f, 0.015f, 0.05f, -0.03f, 0.04f;
  Isometry3T delta;
  math::Exp(delta, perturbation);

  cuvslam::pnp::MultisensorSolverSettings settings;
  settings.max_iteration = 30;
  cuvslam::pnp::MultisensorPoseEstimator estimator(rig, settings);

  Isometry3T rig_from_world = gt_rig_from_world * delta;
  Matrix6T info;
  ASSERT_TRUE(estimator.solve(rig_from_world, info, observations, landmarks));

  Vector6T error;
  math::Log(error, gt_rig_from_world.inverse() * rig_from_world);
  EXPECT_LT(error.head<3>().norm(), 0.01f) << "Angular error too large: " << error.head<3>().norm();
  EXPECT_LT(error.tail<3>().norm(), 0.02f) << "Translation error too large: " << error.tail<3>().norm();
}

// The inertial factor alone must pull the pose to the IMU prediction from a previous pose away
// from the origin, with non-identity rig_from_imu and a preintegration that rotates and translates.
TEST(MultisensorPoseEstimatorTest, InertialPriorConvergesToPrediction) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  Vector6T prev_twist, rig_from_imu_twist, delta_rotation_twist;
  prev_twist << 0.4f, 0.2f, -0.3f, 1.f, 2.f, -3.f;
  rig_from_imu_twist << 0.f, 0.1f, 0.05f, 0.02f, -0.03f, 0.01f;
  delta_rotation_twist << 0.05f, -0.08f, 0.03f, 0.f, 0.f, 0.f;

  cuvslam::pnp::InertialPriorInput imu_in;
  math::Exp(imu_in.prev_rig_from_world, prev_twist);
  math::Exp(imu_in.rig_from_imu, rig_from_imu_twist);
  imu_in.prev_velocity = Vector3T(0.5f, -0.2f, 0.1f);
  imu_in.gravity_world = Vector3T(0.f, 0.f, -9.81f);

  Isometry3T delta_rotation;
  math::Exp(delta_rotation, delta_rotation_twist);
  sba_imu::IMUPreintegration preint;
  preint.dR = delta_rotation.linear();
  preint.dP = Vector3T(0.03f, 0.01f, -0.02f);
  preint.dT_s = 0.1f;
  imu_in.preint = &preint;

  sba_imu::Pose prev_pose_imu;
  prev_pose_imu.w_from_imu = imu_in.prev_rig_from_world.inverse() * imu_in.rig_from_imu;
  prev_pose_imu.velocity = imu_in.prev_velocity;
  sba_imu::Pose predicted;
  ASSERT_TRUE(prev_pose_imu.predict_pose(imu_in.gravity_world, preint, predicted));
  const Isometry3T expected_rig_from_world = (predicted.w_from_imu * imu_in.rig_from_imu.inverse()).inverse();

  Vector6T perturbation;
  perturbation << 0.02f, -0.01f, 0.015f, 0.05f, -0.03f, 0.04f;
  Isometry3T delta;
  math::Exp(delta, perturbation);

  // With min_observations = 0 the solve must still leave out the PnP factor, whose loss is normalized by the
  // (zero) observation count.
  for (const size_t min_observations : {size_t{5}, size_t{0}}) {
    // The default inertial weight puts this problem's initial cost below the minimizer's absolute
    // cost_tolerance, which would end the solve before the first step.
    cuvslam::pnp::MultisensorSolverSettings settings;
    settings.max_iteration = 30;
    settings.factor_weights.inertial = 1.f;
    settings.min_observations = min_observations;
    cuvslam::pnp::MultisensorPoseEstimator estimator(rig, settings);

    Isometry3T rig_from_world = expected_rig_from_world * delta;
    Matrix6T info;
    std::unordered_map<TrackId, Vector3T> landmarks;
    std::vector<camera::Observation> obs;
    ASSERT_TRUE(estimator.solve(rig_from_world, info, obs, landmarks, {}, {}, {}, imu_in)) << min_observations;

    Vector6T error;
    math::Log(error, expected_rig_from_world.inverse() * rig_from_world);
    EXPECT_LT(error.head<3>().norm(), 1e-3f)
        << "min_observations " << min_observations << ": angular error " << error.head<3>().norm();
    EXPECT_LT(error.tail<3>().norm(), 1e-3f)
        << "min_observations " << min_observations << ": translation error " << error.tail<3>().norm();
  }
}

// An estimator keeps its problem between solves and lets the minimizer reuse its structure. Over a sequence
// of solves that keeps the same factor batches for several solves in a row (so the cached structure is
// reused) and then switches the inertial, ICP and point-to-plane factors on and off (which rebuilds it), with
// different observation and depth point counts (including one that grows the ICP capacity) and alternating
// depth images, each result must match a fresh estimator solving the same input.
TEST(MultisensorPoseEstimatorTest, RepeatedSolvesMatchFreshEstimator) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  // Two depth images of a fronto-parallel wall, used on alternating solves so the depth factors are rebound to
  // a different texture each time. The depths are close enough that a stale texture stays within the ICP and
  // point-to-plane gates and biases the pose instead of being rejected.
  const int width = static_cast<int>(kResolution.x());
  const int height = static_cast<int>(kResolution.y());
  const float wall_depths[] = {2.f, 2.02f};
  std::array<cuvslam::cuda::GaussianGPUImagePyramid, 2> depth_pyramids;
  for (size_t i = 0; i < depth_pyramids.size(); ++i) {
    depth_pyramids[i].init(width, height);
    std::vector<float> depth_host(width * height, wall_depths[i]);
    depth_pyramids[i][0].copy(cuvslam::cuda::ToGPU, depth_host.data(), nullptr);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const cuvslam::pnp::RGBDInfo depth_infos[] = {{CameraId{0}, depth_pyramids[0]}, {CameraId{0}, depth_pyramids[1]}};

  Vector6T rig_from_imu_twist, delta_rotation_twist;
  rig_from_imu_twist << 0.f, 0.1f, 0.05f, 0.02f, -0.03f, 0.01f;
  delta_rotation_twist << 0.01f, -0.02f, 0.01f, 0.f, 0.f, 0.f;
  Isometry3T delta_rotation;
  math::Exp(delta_rotation, delta_rotation_twist);
  sba_imu::IMUPreintegration preint;
  preint.dR = delta_rotation.linear();
  preint.dP = Vector3T(0.01f, 0.f, -0.01f);
  preint.dT_s = 0.05f;

  struct SolveInput {
    size_t num_landmarks;
    size_t num_depth_points;  // 0 drops the ICP factors
    bool planes;              // false drops the point-to-plane factors
    bool inertial;
  };
  const SolveInput solves[] = {
      {40, 300, true, false},   {120, 50, true, false},   // reused
      {200, 1500, true, false},                           // more depth points than the ICP capacity: grows it
      {60, 0, true, false},                               // no ICP
      {25, 120, true, true},    {150, 600, true, true},   // inertial, reused
      {90, 200, false, true},                             // no point-to-plane
      {70, 400, true, false},   {180, 100, true, false},  // reused with the grown ICP capacity
  };

  cuvslam::pnp::MultisensorSolverSettings settings;
  settings.max_iteration = 30;
  cuvslam::pnp::MultisensorPoseEstimator reused(rig, settings);

  std::mt19937 rng(3);
  for (size_t k = 0; k < std::size(solves); ++k) {
    const SolveInput& in = solves[k];
    Vector6T gt_twist;
    gt_twist << 0.05f * k, -0.03f * k, 0.02f, 0.2f * k, -0.1f, 0.3f * k;
    Isometry3T gt_rig_from_world;
    math::Exp(gt_rig_from_world, gt_twist);
    const Isometry3T gt_world_from_cam = gt_rig_from_world.inverse();

    auto landmarks = GenerateLandmarks(gt_world_from_cam, in.num_landmarks, rng);
    auto observations = ProjectLandmarks(landmarks, gt_rig_from_world);

    // The wall seen in this solve's depth image, in world coordinates.
    const float wall_depth = wall_depths[k % 2];
    const cuvslam::pnp::RGBDInfos rgbd_infos{&depth_infos[k % 2]};
    std::uniform_real_distribution<float> u_dist(10.f, width - 10.f);
    std::uniform_real_distribution<float> v_dist(10.f, height - 10.f);
    // The first half of the depth points projects outside the image and has no ICP residual. The second half
    // lies slightly behind the wall, within the ICP gate, so it pulls the pose: an ICP batch that dropped the
    // trailing points would move the result.
    std::vector<Vector3T> depth_points;
    for (size_t i = 0; i < in.num_depth_points; ++i) {
      const bool in_view = i >= in.num_depth_points / 2;
      const float u = in_view ? u_dist(rng) : -100.f - u_dist(rng);
      const float v = v_dist(rng);
      const float depth = in_view ? wall_depth * 1.003f : wall_depth;
      const Vector3T p_cam((u - kPrincipal.x()) / kFocal.x() * depth, (v - kPrincipal.y()) / kFocal.y() * depth, depth);
      depth_points.push_back(gt_world_from_cam * p_cam);
    }
    std::vector<map::Plane> planes;
    if (in.planes) {
      map::Plane wall;
      wall.centroid = gt_world_from_cam * Vector3T(0.f, 0.f, wall_depth);
      wall.normal = gt_world_from_cam.linear() * Vector3T(0.f, 0.f, 1.f);
      planes.push_back(wall);
    }

    std::optional<cuvslam::pnp::InertialPriorInput> imu_in;
    if (in.inertial) {
      imu_in.emplace();
      Vector6T prev_twist = gt_twist;
      prev_twist[3] -= 0.02f;
      math::Exp(imu_in->prev_rig_from_world, prev_twist);
      math::Exp(imu_in->rig_from_imu, rig_from_imu_twist);
      imu_in->gravity_world = Vector3T(0.f, 0.f, -9.81f);
      imu_in->preint = &preint;
    }

    Vector6T perturbation;
    perturbation << 0.02f, -0.01f, 0.015f, 0.05f, -0.03f, 0.04f;
    Isometry3T delta;
    math::Exp(delta, perturbation);
    const Isometry3T init_rig_from_world = gt_rig_from_world * delta;

    Isometry3T reused_rig_from_world = init_rig_from_world;
    Isometry3T fresh_rig_from_world = init_rig_from_world;
    Matrix6T info;
    ASSERT_TRUE(
        reused.solve(reused_rig_from_world, info, observations, landmarks, rgbd_infos, planes, depth_points, imu_in))
        << k;
    cuvslam::pnp::MultisensorPoseEstimator fresh(rig, settings);
    ASSERT_TRUE(
        fresh.solve(fresh_rig_from_world, info, observations, landmarks, rgbd_infos, planes, depth_points, imu_in))
        << k;

    Vector6T diff;
    math::Log(diff, fresh_rig_from_world.inverse() * reused_rig_from_world);
    EXPECT_LT(diff.norm(), 1e-4f) << "solve " << k << " (" << observations.size() << " observations, "
                                  << depth_points.size() << " depth points, " << planes.size() << " planes, "
                                  << (imu_in ? "with" : "without") << " inertial factor)";

    Vector6T error;
    math::Log(error, gt_rig_from_world.inverse() * reused_rig_from_world);
    EXPECT_LT(error.norm(), 0.02f) << "solve " << k;
  }
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
