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

#include <cmath>
#include <limits>

#include "gtest/gtest.h"

#include "common/imu_calibration.h"
#include "common/imu_measurement.h"
#include "imu/imu_preintegration.h"
#include "imu/imu_sba.h"
#include "imu/imu_sba_problem.h"
#ifdef USE_CUDA
#include "imu/imu_sba_gpu.h"
#endif

namespace cuvslam::sba_imu {
namespace {

// Two keyframes watching a 3x3 grid of points at the given depth from the first one. The camera sits
// at the rig origin and the world is the first rig frame.
ImuBAProblem MakeProblem(const imu::ImuCalibration& calib, float depth) {
  ImuBAProblem problem;
  problem.rig.num_cameras = 1;
  problem.rig.camera_from_rig[0] = Isometry3T::Identity();
  problem.num_fixed_key_frames = 1;
  problem.robustifier_scale_pose = -1.f;  // as run_imu_sba sets it

  // The second keyframe turns 0.02 rad while the gyro reports half of that. The rotation error then is
  // small but not an exact identity, where the GPU rotation log is undefined.
  IMUPreintegration preintegration;
  for (int64_t i = 0; i < 20; ++i) {
    preintegration.IntegrateNewMeasurement(calib, {i * 5'000'000, Vector3T::Zero(), Vector3T(0.f, 0.f, 0.1f)});
  }
  Isometry3T turn = Isometry3T::Identity();
  turn.linear() = Eigen::AngleAxisf(0.02f, Vector3T::UnitZ()).toRotationMatrix();
  for (const Isometry3T& w_from_imu : {calib.rig_from_imu(), calib.rig_from_imu() * turn}) {
    Pose pose;
    pose.w_from_imu = w_from_imu;
    pose.preintegration = preintegration;
    problem.rig_poses.push_back(pose);
  }

  for (float x : {-0.5f, 0.f, 0.5f}) {
    for (float y : {-0.5f, 0.f, 0.5f}) {
      const auto point_id = static_cast<int32_t>(problem.points.size());
      problem.points.emplace_back(x * depth, y * depth, depth);
      for (int32_t pose_id = 0; pose_id < 2; ++pose_id) {
        const Vector3T p_c =
            calib.rig_from_imu() * problem.rig_poses[pose_id].w_from_imu.inverse() * problem.points.back();
        // off by a little so the reprojection residual is not zero
        problem.observation_xys.push_back(p_c.head<2>() / p_c.z() + Vector2T(0.01f, 0.f));
        problem.observation_infos.push_back(Matrix2T::Identity());
        problem.point_ids.push_back(point_id);
        problem.pose_ids.push_back(pose_id);
        problem.camera_ids.push_back(0);
      }
    }
  }
  return problem;
}

// Between the camera and MINIMUM_HITHER (0.1 m).
constexpr float kNearFieldDepth = 0.05f;

}  // namespace

// The IMU bundlers skip only observations behind the camera. A skipped observation costs nothing, so a
// 0.1 m near plane let an LM step hide points it had pushed close to the camera, and the step got
// accepted. A problem made entirely of near-field observations must therefore keep a finite cost.
TEST(ImuSba, NearFieldObservationsCarryWeightCpu) {
  const imu::ImuCalibration calib;
  ImuBAProblem problem = MakeProblem(calib, kNearFieldDepth);

  IMUBundlerCpuFixedVel bundler(calib);
  bundler.solve(problem);
  EXPECT_TRUE(std::isfinite(problem.initial_cost)) << "every near-field observation was skipped";
  EXPECT_GT(problem.initial_cost, 0.f);
}

// Observations behind the camera are still skipped, so a problem made only of them is infeasible.
TEST(ImuSba, ObservationsBehindTheCameraAreSkippedCpu) {
  const imu::ImuCalibration calib;
  ImuBAProblem problem = MakeProblem(calib, -kNearFieldDepth);

  IMUBundlerCpuFixedVel bundler(calib);
  EXPECT_FALSE(bundler.solve(problem));
  EXPECT_EQ(problem.initial_cost, std::numeric_limits<float>::infinity());
}

#ifdef USE_CUDA
// The GPU bundler carries its own copy of the guard and has to agree with the CPU one.
TEST(ImuSba, NearFieldObservationsCarryWeightGpu) {
  const imu::ImuCalibration calib;
  ImuBAProblem problem = MakeProblem(calib, kNearFieldDepth);

  IMUBundlerGpuFixedVel bundler(calib);
  bundler.solve(problem);
  EXPECT_TRUE(std::isfinite(problem.initial_cost)) << "every near-field observation was skipped on the GPU";
  EXPECT_GT(problem.initial_cost, 0.f);
}

TEST(ImuSba, ObservationsBehindTheCameraAreSkippedGpu) {
  const imu::ImuCalibration calib;
  ImuBAProblem problem = MakeProblem(calib, -kNearFieldDepth);

  IMUBundlerGpuFixedVel bundler(calib);
  EXPECT_FALSE(bundler.solve(problem));
  EXPECT_EQ(problem.initial_cost, std::numeric_limits<float>::infinity());
}
#endif

}  // namespace cuvslam::sba_imu
