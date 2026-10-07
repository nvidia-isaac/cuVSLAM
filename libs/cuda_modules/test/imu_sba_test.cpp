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

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common/imu_calibration.h"
#include "common/imu_measurement.h"
#include "common/include_gtest.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_sba_imu_v1.h"
#include "math/twist.h"

#include "benchmark_utils.h"

#include "imu/imu_sba.h"
#include "imu/imu_sba_gpu.h"
#include "imu/imu_sba_problem.h"

namespace cuvslam::sba_imu {
namespace {

constexpr int kSamplesPerKeyframe = 40;
constexpr float kFocal = 400.f;

Matrix3T ExpSO3(const Vector3T& w) {
  Matrix3T r;
  math::Exp(r, w);
  return r;
}

// ~sqrt(2) * angle for small rotations; unlike acos of the trace it stays accurate near identity in float.
float RotationDistance(const Matrix3T& a, const Matrix3T& b) {
  return (a.transpose() * b - Matrix3T::Identity()).norm();
}

struct Shape {
  int num_poses;
  int num_fixed;
  int num_points;
  // Number of consecutive keyframes that observe each point.
  int track_length;
};

// Every point is seen by every keyframe.
constexpr Shape kParityShape{7, 2, 200, 7};

struct Scenario {
  imu::ImuCalibration calib;
  ImuBAProblem problem;
};

// Ground truth is generated with the same discrete integration scheme as IMUPreintegration::update_state,
// so keyframe states agree exactly with the preintegrated measurements.
// exact: pure translation with no gyro bias, image noise or initial perturbation, so every inertial rotation
// residual is the identity.
Scenario MakeScenario(uint32_t seed, const Shape& shape = kParityShape, bool exact = false) {
  Scenario s;
  std::mt19937 rng(seed);

  Isometry3T rig_from_imu = Isometry3T::Identity();
  rig_from_imu.linear() = ExpSO3(Vector3T(0.02f, -0.03f, 0.01f));
  rig_from_imu.translation() = Vector3T(0.05f, -0.01f, 0.02f);
  const float freq = 200.f;
  s.calib = imu::ImuCalibration(rig_from_imu, 0.00016968f, 0.000019393f, 0.002f, 0.003f, freq);
  const float dt = 1.f / freq;
  const int64_t dt_ns = static_cast<int64_t>(1e9f / freq);

  const Vector3T gravity(0.f, 9.81f, 0.f);
  const Vector3T true_gyro_bias = exact ? Vector3T::Zero() : Vector3T(0.012f, -0.02f, 0.016f);
  const Vector3T true_acc_bias = Vector3T::Zero();
  const Vector3T omega = exact ? Vector3T::Zero() : Vector3T(0.15f, 0.3f, -0.25f);

  ImuBAProblem& p = s.problem;
  p.gravity = gravity;
  p.rig.num_cameras = 2;
  p.rig.camera_from_rig[0].setIdentity();
  p.rig.camera_from_rig[1].setIdentity();
  p.rig.camera_from_rig[1].translate(Vector3T(-0.12f, 0.f, 0.f));

  std::vector<Pose> truth;
  Matrix3T R = Matrix3T::Identity();
  Vector3T pos = Vector3T::Zero();
  Vector3T vel(0.6f, 0.f, 0.1f);
  int64_t t_ns = 0;
  float t = 0.f;

  for (int k = 0; k < shape.num_poses; ++k) {
    Pose pose;
    pose.w_from_imu.linear() = R;
    pose.w_from_imu.translation() = pos;
    pose.velocity = vel;
    pose.gyro_bias = true_gyro_bias;
    pose.acc_bias = true_acc_bias;
    pose.preintegration = IMUPreintegration(Vector3T::Zero(), Vector3T::Zero());

    if (k + 1 < shape.num_poses) {
      for (int j = 0; j < kSamplesPerKeyframe; ++j) {
        t_ns += dt_ns;
        t += dt;
        const Vector3T acc_w(0.4f * std::sin(2.f * t), 0.3f * std::cos(1.5f * t), -0.2f * std::sin(t));
        const Vector3T acc_specific = R.transpose() * (acc_w - gravity);

        imu::ImuMeasurement m;
        m.time_ns = t_ns;
        m.angular_velocity = omega + true_gyro_bias;
        m.linear_acceleration = acc_specific + true_acc_bias;
        pose.preintegration.IntegrateNewMeasurement(s.calib, m);

        pos += vel * dt + 0.5f * gravity * dt * dt + 0.5f * R * acc_specific * dt * dt;
        vel += (gravity + R * acc_specific) * dt;
        R = R * ExpSO3(omega * dt);
      }
    }
    truth.push_back(pose);
  }

  std::uniform_real_distribution<float> xy(-4.f, 4.f);
  std::uniform_real_distribution<float> depth(4.f, 12.f);
  std::normal_distribution<float> pixel_noise(0.f, 0.3f / kFocal);
  std::uniform_int_distribution<int> first_pose(0, shape.num_poses - shape.track_length);
  std::vector<int> point_first_pose;
  for (int i = 0; i < shape.num_points; ++i) {
    p.points.emplace_back(xy(rng) + 1.f, xy(rng), depth(rng));
    point_first_pose.push_back(shape.track_length < shape.num_poses ? first_pose(rng) : 0);
  }

  const Matrix2T info = Matrix2T::Identity() * kFocal * kFocal;
  for (int pose_id = 0; pose_id < shape.num_poses; ++pose_id) {
    const Isometry3T imu_from_w = truth[pose_id].w_from_imu.inverse();
    for (int point_id = 0; point_id < static_cast<int>(p.points.size()); ++point_id) {
      const int first = point_first_pose[point_id];
      if (pose_id < first || pose_id >= first + shape.track_length) {
        continue;
      }
      for (int8_t cam = 0; cam < p.rig.num_cameras; ++cam) {
        const Vector3T p_c = p.rig.camera_from_rig[cam] * rig_from_imu * imu_from_w * p.points[point_id];
        if (p_c.z() < 1.f) {
          continue;
        }
        Vector2T uv = p_c.head<2>() / p_c.z();
        if (uv.cwiseAbs().maxCoeff() > 1.f) {
          continue;
        }
        if (!exact) {
          uv += Vector2T(pixel_noise(rng), pixel_noise(rng));
        }
        p.observation_xys.push_back(uv);
        p.observation_infos.push_back(info);
        p.point_ids.push_back(point_id);
        p.pose_ids.push_back(pose_id);
        p.camera_ids.push_back(cam);
      }
    }
  }

  // Initial guess: fixed poses at truth, optimized poses perturbed; every bias starts at zero.
  std::normal_distribution<float> rot_noise(0.f, 0.01f);
  std::normal_distribution<float> trans_noise(0.f, 0.03f);
  std::normal_distribution<float> point_noise(0.f, 0.02f);
  for (int k = 0; k < shape.num_poses; ++k) {
    Pose pose = truth[k];
    pose.gyro_bias.setZero();
    pose.acc_bias.setZero();
    if (!exact && k >= shape.num_fixed) {
      pose.w_from_imu.linear() =
          pose.w_from_imu.linear() * ExpSO3(Vector3T(rot_noise(rng), rot_noise(rng), rot_noise(rng)));
      pose.w_from_imu.translation() += Vector3T(trans_noise(rng), trans_noise(rng), trans_noise(rng));
    }
    p.rig_poses.push_back(pose);
  }
  for (auto& pt : p.points) {
    if (!exact) {
      pt += Vector3T(point_noise(rng), point_noise(rng), point_noise(rng));
    }
  }

  // Same solver settings as run_imu_sba() in libs/pipelines/service_sba.h.
  p.num_fixed_key_frames = shape.num_fixed;
  p.robustifier_scale = 1.5f;
  p.robustifier_scale_pose = -1.0f;
  p.prior_acc = 0;
  p.prior_gyro = 0;
  p.imu_penalty = 1e-2f;
  p.boundary_imu_penalty = 1e-2f;
  p.acc_rw_penalty = 0.1f;
  p.reintegration_thresh = 1e-3f;
  p.max_iterations = 10;
  return s;
}

TEST(ImuSbaGpu, MatchesCpuBundler) {
  for (uint32_t seed : {1u, 2u, 3u}) {
    SCOPED_TRACE(seed);
    const Scenario s = MakeScenario(seed);

    ImuBAProblem cpu_problem = s.problem;
    ImuBAProblem gpu_problem = s.problem;

    IMUBundlerCpuFixedVel cpu(s.calib);
    IMUBundlerGpuFixedVel gpu(s.calib);
    ASSERT_TRUE(cpu.solve(cpu_problem));
    ASSERT_TRUE(gpu.solve(gpu_problem));
    EXPECT_NEAR(cpu_problem.initial_cost, gpu_problem.initial_cost, 1e-4f * cpu_problem.initial_cost);

    for (int k = kParityShape.num_fixed; k < kParityShape.num_poses; ++k) {
      SCOPED_TRACE(k);
      const Pose& c = cpu_problem.rig_poses[k];
      const Pose& g = gpu_problem.rig_poses[k];
      EXPECT_LT(RotationDistance(c.w_from_imu.linear(), g.w_from_imu.linear()), 1e-3f);
      EXPECT_LT((c.w_from_imu.translation() - g.w_from_imu.translation()).norm(), 2e-3f);
      EXPECT_LT((c.velocity - g.velocity).norm(), 5e-3f);
      EXPECT_LT((c.gyro_bias - g.gyro_bias).norm(), 1e-3f);
      EXPECT_LT((c.acc_bias - g.acc_bias).norm(), 1e-2f);
    }
  }
}

TEST(ImuSbaGpu, HandlesNearIdentityRotationResiduals) {
  const Scenario s = MakeScenario(1, kParityShape, /*exact=*/true);

  ImuBAProblem cpu_problem = s.problem;
  ImuBAProblem gpu_problem = s.problem;
  IMUBundlerCpuFixedVel cpu(s.calib);
  IMUBundlerGpuFixedVel gpu(s.calib);
  cpu.solve(cpu_problem);
  gpu.solve(gpu_problem);
  ASSERT_TRUE(std::isfinite(cpu_problem.initial_cost));
  ASSERT_TRUE(std::isfinite(gpu_problem.initial_cost));
  EXPECT_NEAR(gpu_problem.initial_cost, cpu_problem.initial_cost, 1e-3f * cpu_problem.initial_cost + 1e-3f);
}

// The inertial rotation residual is Log(dR^T * R1^T * R2). The SVD formulation that computed it before eigen-
// decomposes (R - I)^T (R - I), whose entries are ~angle^4 and underflow in float near the identity, which is where a
// converged residual lives. It now only serves angles near pi; elsewhere the GPU log must match the CPU math::Log.
TEST(ImuSbaGpu, So3LogMatchesCpuNearIdentity) {
  const Vector3T axis = Vector3T(0.3f, -0.5f, 0.8f).normalized();
  const std::vector<float> angles = {0.f, 1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1.f, 2.5f, 3.f};
  const int n = static_cast<int>(angles.size());

  cuda::GPUArrayPinned<cuda::Matf33> rotations(n);
  cuda::GPUArrayPinned<cuda::Vecf3> logs(n);
  cuda::GPUArrayPinned<cuda::Vecf3> svd_logs(n);
  std::vector<Vector3T> cpu_logs(n);
  for (int i = 0; i < n; ++i) {
    const Matrix3T r = ExpSO3(angles[i] * axis);
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        rotations[i].d_[j][k] = r(j, k);
      }
    }
    math::Log(cpu_logs[i], r);
  }

  cuda::Stream stream;
  rotations.copy(cuda::GPUCopyDirection::ToGPU, stream.get_stream());
  ASSERT_EQ(cuda::sba_imu::so3_log(rotations.ptr(), logs.ptr(), n, /*svd_based=*/false, stream.get_stream()),
            cudaSuccess);
  ASSERT_EQ(cuda::sba_imu::so3_log(rotations.ptr(), svd_logs.ptr(), n, /*svd_based=*/true, stream.get_stream()),
            cudaSuccess);
  logs.copy(cuda::GPUCopyDirection::ToCPU, stream.get_stream());
  svd_logs.copy(cuda::GPUCopyDirection::ToCPU, stream.get_stream());
  ASSERT_EQ(cudaStreamSynchronize(stream.get_stream()), cudaSuccess);

  // Rotation angle recovered by each log (the norm of the log vector) and its distance from the CPU log vector.
  std::cout << std::setw(10) << "angle" << std::setw(14) << "CPU angle" << std::setw(14) << "GPU angle" << std::setw(14)
            << "GPU error" << std::setw(14) << "SVD angle" << std::setw(14) << "SVD error" << std::endl;
  for (int i = 0; i < n; ++i) {
    SCOPED_TRACE(angles[i]);
    const Vector3T gpu(logs[i].d_[0], logs[i].d_[1], logs[i].d_[2]);
    const Vector3T svd(svd_logs[i].d_[0], svd_logs[i].d_[1], svd_logs[i].d_[2]);
    std::cout << std::setw(10) << angles[i] << std::setw(14) << cpu_logs[i].norm() << std::setw(14) << gpu.norm()
              << std::setw(14) << (gpu - cpu_logs[i]).norm() << std::setw(14) << svd.norm() << std::setw(14)
              << (svd - cpu_logs[i]).norm() << std::endl;

    EXPECT_LT((gpu - cpu_logs[i]).norm(), 1e-6f);
    if (angles[i] <= 1e-6f) {
      EXPECT_FALSE(svd.allFinite()) << "the SVD-based log is expected to break down near the identity";
    }
  }
}

TEST(ImuSbaGpu, EmptyProblemFailsLikeCpu) {
  Scenario s = MakeScenario(1);
  s.problem.points.clear();
  s.problem.observation_xys.clear();
  s.problem.observation_infos.clear();
  s.problem.point_ids.clear();
  s.problem.pose_ids.clear();
  s.problem.camera_ids.clear();

  ImuBAProblem cpu_problem = s.problem;
  ImuBAProblem gpu_problem = s.problem;
  IMUBundlerCpuFixedVel cpu(s.calib);
  IMUBundlerGpuFixedVel gpu(s.calib);
  EXPECT_FALSE(cpu.solve(cpu_problem));
  EXPECT_FALSE(gpu.solve(gpu_problem));
}

// Whole IMUBundlerGpuFixedVel::solve() vs IMUBundlerCpuFixedVel::solve(), including host<->device copies and
// preintegration packing. Shapes follow run_imu_sba() on EuRoC VIO: 10 keyframes, 1 fixed, and roughly the 10th
// percentile, median and maximum observation counts seen there (~3.6k, ~6.4k, ~8k).
TEST(Cuda, ImuSbaSolveSpeedup) {
  constexpr int kRepeats = 20;
  const Shape shapes[] = {{10, 1, 450, 4}, {10, 1, 800, 4}, {10, 1, 1000, 4}};

  std::chrono::nanoseconds cpu_total{0}, gpu_total{0};
  int total_solves = 0;
  for (const Shape& shape : shapes) {
    const Scenario s = MakeScenario(7, shape);
    IMUBundlerCpuFixedVel cpu(s.calib);
    IMUBundlerGpuFixedVel gpu(s.calib);

    // Production keeps one bundler alive across calls, so buffers are allocated before the timed region.
    ImuBAProblem warmup = s.problem;
    ASSERT_TRUE(gpu.solve(warmup));

    std::vector<ImuBAProblem> cpu_problems(kRepeats, s.problem);
    std::vector<ImuBAProblem> gpu_problems(kRepeats, s.problem);

    const auto cpu_start = std::chrono::steady_clock::now();
    for (auto& q : cpu_problems) {
      ASSERT_TRUE(cpu.solve(q));
    }
    const auto cpu_time = std::chrono::steady_clock::now() - cpu_start;

    const auto gpu_start = std::chrono::steady_clock::now();
    for (auto& q : gpu_problems) {
      ASSERT_TRUE(gpu.solve(q));
    }
    const auto gpu_time = std::chrono::steady_clock::now() - gpu_start;

    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_time).count() / kRepeats;
    const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_time).count() / kRepeats;
    std::cout << "points " << shape.num_points << ", observations " << s.problem.observation_xys.size()
              << ", LM iterations cpu/gpu " << cpu_problems[0].iterations << "/" << gpu_problems[0].iterations
              << ": cpu " << cpu_ms << " ms, gpu " << gpu_ms << " ms, speedup " << cpu_ms / gpu_ms << std::endl;

    cpu_total += cpu_time;
    gpu_total += gpu_time;
    total_solves += kRepeats;
  }
  ::test::ReportSpeedBenchmark(cpu_total, gpu_total, total_solves);
  ASSERT_TRUE(cpu_total >= gpu_total);
}

}  // namespace
}  // namespace cuvslam::sba_imu
