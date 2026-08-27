
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
#include <iostream>
#include <random>

#include "benchmark_utils.h"
#include "common/environment.h"
#include "common/include_gtest.h"
#include "cuda_modules/cuda_kernels/cuda_sba_v1.h"
#include "cuda_modules/sba.h"
#include "math/twist.h"
#include "profiler/profiler.h"
#include "sba/bundle_adjustment_problem.h"
#include "sba/schur_complement_bundler_cpu.h"
namespace {
using namespace cuvslam;

using ProfilerDomain = cuvslam::profiler::DefaultProfiler::DomainHelper;
ProfilerDomain helper("SBA_TEST");

void GenerateProblem(sba::BundleAdjustmentProblem& problem, const camera::Rig& rig,
                     const Matrix2T& obs_info = Matrix2T::Identity(), const int num_points = 400,
                     const int num_poses = 5) {
  using namespace cuvslam;
  problem = {};

  // std::rand() is seeded per-test; each call advances its state, so repeated
  // calls to GenerateProblem within the same test get different seeds.
  auto seed = std::rand();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> depth(2.f, 16.f);
  std::uniform_real_distribution<float> xy(-4.f, 4.f);

  std::uniform_real_distribution<float> dt(-2.f, 2.f);
  std::uniform_real_distribution<float> domega(-0.25f, 0.25f);

  for (int i = 0; i < num_poses; ++i) {
    Vector6T log_pose;
    log_pose << domega(rng), domega(rng), domega(rng), dt(rng), dt(rng), dt(rng);

    Isometry3T pose;
    math::Exp(pose, log_pose);

    problem.rig_from_world.push_back(pose);
  }

  for (int i = 0; i < num_points; ++i) {
    problem.points.emplace_back(xy(rng), xy(rng), depth(rng));
  }

  ASSERT_EQ(num_poses, static_cast<int>(problem.rig_from_world.size()));

  using Vec2 = Eigen::Vector2f;
  using Vec3 = Eigen::Vector3f;

  auto& rig_from_world = problem.rig_from_world;
  auto& points = problem.points;

  for (int pose_idx = 0; pose_idx < num_poses; ++pose_idx) {
    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
      for (int8_t cam_idx = 0; cam_idx < rig.num_cameras; ++cam_idx) {
        // cam space
        Vec3 p = rig.camera_from_rig[cam_idx] * rig_from_world[pose_idx] * points[point_idx];

        if (p.z() <= 1.f) {
          continue;
        }

        Vec2 u = p.topRows(2) / p.z();
        u.x() += domega(rng);
        u.y() += domega(rng);

        problem.observation_xys.push_back(u);
        problem.observation_infos.push_back(obs_info);
        problem.camera_ids.push_back(cam_idx);
        problem.point_ids.push_back(point_idx);
        problem.pose_ids.push_back(pose_idx);
      }
    }
  }

  problem.num_fixed_points = 0;
  problem.num_fixed_key_frames = 1;

  problem.rig.num_cameras = rig.num_cameras;
  problem.rig.camera_from_rig[0] = rig.camera_from_rig[0];
  problem.rig.camera_from_rig[1] = rig.camera_from_rig[1];
}

static camera::Rig MakeDefaultRig() {
  using namespace cuvslam;
  camera::Rig rig;
  rig.num_cameras = 2;
  rig.camera_from_rig[0].setIdentity();
  rig.camera_from_rig[1].setIdentity();
  rig.camera_from_rig[1].translate(Eigen::Vector3f(0.125f, 0., 0.f));

  return rig;
}

cuda::sba::temporary::ReducedSystem to_temporary(
    const sba::schur_complement_bundler_cpu_internal::ReducedSystem& system) {
  cuda::sba::temporary::ReducedSystem out;
  out.pose_block = system.pose_block;
  out.pose_rhs = system.pose_rhs;
  out.camera_backsub_block = system.camera_backsub_block;
  out.point_rhs = system.point_rhs;
  out.inverse_point_block = system.inverse_point_block;
  return out;
}

cuda::sba::temporary::FullSystem to_temporary(const sba::schur_complement_bundler_cpu_internal::FullSystem& system) {
  cuda::sba::temporary::FullSystem out;

  out.pose_block = system.pose_block;
  out.pose_rhs = system.pose_rhs;
  out.point_block = system.point_block;
  out.point_rhs = system.point_rhs;
  out.point_pose_block = system.point_pose_block;
  return out;
}

cuda::sba::temporary::ParameterUpdate to_temporary(
    const sba::schur_complement_bundler_cpu_internal::ParameterUpdate& update) {
  cuda::sba::temporary::ParameterUpdate out;

  out.pose_step = update.pose_step;
  out.pose = update.pose;
  out.point_step = update.point_step;
  out.point = update.point;
  return out;
}

sba::schur_complement_bundler_cpu_internal::ParameterUpdate from_temporary(
    const cuda::sba::temporary::ParameterUpdate& update) {
  sba::schur_complement_bundler_cpu_internal::ParameterUpdate out;

  out.pose_step = update.pose_step;
  out.pose = update.pose;
  out.point_step = update.point_step;
  out.point = update.point;
  return out;
}

}  // namespace

namespace test {
using namespace cuvslam;
using cuda::sba::temporary::FullSystem;
using sba::schur_complement_bundler_cpu_internal::ModelFunction;
using sba::schur_complement_bundler_cpu_internal::ParameterUpdate;
using sba::schur_complement_bundler_cpu_internal::ReducedSystem;

TEST(Cuda, SpeedupSBAUpdateModel) {
  const int num_points = 400;
  const int num_poses = 5;
  const int num_cameras = 2;
  const float robustifier_scale = 1.f;
  camera::Rig rig = MakeDefaultRig();
  cuda::sba::GPUBundleAdjustmentProblem gpu_problem(num_points, num_poses, num_points * num_poses * num_cameras);
  cuda::sba::GPUModelFunction gpu_model_function(num_points * num_poses * num_cameras);
  gpu_problem.set_rig(rig);
  cuda::Stream s;

  sba::BundleAdjustmentProblem problem_input1, problem_input2;
  GenerateProblem(problem_input1, rig);
  problem_input2 = problem_input1;
  ModelFunction mf_cpu, mf_gpu;
  ASSERT_TRUE(gpu_problem.set(problem_input2, s.get_stream()));
  cuda::GPUGraph graph;
  auto lambda = [&](cudaStream_t s_) {
    CUDA_CHECK(update_model(gpu_model_function.meta(), gpu_problem.meta(), robustifier_scale, s_));
  };
  graph.launch(lambda, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("CPU_UpdateModel");
    UpdateModel(mf_cpu, problem_input1);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("GPU_UpdateModel");
    graph.launch(lambda, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, SBAUpdateModel) {
  const float thresh = 1;
  const float robustifier_scale = 1.f;
  const int num_points = 400;
  const int num_poses = 5;
  const int num_cameras = 2;

  camera::Rig rig = MakeDefaultRig();
  cuda::sba::GPUBundleAdjustmentProblem gpu_problem(num_points, num_poses, num_points * num_poses * num_cameras);
  cuda::sba::GPUModelFunction gpu_function(num_points * num_poses * num_cameras);
  gpu_problem.set_rig(rig);
  cuda::Stream s;

  cuda::GPUGraph graph;

  for (int i = 0; i < 100; i++) {
    sba::BundleAdjustmentProblem problem_input1, problem_input2;
    GenerateProblem(problem_input1, rig);
    problem_input2 = problem_input1;
    ModelFunction mf_cpu, mf_gpu;
    UpdateModel(mf_cpu, problem_input1);

    graph.launch(
        [&](cudaStream_t s_) {
          ASSERT_TRUE(gpu_problem.set(problem_input2, s_));
          CUDA_CHECK(update_model(gpu_function.meta(), gpu_problem.meta(), robustifier_scale, s_));
        },
        s.get_stream());
    ASSERT_TRUE(gpu_problem.get(problem_input2, s.get_stream()));
    ASSERT_TRUE(gpu_function.get(problem_input2.observation_xys.size(), mf_gpu, s.get_stream()));

    for (size_t jj = 0; jj < problem_input1.observation_xys.size(); jj++) {
      const int j = gpu_problem.original_observation_index(jj);
      if (!mf_gpu.point_jacobians[jj].isApprox(mf_cpu.point_jacobians[j], thresh)) {
        std::cout << "mf_gpu.point_jacobian[" << jj << "] = " << mf_gpu.point_jacobians[jj] << std::endl;
        std::cout << "mf_cpu.point_jacobian[" << j << "] = " << mf_cpu.point_jacobians[j] << std::endl;
      }
      ASSERT_TRUE(mf_gpu.point_jacobians[jj].isApprox(mf_cpu.point_jacobians[j], thresh));
    }
    for (size_t jj = 0; jj < problem_input1.observation_xys.size(); jj++) {
      const int j = gpu_problem.original_observation_index(jj);
      if (!mf_gpu.pose_jacobians[jj].isApprox(mf_cpu.pose_jacobians[j], thresh)) {
        std::cout << "mf_gpu.pose_jacobians[" << jj << "] = " << std::endl << mf_gpu.pose_jacobians[jj] << std::endl;
        std::cout << "mf_cpu.pose_jacobians[" << j << "] = " << std::endl << mf_cpu.pose_jacobians[j] << std::endl;
      }
      ASSERT_TRUE(mf_gpu.pose_jacobians[jj].isApprox(mf_cpu.pose_jacobians[j], thresh));
    }
    for (size_t jj = 0; jj < problem_input1.observation_xys.size(); jj++) {
      const int j = gpu_problem.original_observation_index(jj);
      if ((mf_cpu.residuals[j] - mf_gpu.residuals[jj]).norm() >= thresh) {
        std::cout << "mf_gpu.residuals[" << jj << "] = " << std::endl << mf_gpu.residuals[jj] << std::endl;
        std::cout << "mf_cpu.residuals[" << j << "] = " << std::endl << mf_cpu.residuals[j] << std::endl;
      }
      ASSERT_TRUE((mf_cpu.residuals[j] - mf_gpu.residuals[jj]).norm() < thresh);
    }

    // BundleAdjustmentProblem::info_matrix used to be compared here. Nothing fills it - not
    // GenerateProblem, not GPUBundleAdjustmentProblem::set/get - so both vectors are empty and the
    // comparison read past the end of them.

    for (size_t jj = 0; jj < problem_input1.observation_xys.size(); jj++) {
      const int j = gpu_problem.original_observation_index(jj);
      if (std::abs(mf_cpu.robustifier_weights[j] - mf_gpu.robustifier_weights[jj]) >= thresh) {
        std::cout << "id = " << std::endl << j << std::endl;
        std::cout << "mf_gpu.robustifier_weights[" << jj << "] = " << mf_gpu.robustifier_weights[jj] << std::endl;
        std::cout << "mf_cpu.robustifier_weights[" << j << "] = " << mf_cpu.robustifier_weights[j] << std::endl;
      }
      ASSERT_TRUE(std::abs(mf_cpu.robustifier_weights[j] - mf_gpu.robustifier_weights[jj]) < thresh);
    }
  }
}

TEST(Cuda, SBABuildFullSystemSpeedUp) {
  const int num_points = 400;
  const int num_poses = 5;
  const int num_cameras = 2;

  const float robustifier_scale = 1.f;
  camera::Rig rig = MakeDefaultRig();
  cuda::sba::GPUBundleAdjustmentProblem gpu_problem(num_points, num_poses, num_points * num_poses * num_cameras);
  cuda::sba::GPUModelFunction gpu_function(num_points * num_poses * num_cameras);
  cuda::sba::GPULinearSystem gpu_full_system(num_points, num_poses);
  gpu_problem.set_rig(rig);

  cuda::GPUGraph graph;
  cuda::Stream s;

  sba::BundleAdjustmentProblem problem_input1, problem_input2;
  ModelFunction mf_cpu, mf_gpu;

  // generate sba problem
  GenerateProblem(problem_input1, rig);
  problem_input2 = problem_input1;
  UpdateModel(mf_cpu, problem_input1);

  ASSERT_TRUE(gpu_problem.set(problem_input2, s.get_stream()));
  CUDA_CHECK(update_model(gpu_function.meta(), gpu_problem.meta(), robustifier_scale, s.get_stream()));

  auto lambda_func = [&](cudaStream_t s_) {
    CUDA_CHECK(build_full_system(gpu_full_system.meta(), gpu_function.meta(), gpu_problem.meta(),
                                 problem_input2.num_fixed_points, problem_input2.num_fixed_key_frames, s_));
  };

  graph.launch(lambda_func, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  FullSystem fs_gpu;
  sba::schur_complement_bundler_cpu_internal::FullSystem fs_cpu;

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("CPU_BuildFullSystem");
    BuildFullSystem(fs_cpu, mf_cpu, problem_input1);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("GPU_BuildFullSystem");
    graph.launch(lambda_func, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, SBABuildFullSystemMethods) {
  float thresh = 0.01f;
  camera::Rig rig = MakeDefaultRig();
  cuda::Stream s;

  int max_points = 400;
  int max_poses = 5;

  cuvslam::cuda::sba::GPULinearSystem gpu_full_system(max_points, max_poses);
  cuvslam::cuda::sba::GPULinearSystem gpu_reduced_system(max_points, max_poses);

  sba::BundleAdjustmentProblem problem_input;
  ModelFunction model;
  sba::schur_complement_bundler_cpu_internal::FullSystem fs_1;
  sba::schur_complement_bundler_cpu_internal::ReducedSystem rs_1;
  cuda::sba::temporary::FullSystem fs_2;
  cuda::sba::temporary::ReducedSystem rs_2;

  for (int i = 0; i < 100; i++) {
    // generate sba problem
    GenerateProblem(problem_input, rig, Matrix2T::Identity(), max_points, max_poses);
    UpdateModel(model, problem_input);
    BuildFullSystem(fs_1, model, problem_input);
    BuildReducedSystem(rs_1, fs_1, 0.001f);

    cuda::sba::temporary::FullSystem fs_11 = to_temporary(fs_1);
    cuda::sba::temporary::ReducedSystem rs_11 = to_temporary(rs_1);

    ASSERT_TRUE(gpu_full_system.set(fs_11, s.get_stream()));
    ASSERT_TRUE(gpu_full_system.get(fs_11.point_block.size(), fs_11.pose_rhs.size() / 6, fs_2, s.get_stream()));

    ASSERT_TRUE(gpu_reduced_system.set(rs_11, s.get_stream()));
    ASSERT_TRUE(
        gpu_reduced_system.get(rs_11.inverse_point_block.size(), rs_1.pose_rhs.size() / 6, rs_2, s.get_stream()));
    cudaStreamSynchronize(s.get_stream());

    ASSERT_TRUE(fs_11.point_block == fs_2.point_block);
    ASSERT_TRUE(fs_11.point_rhs == fs_2.point_rhs);

    ASSERT_TRUE(fs_11.pose_block.rows() == fs_2.pose_block.rows());
    ASSERT_TRUE(fs_11.pose_block.cols() == fs_2.pose_block.cols());

    ASSERT_TRUE(fs_11.pose_block.isApprox(fs_2.pose_block, thresh));
    ASSERT_TRUE(fs_11.pose_rhs == fs_2.pose_rhs);
    ASSERT_TRUE(fs_11.point_pose_block == fs_2.point_pose_block);

    ASSERT_TRUE(rs_1.inverse_point_block == rs_2.inverse_point_block);
    ASSERT_TRUE(rs_1.point_rhs == rs_2.point_rhs);
    ASSERT_TRUE(rs_1.pose_block == rs_2.pose_block);
    ASSERT_TRUE(rs_1.pose_rhs == rs_2.pose_rhs);
    ASSERT_TRUE(rs_1.camera_backsub_block == rs_2.camera_backsub_block);
  }
}

TEST(Cuda, SBABuildFullSystem) {
  const int num_points = 400;
  const int num_poses = 5;
  const int num_cameras = 2;

  const float thresh = 1;
  const float robustifier_scale = 1.f;
  camera::Rig rig = MakeDefaultRig();
  cuda::sba::GPUBundleAdjustmentProblem gpu_problem(num_points, num_poses, num_points * num_poses * num_cameras);
  cuda::sba::GPUModelFunction gpu_function(num_points * num_poses * num_cameras);
  cuda::sba::GPULinearSystem gpu_full_system(num_points, num_poses);
  gpu_problem.set_rig(rig);
  cuda::Stream s;
  cuda::GPUGraph graph;

  for (int i = 0; i < 100; i++) {
    sba::BundleAdjustmentProblem problem_input1, problem_input2;
    ModelFunction mf_cpu, mf_gpu;

    // generate sba problem
    GenerateProblem(problem_input1, rig);
    problem_input2 = problem_input1;

    sba::schur_complement_bundler_cpu_internal::FullSystem fs_cpu;
    FullSystem fs_gpu;
    {
      // calculate cpu part
      UpdateModel(mf_cpu, problem_input1);
      BuildFullSystem(fs_cpu, mf_cpu, problem_input1);
    }

    {
      graph.launch(
          [&](cudaStream_t s_) {
            // calc gpu part
            ASSERT_TRUE(gpu_problem.set(problem_input2, s_));
            CUDA_CHECK(update_model(gpu_function.meta(), gpu_problem.meta(), robustifier_scale, s_));
            CUDA_CHECK(build_full_system(gpu_full_system.meta(), gpu_function.meta(), gpu_problem.meta(),
                                         problem_input2.num_fixed_points, problem_input2.num_fixed_key_frames, s_));
          },
          s.get_stream());

      ASSERT_TRUE(gpu_full_system.get(problem_input2.points.size() - problem_input2.num_fixed_points,
                                      problem_input2.rig_from_world.size() - problem_input2.num_fixed_key_frames,
                                      fs_gpu, s.get_stream()));
    }

    ASSERT_TRUE(fs_cpu.point_block.size() == fs_gpu.point_block.size());
    for (size_t j = 0; j < fs_cpu.point_block.size(); j++) {
      if (!fs_cpu.point_block[j].isApprox(fs_gpu.point_block[j], thresh)) {
        std::cout << "id = " << j << std::endl;
        std::cout << "cpu = " << std::endl << fs_cpu.point_block[j] << std::endl;
        std::cout << "gpu = " << std::endl << fs_gpu.point_block[j] << std::endl;
      }
      ASSERT_TRUE(fs_cpu.point_block[j].isApprox(fs_gpu.point_block[j], thresh));
    }
    ASSERT_TRUE(fs_cpu.point_rhs.isApprox(fs_gpu.point_rhs, thresh));
    if (!fs_cpu.pose_rhs.isApprox(fs_gpu.pose_rhs, thresh)) {
      std::cout << "cpu: " << fs_cpu.pose_rhs.block<5, 1>(0, 0) << std::endl;
      std::cout << "gpu: " << fs_gpu.pose_rhs.block<5, 1>(0, 0) << std::endl;
    }
    ASSERT_TRUE(fs_cpu.pose_rhs.isApprox(fs_gpu.pose_rhs, thresh));
    ASSERT_TRUE(fs_cpu.pose_block.isApprox(fs_gpu.pose_block, thresh));
    ASSERT_TRUE(fs_cpu.point_pose_block.isApprox(fs_gpu.point_pose_block, 0.1));
  }
}

TEST(Cuda, SBASolverSpeedUp) {
  camera::Rig rig = MakeDefaultRig();

  ReducedSystem reduced_system;
  Eigen::VectorXf cpu_sollution, gpu_solution;

  int max_system_order = 200;
  int max_points = 400;
  int max_poses = 5;

  cuda::sba::GPUCholeskySolver solver{max_system_order};
  cuda::sba::GPULinearSystem gpu_reduced_system{max_points, max_poses};
  float* x;

  cuda::Stream s;

  int current_system_order;

  {
    sba::BundleAdjustmentProblem problem_input;
    ModelFunction model;
    sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
    GenerateProblem(problem_input, rig, Matrix2T::Identity(), max_points, max_poses);
    UpdateModel(model, problem_input);
    BuildFullSystem(full_system, model, problem_input);
    BuildReducedSystem(reduced_system, full_system, 0.001f);

    // prepare all the data
    CUDA_CHECK(cudaMalloc((void**)&x, max_system_order * sizeof(float)));
    current_system_order = reduced_system.pose_block.cols();
    gpu_solution.resize(current_system_order);
    ASSERT_TRUE(reduced_system.pose_block.cols() == reduced_system.pose_block.rows());
    // gpu part

    cuda::sba::temporary::ReducedSystem reduced_temp = to_temporary(reduced_system);

    ASSERT_TRUE(gpu_reduced_system.set(reduced_temp, s.get_stream()));
    cudaStreamSynchronize(s.get_stream());
  }

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("CPU_Solver");
    Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::NoQRPreconditioner> usv(reduced_system.pose_block,
                                                                     Eigen::ComputeFullU | Eigen::ComputeFullV);
    // control condition number
    usv.setThreshold(1e-6f);
    cpu_sollution = usv.solve(reduced_system.pose_rhs);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  const auto& meta = gpu_reduced_system.meta();
  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("GPU_Solver");
    solver.solve(meta.pose_block, meta.pose_block_pitch, meta.pose_rhs, x, current_system_order, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  CUDA_CHECK(cudaFree(x));

  std::cout << "Basic time, nano_sec = " << duration_basic.count() / 100 << std::endl;
  std::cout << "Cuda time, nano_sec = " << duration_cuda.count() / 100 << std::endl;
  float speedup = static_cast<float>(duration_basic.count()) / static_cast<float>(duration_cuda.count());
  std::cout << "Speedup, times = " << speedup << std::endl;
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, SBASolver) {
  constexpr float thresh = 0.01;
  camera::Rig rig = MakeDefaultRig();
  int max_points = 400;
  int max_poses = 5;
  int max_system_order = 400;
  cuda::sba::GPUCholeskySolver solver{1};
  cuda::sba::GPULinearSystem gpu_reduced_system{max_points, max_poses};
  float* x;
  CUDA_CHECK(cudaMalloc((void**)&x, max_system_order * sizeof(float)));

  cuda::Stream s;

  for (int i = 0; i < 100; i++) {
    sba::BundleAdjustmentProblem problem_input;
    ModelFunction model;
    sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
    ReducedSystem reduced_system;

    Eigen::VectorXf cpu_sollution;
    // generate sba problem
    GenerateProblem(problem_input, rig, Matrix2T::Identity(), max_points, max_poses);
    {
      // calculate cpu part
      UpdateModel(model, problem_input);
      BuildFullSystem(full_system, model, problem_input);
      BuildReducedSystem(reduced_system, full_system, 0.001f);

      Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::NoQRPreconditioner> usv(reduced_system.pose_block,
                                                                       Eigen::ComputeFullU | Eigen::ComputeFullV);

      // control condition number
      usv.setThreshold(1e-6f);

      cpu_sollution = usv.solve(reduced_system.pose_rhs);
    }

    Eigen::VectorXf gpu_solution;

    {
      int current_system_order = reduced_system.pose_block.cols();
      gpu_solution.resize(current_system_order);
      ASSERT_TRUE(reduced_system.pose_block.cols() == reduced_system.pose_block.rows());
      // gpu part

      cuda::sba::temporary::ReducedSystem reduced_temp = to_temporary(reduced_system);
      ASSERT_TRUE(gpu_reduced_system.set(reduced_temp, s.get_stream()));

      const auto& meta = gpu_reduced_system.meta();
      solver.solve(meta.pose_block, meta.pose_block_pitch, meta.pose_rhs, x, current_system_order, s.get_stream());
      CUDA_CHECK(cudaMemcpyAsync(gpu_solution.data(), x, current_system_order * sizeof(float), cudaMemcpyDeviceToHost,
                                 s.get_stream()));
      cudaStreamSynchronize(s.get_stream());
    }

    if (!gpu_solution.isApprox(cpu_sollution, thresh)) {
      std::cout << "cpu = " << cpu_sollution.block<5, 1>(0, 0) << std::endl;
      std::cout << "gpu = " << gpu_solution.block<5, 1>(0, 0) << std::endl;
    }
    ASSERT_TRUE(gpu_solution.isApprox(cpu_sollution, thresh));
  }

  CUDA_CHECK(cudaFree(x));
}

TEST(Cuda, SBAEvaluateCostSpeedUp) {
  camera::Rig rig = MakeDefaultRig();
  int max_points = 400;
  int max_poses = 5;
  constexpr int num_cameras = 2;

  sba::BundleAdjustmentProblem problem_input_cpu, problem_input_gpu;
  ModelFunction model;
  sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
  ReducedSystem reduced_system;
  ParameterUpdate update;
  update.point.resize(max_points, Vector3T::Zero());
  update.pose.resize(max_poses, Isometry3T::Identity());

  GenerateProblem(problem_input_cpu, rig, Matrix2T::Identity(), max_points, max_poses);
  problem_input_gpu = problem_input_cpu;
  UpdateModel(model, problem_input_cpu);
  BuildFullSystem(full_system, model, problem_input_cpu);
  BuildReducedSystem(reduced_system, full_system, 0.001f);
  ComputeUpdate(update, reduced_system, full_system);

  cuda::sba::GPUBundleAdjustmentProblem gpu_problem(max_points, max_poses, max_points * max_poses * num_cameras);
  cuda::sba::GPUModelFunction gpu_function(max_points * max_poses * num_cameras);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  gpu_problem.set_rig(rig);

  cuda::GPUArrayPinned<float> gpu_cost{1};
  cuda::GPUArrayPinned<int> gpu_num_skipped{1};
  cuda::Stream s;
  cuda::GPUGraph graph;

  ASSERT_TRUE(gpu_problem.set(problem_input_gpu, s.get_stream()));
  CUDA_CHECK(
      update_model(gpu_function.meta(), gpu_problem.meta(), problem_input_gpu.robustifier_scale, s.get_stream()));
  cuda::sba::temporary::ParameterUpdate update_temp = to_temporary(update);
  ASSERT_TRUE(gpu_update.set(update_temp, s.get_stream()));

  auto lambda = [&](cudaStream_t s_) {
    CUDA_CHECK(evaluate_cost(gpu_cost.ptr(), gpu_num_skipped.ptr(), gpu_problem.meta(), gpu_update.meta(),
                             problem_input_gpu.robustifier_scale, s_));
  };
  graph.launch(lambda, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("CPU_EvaluateCost");
    EvaluateCost(problem_input_cpu, update);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("GPU_EvaluateCost");
    graph.launch(lambda, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, SBAEvaluateCost) {
  constexpr float thresh = 10;
  camera::Rig rig = MakeDefaultRig();
  int max_points = 400;
  int max_poses = 5;
  constexpr int num_cameras = 2;
  cuda::sba::GPUBundleAdjustmentProblem gpu_problem(max_points, max_poses, max_points * max_poses * num_cameras);
  cuda::sba::GPUModelFunction gpu_function(max_points * max_poses * num_cameras);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  gpu_problem.set_rig(rig);

  cuda::GPUArrayPinned<float> gpu_cost{1};
  cuda::GPUArrayPinned<int> gpu_num_skipped{1};
  cuda::GPUGraph graph;
  cuda::Stream s;

  for (int i = 0; i < 100; i++) {
    sba::BundleAdjustmentProblem problem_input_cpu, problem_input_gpu;
    ModelFunction model;
    sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
    ReducedSystem reduced_system;
    ParameterUpdate update;
    update.point.resize(max_points, Vector3T::Zero());
    update.pose.resize(max_poses, Isometry3T::Identity());

    // generate sba problem
    GenerateProblem(problem_input_cpu, rig, Matrix2T::Identity(), max_points, max_poses);
    problem_input_gpu = problem_input_cpu;
    float cpu_result_cost, gpu_result_cost;
    {
      // calculate cpu part
      UpdateModel(model, problem_input_cpu);
      BuildFullSystem(full_system, model, problem_input_cpu);
      BuildReducedSystem(reduced_system, full_system, 0.001f);
      ComputeUpdate(update, reduced_system, full_system);
      cpu_result_cost = EvaluateCost(problem_input_cpu, update);
    }

    {
      graph.launch(
          [&](cudaStream_t s_) {
            ASSERT_TRUE(gpu_problem.set(problem_input_gpu, s_));
            CUDA_CHECK(update_model(gpu_function.meta(), gpu_problem.meta(), problem_input_gpu.robustifier_scale, s_));

            cuda::sba::temporary::ParameterUpdate update_temp = to_temporary(update);

            ASSERT_TRUE(gpu_update.set(update_temp, s_));
            CUDA_CHECK(evaluate_cost(gpu_cost.ptr(), gpu_num_skipped.ptr(), gpu_problem.meta(), gpu_update.meta(),
                                     problem_input_gpu.robustifier_scale, s_));
          },
          s.get_stream());

      gpu_cost.copy(cuda::GPUCopyDirection::ToCPU, s.get_stream());
      gpu_num_skipped.copy(cuda::GPUCopyDirection::ToCPU, s.get_stream());
      cudaStreamSynchronize(s.get_stream());

      gpu_result_cost = gpu_cost[0];
      if (gpu_num_skipped[0] == (int)problem_input_gpu.observation_xys.size()) {
        gpu_result_cost = std::numeric_limits<float>::infinity();
      }
    }

    if (isfinite(cpu_result_cost)) {
      ASSERT_TRUE(std::abs(cpu_result_cost - gpu_result_cost) < thresh);
    }
  }
}

TEST(Cuda, SBAParameterUpdaterComputeUpdateSpeedUp) {
  camera::Rig rig = MakeDefaultRig();
  int max_points = 2000;
  int max_poses = 20;
  cuda::sba::GPULinearSystem gpu_reduced_system(max_points, max_poses);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  cuda::sba::GPULevenbergMarquardtStep lm_step(max_points, max_poses);
  cuda::GPUArrayPinned<float> points_poses_update_max{2};
  cuda::Stream s;

  sba::BundleAdjustmentProblem problem_input_cpu, problem_input_gpu;
  // generate sba problem
  GenerateProblem(problem_input_cpu, rig, Matrix2T::Identity(), max_points, max_poses);
  int num_points = problem_input_cpu.points.size() - problem_input_cpu.num_fixed_points;
  int num_poses = problem_input_cpu.rig_from_world.size() - problem_input_cpu.num_fixed_key_frames;

  ModelFunction model;
  sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
  ReducedSystem reduced_system;
  ParameterUpdate update, update_gpu;
  update.point.resize(num_points, Vector3T::Zero());
  update.pose.resize(num_poses, Isometry3T::Identity());

  UpdateModel(model, problem_input_cpu);
  BuildFullSystem(full_system, model, problem_input_cpu);
  BuildReducedSystem(reduced_system, full_system, 0.1f);

  cuda::sba::temporary::ReducedSystem reduced_temp = to_temporary(reduced_system);

  ASSERT_TRUE(gpu_reduced_system.set(reduced_temp, s.get_stream()));
  cudaStreamSynchronize(s.get_stream());

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("CPU_ComputeUpdate");
    ComputeUpdate(update, reduced_system, full_system);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("GPU_ComputeUpdate");
    ASSERT_TRUE(lm_step.compute_update(gpu_update.meta(), gpu_reduced_system.meta(), num_points, num_poses,
                                       points_poses_update_max, s.get_stream()));
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, SBAParameterUpdaterComputeUpdate) {
  const float thresh = 0.01f;
  camera::Rig rig = MakeDefaultRig();
  int max_points = 400;
  int max_poses = 5;
  cuda::sba::GPULinearSystem gpu_reduced_system(max_points, max_poses);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  cuda::sba::GPULevenbergMarquardtStep lm_step(max_points, max_poses);
  cuda::GPUArrayPinned<float> points_poses_update_max{2};
  cuda::Stream s;

  for (int i = 0; i < 100; i++) {
    sba::BundleAdjustmentProblem problem_input_cpu, problem_input_gpu;
    // generate sba problem
    GenerateProblem(problem_input_cpu, rig, Matrix2T ::Identity(), max_points, max_poses);
    int num_points = problem_input_cpu.points.size() - problem_input_cpu.num_fixed_points;
    int num_poses = problem_input_cpu.rig_from_world.size() - problem_input_cpu.num_fixed_key_frames;

    ModelFunction model;
    sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
    ReducedSystem reduced_system;
    ParameterUpdate update, update_gpu;
    update.point.resize(num_points, Vector3T::Zero());
    update.pose.resize(num_poses, Isometry3T::Identity());
    {
      // calculate cpu part
      UpdateModel(model, problem_input_cpu);
      BuildFullSystem(full_system, model, problem_input_cpu);
      BuildReducedSystem(reduced_system, full_system, 0.1f);
      ComputeUpdate(update, reduced_system, full_system);
    }

    cuda::sba::temporary::ParameterUpdate update_temp = to_temporary(update_gpu);

    {
      cuda::sba::temporary::ReducedSystem reduced_temp;
      reduced_temp.pose_block = reduced_system.pose_block;
      reduced_temp.pose_rhs = reduced_system.pose_rhs;
      reduced_temp.camera_backsub_block = reduced_system.camera_backsub_block;
      reduced_temp.point_rhs = reduced_system.point_rhs;
      reduced_temp.inverse_point_block = reduced_system.inverse_point_block;

      ASSERT_TRUE(gpu_reduced_system.set(reduced_temp, s.get_stream()));
      ASSERT_TRUE(lm_step.compute_update(gpu_update.meta(), gpu_reduced_system.meta(), num_points, num_poses,
                                         points_poses_update_max, s.get_stream()));

      points_poses_update_max.copy(cuda::GPUCopyDirection::ToCPU, s.get_stream());
      ASSERT_TRUE(gpu_update.get(num_points, num_poses, update_temp, s.get_stream()));

      cudaStreamSynchronize(s.get_stream());
    }

    update_gpu = from_temporary(update_temp);

    if (isfinite(update.pose_step.sum())) {
      ASSERT_TRUE(update.pose_step.isApprox(update_gpu.pose_step, thresh));
      ASSERT_TRUE(update.pose.size() == update_gpu.pose.size());
      for (size_t j = 0; j < update.pose.size(); j++) {
        ASSERT_TRUE(update.pose[j].matrix().isApprox(update_gpu.pose[j].matrix(), thresh));
      }

      ASSERT_TRUE(update.point_step.isApprox(update_gpu.point_step, thresh));
      ASSERT_TRUE(update.point.size() == update_gpu.point.size());
      for (size_t j = 0; j < update.point.size(); j++) {
        ASSERT_TRUE(update.point[j].isApprox(update_gpu.point[j], thresh));
      }
      ASSERT_TRUE(std::abs(update.point_step.lpNorm<Eigen::Infinity>() - points_poses_update_max[0]) < thresh);
      ASSERT_TRUE(std::abs(update.pose_step.lpNorm<Eigen::Infinity>() - points_poses_update_max[1]) < thresh);
    }
  }
}

TEST(Cuda, SBAParameterUpdaterUpdateStateSpeedUp) {
  camera::Rig rig = MakeDefaultRig();
  int max_points = 2000;
  int max_poses = 2000;
  int max_observations = max_points * max_poses * 2;
  cuda::sba::GPUBundleAdjustmentProblem gpu_problem{max_points, max_poses, max_observations};
  gpu_problem.set_rig(rig);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  cuda::sba::GPULevenbergMarquardtStep lm_step(max_points, max_poses);
  cuda::GPUGraph graph;
  cuda::Stream s;

  sba::BundleAdjustmentProblem problem_input_cpu;
  // generate sba problem
  GenerateProblem(problem_input_cpu, rig, Matrix2T::Identity(), max_points, max_poses);
  int num_points = problem_input_cpu.points.size() - problem_input_cpu.num_fixed_points;
  int num_poses = problem_input_cpu.rig_from_world.size() - problem_input_cpu.num_fixed_key_frames;

  ParameterUpdate update;
  update.point.resize(num_points, Vector3T::Zero());
  update.pose.resize(num_poses, Isometry3T::Identity());
  auto lambda = [&](cudaStream_t s_) {
    ASSERT_TRUE(lm_step.apply_update(gpu_problem.meta(), gpu_update.meta(), num_points, num_poses, s_));
  };
  graph.launch(lambda, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("CPU_UpdateState");
    UpdateState(problem_input_cpu, update);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("GPU_UpdateState");
    graph.launch(lambda, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, SBAParameterUpdaterUpdateState) {
  constexpr float thresh = 0.01f;
  camera::Rig rig = MakeDefaultRig();
  int max_points = 400;
  int max_poses = 5;
  int max_observations = 20000;
  cuda::sba::GPUBundleAdjustmentProblem gpu_problem{max_points, max_poses, max_observations};
  gpu_problem.set_rig(rig);
  cuda::sba::GPULinearSystem gpu_reduced_system(max_points, max_poses);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  cuda::sba::GPULevenbergMarquardtStep lm_step(max_points, max_poses);
  cuda::GPUArrayPinned<float> points_poses_update_max{2};
  cuda::Stream s;

  for (int i = 0; i < 100; i++) {
    sba::BundleAdjustmentProblem problem_input_cpu, problem_input_gpu;
    // generate sba problem
    GenerateProblem(problem_input_cpu, rig, Matrix2T::Identity(), max_points, max_poses);
    problem_input_gpu = problem_input_cpu;
    int num_points = problem_input_cpu.points.size() - problem_input_cpu.num_fixed_points;
    int num_poses = problem_input_cpu.rig_from_world.size() - problem_input_cpu.num_fixed_key_frames;

    ModelFunction model;
    sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
    ReducedSystem reduced_system;
    ParameterUpdate update, update_gpu;
    update.point.resize(num_points, Vector3T::Zero());
    update.pose.resize(num_poses, Isometry3T::Identity());
    {
      // calculate cpu part
      UpdateModel(model, problem_input_cpu);
      BuildFullSystem(full_system, model, problem_input_cpu);
      BuildReducedSystem(reduced_system, full_system, 0.1f);
      ComputeUpdate(update, reduced_system, full_system);
      UpdateState(problem_input_cpu, update);
    }

    cuda::sba::temporary::ReducedSystem reduced_temp = to_temporary(reduced_system);

    {
      ASSERT_TRUE(gpu_problem.set(problem_input_gpu, s.get_stream()));
      ASSERT_TRUE(gpu_reduced_system.set(reduced_temp, s.get_stream()));
      ASSERT_TRUE(lm_step.compute_update(gpu_update.meta(), gpu_reduced_system.meta(), num_points, num_poses,
                                         points_poses_update_max, s.get_stream()));
      ASSERT_TRUE(lm_step.apply_update(gpu_problem.meta(), gpu_update.meta(), num_points, num_poses, s.get_stream()));
      ASSERT_TRUE(gpu_problem.get(problem_input_gpu, s.get_stream()));

      cudaStreamSynchronize(s.get_stream());
    }

    ASSERT_TRUE(problem_input_cpu.points.size() == problem_input_gpu.points.size());
    if (isfinite(problem_input_cpu.points[0].sum())) {
      for (size_t j = 0; j < problem_input_cpu.points.size(); j++) {
        ASSERT_TRUE(problem_input_cpu.points[j].isApprox(problem_input_gpu.points[j], thresh));
      }

      ASSERT_TRUE(problem_input_cpu.rig_from_world.size() == problem_input_gpu.rig_from_world.size());
      for (size_t j = 0; j < problem_input_cpu.rig_from_world.size(); j++) {
        ASSERT_TRUE(problem_input_cpu.rig_from_world[j].isApprox(problem_input_gpu.rig_from_world[j], thresh));
      }
    }
  }
}

TEST(Cuda, SBAComputePredictedRelativeReductionSpeedUp) {
  camera::Rig rig = MakeDefaultRig();
  int max_points = 2000;
  int max_poses = 200;
  cuda::sba::GPULinearSystem gpu_full_system(max_points, max_poses);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  cuda::sba::GPULevenbergMarquardtStep lm_step(max_points, max_poses);
  cuda::GPUGraph graph;
  cuda::GPUArrayPinned<float> prediction{1};
  cuda::Stream s;

  sba::BundleAdjustmentProblem problem_input_cpu;
  // generate sba problem
  GenerateProblem(problem_input_cpu, rig, Matrix2T::Identity(), max_points, max_poses);
  int num_points = problem_input_cpu.points.size() - problem_input_cpu.num_fixed_points;
  int num_poses = problem_input_cpu.rig_from_world.size() - problem_input_cpu.num_fixed_key_frames;

  ModelFunction model;
  sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
  ReducedSystem reduced_system;
  ParameterUpdate update, update_gpu;
  update.point.resize(num_points, Vector3T::Zero());
  update.pose.resize(num_poses, Isometry3T::Identity());

  UpdateModel(model, problem_input_cpu);
  BuildFullSystem(full_system, model, problem_input_cpu);
  BuildReducedSystem(reduced_system, full_system, 0.1f);
  ComputeUpdate(update, reduced_system, full_system);
  UpdateState(problem_input_cpu, update);

  FullSystem full_system_2;
  full_system_2.pose_block = full_system.pose_block;
  full_system_2.pose_rhs = full_system.pose_rhs;
  full_system_2.point_block = full_system.point_block;
  full_system_2.point_rhs = full_system.point_rhs;
  full_system_2.point_pose_block = full_system.point_pose_block;

  cuda::sba::temporary::ParameterUpdate update_temp = to_temporary(update);

  ASSERT_TRUE(gpu_update.set(update_temp, s.get_stream()));
  ASSERT_TRUE(gpu_full_system.set(full_system_2, s.get_stream()));
  auto lambda = [&](cudaStream_t s_) {
    ASSERT_TRUE(lm_step.predict_reduction(10, 0.01, gpu_update.meta(), gpu_full_system.meta(), num_points, num_poses,
                                          prediction.ptr(), s_));
  };
  graph.launch(lambda, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("CPU_RelativeReduction");
    // calculate cpu part
    ComputePredictedRelativeReduction(10, 0.01, update, full_system);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    TRACE_EVENT ev = helper.trace_event("GPU_RelativeReduction");
    graph.launch(lambda, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);

  ReportSpeedBenchmark(duration_basic, duration_cuda, 100);
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

TEST(Cuda, SBAComputePredictedRelativeReduction) {
  const float thresh = 0.01f;
  camera::Rig rig = MakeDefaultRig();
  int max_points = 400;
  int max_poses = 5;
  cuda::sba::GPULinearSystem gpu_full_system(max_points, max_poses);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  cuda::sba::GPULevenbergMarquardtStep lm_step(max_points, max_poses);
  cuda::GPUArrayPinned<float> prediction{1};
  cuda::GPUGraph graph;
  cuda::Stream s;

  for (int i = 0; i < 100; i++) {
    sba::BundleAdjustmentProblem problem_input_cpu;
    // generate sba problem
    GenerateProblem(problem_input_cpu, rig, Matrix2T::Identity(), max_points, max_poses);
    int num_points = problem_input_cpu.points.size() - problem_input_cpu.num_fixed_points;
    int num_poses = problem_input_cpu.rig_from_world.size() - problem_input_cpu.num_fixed_key_frames;

    ModelFunction model;
    sba::schur_complement_bundler_cpu_internal::FullSystem full_system;
    ReducedSystem reduced_system;
    ParameterUpdate update, update_gpu;
    update.point.resize(num_points, Vector3T::Zero());
    update.pose.resize(num_poses, Isometry3T::Identity());
    float cpu_prediction, gpu_prediction;
    {
      // calculate cpu part
      UpdateModel(model, problem_input_cpu);
      BuildFullSystem(full_system, model, problem_input_cpu);
      BuildReducedSystem(reduced_system, full_system, 0.1f);
      ComputeUpdate(update, reduced_system, full_system);
      UpdateState(problem_input_cpu, update);
      cpu_prediction = ComputePredictedRelativeReduction(10, 0.01, update, full_system);
    }

    {
      cuda::sba::temporary::FullSystem full_system_2 = to_temporary(full_system);
      cuda::sba::temporary::ParameterUpdate update_temp = to_temporary(update);

      ASSERT_TRUE(gpu_update.set(update_temp, s.get_stream()));
      ASSERT_TRUE(gpu_full_system.set(full_system_2, s.get_stream()));
      graph.launch(
          [&](cudaStream_t s_) {
            ASSERT_TRUE(lm_step.predict_reduction(10, 0.01, gpu_update.meta(), gpu_full_system.meta(), num_points,
                                                  num_poses, prediction.ptr(), s_));
          },
          s.get_stream());
      CUDA_CHECK(
          cudaMemcpyAsync(&gpu_prediction, prediction.ptr(), sizeof(float), cudaMemcpyDeviceToHost, s.get_stream()));

      cudaStreamSynchronize(s.get_stream());
    }
    if (std::abs(cpu_prediction - gpu_prediction) >= thresh) {
      std::cout << "cpu :" << cpu_prediction << std::endl;
      std::cout << "gpu :" << gpu_prediction << std::endl;
    }
    ASSERT_TRUE(std::abs(cpu_prediction - gpu_prediction) < thresh);
  }
}

TEST(Cuda, DISABLED_SBABuildReducedSystem) {
  const int max_points = 400;
  const int max_poses = 5;
  const int num_cameras = 2;

  const float thresh = 10;

  const float threshold = 1e-4;
  const float lambda = 0.1;
  const float robustifier_scale = 1.f;
  camera::Rig rig = MakeDefaultRig();
  cuda::sba::GPUBundleAdjustmentProblem gpu_problem(max_points, max_poses, max_points * max_poses * num_cameras);
  cuda::sba::GPUModelFunction gpu_function(max_points * max_poses * num_cameras);
  cuda::sba::GPULinearSystem gpu_full_system(max_points, max_points);
  cuda::sba::GPUParameterUpdate gpu_update(max_points, max_poses);
  gpu_problem.set_rig(rig);
  cuda::Stream s;
  cuda::GPUGraph graph;

  for (int i = 0; i < 100; i++) {
    sba::BundleAdjustmentProblem problem_input1, problem_input2;
    ModelFunction mf_cpu, mf_gpu;

    // generate sba problem
    GenerateProblem(problem_input1, rig);
    problem_input2 = problem_input1;

    sba::schur_complement_bundler_cpu_internal::FullSystem fs_cpu;
    FullSystem fs_gpu;
    cuda::sba::GPULinearSystem ls_gpu(max_points, max_poses);
    sba::schur_complement_bundler_cpu_internal::ReducedSystem rs_cpu;
    cuda::sba::temporary::ReducedSystem rs_gpu;

    const int num_points = problem_input2.points.size() - problem_input2.num_fixed_points;
    const int num_poses = problem_input2.rig_from_world.size() - problem_input2.num_fixed_key_frames;

    {
      // calculate cpu part
      UpdateModel(mf_cpu, problem_input1);
      BuildFullSystem(fs_cpu, mf_cpu, problem_input1);
      BuildReducedSystem(rs_cpu, fs_cpu, lambda);

#if 0
            graph.launch([&](cudaStream_t s_) {
                //calc gpu part
#endif
      ASSERT_TRUE(gpu_problem.set(problem_input2, s.get_stream()));
      CUDA_CHECK(update_model(gpu_function.meta(), gpu_problem.meta(), robustifier_scale, s.get_stream()));
      CUDA_CHECK(build_full_system(gpu_full_system.meta(), gpu_function.meta(), gpu_problem.meta(),
                                   problem_input2.num_fixed_points, problem_input2.num_fixed_key_frames,
                                   s.get_stream()));

      CUDA_CHECK(build_reduced_system(gpu_full_system.meta(), ls_gpu.meta(), num_points, num_poses, lambda, threshold,
                                      s.get_stream()));

      ls_gpu.get(num_points, num_poses, rs_gpu, s.get_stream());

      cudaStreamSynchronize(s.get_stream());

#if 0
            }, s.get_stream());
#endif
    }

    // std::vector<Matrix3T> inverse_point_block;
    for (int k = 0; k < num_points; k++) {
      if (!rs_gpu.inverse_point_block[k].isApprox(rs_cpu.inverse_point_block[k], thresh)) {
        std::cout << "rs_cpu.inverse_point_block[" << k << "] =\n" << rs_cpu.inverse_point_block[k] << '\n';
        std::cout << "rs_gpu.inverse_point_block[" << k << "] =\n" << rs_gpu.inverse_point_block[k] << '\n';
        ASSERT_TRUE(false);
      }
    }

    // Eigen::MatrixXf camera_backsub_block;
    ASSERT_TRUE(rs_gpu.camera_backsub_block.rows() == rs_cpu.camera_backsub_block.rows());
    ASSERT_TRUE(rs_gpu.camera_backsub_block.cols() == rs_cpu.camera_backsub_block.cols());
    if (!rs_gpu.camera_backsub_block.isApprox(rs_cpu.camera_backsub_block, thresh)) {
      std::cout << "rs_cpu.camera_backsub_block =\n" << rs_cpu.camera_backsub_block << '\n';
      std::cout << "rs_gpu.camera_backsub_block =\n" << rs_gpu.camera_backsub_block << '\n';
      ASSERT_TRUE(false);
    }

    // Eigen::VectorXf point_rhs;
    ASSERT_TRUE(rs_gpu.point_rhs.size() == rs_cpu.point_rhs.size());
    if (!rs_gpu.point_rhs.isApprox(rs_cpu.point_rhs, thresh)) {
      std::cout << "rs_cpu.point_rhs =\n" << rs_cpu.point_rhs << '\n';
      std::cout << "rs_gpu.point_rhs =\n" << rs_gpu.point_rhs << '\n';
      ASSERT_TRUE(false);
    }

    // Eigen::VectorXf pose_rhs;
    ASSERT_TRUE(rs_gpu.pose_rhs.size() == rs_cpu.pose_rhs.size());
    if (!rs_gpu.pose_rhs.isApprox(rs_cpu.pose_rhs, thresh)) {
      std::cout << "rs_cpu.pose_rhs =\n" << rs_cpu.pose_rhs << '\n';
      std::cout << "rs_gpu.pose_rhs =\n" << rs_gpu.pose_rhs << '\n';
      ASSERT_TRUE(false);
    }

    // Eigen::MatrixXf pose_block;
    ASSERT_TRUE(rs_gpu.pose_block.rows() == rs_cpu.pose_block.rows());
    ASSERT_TRUE(rs_gpu.pose_block.cols() == rs_cpu.pose_block.cols());
    if (!rs_gpu.pose_block.isApprox(rs_cpu.pose_block, thresh)) {
      std::cout << "rs_cpu.pose_block =\n" << rs_cpu.pose_block << '\n';
      std::cout << "rs_gpu.pose_block =\n" << rs_gpu.pose_block << '\n';
      ASSERT_TRUE(false);
    }
  }
}  // TEST(Cuda, SBABuildFullSystem)

}  // namespace test
