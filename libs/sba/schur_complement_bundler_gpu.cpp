
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

#include "sba/schur_complement_bundler_gpu.h"

#include "common/log_types.h"
#include "common/statistic.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_sba_v1.h"
#include "cuda_modules/sba.h"
#include "math/robust_cost_function.h"
#include "math/twist.h"
#include "profiler/profiler.h"

#define RETURN_IF_FALSE(condition) \
  if (!(condition)) {              \
    return false;                  \
  }

#define RETURN_IF_FAILED(status) \
  if ((status) != cudaSuccess) { \
    return false;                \
  }

namespace {
using namespace cuvslam;

using ProfilerDomain = profiler::SBAProfiler::DomainHelper;

struct ParameterUpdate_ {
  std::vector<Isometry3T> pose;
  std::vector<Vector3T> point;

  Eigen::VectorXf pose_step;
  Eigen::VectorXf point_step;
};

}  // namespace

namespace cuvslam::sba {

class SchurComplementBundlerGpu::Impl {
public:
  Impl(int max_points, int max_poses);
  Impl() = default;
  bool solve(BundleAdjustmentProblem& problem);

  const int max_points = 3000;
  const int max_poses = 20;

private:
  // Result of the last evaluate_cost(), readable once its stream has been synchronized.
  float evaluated_cost(int num_observations) const;

  cuda::GPUArrayPinned<float> gpu_cost_{1};
  cuda::GPUArrayPinned<int> gpu_num_skipped_{1};
  cuda::GPUArrayPinned<float> predicted_reduction_{1};
  cuda::GPUArrayPinned<float> points_poses_update_max_{2};

  ProfilerDomain profiler_domain_ = ProfilerDomain("SBA GPU");
  const uint32_t profiler_color_ = 0xFF7700;

  cuda::Stream stream_;
  const int max_observations = max_points * max_poses * 2;

  cuda::sba::GPUBundleAdjustmentProblem gpu_problem_{max_points, max_poses, max_observations};
  cuda::sba::GPUModelFunction gpu_function_{max_observations};
  cuda::sba::GPUParameterUpdate gpu_update_{max_points, max_poses};
  cuda::sba::GPULinearSystem gpu_full_system_{max_points, max_poses};
  cuda::sba::GPULinearSystem gpu_reduced_system_{max_points, max_poses};

  cuda::sba::GPULevenbergMarquardtStep lm_step_{max_points, max_poses};
};

SchurComplementBundlerGpu::SchurComplementBundlerGpu() : impl(std::make_unique<Impl>()) {}

SchurComplementBundlerGpu::~SchurComplementBundlerGpu() = default;

bool SchurComplementBundlerGpu::solve(BundleAdjustmentProblem& problem) {
  const int num_points = static_cast<int>(problem.points.size());
  const int num_poses = static_cast<int>(problem.rig_from_world.size());
  if (impl->max_points < num_points || impl->max_poses < num_poses) {
    impl = std::make_unique<Impl>(2 * num_points, 2 * num_poses);
  }
  return impl->solve(problem);
}

SchurComplementBundlerGpu::Impl::Impl(int max_points, int max_poses) : max_points(max_points), max_poses(max_poses) {}

float SchurComplementBundlerGpu::Impl::evaluated_cost(int num_observations) const {
  // An observation closer than the near plane is skipped and adds nothing, so a state that hides
  // every one of them would otherwise score a perfect zero. That is infeasible, not optimal - the
  // IMU bundler reports it the same way.
  if (gpu_num_skipped_[0] == num_observations) {
    return std::numeric_limits<float>::infinity();
  }

  return gpu_cost_[0];
}

bool SchurComplementBundlerGpu::Impl::solve(BundleAdjustmentProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("SchurComplementBundlerGpu::Impl::solve()", profiler_color_);
  const int num_points = static_cast<int>(problem.points.size());
  const int num_poses = static_cast<int>(problem.rig_from_world.size());
  const int num_observations = static_cast<int>(problem.observation_xys.size());

  if (num_points == 0 || num_poses == 0 || num_observations == 0) {
    return false;
  }

  gpu_problem_.set_rig(problem.rig);
  RETURN_IF_FALSE(gpu_problem_.set(problem, stream_.get_stream()));

  {
    // initialize parameter update with default values in order to properly compute
    // initial_cost
    ParameterUpdate_ update;
    update.point.resize(num_points, Vector3T::Zero());
    update.pose.resize(num_poses, Isometry3T::Identity());

    cuvslam::cuda::sba::temporary::ParameterUpdate tpu;
    tpu.pose = update.pose;
    tpu.point = update.point;
    tpu.pose_step = update.pose_step;
    tpu.point_step = update.point_step;
    RETURN_IF_FALSE(gpu_update_.set(tpu, stream_.get_stream()));
  }

  RETURN_IF_FAILED(
      update_model(gpu_function_.meta(), gpu_problem_.meta(), problem.robustifier_scale, stream_.get_stream()));

  RETURN_IF_FAILED(build_full_system(gpu_full_system_.meta(), gpu_function_.meta(), gpu_problem_.meta(),
                                     problem.num_fixed_points, problem.num_fixed_key_frames, stream_.get_stream()));

  RETURN_IF_FAILED(evaluate_cost(gpu_cost_.ptr(), gpu_num_skipped_.ptr(), gpu_problem_.meta(), gpu_update_.meta(),
                                 problem.robustifier_scale, stream_.get_stream()));

  gpu_cost_.copy(cuda::GPUCopyDirection::ToCPU, stream_.get_stream());
  gpu_num_skipped_.copy(cuda::GPUCopyDirection::ToCPU, stream_.get_stream());
  RETURN_IF_FAILED(cudaStreamSynchronize(stream_.get_stream()));

  const float initial_cost = evaluated_cost(num_observations);
  float current_cost = initial_cost;
  problem.initial_cost = initial_cost;
  problem.last_cost = initial_cost;

  if (initial_cost < std::numeric_limits<float>::epsilon()) {
    return true;
  }

  if (initial_cost == std::numeric_limits<float>::infinity()) {
    // starting point is not feasible
    return false;
  }

  int iteration{0};
  const int max_iterations = problem.max_iterations;

  float lambda = 0.001f;

  while (iteration < max_iterations) {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("SBA iter");
    ++iteration;
    problem.iterations = iteration;

    {
      const float threshold = 1e-6f;
      RETURN_IF_FAILED(build_reduced_system(
          gpu_full_system_.meta(), gpu_reduced_system_.meta(), gpu_problem_.num_points() - problem.num_fixed_points,
          gpu_problem_.num_poses() - problem.num_fixed_key_frames, lambda, threshold, stream_.get_stream()));
    }

    RETURN_IF_FALSE(lm_step_.compute_update(
        gpu_update_.meta(), gpu_reduced_system_.meta(), gpu_problem_.num_points() - problem.num_fixed_points,
        gpu_problem_.num_poses() - problem.num_fixed_key_frames, points_poses_update_max_, stream_.get_stream()));

    RETURN_IF_FAILED(evaluate_cost(gpu_cost_.ptr(), gpu_num_skipped_.ptr(), gpu_problem_.meta(), gpu_update_.meta(),
                                   problem.robustifier_scale, stream_.get_stream()));

    RETURN_IF_FALSE(lm_step_.predict_reduction(
        current_cost, lambda, gpu_update_.meta(), gpu_full_system_.meta(), num_points - problem.num_fixed_points,
        num_poses - problem.num_fixed_key_frames, predicted_reduction_.ptr(), stream_.get_stream()));
    points_poses_update_max_.copy(cuda::GPUCopyDirection::ToCPU, stream_.get_stream());
    predicted_reduction_.copy(cuda::GPUCopyDirection::ToCPU, stream_.get_stream());
    gpu_cost_.copy(cuda::GPUCopyDirection::ToCPU, stream_.get_stream());
    gpu_num_skipped_.copy(cuda::GPUCopyDirection::ToCPU, stream_.get_stream());

    RETURN_IF_FAILED(cudaStreamSynchronize(stream_.get_stream()));

    if (!lm_step_.solve_succeeded()) {
      // A pose block that is not positive definite leaves a partial factor, and the step
      // back-substituted from it is meaningless. Damping is the cure, so reject the step.
      lambda *= 5.f;
      continue;
    }

    const float cost = evaluated_cost(num_observations);
    float predicted_relative_reduction = predicted_reduction_[0];

    if (current_cost < initial_cost * std::numeric_limits<float>::epsilon()) {
      break;
    }

    if ((predicted_relative_reduction < sqrt_epsilon()) && (points_poses_update_max_[0] < sqrt_epsilon()) &&
        (points_poses_update_max_[1] < sqrt_epsilon())) {
      break;
    }

    // We guarantee that initial cost is not infinity, and we will
    // never accept a step that leads to an infinite cost.
    // This mean that we will never have a situation when
    // cost = inf and current_cost = inf.
    // The only possible case is cost = inf (step leads outside of feasible region).
    assert(std::isfinite(current_cost));
    assert(std::isfinite(predicted_relative_reduction));
    TraceDebugIf(!std::isfinite(predicted_relative_reduction), "predicted_relative_reduction is not finite");
    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    if (rho > 0.25f) {
      if (rho > 0.75f) {
        if (lambda * 0.125f > 0.f) {
          lambda *= 0.125f;
        }
      }

      current_cost = cost;
      problem.last_cost = cost;
      RETURN_IF_FALSE(lm_step_.apply_update(
          gpu_problem_.meta(), gpu_update_.meta(), gpu_problem_.num_points() - problem.num_fixed_points,
          gpu_problem_.num_poses() - problem.num_fixed_key_frames, stream_.get_stream()));

      RETURN_IF_FAILED(
          update_model(gpu_function_.meta(), gpu_problem_.meta(), problem.robustifier_scale, stream_.get_stream()));
      RETURN_IF_FAILED(build_full_system(gpu_full_system_.meta(), gpu_function_.meta(), gpu_problem_.meta(),
                                         problem.num_fixed_points, problem.num_fixed_key_frames, stream_.get_stream()));
    } else {
      lambda *= 5.f;
    }
  }

  // this invokes cudaStreamSynchronize();
  RETURN_IF_FALSE(gpu_problem_.get(problem, stream_.get_stream()));
  log::Value<LogSba>("SBA iterations", iteration);
  return true;
}

}  // namespace cuvslam::sba
