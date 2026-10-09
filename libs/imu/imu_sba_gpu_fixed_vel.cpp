
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

#include <iostream>

#include <cuda_runtime_api.h>

#include "common/log_types.h"
#include "cuda_modules/cuda_graph_helper.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_matrix.h"
#include "cuda_modules/cuda_kernels/cuda_sba_imu_v1.h"
#include "cuda_modules/culib_helper.h"
#include "math/robust_cost_function.h"
#include "math/twist.h"

#include "imu/imu_sba_gpu.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define ROBUST_COST math::ComputeHuberLoss
#define ROBUST_WEIGHT math::ComputeDHuberLoss

namespace cuvslam::sba_imu {
using namespace cuvslam::cuda;
namespace {

bool is_fixed(const ImuBAProblem& problem, int pid) { return pid < problem.num_fixed_key_frames; }

bool both_fixed(const ImuBAProblem& problem, int pid) { return is_fixed(problem, pid) && is_fixed(problem, pid + 1); }

}  // namespace

IMUBundlerGpuFixedVel::IMUBundlerGpuFixedVel(const imu::ImuCalibration& calib)
    : calib_(calib),
      num_observations_allocated(0),
      num_poses_allocated(0),
      num_cameras_allocated(0),
      num_points_allocated(0),
      num_inertials_allocated(0),
      num_poses_opt_allocated(0) {
  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  for (int j = 0; j < 3; j++) {
    for (int k = 0; k < 3; k++) {
      calib_left_from_imu_linear.d_[j][k] = rig_from_imu.linear()(j, k);
    }
    calib_left_from_imu_translation.d_[j] = rig_from_imu.translation()(j);
  }

  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
}

IMUBundlerGpuFixedVel::~IMUBundlerGpuFixedVel() {
  if (cusolver_handle_) CUSOLVER_CHECK_NOTHROW(cusolverDnDestroy(cusolver_handle_));
  if (graph_exec_1_) CUDA_CHECK_NOTHROW(cudaGraphExecDestroy(graph_exec_1_));
  if (graph_exec_2_) CUDA_CHECK_NOTHROW(cudaGraphExecDestroy(graph_exec_2_));
  if (graph_exec_3_) CUDA_CHECK_NOTHROW(cudaGraphExecDestroy(graph_exec_3_));
}

bool IMUBundlerGpuFixedVel::solve(ImuBAProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU solve", profiler_color_);

  SetValues(problem);
  // Kernels launch one block per point or observation, and an empty grid is a CUDA launch error.
  if (num_inertials <= 0 || num_points == 0 || num_observations == 0) {
    return false;
  }

  cuvslam::cuda::Stream stream(true);
  CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle_, stream.get_stream()));

  //    std::ofstream debug_file("debug.txt", std::ios_base::out | ((call == 0) ? std::ios_base::trunc :
  //    std::ios_base::app));

  AllocateBuffers();

  CopyToDevice(problem, stream.get_stream());
  InitUpdate(stream.get_stream());

  UpdateModel(stream.get_stream());

  CUDA_CHECK(cudaMemsetAsync(working_cost->ptr(), 0, sizeof(CostResult), stream.get_stream()));
  EvaluateCost(stream.get_stream());
  working_cost->copy(cuvslam::cuda::GPUCopyDirection::ToCPU, stream.get_stream());
  CUDA_CHECK(cudaStreamSynchronize(stream.get_stream()));
  const float initial_cost = ((*working_cost)[0].num_skipped == num_observations)
                                 ? std::numeric_limits<float>::infinity()
                                 : (*working_cost)[0].cost;

  //    {
  //        std::stringstream debug_prefix;
  //        debug_prefix << "Call " << call << " start, ";
  //        debug_file << debug_prefix.str() << "initial_cost " << initial_cost << ", num_poses " << num_poses << ",
  //        num_observations "
  //            << num_observations << ", num_points " << num_points << ", num_fixed_key_frames " <<
  //            num_fixed_key_frames << ", imu_penalty " << imu_penalty
  //            << ", prior_gyro " << prior_gyro << ", prior_acc " << prior_acc << ", robustifier_scale_pose " <<
  //            robustifier_scale_pose
  //            << ", robustifier_scale " << robustifier_scale
  //            << ", gravity " << gravity.x << "," << gravity.y << "," << gravity.z <<std::endl;
  //        DebugDump(debug_prefix.str().c_str(), problem, debug_file);
  //    }

  float current_cost = initial_cost;
  problem.initial_cost = initial_cost;

  if (initial_cost < std::numeric_limits<float>::epsilon()) {
    return true;
  }

  if ((std::isnan(initial_cost)) || (initial_cost == std::numeric_limits<float>::infinity())) {
    // starting point is not feasible
    return false;
  }

  int iteration = 0;
  const int max_iterations = problem.max_iterations;

  BuildFullSystem(stream.get_stream());

  (*lambda)[0] = 1e2;

  bool graph_exec_1_need_update = true;
  bool graph_exec_2_need_update = true;
  bool graph_exec_3_need_update = true;

  while (iteration < max_iterations) {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("GPU iteration", profiler_color_);
    ++iteration;
    problem.iterations = iteration;

    if (use_cuda_graph_) {
      if (graph_exec_1_need_update) {
        graph_exec_1_need_update = false;
        cudaGraph_t graph;
        cudaGraphExecUpdateResultInfo updateResultInfo;

        {
          TRACE_EVENT ev = profiler_domain_.trace_event("Graph capture()", profiler_color_cpu_);
          CUDA_CHECK(cudaStreamBeginCapture(stream.get_stream(), cudaStreamCaptureModeGlobal));
          lambda->copy(cuvslam::cuda::GPUCopyDirection::ToGPU, stream.get_stream());
          BuildReducedSystem(stream.get_stream());
          ComputeUpdate(stream.get_stream());
          CUDA_CHECK(cudaMemsetAsync(working_cost->ptr(), 0, sizeof(CostResult), stream.get_stream()));
          EvaluateCost(stream.get_stream());
          ComputePredictedReduction(epsilon(), stream.get_stream());
          working_cost->copy(cuvslam::cuda::GPUCopyDirection::ToCPU, stream.get_stream());
          CUDA_CHECK(cudaStreamEndCapture(stream.get_stream(), &graph));
        }

        if (graph_exec_1_ != nullptr) {
          TRACE_EVENT ev = profiler_domain_.trace_event("Graph exec update()", profiler_color_cpu_);
          CUDA_CHECK(cudaGraphExecUpdate(graph_exec_1_, graph, &updateResultInfo));
        }

        if (graph_exec_1_ == nullptr || updateResultInfo.result != cudaGraphExecUpdateSuccess) {
          TRACE_EVENT ev = profiler_domain_.trace_event("Graph exec instantiate()", profiler_color_cpu_);
          if (graph_exec_1_ != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(graph_exec_1_));
          }
          CUDA_CHECK(cudaGraphInstantiate(&graph_exec_1_, graph, 0));
        }

        CUDA_CHECK(cudaGraphDestroy(graph));
      }

      CUDA_CHECK(cudaGraphLaunch(graph_exec_1_, stream.get_stream()));
    } else {
      lambda->copy(cuvslam::cuda::GPUCopyDirection::ToGPU, stream.get_stream());
      BuildReducedSystem(stream.get_stream());
      ComputeUpdate(stream.get_stream());
      CUDA_CHECK(cudaMemsetAsync(working_cost->ptr(), 0, sizeof(CostResult), stream.get_stream()));
      EvaluateCost(stream.get_stream());
      ComputePredictedReduction(epsilon(), stream.get_stream());
      working_cost->copy(cuvslam::cuda::GPUCopyDirection::ToCPU, stream.get_stream());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream.get_stream()));
    if ((*working_buffer_solver_info)[0] != 0) {
      // Damped reduced system is not positive definite: reject the step instead of propagating NaNs.
      (*lambda)[0] *= 5.f;
      continue;
    }
    float cost = ((*working_cost)[0].num_skipped == num_observations) ? std::numeric_limits<float>::infinity()
                                                                      : (*working_cost)[0].cost;
    float predicted_relative_reduction = (*working_cost)[0].predicted_reduction / current_cost;
    bool max_point_update_significant = (*working_cost)[0].point_and_pose_step_update_significant[0];
    bool max_pose_update_significant = (*working_cost)[0].point_and_pose_step_update_significant[1];

    if ((std::isnan(cost)) || (cost == std::numeric_limits<float>::infinity())) {
      return false;
    }

    if ((current_cost < initial_cost * std::numeric_limits<float>::epsilon()) ||
        ((predicted_relative_reduction < epsilon()) && (!max_point_update_significant) &&
         (!max_pose_update_significant))) {
      break;
    }

    assert(std::isfinite(current_cost));
    assert(std::isfinite(predicted_relative_reduction));
    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;
    //        {
    //            std::stringstream debug_prefix;
    //            debug_prefix << "Call " << call << " progress, ";
    //            debug_file << debug_prefix.str() << "iteration " << iteration << ", predicted_relative_reduction " <<
    //            predicted_relative_reduction
    //                << ", rho " << rho << ", current_cost " << current_cost << ", cost " << cost << ", lambda " <<
    //                lambda << std::endl;
    //        }

    if (rho > 0.25f) {
      if (rho > 0.75f) {
        if ((*lambda)[0] * 0.125f > 0.f) {
          (*lambda)[0] *= 0.125f;
        }
      }

      current_cost = cost;

      if (use_cuda_graph_) {
        if (graph_exec_2_need_update) {
          graph_exec_2_need_update = false;
          cudaGraph_t graph;
          cudaGraphExecUpdateResultInfo updateResultInfo;

          {
            TRACE_EVENT ev = profiler_domain_.trace_event("Graph capture()", profiler_color_cpu_);
            CUDA_CHECK(cudaStreamBeginCapture(stream.get_stream(), cudaStreamCaptureModeGlobal));
            UpdateState(stream.get_stream());
            problem_rig_poses_w_from_imu_linear->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToCPU, num_poses,
                                                            stream.get_stream());
            problem_rig_poses_other->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToCPU, 12 * num_poses,
                                                stream.get_stream());
            CUDA_CHECK(cudaStreamEndCapture(stream.get_stream(), &graph));
          }

          if (graph_exec_2_ != nullptr) {
            TRACE_EVENT ev = profiler_domain_.trace_event("Graph exec update()", profiler_color_cpu_);
            CUDA_CHECK(cudaGraphExecUpdate(graph_exec_2_, graph, &updateResultInfo));
          }

          if (graph_exec_2_ == nullptr || updateResultInfo.result != cudaGraphExecUpdateSuccess) {
            TRACE_EVENT ev = profiler_domain_.trace_event("Graph exec instantiate()", profiler_color_cpu_);
            if (graph_exec_2_ != nullptr) {
              CUDA_CHECK(cudaGraphExecDestroy(graph_exec_2_));
            }
            CUDA_CHECK(cudaGraphInstantiate(&graph_exec_2_, graph, 0));
          }

          CUDA_CHECK(cudaGraphDestroy(graph));
        }

        CUDA_CHECK(cudaGraphLaunch(graph_exec_2_, stream.get_stream()));
      } else {
        UpdateState(stream.get_stream());
        problem_rig_poses_w_from_imu_linear->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToCPU, num_poses,
                                                        stream.get_stream());
        problem_rig_poses_other->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToCPU, 12 * num_poses,
                                            stream.get_stream());
      }

      CUDA_CHECK(cudaStreamSynchronize(stream.get_stream()));
      UnpackPoseChangesFromDevice(problem);

      // No need to update data on GPU for the last iteration
      if (iteration == max_iterations) break;

      {
        TRACE_EVENT ev = profiler_domain_.trace_event("Reintegrate()", profiler_color_cpu_);
        for (int i = 0; i < num_poses; ++i) {
          if (!is_fixed(problem, i)) {
            Pose& p = problem.rig_poses[i];

            // TODO maybe reintegrate measurements here
            p.preintegration.SetNewBias(p.gyro_bias, p.acc_bias);

            Vector3T gyro_bias_diff, acc_bias_diff;
            p.preintegration.GetDeltaBias(gyro_bias_diff, acc_bias_diff);
            if (gyro_bias_diff.norm() > problem.reintegration_thresh ||
                acc_bias_diff.norm() > problem.reintegration_thresh) {
              p.preintegration.Reintegrate(calib_);
            }
          }
        }
      }
      PackPreintegrationData(problem);

      if (use_cuda_graph_) {
        if (graph_exec_3_need_update) {
          graph_exec_3_need_update = false;
          cudaGraph_t graph;
          cudaGraphExecUpdateResultInfo updateResultInfo;

          {
            TRACE_EVENT ev = profiler_domain_.trace_event("Graph capture()", profiler_color_cpu_);
            CUDA_CHECK(cudaStreamBeginCapture(stream.get_stream(), cudaStreamCaptureModeGlobal));
            problem_rig_poses_preint->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, problem_rig_poses_preint_elems,
                                                 stream.get_stream());
            UpdateModel(stream.get_stream());
            BuildFullSystem(stream.get_stream());
            CUDA_CHECK(cudaStreamEndCapture(stream.get_stream(), &graph));
          }

          if (graph_exec_3_ != nullptr) {
            TRACE_EVENT ev = profiler_domain_.trace_event("Graph exec update()", profiler_color_cpu_);
            CUDA_CHECK(cudaGraphExecUpdate(graph_exec_3_, graph, &updateResultInfo));
          }

          if (graph_exec_3_ == nullptr || updateResultInfo.result != cudaGraphExecUpdateSuccess) {
            TRACE_EVENT ev = profiler_domain_.trace_event("Graph exec instantiate()", profiler_color_cpu_);
            if (graph_exec_3_ != nullptr) {
              CUDA_CHECK(cudaGraphExecDestroy(graph_exec_3_));
            }
            CUDA_CHECK(cudaGraphInstantiate(&graph_exec_3_, graph, 0));
          }

          CUDA_CHECK(cudaGraphDestroy(graph));
        }

        CUDA_CHECK(cudaGraphLaunch(graph_exec_3_, stream.get_stream()));
      } else {
        problem_rig_poses_preint->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, problem_rig_poses_preint_elems,
                                             stream.get_stream());
        UpdateModel(stream.get_stream());
        BuildFullSystem(stream.get_stream());
      }
    } else {
      (*lambda)[0] *= 5.f;
    }
  }

  if (current_cost < initial_cost) {
    TRACE_EVENT ev = profiler_domain_.trace_event("Reintegrate()", profiler_color_cpu_);

    Vector3T gyro_diff, acc_diff;
    for (int i = 0; i < num_poses; i++) {
      if (is_fixed(problem, i)) {
        continue;
      }
      Pose& p = problem.rig_poses[i];
      gyro_diff = p.gyro_bias - p.preintegration.GetOriginalGyroBias();
      acc_diff = p.acc_bias - p.preintegration.GetOriginalAccBias();

      if (gyro_diff.norm() > problem.reintegration_thresh || acc_diff.norm() > problem.reintegration_thresh) {
        p.preintegration.SetNewBias(p.gyro_bias, p.acc_bias);
        p.preintegration.Reintegrate(calib_);
      }
    }
  }

  CopyPointChangesFromDevice(problem, stream.get_stream());
  return current_cost < initial_cost;
}

void IMUBundlerGpuFixedVel::EvaluateCost(cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU EvaluateCost()", profiler_color_);

  const float threshold = 1e-4f;
  CUDA_CHECK(cuvslam::cuda::sba_imu::evaluate_cost(
      problem_rig_poses_w_from_imu_linear->ptr(), problem_rig_poses_other->ptr(), update_pose_w_from_imu_linear->ptr(),
      update_pose_other->ptr(), (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JRg_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JVg_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JVa_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JPg_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JPa_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + dR_offset_),
      problem_rig_poses_preint->ptr() + gyro_bias_offset_, problem_rig_poses_preint->ptr() + acc_bias_offset_,
      problem_rig_poses_preint->ptr() + dV_offset_, problem_rig_poses_preint->ptr() + dP_offset_,
      problem_rig_poses_preint->ptr() + dT_s_offset_,
      (cuvslam::cuda::Matf99*)(problem_rig_poses_preint->ptr() + info_matrix_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + acc_random_walk_accum_info_matrix_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + gyro_random_walk_accum_info_matrix_offset_),
      problem_point_ids->ptr(), problem_pose_ids->ptr(), problem_camera_ids->ptr(), problem_points->ptr(),
      update_point->ptr(), problem_rig_camera_from_rig_linear->ptr(), problem_rig_camera_from_rig_translation->ptr(),
      problem_observation_xys->ptr(), problem_observation_infos->ptr(), working_imu_from_w_linear->ptr(),
      working_imu_from_w_translation->ptr(), &(working_cost->ptr()->cost), &(working_cost->ptr()->num_skipped),
      working_partial_costs->ptr(), threshold, num_poses, num_observations, num_fixed_key_frames, prior_gyro, prior_acc,
      gravity, imu_penalty, boundary_imu_penalty, acc_rw_penalty, robustifier_scale_pose, robustifier_scale,
      calib_left_from_imu_linear, calib_left_from_imu_translation, s));
}

void IMUBundlerGpuFixedVel::ComputeUpdate(cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU ComputeUpdate()", profiler_color_);

  // Cholesky factorization
  CUSOLVER_CHECK(cusolverDnSpotrf(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, 15 * num_poses_opt,
                                  reduced_system_pose_block->ptr(), 15 * num_poses_opt, working_buffer_solver->ptr(),
                                  working_buffer_solver->size(), working_buffer_solver_info->ptr()));
  // potrs below overwrites the info value, so fetch the factorization status now.
  working_buffer_solver_info->copy(cuvslam::cuda::GPUCopyDirection::ToCPU, s);

  CUSOLVER_CHECK(cusolverDnSpotrs(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, 15 * num_poses_opt, 1,
                                  reduced_system_pose_block->ptr(), 15 * num_poses_opt,
                                  reduced_system_pose_rhs_and_update_pose_step->ptr(), 15 * num_poses_opt,
                                  working_buffer_solver_info->ptr()));

  CUDA_CHECK(cuvslam::cuda::sba_imu::calc_update(
      reduced_system_point_rhs->ptr(), reduced_system_pose_rhs_and_update_pose_step->ptr(),
      reduced_system_camera_backsub_block_transposed->ptr(), 3 * num_points * sizeof(float), update_point_step->ptr(),
      update_point->ptr(), update_pose_w_from_imu_linear->ptr(), update_pose_other->ptr(), num_points, num_poses_opt,
      s));
}

void IMUBundlerGpuFixedVel::UpdateState(cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU UpdateState()", profiler_color_);

  CUDA_CHECK(cuvslam::cuda::sba_imu::update_state(
      update_point->ptr(), update_pose_w_from_imu_linear->ptr(), update_pose_other->ptr(), problem_points->ptr(),
      problem_rig_poses_w_from_imu_linear->ptr(), problem_rig_poses_other->ptr(), num_points, num_poses,
      num_fixed_key_frames, s));
}

void IMUBundlerGpuFixedVel::ComputePredictedReduction(float max_abs_update_epsilon, cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU ComputePredictedReduction()", profiler_color_);

  CUDA_CHECK(cuvslam::cuda::sba_imu::compute_predicted_reduction(
      full_system_point_block->ptr(), update_point_step->ptr(), reduced_system_pose_rhs_and_update_pose_step->ptr(),
      full_system_point_pose_block_transposed->ptr(), 3 * num_points * sizeof(float), full_system_pose_block->ptr(),
      15 * num_poses_opt * sizeof(float), &(working_cost->ptr()->predicted_reduction),
      &(working_cost->ptr()->point_and_pose_step_update_significant[0]), num_points, num_poses_opt, lambda->ptr(),
      max_abs_update_epsilon, s));
}

void IMUBundlerGpuFixedVel::UpdateModel(cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU UpdateModel()", profiler_color_);

  const float threshold = 1e-4f;
  CUDA_CHECK(cuvslam::cuda::sba_imu::update_model(
      problem_point_ids->ptr(), problem_pose_ids->ptr(), problem_camera_ids->ptr(), problem_points->ptr(),
      problem_rig_poses_w_from_imu_linear->ptr(), problem_rig_poses_other->ptr(),
      problem_rig_camera_from_rig_linear->ptr(), problem_rig_camera_from_rig_translation->ptr(),
      problem_observation_xys->ptr(), problem_observation_infos->ptr(),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JRg_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JVg_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JVa_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JPg_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + JPa_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + dR_offset_),
      problem_rig_poses_preint->ptr() + gyro_bias_offset_, problem_rig_poses_preint->ptr() + acc_bias_offset_,
      problem_rig_poses_preint->ptr() + gyro_bias_diff_offset_, problem_rig_poses_preint->ptr() + dV_offset_,
      problem_rig_poses_preint->ptr() + dP_offset_, problem_rig_poses_preint->ptr() + dT_s_offset_,
      model_reprojection_residuals->ptr(), model_repr_robustifier_weights->ptr(), model_repr_jacobians_jt->ptr(),
      model_repr_jacobians_jr->ptr(), model_repr_jacobians_jp->ptr(), model_inertial_residuals->ptr(),
      model_random_walk_gyro_residuals->ptr(), model_random_walk_acc_residuals->ptr(),
      model_inertial_jacobians_jr_left->ptr(), model_inertial_jacobians_jt_left->ptr(),
      model_inertial_jacobians_jv_left->ptr(), model_inertial_jacobians_jb_acc_left->ptr(),
      model_inertial_jacobians_jb_gyro_left->ptr(), model_inertial_jacobians_jr_right->ptr(),
      model_inertial_jacobians_jt_right->ptr(), model_inertial_jacobians_jv_right->ptr(), threshold, gravity,
      num_observations, num_poses, num_fixed_key_frames, robustifier_scale, calib_left_from_imu_linear,
      calib_left_from_imu_translation, s));
}

void IMUBundlerGpuFixedVel::BuildFullSystem(cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU BuildFullSystem()", profiler_color_);

  CUDA_CHECK(cuvslam::cuda::sba_imu::build_full_system(
      problem_point_num_observations->ptr(), problem_point_start_observation_id->ptr(),
      problem_point_observation_ids->ptr(), problem_pose_num_observations->ptr(),
      problem_pose_start_observation_id->ptr(), problem_pose_observation_ids->ptr(),
      model_repr_robustifier_weights->ptr(), problem_observation_infos->ptr(), model_repr_jacobians_jp->ptr(),
      model_repr_jacobians_jr->ptr(), model_repr_jacobians_jt->ptr(), model_reprojection_residuals->ptr(),
      problem_pose_ids->ptr(), (cuvslam::cuda::Matf99*)(problem_rig_poses_preint->ptr() + info_matrix_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + acc_random_walk_accum_info_matrix_offset_),
      (cuvslam::cuda::Matf33*)(problem_rig_poses_preint->ptr() + gyro_random_walk_accum_info_matrix_offset_),
      model_inertial_residuals->ptr(), model_inertial_jacobians_jr_left->ptr(), model_inertial_jacobians_jt_left->ptr(),
      model_inertial_jacobians_jv_left->ptr(), model_inertial_jacobians_jb_gyro_left->ptr(),
      model_inertial_jacobians_jb_acc_left->ptr(), model_inertial_jacobians_jr_right->ptr(),
      model_inertial_jacobians_jt_right->ptr(), model_inertial_jacobians_jv_right->ptr(),
      model_random_walk_gyro_residuals->ptr(), model_random_walk_acc_residuals->ptr(), problem_rig_poses_other->ptr(),
      full_system_point_block->ptr(), full_system_point_rhs->ptr(), full_system_point_pose_block_transposed->ptr(),
      3 * num_points * sizeof(float), full_system_pose_block->ptr(), 15 * num_poses_opt * sizeof(float),
      full_system_pose_rhs->ptr(), num_observations, num_points, num_poses, num_fixed_key_frames,
      robustifier_scale_pose, imu_penalty, boundary_imu_penalty, acc_rw_penalty, prior_gyro, prior_acc, s));
}

void IMUBundlerGpuFixedVel::BuildReducedSystem(cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU BuildReducedSystem()", profiler_color_);

  const float threshold = 1e-6f;
  CUDA_CHECK(cuvslam::cuda::sba_imu::build_reduced_system(
      full_system_point_block->ptr(), full_system_point_rhs->ptr(), full_system_point_pose_block_transposed->ptr(),
      3 * num_points * sizeof(float), full_system_pose_block->ptr(), 15 * num_poses_opt * sizeof(float),
      full_system_pose_rhs->ptr(), reduced_system_point_rhs->ptr(),
      reduced_system_camera_backsub_block_transposed->ptr(), 3 * num_points * sizeof(float),
      reduced_system_pose_block->ptr(), 15 * num_poses_opt * sizeof(float),
      reduced_system_pose_rhs_and_update_pose_step->ptr(), lambda->ptr(), threshold, num_points, num_poses_opt, s));
}

void IMUBundlerGpuFixedVel::SetValues(const ImuBAProblem& problem) {
  num_observations = static_cast<int>(problem.observation_xys.size());
  num_poses = static_cast<int>(problem.rig_poses.size());
  num_cameras = problem.rig.num_cameras;
  num_points = static_cast<int>(problem.points.size());
  num_inertials = num_poses - 1;
  num_fixed_key_frames = problem.num_fixed_key_frames;
  num_poses_opt = num_poses - num_fixed_key_frames;
  gravity = float3{problem.gravity[0], problem.gravity[1], problem.gravity[2]};
  robustifier_scale_pose = problem.robustifier_scale_pose;
  robustifier_scale = problem.robustifier_scale;
  prior_gyro = problem.prior_gyro;
  prior_acc = problem.prior_acc;
  imu_penalty = problem.imu_penalty;
  boundary_imu_penalty = problem.boundary_imu_penalty;
  acc_rw_penalty = problem.acc_rw_penalty;
}

void IMUBundlerGpuFixedVel::AllocateBuffers() {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU AllocateBuffers()", profiler_color_);

  if (num_observations > num_observations_allocated) {
    problem_point_ids = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(num_observations);
    problem_pose_ids = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(num_observations);
    problem_camera_ids = std::make_unique<cuvslam::cuda::GPUArrayPinned<int8_t>>(num_observations);
    problem_observation_xys = std::make_unique<cuvslam::cuda::GPUArrayPinned<float>>(2 * num_observations);
    problem_observation_infos =
        std::make_unique<cuvslam::cuda::GPUArrayPinned<cuvslam::cuda::Matf22>>(num_observations);
    problem_point_observation_ids = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(num_observations);
    problem_pose_observation_ids = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(num_observations);
    model_repr_jacobians_jr = std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf23>>(num_observations);
    model_repr_jacobians_jt = std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf23>>(num_observations);
    model_repr_jacobians_jp = std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf23>>(num_observations);
    model_repr_robustifier_weights = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(num_observations);
    model_reprojection_residuals = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(2 * num_observations);
    working_partial_costs = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>((num_observations + 31) / 32);
    num_observations_allocated = num_observations;
  }

  if (num_poses > num_poses_allocated) {
    problem_rig_poses_w_from_imu_linear =
        std::make_unique<cuvslam::cuda::GPUArrayPinned<cuvslam::cuda::Matf33>>(num_poses);
    problem_rig_poses_other = std::make_unique<cuvslam::cuda::GPUArrayPinned<float>>(12 * num_poses);
    working_imu_from_w_linear = std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf33>>(num_poses);
    working_imu_from_w_translation = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_poses);

    int current_offset = 0;
    info_matrix_offset_ = current_offset;
    current_offset += 9 * 9 * num_poses;
    acc_random_walk_accum_info_matrix_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    gyro_random_walk_accum_info_matrix_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    gyro_random_walk_accum_info_matrix_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    JRg_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    JVg_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    JVa_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    JPg_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    JPa_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    dR_offset_ = current_offset;
    current_offset += 3 * 3 * num_poses;
    gyro_bias_offset_ = current_offset;
    current_offset += 3 * num_poses;
    acc_bias_offset_ = current_offset;
    current_offset += 3 * num_poses;
    gyro_bias_diff_offset_ = current_offset;
    current_offset += 3 * num_poses;
    dV_offset_ = current_offset;
    current_offset += 3 * num_poses;
    dP_offset_ = current_offset;
    current_offset += 3 * num_poses;
    dT_s_offset_ = current_offset;
    current_offset += num_poses;
    problem_rig_poses_preint_elems = current_offset;
    problem_rig_poses_preint = std::make_unique<cuvslam::cuda::GPUArrayPinned<float>>(problem_rig_poses_preint_elems);

    num_poses_allocated = num_poses;
  }

  if (num_cameras > num_cameras_allocated) {
    problem_rig_camera_from_rig_linear =
        std::make_unique<cuvslam::cuda::GPUArrayPinned<cuvslam::cuda::Matf33>>(num_cameras);
    problem_rig_camera_from_rig_translation = std::make_unique<cuvslam::cuda::GPUArrayPinned<float>>(3 * num_cameras);
    num_cameras_allocated = num_cameras;
  }

  if (num_points > num_points_allocated) {
    problem_points = std::make_unique<cuvslam::cuda::GPUArrayPinned<float>>(3 * num_points);
    problem_point_num_observations = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(num_points);
    problem_point_start_observation_id = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(num_points);
    update_point = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_points);
    update_point_step = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_points);
    full_system_point_block = std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf33>>(num_points);
    full_system_point_rhs = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_points);
    reduced_system_point_rhs = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_points);
    num_points_allocated = num_points;
  }

  if (num_poses_opt > num_poses_opt_allocated) {
    problem_pose_num_observations = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(num_poses_opt);
    problem_pose_start_observation_id = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(num_poses_opt);
    update_pose_w_from_imu_linear = std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf33>>(num_poses_opt);
    update_pose_other = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(12 * num_poses_opt);
    full_system_pose_block =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(15 * num_poses_opt * 15 * num_poses_opt);
    full_system_pose_rhs = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(15 * num_poses_opt);
    reduced_system_pose_block =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(15 * num_poses_opt * 15 * num_poses_opt);
    reduced_system_pose_rhs_and_update_pose_step =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(15 * num_poses_opt);
    num_poses_opt_allocated = num_poses_opt;
  }

  if (num_inertials > num_inertials_allocated) {
    model_inertial_jacobians_jr_left =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>>(num_inertials);
    model_inertial_jacobians_jt_left =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>>(num_inertials);
    model_inertial_jacobians_jv_left =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>>(num_inertials);
    model_inertial_jacobians_jb_acc_left =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>>(num_inertials);
    model_inertial_jacobians_jb_gyro_left =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>>(num_inertials);
    model_inertial_jacobians_jr_right =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>>(num_inertials);
    model_inertial_jacobians_jt_right =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>>(num_inertials);
    model_inertial_jacobians_jv_right =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>>(num_inertials);
    model_random_walk_gyro_residuals = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_inertials);
    model_random_walk_acc_residuals = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_inertials);
    model_inertial_residuals = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(9 * num_inertials);
    num_inertials_allocated = num_inertials;
  }

  if (!lambda) {
    lambda = std::make_unique<cuvslam::cuda::GPUArrayPinned<float>>(1);
  }

  if (!working_cost) {
    working_cost = std::make_unique<cuvslam::cuda::GPUArrayPinned<CostResult>>(1);
  }

  if ((!full_system_point_pose_block_transposed) ||
      ((int)(full_system_point_pose_block_transposed->size()) < 3 * num_points * 6 * num_poses_opt)) {
    full_system_point_pose_block_transposed =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_points * 6 * num_poses_opt);
  }

  if ((!reduced_system_camera_backsub_block_transposed) ||
      ((int)(reduced_system_camera_backsub_block_transposed->size()) < 3 * num_points * 6 * num_poses_opt)) {
    reduced_system_camera_backsub_block_transposed =
        std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(3 * num_points * 6 * num_poses_opt);
  }

  if (!working_buffer_solver_info) {
    working_buffer_solver_info = std::make_unique<cuvslam::cuda::GPUArrayPinned<int>>(1);
  }

  {
    int new_buffer_size;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, 15 * num_poses_opt,
                                               reduced_system_pose_block->ptr(), 15 * num_poses_opt, &new_buffer_size));
    if ((!working_buffer_solver) || (new_buffer_size > (int)(working_buffer_solver->size()))) {
      working_buffer_solver = std::make_unique<cuvslam::cuda::GPUOnlyArray<float>>(new_buffer_size);
    }
  }
}

void IMUBundlerGpuFixedVel::CopyToDevice(const ImuBAProblem& problem, cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU CopyToDevice()", profiler_color_);

  std::vector<int> point_num_observations(num_points);
  std::vector<int> point_start_observation_id(num_points);
  std::vector<int> point_observation_ids(num_observations);
  {
    for (int obs = 0; obs < num_observations; obs++) {
      int point_idx = problem.point_ids[obs];
      point_num_observations[point_idx]++;
    }
    int cur = 0;
    for (int point_idx = 0; point_idx < num_points; point_idx++) {
      point_start_observation_id[point_idx] = cur;
      cur += point_num_observations[point_idx];
    }
    std::vector<int> point_current_observation_ids = point_start_observation_id;
    for (int obs = 0; obs < num_observations; obs++) {
      int point_idx = problem.point_ids[obs];
      point_observation_ids[point_current_observation_ids[point_idx]] = obs;
      point_current_observation_ids[point_idx]++;
    }
    // For each point sort obervations in the order of increasing pose_id
    for (int point_idx = 0; point_idx < num_points; point_idx++) {
      std::sort(
          point_observation_ids.begin() + point_start_observation_id[point_idx],
          point_observation_ids.begin() + point_start_observation_id[point_idx] + point_num_observations[point_idx],
          [&problem](int x, int y) { return problem.pose_ids[x] < problem.pose_ids[y]; });
    }
  }

  std::vector<int> pose_num_observations(num_poses_opt);
  std::vector<int> pose_start_observation_id(num_poses_opt);
  std::vector<int> pose_observation_ids(num_observations);
  {
    for (int obs = 0; obs < num_observations; obs++) {
      int pose_idx = problem.pose_ids[obs];
      if (pose_idx >= problem.num_fixed_key_frames) {
        pose_idx -= problem.num_fixed_key_frames;
        pose_num_observations[pose_idx]++;
      }
    }
    int cur = 0;
    for (int pose_idx = 0; pose_idx < num_poses_opt; pose_idx++) {
      pose_start_observation_id[pose_idx] = cur;
      cur += pose_num_observations[pose_idx];
    }
    std::vector<int> pose_current_observation_ids = pose_start_observation_id;
    for (int obs = 0; obs < num_observations; obs++) {
      int pose_idx = problem.pose_ids[obs];
      if (pose_idx >= problem.num_fixed_key_frames) {
        pose_idx -= problem.num_fixed_key_frames;
        pose_observation_ids[pose_current_observation_ids[pose_idx]] = obs;
        pose_current_observation_ids[pose_idx]++;
      }
    }
  }

  for (int i = 0; i < num_observations; i++) {
    problem_point_ids->operator[](i) = problem.point_ids[i];
    problem_pose_ids->operator[](i) = problem.pose_ids[i];
    problem_camera_ids->operator[](i) = problem.camera_ids[i];
    problem_point_observation_ids->operator[](i) = point_observation_ids[i];
    problem_pose_observation_ids->operator[](i) = pose_observation_ids[i];
    for (int j = 0; j < 2; j++) {
      problem_observation_xys->operator[](2 * i + j) = problem.observation_xys[i][j];
      for (int k = 0; k < 2; k++) {
        problem_observation_infos->operator[](i)[j][k] = problem.observation_infos[i](j, k);
      }
    }
  }
  problem_point_ids->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_observations, s);
  problem_pose_ids->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_observations, s);
  problem_camera_ids->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_observations, s);
  problem_observation_xys->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, 2 * num_observations, s);
  problem_observation_infos->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_observations, s);
  problem_point_observation_ids->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_observations, s);
  problem_pose_observation_ids->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_observations, s);

  for (int i = 0; i < num_poses; ++i) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        problem_rig_poses_w_from_imu_linear->operator[](i)[j][k] = problem.rig_poses[i].w_from_imu.linear()(j, k);
      }
      problem_rig_poses_other->operator[](12 * i + j) = problem.rig_poses[i].w_from_imu.translation()[j];
      problem_rig_poses_other->operator[](12 * i + 3 + j) = problem.rig_poses[i].velocity[j];
      problem_rig_poses_other->operator[](12 * i + 6 + j) = problem.rig_poses[i].gyro_bias[j];
      problem_rig_poses_other->operator[](12 * i + 9 + j) = problem.rig_poses[i].acc_bias[j];
    }
  }
  problem_rig_poses_w_from_imu_linear->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_poses, s);
  problem_rig_poses_other->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, 12 * num_poses, s);
  PackPreintegrationData(problem);
  problem_rig_poses_preint->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, problem_rig_poses_preint_elems, s);

  for (int i = 0; i < num_cameras; ++i) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        problem_rig_camera_from_rig_linear->operator[](i)[j][k] = problem.rig.camera_from_rig[i].linear()(j, k);
      }
      problem_rig_camera_from_rig_translation->operator[](3 * i + j) = problem.rig.camera_from_rig[i].translation()[j];
    }
  }
  problem_rig_camera_from_rig_linear->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_cameras, s);
  problem_rig_camera_from_rig_translation->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, 3 * num_cameras, s);

  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < 3; j++) {
      problem_points->operator[](3 * i + j) = problem.points[i][j];
    }
    problem_point_num_observations->operator[](i) = point_num_observations[i];
    problem_point_start_observation_id->operator[](i) = point_start_observation_id[i];
  }
  problem_points->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, 3 * num_points, s);
  problem_point_num_observations->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_points, s);
  problem_point_start_observation_id->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_points, s);

  for (int i = 0; i < num_poses_opt; i++) {
    problem_pose_num_observations->operator[](i) = pose_num_observations[i];
    problem_pose_start_observation_id->operator[](i) = pose_start_observation_id[i];
  }
  problem_pose_num_observations->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_poses_opt, s);
  problem_pose_start_observation_id->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToGPU, num_poses_opt, s);
}

void IMUBundlerGpuFixedVel::InitUpdate(cudaStream_t s) {
  CUDA_CHECK(cuvslam::cuda::sba_imu::init_update(update_pose_w_from_imu_linear->ptr(), update_pose_other->ptr(),
                                                 update_point->ptr(), num_poses_opt, num_points, s));
}

void IMUBundlerGpuFixedVel::PackPreintegrationData(const ImuBAProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU CopyPreintegrationToDevice()", profiler_color_cpu_);

  for (int i = 0; i < num_poses; ++i) {
    Matrix9T info_matrix;
    problem.rig_poses[i].preintegration.InfoMatrix(info_matrix);
    Matrix3T acc_random_walk_accum_info_matrix;
    problem.rig_poses[i].preintegration.InfoAccRWMatrix(acc_random_walk_accum_info_matrix);
    Matrix3T gyro_random_walk_accum_info_matrix;
    problem.rig_poses[i].preintegration.InfoGyroRWMatrix(gyro_random_walk_accum_info_matrix);
    Vector3T gyro_bias = problem.rig_poses[i].preintegration.GetOriginalGyroBias();
    Vector3T acc_bias = problem.rig_poses[i].preintegration.GetOriginalAccBias();
    Vector3T gyro_bias_delta, acc_bias_delta;
    problem.rig_poses[i].preintegration.GetDeltaBias(gyro_bias_delta, acc_bias_delta);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        ((cuvslam::cuda::Matf33*)(&problem_rig_poses_preint->operator[](JRg_offset_)))[i][j][k] =
            problem.rig_poses[i].preintegration.JRg(j, k);
        ((cuvslam::cuda::Matf33*)(&problem_rig_poses_preint->operator[](JVg_offset_)))[i][j][k] =
            problem.rig_poses[i].preintegration.JVg(j, k);
        ((cuvslam::cuda::Matf33*)(&problem_rig_poses_preint->operator[](JVa_offset_)))[i][j][k] =
            problem.rig_poses[i].preintegration.JVa(j, k);
        ((cuvslam::cuda::Matf33*)(&problem_rig_poses_preint->operator[](JPg_offset_)))[i][j][k] =
            problem.rig_poses[i].preintegration.JPg(j, k);
        ((cuvslam::cuda::Matf33*)(&problem_rig_poses_preint->operator[](JPa_offset_)))[i][j][k] =
            problem.rig_poses[i].preintegration.JPa(j, k);
        ((cuvslam::cuda::Matf33*)(&problem_rig_poses_preint->operator[](dR_offset_)))[i][j][k] =
            problem.rig_poses[i].preintegration.dR(j, k);
        ((cuvslam::cuda::Matf33*)(&problem_rig_poses_preint->operator[](
            acc_random_walk_accum_info_matrix_offset_)))[i][j][k] = acc_random_walk_accum_info_matrix(j, k);
        ((cuvslam::cuda::Matf33*)(&problem_rig_poses_preint->operator[](
            gyro_random_walk_accum_info_matrix_offset_)))[i][j][k] = gyro_random_walk_accum_info_matrix(j, k);
      }
      problem_rig_poses_preint->operator[](gyro_bias_offset_ + 3 * i + j) = gyro_bias[j];
      problem_rig_poses_preint->operator[](acc_bias_offset_ + 3 * i + j) = acc_bias[j];
      problem_rig_poses_preint->operator[](gyro_bias_diff_offset_ + 3 * i + j) = gyro_bias_delta[j];
      problem_rig_poses_preint->operator[](dV_offset_ + 3 * i + j) = problem.rig_poses[i].preintegration.dV[j];
      problem_rig_poses_preint->operator[](dP_offset_ + 3 * i + j) = problem.rig_poses[i].preintegration.dP[j];
    }
    problem_rig_poses_preint->operator[](dT_s_offset_ + i) = problem.rig_poses[i].preintegration.dT_s;
    for (int j = 0; j < 9; j++) {
      for (int k = 0; k < 9; k++) {
        ((cuvslam::cuda::Matf99*)(&problem_rig_poses_preint->operator[](info_matrix_offset_)))[i][j][k] =
            info_matrix(j, k);
      }
    }
  }
}

void IMUBundlerGpuFixedVel::CopyPointChangesFromDevice(ImuBAProblem& problem, cudaStream_t s) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU CopyPointChangesFromDevice()", profiler_color_);

  problem_points->copy_top_n(cuvslam::cuda::GPUCopyDirection::ToCPU, 3 * num_points, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  for (int i = 0; i < num_points; i++) {
    for (int j = 0; j < 3; j++) {
      problem.points[i](j) = problem_points->operator[](i * 3 + j);
    }
  }
}

void IMUBundlerGpuFixedVel::UnpackPoseChangesFromDevice(ImuBAProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("GPU CopyPoseChangesFromDevice()", profiler_color_cpu_);

  for (int i = num_fixed_key_frames; i < num_poses; ++i) {
    const auto& mat = problem_rig_poses_w_from_imu_linear->operator[](i);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        problem.rig_poses[i].w_from_imu.linear()(j, k) = mat[j][k];
      }
      problem.rig_poses[i].w_from_imu.translation()[j] = problem_rig_poses_other->operator[](12 * i + j);
      problem.rig_poses[i].velocity[j] = problem_rig_poses_other->operator[](12 * i + 3 + j);
      problem.rig_poses[i].gyro_bias[j] = problem_rig_poses_other->operator[](12 * i + 6 + j);
      problem.rig_poses[i].acc_bias[j] = problem_rig_poses_other->operator[](12 * i + 9 + j);
    }
    problem.rig_poses[i].w_from_imu.makeAffine();
  }
}
/*
void IMUBundlerGpuFixedVel::DebugDump(const char * prefix, ImuBAProblem& problem, std::ostream& out) const
{
    for (int i = 0; i < problem.points.size(); i++) {
        out << prefix << "problem.points " << i << " :";
        for (int j = 0; j < 3; j++) {
            out << " " << problem.points[i](j);
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.rig_poses.size(); ++i) {
        out << prefix << "problem.rig_poses.w_from_imu linear " << i << " :";
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                out << " " << problem.rig_poses[i].w_from_imu.linear()(j, k);
            }
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.rig_poses.size(); ++i) {
        out << prefix << "problem.rig_poses.w_from_imu translation " << i << " :";
        for (int j = 0; j < 3; j++) {
            out <<  " " << problem.rig_poses[i].w_from_imu.translation()[j];
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.rig_poses.size(); ++i) {
        out << prefix << "problem.rig_poses.velocity " << i << " :";
        for (int j = 0; j < 3; j++) {
            out <<  " " << problem.rig_poses[i].velocity[j];
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.rig_poses.size(); ++i) {
        out << prefix << "problem.rig_poses.gyro_bias " << i << " :";
        for (int j = 0; j < 3; j++) {
            out <<  " " << problem.rig_poses[i].gyro_bias[j];
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.rig_poses.size(); ++i) {
        out << prefix << "problem.rig_poses.acc_bias " << i << " :";
        for (int j = 0; j < 3; j++) {
            out <<  " " << problem.rig_poses[i].acc_bias[j];
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.observation_uvs.size(); ++i) {
        out << prefix << "problem.observation_uvs " << i << " :";
        for (int j = 0; j < 2; j++) {
            out <<  " " << problem.observation_uvs[i][j];
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.observation_infos.size(); ++i) {
        out << prefix << "problem.observation_infos " << i << " :";
        for (int j = 0; j < 2; j++) {
            out <<  " " << problem.observation_infos[i][j];
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.info_matrix.size(); ++i) {
        out << prefix << "problem.info_matrix " << i << " :";
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                out << " " << problem.info_matrix[i](j, k);
            }
        }
        out << std::endl;
    }
    for (int i = 0; i < problem.point_ids.size(); ++i) {
        out << prefix << "problem.point_ids " << i << " : " << problem.point_ids[i] << std::endl;
    }
    for (int i = 0; i < problem.pose_ids.size(); ++i) {
        out << prefix << "problem.pose_ids " << i << " : " << problem.pose_ids[i] << std::endl;
    }
    for (int i = 0; i < problem.camera_ids.size(); ++i) {
        out << prefix << "problem.camera_ids " << i << " : " << (int)(problem.camera_ids[i]) << std::endl;
    }
    for (int i = 0; i < problem.rig_poses.size(); ++i) {
        out << prefix << "problem.rig_poses.info " << i << " :";
        for (int j = 0; j < 15; j++) {
            for (int k = 0; k < 15; k++) {
                out << " " << problem.rig_poses[i].info(j, k);
            }
        }
        out << std::endl;
    }
    for(int i = 0; i < num_cameras; ++i) {
        out << prefix << "problem.rig.camera_from_rig linear " << i << " :";
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                out << " " << problem.rig.camera_from_rig[i].linear()(j, k);
            }
        }
        out << std::endl;
    }
    for (int i = 0; i < num_cameras; ++i) {
        out << prefix << "problem.rig.camera_from_rig translation " << i << " :";
        for (int j = 0; j < 3; j++) {
            out <<  " " << problem.rig.camera_from_rig[i].translation()[j];
        }
        out << std::endl;
    }
}
*/
}  // namespace cuvslam::sba_imu
