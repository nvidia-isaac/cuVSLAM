
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

#include <ostream>
#include <tuple>
#include <vector>

#include <cusolverDn.h>

#include "common/imu_calibration.h"
#include "cuda_modules/sba.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "imu/imu_sba_problem.h"

namespace {
using ProfilerDomain = cuvslam::profiler::SBAProfiler::DomainHelper;
}

namespace cuvslam::sba_imu {

// This is a more or less classic version of bundle adjustment.
// Each step of Levenberg-Marquardt algorithm solves a sparse linear
// system using Schur complement trick.
class IMUBundlerGpuFixedVel {
public:
  IMUBundlerGpuFixedVel(const imu::ImuCalibration& calib);
  virtual ~IMUBundlerGpuFixedVel();

  bool solve(ImuBAProblem& problem);

private:
  imu::ImuCalibration calib_;

  void EvaluateCost(cudaStream_t s);
  void ComputeUpdate(cudaStream_t s);
  void UpdateState(cudaStream_t s);
  void ComputePredictedReduction(float max_abs_update_epsilon, cudaStream_t s);
  void UpdateModel(cudaStream_t s);
  void BuildFullSystem(cudaStream_t s);
  void BuildReducedSystem(cudaStream_t s);

  ProfilerDomain profiler_domain_ = ProfilerDomain("Inertial SBA GPU");
  const uint32_t profiler_color_ = 0x00FF77;
  const uint32_t profiler_color_cpu_ = 0xFF7700;

  void SetValues(const ImuBAProblem& problem);

  void AllocateBuffers();

  void CopyToDevice(const ImuBAProblem& problem, cudaStream_t s);

  void InitUpdate(cudaStream_t s);

  void PackPreintegrationData(const ImuBAProblem& problem);

  void CopyPointChangesFromDevice(ImuBAProblem& problem, cudaStream_t s);

  void UnpackPoseChangesFromDevice(ImuBAProblem& problem);

  //    void DebugDump(const char * prefix, ImuBAProblem& problem, std::ostream& out) const;

  struct CostResult {
    float cost;
    int num_skipped;
    float predicted_reduction;
    int point_and_pose_step_update_significant[2];
  };

private:
  cusolverDnHandle_t cusolver_handle_ = nullptr;
  bool use_cuda_graph_ = false;
  cudaGraphExec_t graph_exec_1_ = nullptr;
  cudaGraphExec_t graph_exec_2_ = nullptr;
  cudaGraphExec_t graph_exec_3_ = nullptr;

  int num_observations_allocated;
  int num_observations;
  int num_poses_allocated;
  int num_poses;
  int num_cameras_allocated;
  int num_cameras;
  int num_points_allocated;
  int num_points;
  int num_inertials_allocated;
  int num_inertials;
  int num_fixed_key_frames;
  int num_poses_opt_allocated;
  int num_poses_opt;
  float3 gravity;
  float robustifier_scale_pose;
  float robustifier_scale;
  float prior_gyro;
  float prior_acc;
  float imu_penalty;
  float boundary_imu_penalty;
  float acc_rw_penalty;

  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> problem_point_ids;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> problem_pose_ids;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int8_t>> problem_camera_ids;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<float>> problem_points;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<cuvslam::cuda::Matf33>> problem_rig_poses_w_from_imu_linear;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<float>> problem_rig_poses_other;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<cuvslam::cuda::Matf33>> problem_rig_camera_from_rig_linear;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<float>> problem_rig_camera_from_rig_translation;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<float>> problem_observation_xys;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<cuvslam::cuda::Matf22>> problem_observation_infos;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> problem_point_num_observations;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> problem_point_start_observation_id;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> problem_point_observation_ids;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> problem_pose_num_observations;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> problem_pose_start_observation_id;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> problem_pose_observation_ids;

  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<float>> problem_rig_poses_preint;
  int info_matrix_offset_;
  int acc_random_walk_accum_info_matrix_offset_;
  int gyro_random_walk_accum_info_matrix_offset_;
  int JRg_offset_;
  int JVg_offset_;
  int JVa_offset_;
  int JPg_offset_;
  int JPa_offset_;
  int dR_offset_;
  int gyro_bias_offset_;
  int acc_bias_offset_;
  int gyro_bias_diff_offset_;
  int dV_offset_;
  int dP_offset_;
  int dT_s_offset_;
  int problem_rig_poses_preint_elems;

  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf33>> update_pose_w_from_imu_linear;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> update_pose_other;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> update_point;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> update_point_step;

  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf33>> working_imu_from_w_linear;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> working_imu_from_w_translation;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<CostResult>> working_cost;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> working_partial_costs;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> working_buffer_solver;
  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<int>> working_buffer_solver_info;

  std::unique_ptr<cuvslam::cuda::GPUArrayPinned<float>> lambda;

  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> full_system_pose_block;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> full_system_pose_rhs;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf33>> full_system_point_block;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> full_system_point_rhs;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> full_system_point_pose_block_transposed;

  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> reduced_system_pose_block;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> reduced_system_pose_rhs_and_update_pose_step;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> reduced_system_point_rhs;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> reduced_system_camera_backsub_block_transposed;

  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>> model_inertial_jacobians_jr_left;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>> model_inertial_jacobians_jt_left;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>> model_inertial_jacobians_jv_left;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>> model_inertial_jacobians_jb_acc_left;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>> model_inertial_jacobians_jb_gyro_left;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>> model_inertial_jacobians_jr_right;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>> model_inertial_jacobians_jt_right;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf93>> model_inertial_jacobians_jv_right;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf23>> model_repr_jacobians_jr;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf23>> model_repr_jacobians_jt;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<cuvslam::cuda::Matf23>> model_repr_jacobians_jp;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> model_repr_robustifier_weights;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> model_reprojection_residuals;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> model_random_walk_gyro_residuals;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> model_random_walk_acc_residuals;
  std::unique_ptr<cuvslam::cuda::GPUOnlyArray<float>> model_inertial_residuals;

  cuvslam::cuda::Matf33 calib_left_from_imu_linear;
  cuvslam::cuda::Vecf3 calib_left_from_imu_translation;
};

}  // namespace cuvslam::sba_imu
