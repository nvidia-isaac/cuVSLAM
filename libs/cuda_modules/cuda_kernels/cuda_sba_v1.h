
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

#include <driver_types.h>
#include <vector_types.h>

#include "cuda_modules/cuda_kernels/cuda_common.h"
#include "cuda_modules/cuda_kernels/cuda_matrix.h"

namespace cuvslam::cuda::sba {

struct GPUModelFunctionMeta {
  // one block per observation
  Mat<float, 2, 6> *__restrict__ pose_jacobians = nullptr;
  Mat<float, 2, 3> *__restrict__ point_jacobians = nullptr;
  float2 *__restrict__ residuals = nullptr;
  float *__restrict__ robustifier_weights = nullptr;
};

struct float6 {
  float x1, x2, x3, x4, x5, x6;
};

struct GPULinearSystemMeta {
  float *__restrict__ pose_block;
  size_t pose_block_pitch;

  float *__restrict__ point_pose_block_transposed;  // camera_backsub_block in reduced
  size_t point_pose_block_transposed_pitch;

  float *__restrict__ pose_rhs;
  Matf33 *__restrict__ point_block;  // inverse_point_block in reduced
  float *__restrict__ point_rhs;
};

struct GPUParameterUpdateMeta {
  Pose *__restrict__ pose;
  float3 *__restrict__ point;

  float6 *__restrict__ pose_step;
  float3 *__restrict__ points_step;
};

struct Observation {
  float2 xy;
  Matf22 info;

  int point_id;
  int pose_id;
  int camera_id;
};

struct Point {
  float3 coords;
  Matf22 info_matrix;
  int num_observations;
  int first_observation_id;
};

struct Rig {
  static constexpr const int kMaxCameras = 32;
  Pose camera_from_rig[kMaxCameras];
  int num_cameras = 0;
};

struct GPUBundleAdjustmentProblemMeta {
  // last num_fixed_points points are not allowed to move
  Point *__restrict__ points;
  int num_points;

  Observation *__restrict__ observations;
  int num_observations;

  // Last num_fixed_key_frames are not allowed to move.
  Pose *__restrict__ rig_from_world;
  int num_poses;
  Rig *__restrict__ rig;
  int num_cameras;  // to access the value from a host
};

cudaError_t update_model(const GPUModelFunctionMeta &function_meta, const GPUBundleAdjustmentProblemMeta &problem_meta,
                         float robustifier_scale, cudaStream_t s);

cudaError_t build_full_system(GPULinearSystemMeta system, GPUModelFunctionMeta function_meta,
                              GPUBundleAdjustmentProblemMeta problem_meta, int num_fixed_points,
                              int num_fixed_key_frames, cudaStream_t s);

cudaError_t build_reduced_system(GPULinearSystemMeta full_system, GPULinearSystemMeta reduced_system, int num_points,
                                 int num_poses, float lambda, float threshold, cudaStream_t s);

// partials: nullptr adds with float atomics, else one float per MAX_THREADS observations, added in a fixed order.
cudaError_t evaluate_cost(float *cost, int *num_skipped, GPUBundleAdjustmentProblemMeta problem_meta,
                          GPUParameterUpdateMeta update, float robustifier_scale, float *partials, cudaStream_t s);

cudaError_t calc_update(float *point, float *camera_backsub_block_T, size_t pitch, float *point_rhs, Pose *poses,
                        float6 *twists, int num_points, int num_poses, cudaStream_t s);

cudaError_t reduce_abs_max(float *array, size_t size, float *result, cudaStream_t s);

cudaError_t update_parameters(Point *points, float3 *point_update, Pose *poses, Pose *pose_update, int num_points,
                              int num_poses, cudaStream_t s);

// partials: as in evaluate_cost, one float per row, or per column if use_M_transposed.
cudaError_t v1T_x_M_x_v2(float *v1, float *matrix, size_t pitch, int rows, int cols, float *v2, float *result,
                         bool use_M_transposed, float *partials, cudaStream_t s);

// partials: as in evaluate_cost, one float per MAX_THREADS elements.
cudaError_t pose_scaling_term(float *vector, float *matrix, size_t matrix_pitch, int vector_size, float *res,
                              float *partials, cudaStream_t s);

// partials: as in evaluate_cost, one float per MAX_THREADS blocks.
cudaError_t point_term(float3 *vector, Matf33 *blocks, int num_blocks, int is_scaling, float *res, float *partials,
                       cudaStream_t s);

cudaError_t make_prediction(float current_cost, float lambda, float *pose_hessian_term_, float *point_hessian_term_,
                            float *pose_scaling_term_, float *point_scaling_term_, float *prediction, cudaStream_t s);

}  // namespace cuvslam::cuda::sba
