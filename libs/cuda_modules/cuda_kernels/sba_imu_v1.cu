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

#include <cfloat>

#include <cuda_runtime.h>

#if __CUDA_ARCH__ >= 700
#include <cuda_pipeline_primitives.h>
#endif

#include "cuda_modules/cuda_kernels/cuda_common.h"
#include "cuda_modules/cuda_kernels/cuda_matrix.h"
#include "cuda_modules/cuda_kernels/cuda_sba_common.h"
#include "cuda_modules/cuda_kernels/cuda_sba_v1.h"
#include "epipolar/near_plane.h"

namespace cuvslam::cuda::sba_imu {

__device__ __forceinline__ void Exp(cuvslam::cuda::Matf33& result, const cuvslam::cuda::Vecf3& twist) {
  float3 w = {twist.d_[0], twist.d_[1], twist.d_[2]};
  float w_norm = fast_sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
  float theta = max(w_norm, kFloatEpsilon);
  float c = fast_cos(theta);
  float theta_inv = fast_div(1.f, theta);
  float s_theta = fast_sin(theta) * theta_inv;
  float3 n = {w.x * theta_inv, w.y * theta_inv, w.z * theta_inv};

  result.d_[0][0] = c;
  result.d_[0][1] = -w.z * s_theta;
  result.d_[0][2] = w.y * s_theta;
  result.d_[1][0] = w.z * s_theta;
  result.d_[1][1] = c;
  result.d_[1][2] = -w.x * s_theta;
  result.d_[2][0] = -w.y * s_theta;
  result.d_[2][1] = w.x * s_theta;
  result.d_[2][2] = c;

  result.d_[0][0] += (1 - c) * n.x * n.x;
  result.d_[0][1] += (1 - c) * n.y * n.x;
  result.d_[0][2] += (1 - c) * n.z * n.x;

  result.d_[1][0] += (1 - c) * n.x * n.y;
  result.d_[1][1] += (1 - c) * n.y * n.y;
  result.d_[1][2] += (1 - c) * n.z * n.y;

  result.d_[2][0] += (1 - c) * n.x * n.z;
  result.d_[2][1] += (1 - c) * n.y * n.z;
  result.d_[2][2] += (1 - c) * n.z * n.z;
}

__device__ __forceinline__ cuvslam::cuda::Matf33 Exp(const cuvslam::cuda::Vecf3& twist) {
  cuvslam::cuda::Matf33 res;
  Exp(res, twist);
  return res;
}

__device__ __forceinline__ void Log(cuvslam::cuda::Vecf3& result, const cuvslam::cuda::Matf33& m, float threshold) {
  cuvslam::cuda::Matf33 a = m;
  a.d_[0][0] -= 1.f;
  a.d_[1][1] -= 1.f;
  a.d_[2][2] -= 1.f;

  SVD<float> usv(a, threshold);
  const auto& v = usv.v();
  cuvslam::cuda::Vecf3 wvec;
  wvec.d_[0] = v.d_[0][2];
  wvec.d_[1] = v.d_[1][2];
  wvec.d_[2] = v.d_[2][2];
  cuvslam::cuda::Vecf3 rvec;
  rvec.d_[0] = a.d_[2][1] - a.d_[1][2];
  rvec.d_[1] = a.d_[0][2] - a.d_[2][0];
  rvec.d_[2] = a.d_[1][0] - a.d_[0][1];

  float wmag = atan2f(dot(rvec, wvec), trace(a) + 2.f);
  mul(wmag, wvec, result);
}

__device__ __forceinline__ cuvslam::cuda::Vecf3 Log(const cuvslam::cuda::Matf33& m, float threshold) {
  cuvslam::cuda::Vecf3 res;
  Log(res, m, threshold);
  return res;
}

__device__ __forceinline__ cuvslam::cuda::Matf33 twist_left_inverse_jacobian(const cuvslam::cuda::Vecf3& twist) {
  float phi_inverse = fast_rsqrt(twist.d_[0] * twist.d_[0] + twist.d_[1] * twist.d_[1] + twist.d_[2] * twist.d_[2]);
  float phi = fast_div(1.f, phi_inverse);
  if (phi < sqrtf((float)FLT_EPSILON)) {
    return identity<float, 3>() - 0.5f * SkewSymmetric(twist);
  }
  cuvslam::cuda::Vecf3 a = phi_inverse * twist;
  float phi_half = phi * 0.5f;
  float cot_2 = phi_half * fast_cotan(phi_half);

  cuvslam::cuda::Matf33 b;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) b.d_[i][j] = a.d_[i] * a.d_[j];

  return cot_2 * identity<float, 3>() + (1.f - cot_2) * b - phi_half * SkewSymmetric(a);
}

__device__ __forceinline__ cuvslam::cuda::Mat<float, 1, 3> twist_left_inverse_jacobian_row(
    const cuvslam::cuda::Vecf3& twist, int row_id) {
  float phi_inverse = fast_rsqrt(twist.d_[0] * twist.d_[0] + twist.d_[1] * twist.d_[1] + twist.d_[2] * twist.d_[2]);
  float phi = fast_div(1.f, phi_inverse);
  cuvslam::cuda::Mat<float, 1, 3> identity_slice;
  identity_slice.d_[0][0] = (row_id == 0) ? 1.f : 0.f;
  identity_slice.d_[0][1] = (row_id == 1) ? 1.f : 0.f;
  identity_slice.d_[0][2] = (row_id == 2) ? 1.f : 0.f;
  if (phi < sqrtf(FLT_EPSILON)) {
    return identity_slice - 0.5f * SkewSymmetric_row(twist, row_id);
  }
  cuvslam::cuda::Vecf3 a = phi_inverse * twist;
  float phi_half = phi * 0.5f;
  float cot_2 = phi_half * fast_cotan(phi_half);

  return cot_2 * identity_slice + ((1.f - cot_2) * extract(a, row_id)) * transp(a) -
         phi_half * SkewSymmetric_row(a, row_id);
}

__device__ __forceinline__ float twist_left_inverse_jacobian(const cuvslam::cuda::Vecf3& twist, int row_id,
                                                             int col_id) {
  float phi_inverse = fast_rsqrt(twist.d_[0] * twist.d_[0] + twist.d_[1] * twist.d_[1] + twist.d_[2] * twist.d_[2]);
  float phi = fast_div(1.f, phi_inverse);
  if (phi < sqrtf(FLT_EPSILON)) {
    return ((row_id == col_id) ? 1.f : 0.f) - 0.5f * SkewSymmetric(twist, row_id, col_id);
  }
  cuvslam::cuda::Vecf3 a = phi_inverse * twist;
  float phi_half = phi * 0.5f;
  float cot_2 = phi_half * fast_cotan(phi_half);

  return ((row_id == col_id) ? cot_2 : 0.f) + (1.f - cot_2) * extract(a, row_id) * extract(a, col_id) -
         phi_half * SkewSymmetric(a, row_id, col_id);
}

__device__ __forceinline__ cuvslam::cuda::Matf33 twist_left_jacobian(const cuvslam::cuda::Vecf3& twist) {
  float phi_inverse = fast_rsqrt(twist.d_[0] * twist.d_[0] + twist.d_[1] * twist.d_[1] + twist.d_[2] * twist.d_[2]);
  float phi = fast_div(1.f, phi_inverse);
  if (phi < sqrtf(FLT_EPSILON)) {
    return identity<float, 3>() + 0.5f * SkewSymmetric(twist);
  }
  cuvslam::cuda::Vecf3 a = phi_inverse * twist;

  float wonderfull_limm = fast_div(fast_sin(phi), phi);
  float k = fast_div(1.f - fast_cos(phi), phi);

  cuvslam::cuda::Matf33 b;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) b.d_[i][j] = a.d_[i] * a.d_[j];

  return wonderfull_limm * identity<float, 3>() + (1.f - wonderfull_limm) * b + k * SkewSymmetric(a);
}

__device__ __forceinline__ cuvslam::cuda::Matf33 twist_right_inverse_jacobian(const cuvslam::cuda::Vecf3& twist) {
  cuvslam::cuda::Vecf3 neg_twist;
  neg_twist.d_[0] = -twist.d_[0];
  neg_twist.d_[1] = -twist.d_[1];
  neg_twist.d_[2] = -twist.d_[2];
  return twist_left_inverse_jacobian(neg_twist);
}

__device__ __forceinline__ cuvslam::cuda::Matf13 twist_right_inverse_jacobian_row(const cuvslam::cuda::Vecf3& twist,
                                                                                  int row_id) {
  cuvslam::cuda::Vecf3 neg_twist;
  neg_twist.d_[0] = -twist.d_[0];
  neg_twist.d_[1] = -twist.d_[1];
  neg_twist.d_[2] = -twist.d_[2];
  return twist_left_inverse_jacobian_row(neg_twist, row_id);
}

__device__ __forceinline__ float twist_right_inverse_jacobian(const cuvslam::cuda::Vecf3& twist, int row_id,
                                                              int col_id) {
  cuvslam::cuda::Vecf3 neg_twist;
  neg_twist.d_[0] = -twist.d_[0];
  neg_twist.d_[1] = -twist.d_[1];
  neg_twist.d_[2] = -twist.d_[2];
  return twist_left_inverse_jacobian(neg_twist, row_id, col_id);
}

__device__ __forceinline__ cuvslam::cuda::Matf33 twist_right_jacobian(const cuvslam::cuda::Vecf3& twist) {
  cuvslam::cuda::Vecf3 neg_twist;
  neg_twist.d_[0] = -twist.d_[0];
  neg_twist.d_[1] = -twist.d_[1];
  neg_twist.d_[2] = -twist.d_[2];
  return twist_left_jacobian(neg_twist);
}

template <typename T>
__device__ __forceinline__ void pipeline_memcpy_async(T* dst, const T* src) {
#if __CUDA_ARCH__ >= 700
  __pipeline_memcpy_async(dst, src, sizeof(T));
#else
  *dst = *src;
#endif
}

__device__ __forceinline__ void pipeline_commit() {
#if __CUDA_ARCH__ >= 700
  __pipeline_commit();
#endif
}

__device__ __forceinline__ void pipeline_wait_prior(int N = 0) {
#if __CUDA_ARCH__ >= 700
  __pipeline_wait_prior(N);
#endif
  __syncwarp();
}

#define GET_ELEMENT(ptr, pitch, row, col) ((float*)((char*)ptr + (row) * (pitch)) + (col))

template <int THREADBLOCK_SIZE>
__global__ void build_reduced_system_stage_1_kernel(const cuvslam::cuda::Matf33* __restrict__ full_system_point_block,
                                                    const float* __restrict__ full_system_point_rhs,
                                                    const float* __restrict__ full_system_point_pose_block_transposed,
                                                    int full_system_point_pose_block_transposed_pitch,
                                                    float* __restrict__ reduced_system_point_rhs,
                                                    float* __restrict__ reduced_system_camera_backsub_block_transposed,
                                                    int reduced_system_camera_backsub_block_transposed_pitch,
                                                    const float* __restrict__ lambda_ptr, float threshold,
                                                    int num_points, int num_poses) {
  int warp_id = (blockIdx.x * (THREADBLOCK_SIZE / 32)) + (threadIdx.x >> 5);
  int thread_group_id = (threadIdx.x & 31) / 9;
  if (thread_group_id >= 3) return;
  int i = warp_id * 3 + thread_group_id;
  if (i >= num_points) {
    return;
  }
  int local_id = (threadIdx.x & 31) - thread_group_id * 9;

  int slice_id = local_id / 3;
  int elem_id = local_id - slice_id * 3;

  Matf33 point_block = full_system_point_block[i];  // 3x3

  dampen(point_block, (*lambda_ptr));
  LDLT<float> ldlt(point_block);
  Vecf3 u{(slice_id == 0) ? 1.f : 0.f, (slice_id == 1) ? 1.f : 0.f, (slice_id == 2) ? 1.f : 0.f};
  Vecf3 point_block_inv_slice = ldlt.solve(u);

  float full_system_point_rhs_val = full_system_point_rhs[3 * i + elem_id];
  const Vecf3 a = {full_system_point_rhs_val, __shfl_down_sync(0xffffffff, full_system_point_rhs_val, 1),
                   __shfl_down_sync(0xffffffff, full_system_point_rhs_val, 2)};  // 3x1
  if (elem_id == 0) {
    float b = dot(point_block_inv_slice, a);  // 3x1
    reduced_system_point_rhs[3 * i + slice_id] = b;
  }

  for (int j = elem_id; j < 6 * num_poses; j += 3) {
    Vecf3 a;
    for (int l = 0; l < 3; ++l) {
      a.d_[l] = *GET_ELEMENT(full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch, j,
                             3 * i + l);
    }
    float c = dot(point_block_inv_slice, a);
    *GET_ELEMENT(reduced_system_camera_backsub_block_transposed, reduced_system_camera_backsub_block_transposed_pitch,
                 j, 3 * i + slice_id) = c;
  }
}

template <int THREADBLOCK_SIZE>
__global__ void build_reduced_system_stage_2_kernel(
    const float* __restrict__ full_system_point_rhs, const float* __restrict__ full_system_pose_block,
    int full_system_pose_block_pitch, const float* __restrict__ full_system_point_pose_block_transposed,
    int full_system_point_pose_block_transposed_pitch, const float* __restrict__ full_system_pose_rhs,
    const float* __restrict__ reduced_system_camera_backsub_block_transposed,
    int reduced_system_camera_backsub_block_transposed_pitch, float* __restrict__ reduced_system_pose_block,
    int reduced_system_pose_block_pitch, float* __restrict__ reduced_system_pose_rhs,
    const float* __restrict__ lambda_ptr, int num_points, int num_poses) {
  static __shared__ float shared[THREADBLOCK_SIZE / 32];  // Shared mem for partial sums

  int out_row = blockIdx.x;
  int out_col = blockIdx.y;

  int pose_x = out_row / 15;
  int shift_x = out_row - pose_x * 15;

  int pose_y = out_col / 15;
  int shift_y = out_col - pose_y * 15;

  float row_dot_col = 0.f;
  if ((shift_x < 6) && (shift_y < 6)) {
    int in_row = pose_x * 6 + shift_x;
    int in_col = pose_y * 6 + shift_y;
    if (pose_y < num_poses) {
      for (int i = threadIdx.x; i < 3 * num_points; i += THREADBLOCK_SIZE) {
        float Y_ij = *GET_ELEMENT(reduced_system_camera_backsub_block_transposed,
                                  reduced_system_camera_backsub_block_transposed_pitch, in_row, i);
        float transp_W = *GET_ELEMENT(full_system_point_pose_block_transposed,
                                      full_system_point_pose_block_transposed_pitch, in_col, i);
        row_dot_col += Y_ij * transp_W;
      }
    } else {
      for (int i = threadIdx.x; i < 3 * num_points; i += THREADBLOCK_SIZE) {
        float Y_ij = *GET_ELEMENT(reduced_system_camera_backsub_block_transposed,
                                  reduced_system_camera_backsub_block_transposed_pitch, in_row, i);
        float rhs = full_system_point_rhs[i];
        row_dot_col += Y_ij * rhs;
      }
    }

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      row_dot_col += __shfl_down_sync(0xffffffff, row_dot_col, offset);
    }
    if (lane == 0) {
      shared[wid] = row_dot_col;
    }
    __syncthreads();

    row_dot_col = lane < (THREADBLOCK_SIZE / 32) ? shared[lane] : 0;

    if (wid == 0) {
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        row_dot_col += __shfl_down_sync(0xffffffff, row_dot_col, offset);
      }
    }
  }

  if (threadIdx.x == 0) {
    if (pose_y < num_poses) {
      float full_sys_pb = *GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch, out_row, out_col);
      float out = full_sys_pb - row_dot_col;
      if (out_row == out_col) {  // on diagonal
        // dampening
        // HZ: S_jj* = S_jj + lambda * U_j
        out += (*lambda_ptr) * full_sys_pb;
      }
      *GET_ELEMENT(reduced_system_pose_block, reduced_system_pose_block_pitch, out_row, out_col) = out;
    } else {
      reduced_system_pose_rhs[out_row] = full_system_pose_rhs[out_row] - row_dot_col;
    }
  }
}

__global__ void calc_update_stage_1_kernel(const float* __restrict__ reduced_system_point_rhs,
                                           const float* __restrict__ update_pose_step,
                                           const float* __restrict__ reduced_system_camera_backsub_block_transposed,
                                           int reduced_system_camera_backsub_block_transposed_pitch,
                                           float* __restrict__ update_point_step, float* __restrict__ update_point,
                                           int num_points, int num_poses) {
  const int row_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_id >= num_points * 3) {
    return;
  }

  float row_dot = 0.f;
  for (int i = 0; i < num_poses; ++i) {
    for (int j = 0; j < 6; ++j) {
      float* ptr = GET_ELEMENT(reduced_system_camera_backsub_block_transposed,
                               reduced_system_camera_backsub_block_transposed_pitch, i * 6 + j, row_id);
      row_dot += *ptr * update_pose_step[i * 15 + j];
    }
  }

  float val = reduced_system_point_rhs[row_id] - row_dot;
  update_point_step[row_id] = val;
  update_point[row_id] = val;
}

__global__ void calc_update_stage_2_kernel(const float* __restrict__ update_pose_step,
                                           cuvslam::cuda::Matf33* __restrict__ update_pose_w_from_imu_linear,
                                           float* __restrict__ update_pose_other, int num_poses) {
  const int pose_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (pose_id >= num_poses) {
    return;
  }

  cuvslam::cuda::Vecf3 twist;
  twist.d_[0] = update_pose_step[pose_id * 15];
  twist.d_[1] = update_pose_step[pose_id * 15 + 1];
  twist.d_[2] = update_pose_step[pose_id * 15 + 2];
  cuvslam::cuda::Matf33 mat = Exp(twist);
  update_pose_w_from_imu_linear[pose_id] = mat;
  for (int i = 0; i < 12; ++i) {
    update_pose_other[pose_id * 12 + i] = update_pose_step[pose_id * 15 + 3 + i];
  }
}

__global__ void evaluate_cost_stage_1_kernel(
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_w_from_imu_linear,
    const float* __restrict__ problem_rig_poses_other,
    const cuvslam::cuda::Matf33* __restrict__ update_pose_w_from_imu_linear,
    const float* __restrict__ update_pose_other,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JRg_ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JVg_ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JVa_ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JPg_ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JPa_ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_dR_ptr,
    const float* __restrict__ problem_rig_poses_preint_gyro_bias__ptr,
    const float* __restrict__ problem_rig_poses_preint_acc_bias__ptr,
    const float* __restrict__ problem_rig_poses_preint_dV_ptr,
    const float* __restrict__ problem_rig_poses_preint_dP_ptr,
    const float* __restrict__ problem_rig_poses_preint_dT_s_ptr,
    const cuvslam::cuda::Matf99* __restrict__ problem_rig_poses_preint_info_matrix__ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_acc_random_walk_accum_info_matrix__ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_gyro_random_walk_accum_info_matrix__ptr,
    cuvslam::cuda::Matf33* __restrict__ imu_from_w_linear, float* __restrict__ imu_from_w_translation,
    float* __restrict__ cost_ptr, float threshold, int num_poses, int num_fixed_key_frames, float prior_gyro,
    float prior_acc, float3 gravity, float imu_penalty, float boundary_imu_penalty, float acc_rw_penalty,
    float robustifier_scale_pose) {
  float cost = 0.f;

  const int pose_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (pose_id < num_poses) {
    cuvslam::cuda::Matf33 pu1_w_from_imu_linear = problem_rig_poses_w_from_imu_linear[pose_id];
    cuvslam::cuda::Vecf3 pu1_w_from_imu_translation;
    pu1_w_from_imu_translation.d_[0] = problem_rig_poses_other[12 * pose_id];
    pu1_w_from_imu_translation.d_[1] = problem_rig_poses_other[12 * pose_id + 1];
    pu1_w_from_imu_translation.d_[2] = problem_rig_poses_other[12 * pose_id + 2];
    cuvslam::cuda::Vecf3 pu1_velocity;
    pu1_velocity.d_[0] = problem_rig_poses_other[12 * pose_id + 3];
    pu1_velocity.d_[1] = problem_rig_poses_other[12 * pose_id + 4];
    pu1_velocity.d_[2] = problem_rig_poses_other[12 * pose_id + 5];
    cuvslam::cuda::Vecf3 pu1_gyro_bias;
    pu1_gyro_bias.d_[0] = problem_rig_poses_other[12 * pose_id + 6];
    pu1_gyro_bias.d_[1] = problem_rig_poses_other[12 * pose_id + 7];
    pu1_gyro_bias.d_[2] = problem_rig_poses_other[12 * pose_id + 8];
    cuvslam::cuda::Vecf3 pu1_acc_bias;
    pu1_acc_bias.d_[0] = problem_rig_poses_other[12 * pose_id + 9];
    pu1_acc_bias.d_[1] = problem_rig_poses_other[12 * pose_id + 10];
    pu1_acc_bias.d_[2] = problem_rig_poses_other[12 * pose_id + 11];

    // if not fixed
    if (pose_id >= num_fixed_key_frames) {
      int id = pose_id - num_fixed_key_frames;

      cuvslam::cuda::Matf33 update_pose_w_from_imu_linear_s = update_pose_w_from_imu_linear[id];
      cuvslam::cuda::Vecf3 update_pose_w_from_imu_translation;
      update_pose_w_from_imu_translation.d_[0] = update_pose_other[12 * id];
      update_pose_w_from_imu_translation.d_[1] = update_pose_other[12 * id + 1];
      update_pose_w_from_imu_translation.d_[2] = update_pose_other[12 * id + 2];
      cuvslam::cuda::Vecf3 update_pose_velocity;
      update_pose_velocity.d_[0] = update_pose_other[12 * id + 3];
      update_pose_velocity.d_[1] = update_pose_other[12 * id + 4];
      update_pose_velocity.d_[2] = update_pose_other[12 * id + 5];
      cuvslam::cuda::Vecf3 update_pose_gyro_bias;
      update_pose_gyro_bias.d_[0] = update_pose_other[12 * id + 6];
      update_pose_gyro_bias.d_[1] = update_pose_other[12 * id + 7];
      update_pose_gyro_bias.d_[2] = update_pose_other[12 * id + 8];
      cuvslam::cuda::Vecf3 update_pose_acc_bias;
      update_pose_acc_bias.d_[0] = update_pose_other[12 * id + 9];
      update_pose_acc_bias.d_[1] = update_pose_other[12 * id + 10];
      update_pose_acc_bias.d_[2] = update_pose_other[12 * id + 11];

      cuvslam::cuda::Vecf3 p_translation;
      mul_add(pu1_w_from_imu_linear, update_pose_w_from_imu_translation, pu1_w_from_imu_translation, p_translation);
      cuvslam::cuda::Matf33 p_linear = pu1_w_from_imu_linear * update_pose_w_from_imu_linear_s;

      pu1_w_from_imu_linear = p_linear;
      pu1_w_from_imu_translation = p_translation;
      pu1_velocity.d_[0] += update_pose_velocity.d_[0];
      pu1_velocity.d_[1] += update_pose_velocity.d_[1];
      pu1_velocity.d_[2] += update_pose_velocity.d_[2];
      pu1_gyro_bias.d_[0] += update_pose_gyro_bias.d_[0];
      pu1_gyro_bias.d_[1] += update_pose_gyro_bias.d_[1];
      pu1_gyro_bias.d_[2] += update_pose_gyro_bias.d_[2];
      pu1_acc_bias.d_[0] += update_pose_acc_bias.d_[0];
      pu1_acc_bias.d_[1] += update_pose_acc_bias.d_[1];
      pu1_acc_bias.d_[2] += update_pose_acc_bias.d_[2];

      // bias priors on oldest non-fixed pose only
      if (pose_id == num_fixed_key_frames) {
        cost += prior_gyro * dot(pu1_gyro_bias, pu1_gyro_bias);
        cost += prior_acc * dot(pu1_acc_bias, pu1_acc_bias);
      }
    }
    cuvslam::cuda::Matf33 imu_from_w_linear_s = transp(pu1_w_from_imu_linear);
    cuvslam::cuda::Vecf3 pu1_w_from_imu_translation_neg;
    pu1_w_from_imu_translation_neg.d_[0] = -pu1_w_from_imu_translation.d_[0];
    pu1_w_from_imu_translation_neg.d_[1] = -pu1_w_from_imu_translation.d_[1];
    pu1_w_from_imu_translation_neg.d_[2] = -pu1_w_from_imu_translation.d_[2];
    cuvslam::cuda::Vecf3 imu_from_w_translation_s = imu_from_w_linear_s * pu1_w_from_imu_translation_neg;
    imu_from_w_linear[pose_id] = imu_from_w_linear_s;
    imu_from_w_translation[3 * pose_id] = imu_from_w_translation_s.d_[0];
    imu_from_w_translation[3 * pose_id + 1] = imu_from_w_translation_s.d_[1];
    imu_from_w_translation[3 * pose_id + 2] = imu_from_w_translation_s.d_[2];

    if ((pose_id != num_poses - 1) && (pose_id + 1 >= num_fixed_key_frames)) {
      const float eff_penalty = (pose_id < num_fixed_key_frames) ? boundary_imu_penalty : imu_penalty;
      cuvslam::cuda::Matf33 pu2_w_from_imu_linear = problem_rig_poses_w_from_imu_linear[pose_id + 1];
      cuvslam::cuda::Vecf3 pu2_w_from_imu_translation;
      pu2_w_from_imu_translation.d_[0] = problem_rig_poses_other[12 * (pose_id + 1)];
      pu2_w_from_imu_translation.d_[1] = problem_rig_poses_other[12 * (pose_id + 1) + 1];
      pu2_w_from_imu_translation.d_[2] = problem_rig_poses_other[12 * (pose_id + 1) + 2];
      cuvslam::cuda::Vecf3 pu2_velocity;
      pu2_velocity.d_[0] = problem_rig_poses_other[12 * (pose_id + 1) + 3];
      pu2_velocity.d_[1] = problem_rig_poses_other[12 * (pose_id + 1) + 4];
      pu2_velocity.d_[2] = problem_rig_poses_other[12 * (pose_id + 1) + 5];
      cuvslam::cuda::Vecf3 pu2_gyro_bias;
      pu2_gyro_bias.d_[0] = problem_rig_poses_other[12 * (pose_id + 1) + 6];
      pu2_gyro_bias.d_[1] = problem_rig_poses_other[12 * (pose_id + 1) + 7];
      pu2_gyro_bias.d_[2] = problem_rig_poses_other[12 * (pose_id + 1) + 8];
      cuvslam::cuda::Vecf3 pu2_acc_bias;
      pu2_acc_bias.d_[0] = problem_rig_poses_other[12 * (pose_id + 1) + 9];
      pu2_acc_bias.d_[1] = problem_rig_poses_other[12 * (pose_id + 1) + 10];
      pu2_acc_bias.d_[2] = problem_rig_poses_other[12 * (pose_id + 1) + 11];

      {
        int id = pose_id + 1 - num_fixed_key_frames;

        cuvslam::cuda::Matf33 update_pose_w_from_imu_linear_s = update_pose_w_from_imu_linear[id];
        cuvslam::cuda::Vecf3 update_pose_w_from_imu_translation;
        update_pose_w_from_imu_translation.d_[0] = update_pose_other[12 * id];
        update_pose_w_from_imu_translation.d_[1] = update_pose_other[12 * id + 1];
        update_pose_w_from_imu_translation.d_[2] = update_pose_other[12 * id + 2];
        cuvslam::cuda::Vecf3 update_pose_velocity;
        update_pose_velocity.d_[0] = update_pose_other[12 * id + 3];
        update_pose_velocity.d_[1] = update_pose_other[12 * id + 4];
        update_pose_velocity.d_[2] = update_pose_other[12 * id + 5];
        cuvslam::cuda::Vecf3 update_pose_gyro_bias;
        update_pose_gyro_bias.d_[0] = update_pose_other[12 * id + 6];
        update_pose_gyro_bias.d_[1] = update_pose_other[12 * id + 7];
        update_pose_gyro_bias.d_[2] = update_pose_other[12 * id + 8];
        cuvslam::cuda::Vecf3 update_pose_acc_bias;
        update_pose_acc_bias.d_[0] = update_pose_other[12 * id + 9];
        update_pose_acc_bias.d_[1] = update_pose_other[12 * id + 10];
        update_pose_acc_bias.d_[2] = update_pose_other[12 * id + 11];

        cuvslam::cuda::Vecf3 p_translation;
        mul_add(pu2_w_from_imu_linear, update_pose_w_from_imu_translation, pu2_w_from_imu_translation, p_translation);
        cuvslam::cuda::Matf33 p_linear = pu2_w_from_imu_linear * update_pose_w_from_imu_linear_s;

        pu2_w_from_imu_linear = p_linear;
        pu2_w_from_imu_translation = p_translation;
        pu2_velocity.d_[0] += update_pose_velocity.d_[0];
        pu2_velocity.d_[1] += update_pose_velocity.d_[1];
        pu2_velocity.d_[2] += update_pose_velocity.d_[2];
        pu2_gyro_bias.d_[0] += update_pose_gyro_bias.d_[0];
        pu2_gyro_bias.d_[1] += update_pose_gyro_bias.d_[1];
        pu2_gyro_bias.d_[2] += update_pose_gyro_bias.d_[2];
        pu2_acc_bias.d_[0] += update_pose_acc_bias.d_[0];
        pu2_acc_bias.d_[1] += update_pose_acc_bias.d_[1];
        pu2_acc_bias.d_[2] += update_pose_acc_bias.d_[2];
      }

      cuvslam::cuda::Matf33 R1T = transp(pu1_w_from_imu_linear);
      cuvslam::cuda::Matf33 R2 = pu2_w_from_imu_linear;

      cuvslam::cuda::Matf33 problem_rig_poses_preint_JRg = problem_rig_poses_preint_JRg_ptr[pose_id];
      cuvslam::cuda::Matf33 problem_rig_poses_preint_JVg = problem_rig_poses_preint_JVg_ptr[pose_id];
      cuvslam::cuda::Matf33 problem_rig_poses_preint_JVa = problem_rig_poses_preint_JVa_ptr[pose_id];
      cuvslam::cuda::Matf33 problem_rig_poses_preint_JPg = problem_rig_poses_preint_JPg_ptr[pose_id];
      cuvslam::cuda::Matf33 problem_rig_poses_preint_JPa = problem_rig_poses_preint_JPa_ptr[pose_id];
      cuvslam::cuda::Matf33 problem_rig_poses_preint_dR = problem_rig_poses_preint_dR_ptr[pose_id];
      cuvslam::cuda::Vecf3 problem_rig_poses_preint_gyro_bias_;
      problem_rig_poses_preint_gyro_bias_.d_[0] = problem_rig_poses_preint_gyro_bias__ptr[3 * pose_id];
      problem_rig_poses_preint_gyro_bias_.d_[1] = problem_rig_poses_preint_gyro_bias__ptr[3 * pose_id + 1];
      problem_rig_poses_preint_gyro_bias_.d_[2] = problem_rig_poses_preint_gyro_bias__ptr[3 * pose_id + 2];
      cuvslam::cuda::Vecf3 problem_rig_poses_preint_acc_bias_;
      problem_rig_poses_preint_acc_bias_.d_[0] = problem_rig_poses_preint_acc_bias__ptr[3 * pose_id];
      problem_rig_poses_preint_acc_bias_.d_[1] = problem_rig_poses_preint_acc_bias__ptr[3 * pose_id + 1];
      problem_rig_poses_preint_acc_bias_.d_[2] = problem_rig_poses_preint_acc_bias__ptr[3 * pose_id + 2];
      cuvslam::cuda::Vecf3 problem_rig_poses_preint_dV;
      problem_rig_poses_preint_dV.d_[0] = problem_rig_poses_preint_dV_ptr[3 * pose_id];
      problem_rig_poses_preint_dV.d_[1] = problem_rig_poses_preint_dV_ptr[3 * pose_id + 1];
      problem_rig_poses_preint_dV.d_[2] = problem_rig_poses_preint_dV_ptr[3 * pose_id + 2];
      cuvslam::cuda::Vecf3 problem_rig_poses_preint_dP;
      problem_rig_poses_preint_dP.d_[0] = problem_rig_poses_preint_dP_ptr[3 * pose_id];
      problem_rig_poses_preint_dP.d_[1] = problem_rig_poses_preint_dP_ptr[3 * pose_id + 1];
      problem_rig_poses_preint_dP.d_[2] = problem_rig_poses_preint_dP_ptr[3 * pose_id + 2];
      float dT = problem_rig_poses_preint_dT_s_ptr[pose_id];
      cuvslam::cuda::Matf99 problem_rig_poses_preint_info_matrix_ = problem_rig_poses_preint_info_matrix__ptr[pose_id];
      cuvslam::cuda::Matf33 problem_rig_poses_preint_acc_random_walk_accum_info_matrix_ =
          problem_rig_poses_preint_acc_random_walk_accum_info_matrix__ptr[pose_id];
      cuvslam::cuda::Matf33 problem_rig_poses_preint_gyro_random_walk_accum_info_matrix_ =
          problem_rig_poses_preint_gyro_random_walk_accum_info_matrix__ptr[pose_id];

      // = preint.GetDeltaRotation(pu1.gyro_bias);
      cuvslam::cuda::Matf33 dR =
          problem_rig_poses_preint_dR *
          Exp(problem_rig_poses_preint_JRg * (pu1_gyro_bias - problem_rig_poses_preint_gyro_bias_));
      cuvslam::cuda::Vecf3 dV;  // = preint.GetDeltaVelocity(pu1.gyro_bias, pu1.acc_bias);
      {
        cuvslam::cuda::Vecf3 dbg = pu1_gyro_bias - problem_rig_poses_preint_gyro_bias_;
        cuvslam::cuda::Vecf3 dba = pu1_acc_bias - problem_rig_poses_preint_acc_bias_;
        cuvslam::cuda::Vecf3 v1;
        mul_add(problem_rig_poses_preint_JVg, dbg, problem_rig_poses_preint_dV, v1);
        mul_add(problem_rig_poses_preint_JVa, dba, v1, dV);
      }
      cuvslam::cuda::Vecf3 dP;  // = preint.GetDeltaPosition(pu1.gyro_bias, pu1.acc_bias);
      {
        cuvslam::cuda::Vecf3 dbg = pu1_gyro_bias - problem_rig_poses_preint_gyro_bias_;
        cuvslam::cuda::Vecf3 dba = pu1_acc_bias - problem_rig_poses_preint_acc_bias_;
        cuvslam::cuda::Vecf3 v1;
        mul_add(problem_rig_poses_preint_JPg, dbg, problem_rig_poses_preint_dP, v1);
        mul_add(problem_rig_poses_preint_JPa, dba, v1, dP);
      }

      cuvslam::cuda::Vecf3 rot_error = Log(transp(dR) * R1T * R2, threshold);

      cuvslam::cuda::Vecf3 g;
      g.d_[0] = gravity.x;
      g.d_[1] = gravity.y;
      g.d_[2] = gravity.z;

      cuvslam::cuda::Vecf3 velocity_error_term = R1T * (pu2_velocity - pu1_velocity - dT * g);
      cuvslam::cuda::Vecf3 trans_error_term =
          R1T * (pu2_w_from_imu_translation - pu1_w_from_imu_translation - dT * pu1_velocity - (0.5f * dT * dT) * g);

      cuvslam::cuda::Vecf3 random_walk_gyro_error = pu1_gyro_bias - pu2_gyro_bias;
      cuvslam::cuda::Vecf3 random_walk_acc_error = pu1_acc_bias - pu2_acc_bias;

      cost += eff_penalty * dot(random_walk_gyro_error,
                                problem_rig_poses_preint_gyro_random_walk_accum_info_matrix_ * random_walk_gyro_error);
      // Acc random walk is NOT boundary-downweighted to keep bias chain strong
      const float eff_acc_rw_penalty = (acc_rw_penalty >= 0) ? acc_rw_penalty : imu_penalty;
      cost +=
          eff_acc_rw_penalty * dot(random_walk_acc_error,
                                   problem_rig_poses_preint_acc_random_walk_accum_info_matrix_ * random_walk_acc_error);

      cuvslam::cuda::Vecf9 inertial_error;
      inertial_error.d_[0] = rot_error.d_[0];
      inertial_error.d_[1] = rot_error.d_[1];
      inertial_error.d_[2] = rot_error.d_[2];
      cuvslam::cuda::Vecf3 v1 = velocity_error_term - dV;
      inertial_error.d_[3] = v1.d_[0];
      inertial_error.d_[4] = v1.d_[1];
      inertial_error.d_[5] = v1.d_[2];
      cuvslam::cuda::Vecf3 v2 = trans_error_term - dP;
      inertial_error.d_[6] = v2.d_[0];
      inertial_error.d_[7] = v2.d_[1];
      inertial_error.d_[8] = v2.d_[2];

      cost +=
          ComputeHuberLoss(eff_penalty * dot(inertial_error, problem_rig_poses_preint_info_matrix_ * inertial_error),
                           robustifier_scale_pose);
    }
  }

  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    cost += __shfl_down_sync(0xffffffff, cost, offset);
  }

  if ((threadIdx.x & 31) == 0) {
    atomicAdd(cost_ptr, cost);
  }
}

__global__ void evaluate_cost_stage_2_kernel(
    const cuvslam::cuda::Matf33* __restrict__ imu_from_w_linear_ptr,
    const float* __restrict__ imu_from_w_translation_ptr, const int* __restrict__ problem_point_ids,
    const int* __restrict__ problem_pose_ids, const int8_t* __restrict__ problem_camera_ids,
    const float* __restrict__ problem_points_ptr, const float* __restrict__ update_point_ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_camera_from_rig_linear_ptr,
    const float* __restrict__ problem_rig_camera_from_rig_translation_ptr,
    const float* __restrict__ problem_observation_xys_ptr,
    const cuvslam::cuda::Matf22* __restrict__ problem_observation_infos_ptr, float* __restrict__ partial_costs_ptr,
    int max_partial_costs, int* __restrict__ num_skipped_ptr, int num_observations,
    cuvslam::cuda::Matf33 calib_left_from_imu_linear, cuvslam::cuda::Vecf3 calib_left_from_imu_translation,
    float robustifier_scale) {
  int num_skipped = 0;
  float cost = 0.f;
  int lane = threadIdx.x & 31;

  const int obs = blockIdx.x * blockDim.x + threadIdx.x;
  if (obs < num_observations) {
    int point_idx = problem_point_ids[obs];
    int pose_idx = problem_pose_ids[obs];
    int camera_idx = problem_camera_ids[obs];

    cuvslam::cuda::Vecf3 problem_point;
    problem_point.d_[0] = problem_points_ptr[3 * point_idx];
    problem_point.d_[1] = problem_points_ptr[3 * point_idx + 1];
    problem_point.d_[2] = problem_points_ptr[3 * point_idx + 2];
    cuvslam::cuda::Vecf3 update_point;
    update_point.d_[0] = update_point_ptr[3 * point_idx];
    update_point.d_[1] = update_point_ptr[3 * point_idx + 1];
    update_point.d_[2] = update_point_ptr[3 * point_idx + 2];
    cuvslam::cuda::Matf33 imu_from_w_linear = imu_from_w_linear_ptr[pose_idx];
    cuvslam::cuda::Vecf3 imu_from_w_translation;
    imu_from_w_translation.d_[0] = imu_from_w_translation_ptr[3 * pose_idx];
    imu_from_w_translation.d_[1] = imu_from_w_translation_ptr[3 * pose_idx + 1];
    imu_from_w_translation.d_[2] = imu_from_w_translation_ptr[3 * pose_idx + 2];
    cuvslam::cuda::Matf33 problem_rig_camera_from_rig_linear = problem_rig_camera_from_rig_linear_ptr[camera_idx];
    cuvslam::cuda::Vecf3 problem_rig_camera_from_rig_translation;
    problem_rig_camera_from_rig_translation.d_[0] = problem_rig_camera_from_rig_translation_ptr[3 * camera_idx];
    problem_rig_camera_from_rig_translation.d_[1] = problem_rig_camera_from_rig_translation_ptr[3 * camera_idx + 1];
    problem_rig_camera_from_rig_translation.d_[2] = problem_rig_camera_from_rig_translation_ptr[3 * camera_idx + 2];
    cuvslam::cuda::Vecf2 problem_observation_xys;
    problem_observation_xys.d_[0] = problem_observation_xys_ptr[2 * obs];
    problem_observation_xys.d_[1] = problem_observation_xys_ptr[2 * obs + 1];
    cuvslam::cuda::Matf22 problem_observation_infos = problem_observation_infos_ptr[obs];

    cuvslam::cuda::Vecf3 p_w = problem_point + update_point;

    cuvslam::cuda::Vecf3 v1;
    mul_add(imu_from_w_linear, p_w, imu_from_w_translation, v1);
    cuvslam::cuda::Vecf3 v2;
    mul_add(calib_left_from_imu_linear, v1, calib_left_from_imu_translation, v2);
    cuvslam::cuda::Vecf3 p_c;
    mul_add(problem_rig_camera_from_rig_linear, v2, problem_rig_camera_from_rig_translation, p_c);

    if (cuvslam::epipolar::IsDepthInFront(p_c.d_[2])) {
      cuvslam::cuda::Vecf2 r;
      r.d_[0] = p_c.d_[0] / p_c.d_[2] - problem_observation_xys.d_[0];
      r.d_[1] = p_c.d_[1] / p_c.d_[2] - problem_observation_xys.d_[1];
      cost = ComputeHuberLoss(dot(r, problem_observation_infos * r), robustifier_scale);
    } else {
      num_skipped = 1;
    }
  }

  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    cost += __shfl_down_sync(0xffffffff, cost, offset);
    num_skipped += __shfl_down_sync(0xffffffff, num_skipped, offset);
  }
  if (lane == 0) {
    int partial_cost_id = obs / 32;
    if (partial_cost_id < max_partial_costs) partial_costs_ptr[partial_cost_id] = cost;
    atomicAdd(num_skipped_ptr, num_skipped);
  }
}

__global__ void evaluate_cost_stage_2_reduce_kernel(const float* __restrict__ partial_costs_ptr, int max_partial_costs,
                                                    float* __restrict__ cost_ptr) {
  float cost = 0.f;
  int lane = threadIdx.x & 31;

  for (int partial_cost_id = lane; partial_cost_id < max_partial_costs; partial_cost_id += 32) {
    cost += partial_costs_ptr[partial_cost_id];
  }
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    cost += __shfl_down_sync(0xffffffff, cost, offset);
  }
  if (lane == 0) {
    *cost_ptr += cost;
  }
}

template <int THREADBLOCK_SIZE>
__global__ void compute_predicted_reduction_stage_1_kernel(
    cuvslam::cuda::Matf33* __restrict__ full_system_point_blocks_ptr, const float* __restrict__ update_point_step_ptr,
    const float* __restrict__ update_pose_step_ptr, const float* __restrict__ full_system_point_pose_block_transposed,
    int full_system_point_pose_block_transposed_pitch, float* __restrict__ prediction_ptr,
    int* __restrict__ working_update_point_step_significant_ptr, int num_points, int num_poses,
    const float* __restrict__ lambda_ptr, float max_abs_update_epsilon) {
  const int point_id = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  float prediction = 0.f;
  float max_abs_update = 0.f;

  if (point_id < num_points) {
    Matf33 point_block = full_system_point_blocks_ptr[point_id];
    float lambda = *lambda_ptr;
    point_block.d_[0][0] *= (1.f + 2.f * lambda);
    point_block.d_[1][1] *= (1.f + 2.f * lambda);
    point_block.d_[2][2] *= (1.f + 2.f * lambda);
    cuvslam::cuda::Vecf3 update_point_step;
    update_point_step.d_[0] = update_point_step_ptr[3 * point_id];
    update_point_step.d_[1] = update_point_step_ptr[3 * point_id + 1];
    update_point_step.d_[2] = update_point_step_ptr[3 * point_id + 2];

    for (int i = 0; i < 3; ++i) max_abs_update = max(fabsf(update_point_step.d_[i]), max_abs_update);

    prediction += dot(update_point_step, point_block * update_point_step);
    ;

    cuvslam::cuda::Vecf3 v{0.f, 0.f, 0.f};
    for (int i = 0; i < num_poses; i++) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 6; ++k) {
          float a = *GET_ELEMENT(full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch,
                                 i * 6 + k, 3 * point_id + j);
          float b = update_pose_step_ptr[15 * i + k];
          v.d_[j] += a * b;
        }
      }
    }
    prediction += dot(update_point_step, v) * 2.f;
  }

  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    prediction += __shfl_down_sync(0xffffffff, prediction, offset);
  }

  unsigned int significant_update = __ballot_sync(0xffffffff, max_abs_update >= max_abs_update_epsilon);

  if (lane == 0) {
    atomicAdd(prediction_ptr, prediction);
    if (significant_update != 0) *working_update_point_step_significant_ptr = 1;
  }
}

template <int THREADBLOCK_SIZE>
__global__ void compute_predicted_reduction_stage_2_kernel(const float* __restrict__ update_pose_step,
                                                           const float* __restrict__ full_system_pose_block,
                                                           int full_system_pose_block_pitch,
                                                           float* __restrict__ prediction_ptr,
                                                           int* __restrict__ working_update_pose_step_significant_ptr,
                                                           int num_poses, const float* __restrict__ lambda_ptr,
                                                           float max_abs_update_epsilon) {
  const int col_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_id = blockIdx.y * blockDim.y + threadIdx.y;
  int lane = (threadIdx.y * blockDim.x + threadIdx.x) & 31;

  float prediction = 0.f;
  float max_abs_update = 0.f;

  if ((row_id < num_poses * 15) && (col_id < num_poses * 15)) {
    float full_sys_pb = *GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch, row_id, col_id);
    if (row_id == col_id) full_sys_pb *= (1.f + 2.f * (*lambda_ptr));
    prediction = update_pose_step[row_id] * update_pose_step[col_id] * full_sys_pb;

    if (row_id == 0) max_abs_update = max(fabsf(update_pose_step[col_id]), max_abs_update);
  }

  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    prediction += __shfl_down_sync(0xffffffff, prediction, offset);
  }

  unsigned int significant_update = __ballot_sync(0xffffffff, max_abs_update >= max_abs_update_epsilon);

  if (lane == 0) {
    atomicAdd(prediction_ptr, prediction);
    if (significant_update != 0) *working_update_pose_step_significant_ptr = 1;
  }
}

__global__ void update_state_stage_1_kernel(const float* __restrict__ update_point, float* __restrict__ problem_points,
                                            int num_points) {
  const int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (elem_id >= num_points * 3) {
    return;
  }

  problem_points[elem_id] += update_point[elem_id];
}

__global__ void update_state_stage_2_kernel(const cuvslam::cuda::Matf33* __restrict__ update_pose_w_from_imu_linear_ptr,
                                            const float* __restrict__ update_pose_other_ptr,
                                            cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_w_from_imu_linear_ptr,
                                            float* __restrict__ problem_rig_poses_other_ptr, int num_poses,
                                            int num_fixed_key_frames) {
  int update_id = blockIdx.x * blockDim.x + threadIdx.x;
  int pose_id = update_id + num_fixed_key_frames;

  if (update_id >= num_poses - num_fixed_key_frames) return;

  cuvslam::cuda::Matf33 update_pose_w_from_imu_linear = update_pose_w_from_imu_linear_ptr[update_id];
  cuvslam::cuda::Vecf3 update_pose_w_from_imu_translation;
  update_pose_w_from_imu_translation.d_[0] = update_pose_other_ptr[12 * update_id];
  update_pose_w_from_imu_translation.d_[1] = update_pose_other_ptr[12 * update_id + 1];
  update_pose_w_from_imu_translation.d_[2] = update_pose_other_ptr[12 * update_id + 2];

  cuvslam::cuda::Matf33 problem_rig_poses_w_from_imu_linear = problem_rig_poses_w_from_imu_linear_ptr[pose_id];
  cuvslam::cuda::Vecf3 problem_rig_poses_w_from_imu_translation;
  problem_rig_poses_w_from_imu_translation.d_[0] = problem_rig_poses_other_ptr[12 * pose_id];
  problem_rig_poses_w_from_imu_translation.d_[1] = problem_rig_poses_other_ptr[12 * pose_id + 1];
  problem_rig_poses_w_from_imu_translation.d_[2] = problem_rig_poses_other_ptr[12 * pose_id + 2];

  mul_add(problem_rig_poses_w_from_imu_linear, update_pose_w_from_imu_translation,
          problem_rig_poses_w_from_imu_translation, problem_rig_poses_w_from_imu_translation);
  problem_rig_poses_w_from_imu_linear = problem_rig_poses_w_from_imu_linear * update_pose_w_from_imu_linear;

  problem_rig_poses_w_from_imu_linear_ptr[pose_id] = problem_rig_poses_w_from_imu_linear;
  problem_rig_poses_other_ptr[12 * pose_id] = problem_rig_poses_w_from_imu_translation.d_[0];
  problem_rig_poses_other_ptr[12 * pose_id + 1] = problem_rig_poses_w_from_imu_translation.d_[1];
  problem_rig_poses_other_ptr[12 * pose_id + 2] = problem_rig_poses_w_from_imu_translation.d_[2];

  for (int i = 3; i < 12; ++i) {
    problem_rig_poses_other_ptr[12 * pose_id + i] += update_pose_other_ptr[12 * update_id + i];
  }
}

__global__ void update_model_stage_1_kernel(
    const int* __restrict__ problem_point_ids, const int* __restrict__ problem_pose_ids,
    const int8_t* __restrict__ problem_camera_ids, const float* __restrict__ problem_points_ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_w_from_imu_linear_ptr,
    const float* __restrict__ problem_rig_poses_other_ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_camera_from_rig_linear_ptr,
    const float* __restrict__ problem_rig_camera_from_rig_translation_ptr,
    const float* __restrict__ problem_observation_xys_ptr,
    const cuvslam::cuda::Matf22* __restrict__ problem_observation_infos_ptr,
    float* __restrict__ model_reprojection_residuals_ptr, float* __restrict__ model_repr_robustifier_weights_ptr,
    cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jt_ptr,
    cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jr_ptr,
    cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jp_ptr, int num_observations, float robustifier_scale,
    cuvslam::cuda::Matf33 calib_left_from_imu_linear, cuvslam::cuda::Vecf3 calib_left_from_imu_translation) {
  const int obs = blockIdx.x * blockDim.x + threadIdx.x;
  if (obs >= num_observations) return;

  int point_idx = problem_point_ids[obs];
  int pose_idx = problem_pose_ids[obs];
  int camera_idx = problem_camera_ids[obs];

  cuvslam::cuda::Vecf3 p_w;
  p_w.d_[0] = problem_points_ptr[3 * point_idx];
  p_w.d_[1] = problem_points_ptr[3 * point_idx + 1];
  p_w.d_[2] = problem_points_ptr[3 * point_idx + 2];

  cuvslam::cuda::Matf33 problem_rig_poses_w_from_imu_linear = problem_rig_poses_w_from_imu_linear_ptr[pose_idx];
  cuvslam::cuda::Vecf3 problem_rig_poses_w_from_imu_translation;
  problem_rig_poses_w_from_imu_translation.d_[0] = problem_rig_poses_other_ptr[12 * pose_idx];
  problem_rig_poses_w_from_imu_translation.d_[1] = problem_rig_poses_other_ptr[12 * pose_idx + 1];
  problem_rig_poses_w_from_imu_translation.d_[2] = problem_rig_poses_other_ptr[12 * pose_idx + 2];

  cuvslam::cuda::Matf33 imu_from_w_linear = transp(problem_rig_poses_w_from_imu_linear);
  cuvslam::cuda::Vecf3 problem_rig_poses_w_from_imu_translation_neg;
  problem_rig_poses_w_from_imu_translation_neg.d_[0] = -problem_rig_poses_w_from_imu_translation.d_[0];
  problem_rig_poses_w_from_imu_translation_neg.d_[1] = -problem_rig_poses_w_from_imu_translation.d_[1];
  problem_rig_poses_w_from_imu_translation_neg.d_[2] = -problem_rig_poses_w_from_imu_translation.d_[2];
  cuvslam::cuda::Vecf3 imu_from_w_translation = imu_from_w_linear * problem_rig_poses_w_from_imu_translation_neg;

  cuvslam::cuda::Vecf3 p_imu;
  mul_add(imu_from_w_linear, p_w, imu_from_w_translation, p_imu);

  cuvslam::cuda::Matf33 problem_rig_camera_from_rig_linear = problem_rig_camera_from_rig_linear_ptr[camera_idx];
  cuvslam::cuda::Vecf3 problem_rig_camera_from_rig_translation;
  problem_rig_camera_from_rig_translation.d_[0] = problem_rig_camera_from_rig_translation_ptr[3 * camera_idx];
  problem_rig_camera_from_rig_translation.d_[1] = problem_rig_camera_from_rig_translation_ptr[3 * camera_idx + 1];
  problem_rig_camera_from_rig_translation.d_[2] = problem_rig_camera_from_rig_translation_ptr[3 * camera_idx + 2];

  cuvslam::cuda::Matf33 cam_from_imu_linear = problem_rig_camera_from_rig_linear * calib_left_from_imu_linear;
  cuvslam::cuda::Vecf3 cam_from_imu_translation;
  mul_add(problem_rig_camera_from_rig_linear, calib_left_from_imu_translation, problem_rig_camera_from_rig_translation,
          cam_from_imu_translation);

  cuvslam::cuda::Vecf3 p_c;
  mul_add(cam_from_imu_linear, p_imu, cam_from_imu_translation, p_c);

  cuvslam::cuda::Vecf2 r{0.f, 0.f};
  float model_repr_robustifier_weights = 0.f;
  cuvslam::cuda::Matf23 model_repr_jacobians_jt{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  cuvslam::cuda::Matf23 model_repr_jacobians_jr{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  cuvslam::cuda::Matf23 model_repr_jacobians_jp{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (cuvslam::epipolar::IsDepthInFront(p_c.d_[2])) {
    cuvslam::cuda::Matf33 Rimu_from_w = imu_from_w_linear;
    cuvslam::cuda::Matf33 Rcam_from_imu = cam_from_imu_linear;

    cuvslam::cuda::Vecf2 problem_observation_xys;
    problem_observation_xys.d_[0] = problem_observation_xys_ptr[2 * obs];
    problem_observation_xys.d_[1] = problem_observation_xys_ptr[2 * obs + 1];
    cuvslam::cuda::Matf22 problem_observation_infos = problem_observation_infos_ptr[obs];

    cuvslam::cuda::Vecf2 prediction;
    prediction.d_[0] = p_c.d_[0] / p_c.d_[2];
    prediction.d_[1] = p_c.d_[1] / p_c.d_[2];

    r.d_[0] = prediction.d_[0] - problem_observation_xys.d_[0];
    r.d_[1] = prediction.d_[1] - problem_observation_xys.d_[1];

    model_repr_robustifier_weights = ComputeDHuberLoss(dot(r, problem_observation_infos * r), robustifier_scale);

    float inv_z = 1.f / p_c.d_[2];
    cuvslam::cuda::Matf23 dproj;
    dproj.d_[0][0] = inv_z;
    dproj.d_[0][1] = 0.f;
    dproj.d_[0][2] = -prediction.d_[0] * inv_z;
    dproj.d_[1][0] = 0.f;
    dproj.d_[1][1] = inv_z;
    dproj.d_[1][2] = -prediction.d_[1] * inv_z;

    cuvslam::cuda::Matf23 M;
    // Mat23 M = dproj * Rcam_from_imu;
    M.d_[0][0] = dproj.d_[0][0] * Rcam_from_imu.d_[0][0] + dproj.d_[0][2] * Rcam_from_imu.d_[2][0];
    M.d_[0][1] = dproj.d_[0][0] * Rcam_from_imu.d_[0][1] + dproj.d_[0][2] * Rcam_from_imu.d_[2][1];
    M.d_[0][2] = dproj.d_[0][0] * Rcam_from_imu.d_[0][2] + dproj.d_[0][2] * Rcam_from_imu.d_[2][2];
    M.d_[1][0] = dproj.d_[1][1] * Rcam_from_imu.d_[1][0] + dproj.d_[1][2] * Rcam_from_imu.d_[2][0];
    M.d_[1][1] = dproj.d_[1][1] * Rcam_from_imu.d_[1][1] + dproj.d_[1][2] * Rcam_from_imu.d_[2][1];
    M.d_[1][2] = dproj.d_[1][1] * Rcam_from_imu.d_[1][2] + dproj.d_[1][2] * Rcam_from_imu.d_[2][2];

    // Jr = M * Skew(p_imu)
    model_repr_jacobians_jr.d_[0][0] = M.d_[0][1] * p_imu.d_[2] - M.d_[0][2] * p_imu.d_[1];
    model_repr_jacobians_jr.d_[0][1] = -M.d_[0][0] * p_imu.d_[2] + M.d_[0][2] * p_imu.d_[0];
    model_repr_jacobians_jr.d_[0][2] = M.d_[0][0] * p_imu.d_[1] - M.d_[0][1] * p_imu.d_[0];
    model_repr_jacobians_jr.d_[1][0] = M.d_[1][1] * p_imu.d_[2] - M.d_[1][2] * p_imu.d_[1];
    model_repr_jacobians_jr.d_[1][1] = -M.d_[1][0] * p_imu.d_[2] + M.d_[1][2] * p_imu.d_[0];
    model_repr_jacobians_jr.d_[1][2] = M.d_[1][0] * p_imu.d_[1] - M.d_[1][1] * p_imu.d_[0];

    // Jr = M * Rimu_from_w
    model_repr_jacobians_jp.d_[0][0] =
        M.d_[0][0] * Rimu_from_w.d_[0][0] + M.d_[0][1] * Rimu_from_w.d_[1][0] + M.d_[0][2] * Rimu_from_w.d_[2][0];
    model_repr_jacobians_jp.d_[0][1] =
        M.d_[0][0] * Rimu_from_w.d_[0][1] + M.d_[0][1] * Rimu_from_w.d_[1][1] + M.d_[0][2] * Rimu_from_w.d_[2][1];
    model_repr_jacobians_jp.d_[0][2] =
        M.d_[0][0] * Rimu_from_w.d_[0][2] + M.d_[0][1] * Rimu_from_w.d_[1][2] + M.d_[0][2] * Rimu_from_w.d_[2][2];
    model_repr_jacobians_jp.d_[1][0] =
        M.d_[1][0] * Rimu_from_w.d_[0][0] + M.d_[1][1] * Rimu_from_w.d_[1][0] + M.d_[1][2] * Rimu_from_w.d_[2][0];
    model_repr_jacobians_jp.d_[1][1] =
        M.d_[1][0] * Rimu_from_w.d_[0][1] + M.d_[1][1] * Rimu_from_w.d_[1][1] + M.d_[1][2] * Rimu_from_w.d_[2][1];
    model_repr_jacobians_jp.d_[1][2] =
        M.d_[1][0] * Rimu_from_w.d_[0][2] + M.d_[1][1] * Rimu_from_w.d_[1][2] + M.d_[1][2] * Rimu_from_w.d_[2][2];

    model_repr_jacobians_jt.d_[0][0] = -M.d_[0][0];
    model_repr_jacobians_jt.d_[0][1] = -M.d_[0][1];
    model_repr_jacobians_jt.d_[0][2] = -M.d_[0][2];
    model_repr_jacobians_jt.d_[1][0] = -M.d_[1][0];
    model_repr_jacobians_jt.d_[1][1] = -M.d_[1][1];
    model_repr_jacobians_jt.d_[1][2] = -M.d_[1][2];
  }

  model_reprojection_residuals_ptr[2 * obs] = r.d_[0];
  model_reprojection_residuals_ptr[2 * obs + 1] = r.d_[1];

  model_repr_robustifier_weights_ptr[obs] = model_repr_robustifier_weights;
  model_repr_jacobians_jt_ptr[obs] = model_repr_jacobians_jt;
  model_repr_jacobians_jr_ptr[obs] = model_repr_jacobians_jr;
  model_repr_jacobians_jp_ptr[obs] = model_repr_jacobians_jp;
}

template <int THREADBLOCK_SIZE>
__global__ void __launch_bounds__(THREADBLOCK_SIZE)
    update_model_stage_2_kernel(const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_w_from_imu_linear_ptr,
                                const float* __restrict__ problem_rig_poses_other_ptr,
                                const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JRg_ptr,
                                const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JVg_ptr,
                                const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JVa_ptr,
                                const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JPg_ptr,
                                const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_JPa_ptr,
                                const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_dR_ptr,
                                const float* __restrict__ problem_rig_poses_preint_gyro_bias__ptr,
                                const float* __restrict__ problem_rig_poses_preint_acc_bias__ptr,
                                const float* __restrict__ problem_rig_poses_preint_gyro_bias_diff__ptr,
                                const float* __restrict__ problem_rig_poses_preint_dV_ptr,
                                const float* __restrict__ problem_rig_poses_preint_dP_ptr,
                                const float* __restrict__ problem_rig_poses_preint_dT_s_ptr,
                                float* __restrict__ model_inertial_residuals_ptr,
                                float* __restrict__ model_random_walk_gyro_residuals_ptr,
                                float* __restrict__ model_random_walk_acc_residuals_ptr,
                                cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jr_left_ptr,
                                cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jt_left_ptr,
                                cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jv_left_ptr,
                                cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jb_acc_left_ptr,
                                cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jb_gyro_left_ptr,
                                cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jr_right_ptr,
                                cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jt_right_ptr,
                                cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jv_right_ptr,
                                float threshold, float3 gravity, int num_poses, int num_fixed_key_frames) {
  int warp_id = (blockIdx.x * (THREADBLOCK_SIZE / 32)) + (threadIdx.x >> 5);
  int elem = warp_id;
  if (elem >= num_poses - num_fixed_key_frames) return;

  int lane_id = threadIdx.x & 31;
  int row93 = lane_id / 3;
  if (row93 >= 9) return;
  int col93 = lane_id - row93 * 3;
  int row33 = row93 % 3;
  int col33 = col93;

  int i = elem + (num_fixed_key_frames - 1);

  __shared__ cuvslam::cuda::Matf33 R1T_s;
  __shared__ cuvslam::cuda::Matf33 R2_s;
  __shared__ cuvslam::cuda::Matf33 problem_rig_poses_preint_JVg_s;
  __shared__ cuvslam::cuda::Matf33 problem_rig_poses_preint_JPg_s;
  __shared__ cuvslam::cuda::Matf33 problem_rig_poses_preint_JPa_s;
  __shared__ cuvslam::cuda::Matf33 problem_rig_poses_preint_JVa_s;
  __shared__ cuvslam::cuda::Matf33 problem_rig_poses_preint_JRg_s;
  __shared__ cuvslam::cuda::Matf33 problem_rig_poses_preint_dR_s;
  if (lane_id < 9) {
    pipeline_memcpy_async(&(R1T_s.d_[row33][col33]), &(problem_rig_poses_w_from_imu_linear_ptr[i].d_[col33][row33]));
    pipeline_memcpy_async(&(R2_s.d_[row33][col33]), &(problem_rig_poses_w_from_imu_linear_ptr[i + 1].d_[row33][col33]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_JVg_s.d_[row33][col33]),
                          &(problem_rig_poses_preint_JVg_ptr[i].d_[row33][col33]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_JPg_s.d_[row33][col33]),
                          &(problem_rig_poses_preint_JPg_ptr[i].d_[row33][col33]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_JPa_s.d_[row33][col33]),
                          &(problem_rig_poses_preint_JPa_ptr[i].d_[row33][col33]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_JVa_s.d_[row33][col33]),
                          &(problem_rig_poses_preint_JVa_ptr[i].d_[row33][col33]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_JRg_s.d_[row33][col33]),
                          &(problem_rig_poses_preint_JRg_ptr[i].d_[row33][col33]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_dR_s.d_[row33][col33]),
                          &(problem_rig_poses_preint_dR_ptr[i].d_[row33][col33]));
  }
  __shared__ float problem_rig_poses_other_s[24];
  if (lane_id < 24) {
    pipeline_memcpy_async(problem_rig_poses_other_s + lane_id, &(problem_rig_poses_other_ptr[12 * i + lane_id]));
  }
  __shared__ cuvslam::cuda::Vecf3 problem_rig_poses_preint_gyro_bias__s;
  __shared__ cuvslam::cuda::Vecf3 problem_rig_poses_preint_acc_bias__s;
  __shared__ cuvslam::cuda::Vecf3 problem_rig_poses_preint_gyro_bias_diff__s;
  __shared__ cuvslam::cuda::Vecf3 problem_rig_poses_preint_dV_s;
  __shared__ cuvslam::cuda::Vecf3 problem_rig_poses_preint_dP_s;
  if (lane_id < 3) {
    pipeline_memcpy_async(&(problem_rig_poses_preint_gyro_bias__s.d_[lane_id]),
                          &(problem_rig_poses_preint_gyro_bias__ptr[3 * i + lane_id]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_acc_bias__s.d_[lane_id]),
                          &(problem_rig_poses_preint_acc_bias__ptr[3 * i + lane_id]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_gyro_bias_diff__s.d_[lane_id]),
                          &(problem_rig_poses_preint_gyro_bias_diff__ptr[3 * i + lane_id]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_dV_s.d_[lane_id]),
                          &(problem_rig_poses_preint_dV_ptr[3 * i + lane_id]));
    pipeline_memcpy_async(&(problem_rig_poses_preint_dP_s.d_[lane_id]),
                          &(problem_rig_poses_preint_dP_ptr[3 * i + lane_id]));
  }
  pipeline_commit();
  pipeline_wait_prior();
  cuvslam::cuda::Matf33 R1T = R1T_s;
  cuvslam::cuda::Matf33 R2 = R2_s;
  cuvslam::cuda::Matf33 problem_rig_poses_preint_JVg = problem_rig_poses_preint_JVg_s;
  cuvslam::cuda::Matf33 problem_rig_poses_preint_JPg = problem_rig_poses_preint_JPg_s;
  cuvslam::cuda::Matf33 problem_rig_poses_preint_JPa = problem_rig_poses_preint_JPa_s;
  cuvslam::cuda::Matf33 problem_rig_poses_preint_JVa = problem_rig_poses_preint_JVa_s;
  cuvslam::cuda::Matf33 problem_rig_poses_preint_JRg = problem_rig_poses_preint_JRg_s;
  cuvslam::cuda::Matf33 problem_rig_poses_preint_dR = problem_rig_poses_preint_dR_s;

  cuvslam::cuda::Vecf3 w_from_imu_left_translation;
  w_from_imu_left_translation.d_[0] = problem_rig_poses_other_s[0];
  w_from_imu_left_translation.d_[1] = problem_rig_poses_other_s[1];
  w_from_imu_left_translation.d_[2] = problem_rig_poses_other_s[2];
  cuvslam::cuda::Vecf3 w_from_imu_right_translation;
  w_from_imu_right_translation.d_[0] = problem_rig_poses_other_s[12];
  w_from_imu_right_translation.d_[1] = problem_rig_poses_other_s[12 + 1];
  w_from_imu_right_translation.d_[2] = problem_rig_poses_other_s[12 + 2];
  cuvslam::cuda::Vecf3 pose_left_velocity;
  pose_left_velocity.d_[0] = problem_rig_poses_other_s[3];
  pose_left_velocity.d_[1] = problem_rig_poses_other_s[4];
  pose_left_velocity.d_[2] = problem_rig_poses_other_s[5];
  cuvslam::cuda::Vecf3 pose_right_velocity;
  pose_right_velocity.d_[0] = problem_rig_poses_other_s[12 + 3];
  pose_right_velocity.d_[1] = problem_rig_poses_other_s[12 + 4];
  pose_right_velocity.d_[2] = problem_rig_poses_other_s[12 + 5];
  cuvslam::cuda::Vecf3 pose_left_gyro_bias;
  pose_left_gyro_bias.d_[0] = problem_rig_poses_other_s[6];
  pose_left_gyro_bias.d_[1] = problem_rig_poses_other_s[7];
  pose_left_gyro_bias.d_[2] = problem_rig_poses_other_s[8];
  //    cuvslam::cuda::Vecf3 pose_right_gyro_bias;
  //    pose_right_gyro_bias.d_[0] = problem_rig_poses_other_s[12 + 6];
  //    pose_right_gyro_bias.d_[1] = problem_rig_poses_other_s[12 + 7];
  //    pose_right_gyro_bias.d_[2] = problem_rig_poses_other_s[12 + 8];
  cuvslam::cuda::Vecf3 pose_left_acc_bias;
  pose_left_acc_bias.d_[0] = problem_rig_poses_other_s[9];
  pose_left_acc_bias.d_[1] = problem_rig_poses_other_s[10];
  pose_left_acc_bias.d_[2] = problem_rig_poses_other_s[11];
  //    cuvslam::cuda::Vecf3 pose_right_acc_bias;
  //    pose_right_acc_bias.d_[0] = problem_rig_poses_other_s[12 + 9];
  //    pose_right_acc_bias.d_[1] = problem_rig_poses_other_s[12 + 10];
  //    pose_right_acc_bias.d_[2] = problem_rig_poses_other_s[12 + 11];

  cuvslam::cuda::Vecf3 problem_rig_poses_preint_gyro_bias_ = problem_rig_poses_preint_gyro_bias__s;
  cuvslam::cuda::Vecf3 problem_rig_poses_preint_acc_bias_ = problem_rig_poses_preint_acc_bias__s;
  cuvslam::cuda::Vecf3 problem_rig_poses_preint_gyro_bias_diff_ = problem_rig_poses_preint_gyro_bias_diff__s;
  cuvslam::cuda::Vecf3 problem_rig_poses_preint_dV = problem_rig_poses_preint_dV_s;
  cuvslam::cuda::Vecf3 problem_rig_poses_preint_dP = problem_rig_poses_preint_dP_s;
  float dT = problem_rig_poses_preint_dT_s_ptr[i];

  // = preint.GetDeltaRotation(pu1.gyro_bias);
  cuvslam::cuda::Matf33 dR =
      problem_rig_poses_preint_dR *
      Exp(problem_rig_poses_preint_JRg * (pose_left_gyro_bias - problem_rig_poses_preint_gyro_bias_));
  cuvslam::cuda::Vecf3 dV;  // = preint.GetDeltaVelocity(pu1.gyro_bias, pu1.acc_bias);
  {
    cuvslam::cuda::Vecf3 dbg = pose_left_gyro_bias - problem_rig_poses_preint_gyro_bias_;
    cuvslam::cuda::Vecf3 dba = pose_left_acc_bias - problem_rig_poses_preint_acc_bias_;
    cuvslam::cuda::Vecf3 v1;
    mul_add(problem_rig_poses_preint_JVg, dbg, problem_rig_poses_preint_dV, v1);
    mul_add(problem_rig_poses_preint_JVa, dba, v1, dV);
  }
  cuvslam::cuda::Vecf3 dP;  // = preint.GetDeltaPosition(pu1.gyro_bias, pu1.acc_bias);
  {
    cuvslam::cuda::Vecf3 dbg = pose_left_gyro_bias - problem_rig_poses_preint_gyro_bias_;
    cuvslam::cuda::Vecf3 dba = pose_left_acc_bias - problem_rig_poses_preint_acc_bias_;
    cuvslam::cuda::Vecf3 v1;
    mul_add(problem_rig_poses_preint_JPg, dbg, problem_rig_poses_preint_dP, v1);
    mul_add(problem_rig_poses_preint_JPa, dba, v1, dP);
  }

  cuvslam::cuda::Matf33 R1TR2 = R1T * R2;
  cuvslam::cuda::Vecf3 rot_error = Log(transp(dR) * R1TR2, threshold);

  cuvslam::cuda::Vecf3 g;
  g.d_[0] = gravity.x;
  g.d_[1] = gravity.y;
  g.d_[2] = gravity.z;

  cuvslam::cuda::Vecf3 vel_or_trans_error_term_right;
  if (row93 < 6)
    vel_or_trans_error_term_right = pose_right_velocity - pose_left_velocity - dT * g;
  else
    vel_or_trans_error_term_right =
        w_from_imu_right_translation - w_from_imu_left_translation - dT * pose_left_velocity - (0.5f * dT * dT) * g;
  cuvslam::cuda::Vecf3 vel_or_trans_error_term = R1T * vel_or_trans_error_term_right;

  if (row33 == 0) {
    float model_inertial_residuals_val;
    if (row93 == 0) {
      model_inertial_residuals_val = extract(rot_error, col33);
    } else {
      model_inertial_residuals_val = extract(vel_or_trans_error_term, col33) - extract((row93 == 3) ? dV : dP, col33);
    }
    model_inertial_residuals_ptr[i * 9 + row93 + col33] = model_inertial_residuals_val;
  }

  float model_inertial_jacobians_jr_left_val;
  if (row93 < 3) {
    cuvslam::cuda::Matf13 m1 = twist_right_inverse_jacobian_row(rot_error, row33) * transp(R1TR2);
    model_inertial_jacobians_jr_left_val = -extract(m1, 0, col33);
  } else {
    float m2_val = SkewSymmetric(vel_or_trans_error_term, row33, col33);
    model_inertial_jacobians_jr_left_val = m2_val;
  }
  model_inertial_jacobians_jr_left_ptr[i].d_[row93][col93] = model_inertial_jacobians_jr_left_val;

  float model_inertial_jacobians_jt_left_val = ((row93 >= 6) && (row33 == col33)) ? -1.f : 0.f;
  model_inertial_jacobians_jt_left_ptr[i].d_[row93][col93] = model_inertial_jacobians_jt_left_val;

  float model_inertial_jacobians_jv_left_val = 0.f;
  if (row93 >= 3) model_inertial_jacobians_jv_left_val = -R1T_s.d_[row33][col33];
  if (row93 >= 6) model_inertial_jacobians_jv_left_val *= dT;
  model_inertial_jacobians_jv_left_ptr[i].d_[row93][col93] = model_inertial_jacobians_jv_left_val;

  float model_inertial_jacobians_jb_acc_left_val = 0.f;
  if (row93 >= 6)
    model_inertial_jacobians_jb_acc_left_val = -problem_rig_poses_preint_JPa_s.d_[row33][col33];
  else if (row93 >= 3)
    model_inertial_jacobians_jb_acc_left_val = -problem_rig_poses_preint_JVa_s.d_[row33][col33];
  model_inertial_jacobians_jb_acc_left_ptr[i].d_[row93][col93] = model_inertial_jacobians_jb_acc_left_val;

  float model_inertial_jacobians_jb_gyro_left_val;
  if (row93 < 3) {
    cuvslam::cuda::Vecf3 problem_rig_poses_preint_JRg_col;
    problem_rig_poses_preint_JRg_col.d_[0] = problem_rig_poses_preint_JRg_s.d_[col33][0];
    problem_rig_poses_preint_JRg_col.d_[1] = problem_rig_poses_preint_JRg_s.d_[col33][1];
    problem_rig_poses_preint_JRg_col.d_[2] = problem_rig_poses_preint_JRg_s.d_[col33][2];
    float m1_val = dot(to_vector(transp(twist_left_inverse_jacobian_row(rot_error, row33) *
                                        twist_right_jacobian(problem_rig_poses_preint_JRg *
                                                             problem_rig_poses_preint_gyro_bias_diff_))),
                       problem_rig_poses_preint_JRg_col);
    model_inertial_jacobians_jb_gyro_left_val = -m1_val;
  } else if (row93 < 6) {
    model_inertial_jacobians_jb_gyro_left_val = -problem_rig_poses_preint_JVg_s.d_[row33][col33];
  } else {
    model_inertial_jacobians_jb_gyro_left_val = -problem_rig_poses_preint_JPg_s.d_[row33][col33];
  }
  model_inertial_jacobians_jb_gyro_left_ptr[i].d_[row93][col93] = model_inertial_jacobians_jb_gyro_left_val;

  float model_inertial_jacobians_jr_right_val = 0.f;
  if (row93 < 3) {
    float m1_val = twist_right_inverse_jacobian(rot_error, row33, col33);
    model_inertial_jacobians_jr_right_val = m1_val;
  }
  model_inertial_jacobians_jr_right_ptr[i].d_[row93][col93] = model_inertial_jacobians_jr_right_val;

  float model_inertial_jacobians_jt_right_val = 0.f;
  if (row93 >= 6) {
    float m3_val = R1T_s.d_[row33][0] * R2_s.d_[0][col33];
    for (int k = 1; k < 3; ++k) m3_val += R1T_s.d_[row33][k] * R2_s.d_[k][col33];
    model_inertial_jacobians_jt_right_val = m3_val;
  }
  model_inertial_jacobians_jt_right_ptr[i].d_[row93][col93] = model_inertial_jacobians_jt_right_val;

  float model_inertial_jacobians_jv_right_val = 0.f;
  if ((row93 >= 3) && (row93 < 6)) model_inertial_jacobians_jv_right_val = R1T_s.d_[row33][col33];
  model_inertial_jacobians_jv_right_ptr[i].d_[row93][col93] = model_inertial_jacobians_jv_right_val;

  if (lane_id < 3) {
    float model_random_walk_gyro_residuals_val =
        problem_rig_poses_other_s[6 + lane_id] - problem_rig_poses_other_s[12 + 6 + lane_id];
    model_random_walk_gyro_residuals_ptr[i * 3 + lane_id] = model_random_walk_gyro_residuals_val;
  }

  if (lane_id < 3) {
    float model_random_walk_acc_residuals_val =
        problem_rig_poses_other_s[9 + lane_id] - problem_rig_poses_other_s[12 + 9 + lane_id];
    model_random_walk_acc_residuals_ptr[i * 3 + lane_id] = model_random_walk_acc_residuals_val;
  }
}

__global__ void clear_full_system_stage_1_kernel(float* __restrict__ full_system_point_pose_block_transposed,
                                                 int full_system_point_pose_block_transposed_pitch, int num_poses_opt,
                                                 int num_points) {
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
  if ((idx_x < 6 * num_poses_opt) && (idx_y < 3 * num_points)) {
    float* ptr = GET_ELEMENT(full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch,
                             idx_x, idx_y);
    *ptr = 0.f;
  }
}

__global__ void clear_full_system_stage_2_kernel(float* __restrict__ full_system_pose_block,
                                                 int full_system_pose_block_pitch,
                                                 float* __restrict__ full_system_pose_rhs, int num_poses_opt) {
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
  if ((idx_x < 15 * num_poses_opt) && (idx_y < 15 * num_poses_opt)) {
    float* ptr = GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch, idx_x, idx_y);
    *ptr = 0.f;
  }
  if ((idx_x < 15 * num_poses_opt) && (idx_y == 0)) {
    full_system_pose_rhs[idx_x] = 0.f;
  }
}

template <int THREADBLOCK_SIZE>
__global__ void build_full_system_stage_1_kernel(
    const int* __restrict__ problem_point_num_observations, const int* __restrict__ problem_point_start_observation_id,
    const int* __restrict__ problem_point_observation_ids, const float* __restrict__ model_repr_robustifier_weights_ptr,
    const cuvslam::cuda::Matf22* __restrict__ problem_observation_infos_ptr,
    const cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jp_ptr,
    const cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jr_ptr,
    const cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jt_ptr,
    const float* __restrict__ model_reprojection_residuals_ptr, const int* __restrict__ problem_pose_ids,
    cuvslam::cuda::Matf33* __restrict__ full_system_point_block_ptr, float* __restrict__ full_system_point_rhs_ptr,
    float* __restrict__ full_system_point_pose_block_transposed, int full_system_point_pose_block_transposed_pitch,
    int num_observations, int num_points, int num_fixed_key_frames) {
  int warp_id = (blockIdx.x * (THREADBLOCK_SIZE / 32)) + (threadIdx.x >> 5);
  int thread_group_id = (threadIdx.x & 31) / 9;
  if (thread_group_id >= 3) return;
  int idx = warp_id * 3 + thread_group_id;
  if (idx >= num_points) return;
  int elem_id = (threadIdx.x & 31) - thread_group_id * 9;

  int row_id = elem_id / 3;
  int col_id = elem_id - row_id * 3;

  bool full_system_point_rhs_valid = (col_id == 0);

  float full_system_point_block = 0.f;
  float full_system_point_rhs = 0.f;

  int point_num_observations = problem_point_num_observations[idx];
  int point_start_observation_id = problem_point_start_observation_id[idx];
  float hpr_sum = 0.f;
  float hpt_sum = 0.f;
  int id_accum = -1;
  for (int k = 0; k < point_num_observations; ++k) {
    int obs = problem_point_observation_ids[point_start_observation_id + k];
    float weight = model_repr_robustifier_weights_ptr[obs];
    cuvslam::cuda::Matf22 problem_observation_infos = problem_observation_infos_ptr[obs];
    cuvslam::cuda::Vecf2 model_repr_jacobians_jp_col_slice;
    model_repr_jacobians_jp_col_slice.d_[0] = model_repr_jacobians_jp_ptr[obs].d_[0][col_id];
    model_repr_jacobians_jp_col_slice.d_[1] = model_repr_jacobians_jp_ptr[obs].d_[1][col_id];
    cuvslam::cuda::Vecf2 model_repr_jacobians_jp_row_slice;
    model_repr_jacobians_jp_row_slice.d_[0] = model_repr_jacobians_jp_ptr[obs].d_[0][row_id];
    model_repr_jacobians_jp_row_slice.d_[1] = model_repr_jacobians_jp_ptr[obs].d_[1][row_id];
    cuvslam::cuda::Vecf2 model_repr_jacobians_jr;
    model_repr_jacobians_jr.d_[0] = model_repr_jacobians_jr_ptr[obs].d_[0][col_id];
    model_repr_jacobians_jr.d_[1] = model_repr_jacobians_jr_ptr[obs].d_[1][col_id];
    cuvslam::cuda::Vecf2 model_repr_jacobians_jt;
    model_repr_jacobians_jt.d_[0] = model_repr_jacobians_jt_ptr[obs].d_[0][col_id];
    model_repr_jacobians_jt.d_[1] = model_repr_jacobians_jt_ptr[obs].d_[1][col_id];
    cuvslam::cuda::Vecf2 model_reprojection_residuals;
    model_reprojection_residuals.d_[0] = model_reprojection_residuals_ptr[obs * 2];
    model_reprojection_residuals.d_[1] = model_reprojection_residuals_ptr[obs * 2 + 1];
    int pose_idx = problem_pose_ids[obs];
    const int id = pose_idx - num_fixed_key_frames;

    cuvslam::cuda::Vecf2 K = transp(problem_observation_infos) * model_repr_jacobians_jp_row_slice;
    float hpp = dot(K, model_repr_jacobians_jp_col_slice);

    if ((id != id_accum) && (id_accum >= 0)) {
      {
        float* ptr = GET_ELEMENT(full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch,
                                 6 * id_accum + col_id, 3 * idx + row_id);
        *ptr = hpr_sum;
        hpr_sum = 0.f;
      }
      {
        float* ptr = GET_ELEMENT(full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch,
                                 6 * id_accum + 3 + col_id, 3 * idx + row_id);
        *ptr = hpt_sum;
        hpt_sum = 0.f;
      }
    }

    if (pose_idx >= num_fixed_key_frames) {
      id_accum = id;
      float hpr = dot(K, model_repr_jacobians_jr);
      float hpt = dot(K, model_repr_jacobians_jt);
      hpr_sum += hpr * weight;
      hpt_sum += hpt * weight;
    }

    cuvslam::cuda::Vecf2 m = (problem_observation_infos * model_reprojection_residuals);
    full_system_point_block += weight * hpp;
    full_system_point_rhs -= weight * dot(model_repr_jacobians_jp_row_slice, m);
  }

  if (id_accum >= 0) {
    {
      float* ptr = GET_ELEMENT(full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch,
                               6 * id_accum + col_id, 3 * idx + row_id);
      *ptr = hpr_sum;
    }
    {
      float* ptr = GET_ELEMENT(full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch,
                               6 * id_accum + 3 + col_id, 3 * idx + row_id);
      *ptr = hpt_sum;
    }
  }

  full_system_point_block_ptr[idx].d_[row_id][col_id] = full_system_point_block;
  if (full_system_point_rhs_valid) full_system_point_rhs_ptr[idx * 3 + row_id] = full_system_point_rhs;
}

template <int WARP_COUNT>
__global__ void build_full_system_stage_2_kernel(
    const int* __restrict__ problem_pose_num_observations, const int* __restrict__ problem_pose_start_observation_id,
    const int* __restrict__ problem_pose_observation_ids, const float* __restrict__ model_repr_robustifier_weights_ptr,
    const cuvslam::cuda::Matf22* __restrict__ problem_observation_infos_ptr,
    const cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jp_ptr,
    const cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jr_ptr,
    const cuvslam::cuda::Matf23* __restrict__ model_repr_jacobians_jt_ptr,
    const float* __restrict__ model_reprojection_residuals_ptr, float* __restrict__ full_system_pose_block,
    int full_system_pose_block_pitch, float* __restrict__ full_system_pose_rhs, int num_poses,
    int num_fixed_key_frames) {
  int id = blockIdx.x;
  int lane_id = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;

  static __shared__ Matf33 shared_hcc[WARP_COUNT][3];
  static __shared__ Vecf6 shared_rhs[WARP_COUNT];

  cuvslam::cuda::Matf33 sum_hcc[3];
  cuvslam::cuda::Vecf6 sum_rhs;
  for (int i = 0; i < 6; ++i) sum_rhs.d_[i] = 0.f;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) sum_hcc[k].d_[i][j] = 0.f;
  int pose_num_observations = problem_pose_num_observations[id];
  int pose_start_observation_id = problem_pose_start_observation_id[id];
  for (int idx = threadIdx.x; idx < pose_num_observations; idx += WARP_COUNT * 32) {
    int obs = problem_pose_observation_ids[pose_start_observation_id + idx];
    float weight = model_repr_robustifier_weights_ptr[obs];
    cuvslam::cuda::Matf22 problem_observation_infos = problem_observation_infos_ptr[obs];
    cuvslam::cuda::Matf23 model_repr_jacobians_jr = model_repr_jacobians_jr_ptr[obs];
    cuvslam::cuda::Matf23 model_repr_jacobians_jt = model_repr_jacobians_jt_ptr[obs];
    cuvslam::cuda::Vecf2 model_reprojection_residuals;
    model_reprojection_residuals.d_[0] = model_reprojection_residuals_ptr[obs * 2];
    model_reprojection_residuals.d_[1] = model_reprojection_residuals_ptr[obs * 2 + 1];

    cuvslam::cuda::Matf32 K = transp(model_repr_jacobians_jr) * problem_observation_infos;
    Matf33 hrr = K * model_repr_jacobians_jr;
    Matf33 hrt = K * model_repr_jacobians_jt;
    Matf33 htt = transp(model_repr_jacobians_jt) * problem_observation_infos * model_repr_jacobians_jt;

    Vecf2 m = problem_observation_infos * model_reprojection_residuals;
    cuvslam::cuda::Vecf3 v1 = transp(model_repr_jacobians_jr) * m;
    cuvslam::cuda::Vecf3 v2 = transp(model_repr_jacobians_jt) * m;

    for (int i = 0; i < 3; ++i) {
      sum_rhs.d_[i] -= v1.d_[i] * weight;
      sum_rhs.d_[i + 3] -= v2.d_[i] * weight;
      for (int j = 0; j < 3; ++j) {
        sum_hcc[0].d_[i][j] += hrr.d_[i][j] * weight;
        sum_hcc[1].d_[i][j] += hrt.d_[i][j] * weight;
        sum_hcc[2].d_[i][j] += htt.d_[i][j] * weight;
      }
    }
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    for (int i = 0; i < 6; ++i) sum_rhs.d_[i] += __shfl_down_sync(0xffffffff, sum_rhs.d_[i], offset);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) sum_hcc[k].d_[i][j] += __shfl_down_sync(0xffffffff, sum_hcc[k].d_[i][j], offset);
  }

  if (lane_id == 0) {
    shared_rhs[warp_id] = sum_rhs;
    for (int k = 0; k < 3; ++k) shared_hcc[warp_id][k] = sum_hcc[k];
  }

  __syncthreads();

  if (threadIdx.x < 27) {
    int k = lane_id / 9;
    int elem_id = lane_id - 9 * k;
    int i = elem_id / 3;
    int j = elem_id - 3 * i;
    float hcc = 0.f;
    for (int w = 0; w < WARP_COUNT; ++w) hcc += shared_hcc[w][k].d_[i][j];
    float hcc1 = __shfl_down_sync(0xffffffff, hcc, 9);
    float hcc1a = __shfl_sync(0xffffffff, hcc, 9 + j * 3 + i);
    float hcc2 = __shfl_down_sync(0xffffffff, hcc, 18);
    if (lane_id < 9) {
      *GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch, 15 * id + i, 15 * id + j) = hcc;
      *GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch, 15 * id + i, 15 * id + j + 3) = hcc1;
      *GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch, 15 * id + i + 3, 15 * id + j) = hcc1a;
      *GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch, 15 * id + i + 3, 15 * id + j + 3) = hcc2;
    }
  }
  if (threadIdx.x < 6) {
    float rhs = 0.f;
    for (int w = 0; w < WARP_COUNT; ++w) rhs += shared_rhs[w].d_[lane_id];
    full_system_pose_rhs[15 * id + lane_id] = rhs;
  }
}

__global__ void build_full_system_stage_3_kernel(
    const cuvslam::cuda::Matf99* __restrict__ problem_rig_poses_preint_info_matrix__ptr,
    const float* __restrict__ model_inertial_residuals,
    const cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jr_left,
    const cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jt_left,
    const cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jv_left,
    const cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jb_gyro_left,
    const cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jb_acc_left,
    const cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jr_right,
    const cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jt_right,
    const cuvslam::cuda::Matf93* __restrict__ model_inertial_jacobians_jv_right,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_acc_random_walk_accum_info_matrix__ptr,
    const cuvslam::cuda::Matf33* __restrict__ problem_rig_poses_preint_gyro_random_walk_accum_info_matrix__ptr,
    const float* __restrict__ model_random_walk_gyro_residuals_ptr,
    const float* __restrict__ model_random_walk_acc_residuals_ptr, const float* __restrict__ problem_rig_poses_other,
    float* __restrict__ full_system_pose_block, int full_system_pose_block_pitch,
    float* __restrict__ full_system_pose_rhs, int num_poses, int num_fixed_key_frames, float robustifier_scale_pose,
    float imu_penalty, float boundary_imu_penalty, float acc_rw_penalty, float prior_gyro, float prior_acc) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  int id = blockIdx.x;

  int i = id + num_fixed_key_frames;

  if ((x >= 5) && (y >= 5)) return;
  if ((i == (num_poses - 1)) && ((x >= 5) || (y >= 5))) return;

  cuvslam::cuda::Matf33 m;
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      float* full_sys_ptr = GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch,
                                        15 * id + y * 3 + j,  // row
                                        15 * id + x * 3 + k   // col
      );
      m.d_[j][k] = *full_sys_ptr;
    }
  }

  cuvslam::cuda::Vecf3 v;
  if ((x == y) && (x < 5)) {
    for (int j = 0; j < 3; ++j) v.d_[j] = full_system_pose_rhs[15 * id + 3 * x + j];
  }

  if (i < num_poses - 1) {
    int min_xy = min(x, y);
    int max_xy = max(x, y);

    cuvslam::cuda::Matf99 info = problem_rig_poses_preint_info_matrix__ptr[i];
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix__ptr =
        problem_rig_poses_preint_gyro_random_walk_accum_info_matrix__ptr;
    if (((x == 4) || (x == 9)) || ((y == 4) || (y == 9)))
      problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix__ptr =
          problem_rig_poses_preint_acc_random_walk_accum_info_matrix__ptr;

    cuvslam::cuda::Matf33 problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix_ =
        problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix__ptr[i];

    cuvslam::cuda::Vecf9 e;
    for (int j = 0; j < 9; ++j) e.d_[j] = model_inertial_residuals[i * 9 + j];
    float w = ComputeDHuberLoss(dot(e, info * e), robustifier_scale_pose);

    const bool is_acc_block = ((x == 4) || (x == 9)) || ((y == 4) || (y == 9));
    const float rw_penalty_this_edge =
        is_acc_block ? ((acc_rw_penalty >= 0) ? acc_rw_penalty : imu_penalty) : imu_penalty;
    cuvslam::cuda::Matf33 info_gyro_or_acc_rw =
        rw_penalty_this_edge * problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix_;

    if (max_xy < 8) {
      const cuvslam::cuda::Matf93* m_left_ptr;
      switch (min_xy) {
        case 0:
          m_left_ptr = model_inertial_jacobians_jr_left;
          break;
        case 1:
          m_left_ptr = model_inertial_jacobians_jt_left;
          break;
        case 2:
          m_left_ptr = model_inertial_jacobians_jv_left;
          break;
        case 3:
          m_left_ptr = model_inertial_jacobians_jb_gyro_left;
          break;
        case 4:
          m_left_ptr = model_inertial_jacobians_jb_acc_left;
          break;
        case 5:
          m_left_ptr = model_inertial_jacobians_jr_right;
          break;
        case 6:
          m_left_ptr = model_inertial_jacobians_jt_right;
          break;
        case 7:
          m_left_ptr = model_inertial_jacobians_jv_right;
          break;
      }
      cuvslam::cuda::Matf93 m_left = m_left_ptr[i];

      const cuvslam::cuda::Matf93* m_right_ptr;
      switch (max_xy) {
        case 0:
          m_right_ptr = model_inertial_jacobians_jr_left;
          break;
        case 1:
          m_right_ptr = model_inertial_jacobians_jt_left;
          break;
        case 2:
          m_right_ptr = model_inertial_jacobians_jv_left;
          break;
        case 3:
          m_right_ptr = model_inertial_jacobians_jb_gyro_left;
          break;
        case 4:
          m_right_ptr = model_inertial_jacobians_jb_acc_left;
          break;
        case 5:
          m_right_ptr = model_inertial_jacobians_jr_right;
          break;
        case 6:
          m_right_ptr = model_inertial_jacobians_jt_right;
          break;
        case 7:
          m_right_ptr = model_inertial_jacobians_jv_right;
          break;
      }
      cuvslam::cuda::Matf93 m_right = m_right_ptr[i];

      cuvslam::cuda::Matf33 h = (w * imu_penalty) * (transp(m_left) * info * m_right);
      if (y > x) h = transp(h);

      m = m + h;
    }  // if ((x < 8) && (y < 8))

    if (((min_xy == 3) && (max_xy == 8)) || ((min_xy == 4) && (max_xy == 9))) m = m - info_gyro_or_acc_rw;

    if ((x == y) && ((x == 3) || (x == 4))) m = m + info_gyro_or_acc_rw;

    if ((x == y) && (x < 5)) {
      const float* model_random_walk_gyro_or_acc_residuals_ptr = model_random_walk_gyro_residuals_ptr;
      if (x == 4) model_random_walk_gyro_or_acc_residuals_ptr = model_random_walk_acc_residuals_ptr;
      cuvslam::cuda::Vecf3 model_random_walk_gyro_or_acc_residuals;
      for (int j = 0; j < 3; ++j)
        model_random_walk_gyro_or_acc_residuals.d_[j] = model_random_walk_gyro_or_acc_residuals_ptr[i * 3 + j];

      const cuvslam::cuda::Matf93* m_left_ptr;
      switch (x) {
        case 0:
          m_left_ptr = model_inertial_jacobians_jr_left;
          break;
        case 1:
          m_left_ptr = model_inertial_jacobians_jt_left;
          break;
        case 2:
          m_left_ptr = model_inertial_jacobians_jv_left;
          break;
        case 3:
          m_left_ptr = model_inertial_jacobians_jb_gyro_left;
          break;
        case 4:
          m_left_ptr = model_inertial_jacobians_jb_acc_left;
          break;
      }
      cuvslam::cuda::Matf93 m_left = m_left_ptr[i];

      v = v - (w * imu_penalty) * (transp(m_left) * (info * e));

      if ((x == 3) || (x == 4)) {
        v = v - info_gyro_or_acc_rw * model_random_walk_gyro_or_acc_residuals;
      }
    }  // if ((x == y) && (x < 5))
  }    // if (i < num_poses - 1)

  // edge i-1: boundary when i-1 is the last fixed frame (i == num_fixed_key_frames)
  const float eff_penalty_prev = (i == num_fixed_key_frames) ? boundary_imu_penalty : imu_penalty;

  if ((x < 3) && (y < 3)) {
    cuvslam::cuda::Matf99 info = problem_rig_poses_preint_info_matrix__ptr[i - 1];

    cuvslam::cuda::Vecf9 e;
    for (int j = 0; j < 9; ++j) e.d_[j] = model_inertial_residuals[(i - 1) * 9 + j];

    float w = ComputeDHuberLoss(dot(e, info * e), robustifier_scale_pose);

    int min_xy = min(x, y);
    int max_xy = max(x, y);

    const cuvslam::cuda::Matf93* m_left_ptr;
    switch (min_xy) {
      case 0:
        m_left_ptr = model_inertial_jacobians_jr_right;
        break;
      case 1:
        m_left_ptr = model_inertial_jacobians_jt_right;
        break;
      case 2:
        m_left_ptr = model_inertial_jacobians_jv_right;
        break;
    }
    cuvslam::cuda::Matf93 m_left = m_left_ptr[i - 1];

    const cuvslam::cuda::Matf93* m_right_ptr;
    switch (max_xy) {
      case 0:
        m_right_ptr = model_inertial_jacobians_jr_right;
        break;
      case 1:
        m_right_ptr = model_inertial_jacobians_jt_right;
        break;
      case 2:
        m_right_ptr = model_inertial_jacobians_jv_right;
        break;
    }
    cuvslam::cuda::Matf93 m_right = m_right_ptr[i - 1];

    cuvslam::cuda::Matf33 h = (w * eff_penalty_prev) * (transp(m_left) * info * m_right);
    if (y > x) h = transp(h);

    m = m + h;

    if (x == y) {
      const cuvslam::cuda::Matf93* m_left_ptr;
      switch (x) {
        case 0:
          m_left_ptr = model_inertial_jacobians_jr_right;
          break;
        case 1:
          m_left_ptr = model_inertial_jacobians_jt_right;
          break;
        case 2:
          m_left_ptr = model_inertial_jacobians_jv_right;
          break;
      }
      cuvslam::cuda::Matf93 m_left = m_left_ptr[i - 1];
      v = v - (w * eff_penalty_prev) * (transp(m_left) * (info * e));
    }
  }  // if ((x < 3) && (y < 3))

  // random walk
  if ((x == y) && ((x == 3) || (x == 4))) {
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix__ptr =
        problem_rig_poses_preint_gyro_random_walk_accum_info_matrix__ptr;
    if (x == 4)
      problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix__ptr =
          problem_rig_poses_preint_acc_random_walk_accum_info_matrix__ptr;
    cuvslam::cuda::Matf33 problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix_ =
        problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix__ptr[i - 1];
    // Acc random walk (x==4) is NOT boundary-downweighted to keep bias chain strong
    const float rw_penalty = (x == 4) ? ((acc_rw_penalty >= 0) ? acc_rw_penalty : imu_penalty) : eff_penalty_prev;
    cuvslam::cuda::Matf33 info_gyro_or_acc_rw =
        rw_penalty * problem_rig_poses_preint_gyro_or_acc_random_walk_accum_info_matrix_;

    m = m + info_gyro_or_acc_rw;

    {
      const float* model_random_walk_gyro_or_acc_residuals_ptr = model_random_walk_gyro_residuals_ptr;
      if (x == 4) model_random_walk_gyro_or_acc_residuals_ptr = model_random_walk_acc_residuals_ptr;
      cuvslam::cuda::Vecf3 model_random_walk_gyro_or_acc_residuals;
      for (int j = 0; j < 3; ++j)
        model_random_walk_gyro_or_acc_residuals.d_[j] = model_random_walk_gyro_or_acc_residuals_ptr[(i - 1) * 3 + j];

      v = v + info_gyro_or_acc_rw * model_random_walk_gyro_or_acc_residuals;
    }
  }

  // bias priors on oldest non-fixed pose only
  if ((id == 0) && (x == y) && ((x == 3) || (x == 4))) {
    float prior = (x == 3) ? prior_gyro : prior_acc;

    for (int j = 0; j < 3; ++j) m.d_[j][j] += prior;

    cuvslam::cuda::Vecf3 bias;
    for (int j = 0; j < 3; ++j) bias.d_[j] = problem_rig_poses_other[12 * i + x * 3 - 3 + j];

    v = v - prior * bias;
  }

  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      float* full_sys_ptr = GET_ELEMENT(full_system_pose_block, full_system_pose_block_pitch,
                                        15 * id + y * 3 + j,  // row
                                        15 * id + x * 3 + k   // col
      );
      *full_sys_ptr = m.d_[j][k];
    }
  }
  if ((x == y) && (x < 5)) {
    for (int j = 0; j < 3; ++j) full_system_pose_rhs[15 * id + 3 * x + j] = v.d_[j];
  }
}

__global__ void init_update_kernel(cuvslam::cuda::Matf33* __restrict__ update_pose_w_from_imu_linear,
                                   int num_poses_opt) {
  int x = threadIdx.x;
  int y = threadIdx.y;
  int pose_id = threadIdx.z + blockIdx.x * blockDim.z;
  if (pose_id < num_poses_opt) update_pose_w_from_imu_linear[pose_id].d_[y][x] = ((x == y) ? 1.0f : 0.0f);
}

cudaError_t build_reduced_system(const cuvslam::cuda::Matf33* full_system_point_block,
                                 const float* full_system_point_rhs,
                                 const float* full_system_point_pose_block_transposed,
                                 int full_system_point_pose_block_transposed_pitch, const float* full_system_pose_block,
                                 int full_system_pose_block_pitch, const float* full_system_pose_rhs,
                                 float* reduced_system_point_rhs, float* reduced_system_camera_backsub_block_transposed,
                                 int reduced_system_camera_backsub_block_transposed_pitch,
                                 float* reduced_system_pose_block, int reduced_system_pose_block_pitch,
                                 float* reduced_system_pose_rhs, const float* lambda, float threshold, int num_points,
                                 int num_poses, cudaStream_t s) {
  {
    const int WARPS_IN_BLOCK = 1;
    const int THREADBLOCK_SIZE = 32 * WARPS_IN_BLOCK;
    int num_warps = (num_points + 3 - 1) / 3;
    int blocks = (num_warps + WARPS_IN_BLOCK - 1) / WARPS_IN_BLOCK;
    build_reduced_system_stage_1_kernel<THREADBLOCK_SIZE><<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        full_system_point_block, full_system_point_rhs, full_system_point_pose_block_transposed,
        full_system_point_pose_block_transposed_pitch, reduced_system_point_rhs,
        reduced_system_camera_backsub_block_transposed, reduced_system_camera_backsub_block_transposed_pitch, lambda,
        threshold, num_points, num_poses);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  {
    const int THREADBLOCK_SIZE = 64;
    dim3 blocks(15 * num_poses, 15 * num_poses + 1);
    build_reduced_system_stage_2_kernel<THREADBLOCK_SIZE><<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        full_system_point_rhs, full_system_pose_block, full_system_pose_block_pitch,
        full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch, full_system_pose_rhs,
        reduced_system_camera_backsub_block_transposed, reduced_system_camera_backsub_block_transposed_pitch,
        reduced_system_pose_block, reduced_system_pose_block_pitch, reduced_system_pose_rhs, lambda, num_points,
        num_poses);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  return cudaSuccess;
}

cudaError_t calc_update(const float* reduced_system_point_rhs, const float* update_pose_step,
                        const float* reduced_system_camera_backsub_block_transposed,
                        int reduced_system_camera_backsub_block_transposed_pitch, float* update_point_step,
                        float* update_point, cuvslam::cuda::Matf33* update_pose_w_from_imu_linear,
                        float* update_pose_other, int num_points, int num_poses, cudaStream_t s) {
  {
    const int THREADBLOCK_SIZE = 32;
    int blocks = (num_points * 3 + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    calc_update_stage_1_kernel<<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        reduced_system_point_rhs, update_pose_step, reduced_system_camera_backsub_block_transposed,
        reduced_system_camera_backsub_block_transposed_pitch, update_point_step, update_point, num_points, num_poses);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    const int THREADBLOCK_SIZE = 32;
    int blocks = (num_poses + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    calc_update_stage_2_kernel<<<blocks, THREADBLOCK_SIZE, 0, s>>>(update_pose_step, update_pose_w_from_imu_linear,
                                                                   update_pose_other, num_poses);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  return cudaSuccess;
}

cudaError_t evaluate_cost(
    const cuvslam::cuda::Matf33* problem_rig_poses_w_from_imu_linear, const float* problem_rig_poses_other,
    const cuvslam::cuda::Matf33* update_pose_w_from_imu_linear, const float* update_pose_other,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JRg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JVg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JVa,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JPg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JPa, const cuvslam::cuda::Matf33* problem_rig_poses_preint_dR,
    const float* problem_rig_poses_preint_gyro_bias_, const float* problem_rig_poses_preint_acc_bias_,
    const float* problem_rig_poses_preint_dV, const float* problem_rig_poses_preint_dP,
    const float* problem_rig_poses_preint_dT_s, const cuvslam::cuda::Matf99* problem_rig_poses_preint_info_matrix_,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_acc_random_walk_accum_info_matrix_,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_gyro_random_walk_accum_info_matrix_,
    const int* problem_point_ids, const int* problem_pose_ids, const int8_t* problem_camera_ids,
    const float* problem_points, const float* update_point,
    const cuvslam::cuda::Matf33* problem_rig_camera_from_rig_linear,
    const float* problem_rig_camera_from_rig_translation, const float* problem_observation_xys,
    const cuvslam::cuda::Matf22* problem_observation_infos, cuvslam::cuda::Matf33* imu_from_w_linear,
    float* imu_from_w_translation, float* cost, int* num_skipped, float* partial_costs, float threshold, int num_poses,
    int num_observations, int num_fixed_key_frames, float prior_gyro, float prior_acc, float3 gravity,
    float imu_penalty, float boundary_imu_penalty, float acc_rw_penalty, float robustifier_scale_pose,
    float robustifier_scale, const cuvslam::cuda::Matf33& calib_left_from_imu_linear,
    const cuvslam::cuda::Vecf3& calib_left_from_imu_translation, cudaStream_t s) {
  {
    const int THREADBLOCK_SIZE = 32;
    int blocks = (num_poses + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    evaluate_cost_stage_1_kernel<<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        problem_rig_poses_w_from_imu_linear, problem_rig_poses_other, update_pose_w_from_imu_linear, update_pose_other,
        problem_rig_poses_preint_JRg, problem_rig_poses_preint_JVg, problem_rig_poses_preint_JVa,
        problem_rig_poses_preint_JPg, problem_rig_poses_preint_JPa, problem_rig_poses_preint_dR,
        problem_rig_poses_preint_gyro_bias_, problem_rig_poses_preint_acc_bias_, problem_rig_poses_preint_dV,
        problem_rig_poses_preint_dP, problem_rig_poses_preint_dT_s, problem_rig_poses_preint_info_matrix_,
        problem_rig_poses_preint_acc_random_walk_accum_info_matrix_,
        problem_rig_poses_preint_gyro_random_walk_accum_info_matrix_, imu_from_w_linear, imu_from_w_translation, cost,
        threshold, num_poses, num_fixed_key_frames, prior_gyro, prior_acc, gravity, imu_penalty, boundary_imu_penalty,
        acc_rw_penalty, robustifier_scale_pose);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    const int THREADBLOCK_SIZE = 32;
    int blocks = (num_observations + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    int max_partial_costs = (num_observations + 31) / 32;
    evaluate_cost_stage_2_kernel<<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        imu_from_w_linear, imu_from_w_translation, problem_point_ids, problem_pose_ids, problem_camera_ids,
        problem_points, update_point, problem_rig_camera_from_rig_linear, problem_rig_camera_from_rig_translation,
        problem_observation_xys, problem_observation_infos, partial_costs, max_partial_costs, num_skipped,
        num_observations, calib_left_from_imu_linear, calib_left_from_imu_translation, robustifier_scale);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;

    evaluate_cost_stage_2_reduce_kernel<<<1, 32, 0, s>>>(partial_costs, max_partial_costs, cost);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  return cudaSuccess;
}

cudaError_t compute_predicted_reduction(cuvslam::cuda::Matf33* full_system_point_blocks, const float* update_point_step,
                                        const float* update_pose_step,
                                        const float* full_system_point_pose_block_transposed,
                                        int full_system_point_pose_block_transposed_pitch,
                                        const float* full_system_pose_block, int full_system_pose_block_pitch,
                                        float* prediction, int* working_update_point_and_pose_step_significant,
                                        int num_points, int num_poses, const float* lambda,
                                        float max_abs_update_epsilon, cudaStream_t s) {
  {
    const int THREADBLOCK_SIZE = 32;
    int blocks = (num_points + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    compute_predicted_reduction_stage_1_kernel<THREADBLOCK_SIZE><<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        full_system_point_blocks, update_point_step, update_pose_step, full_system_point_pose_block_transposed,
        full_system_point_pose_block_transposed_pitch, prediction, working_update_point_and_pose_step_significant,
        num_points, num_poses, lambda, max_abs_update_epsilon);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    const int TILE_SIZE = 16;
    const int THREADBLOCK_SIZE = TILE_SIZE * TILE_SIZE;
    dim3 blocks((15 * num_poses + TILE_SIZE - 1) / TILE_SIZE, (15 * num_poses + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    compute_predicted_reduction_stage_2_kernel<THREADBLOCK_SIZE><<<blocks, threads, 0, s>>>(
        update_pose_step, full_system_pose_block, full_system_pose_block_pitch, prediction,
        working_update_point_and_pose_step_significant + 1, num_poses, lambda, max_abs_update_epsilon);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  return cudaSuccess;
}

cudaError_t update_state(const float* update_point, const cuvslam::cuda::Matf33* update_pose_w_from_imu_linear,
                         const float* update_pose_other, float* problem_points,
                         cuvslam::cuda::Matf33* problem_rig_poses_w_from_imu_linear, float* problem_rig_poses_other,
                         int num_points, int num_poses, int num_fixed_key_frames, cudaStream_t s) {
  {
    const int THREADBLOCK_SIZE = 64;
    int blocks = (num_points * 3 + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    update_state_stage_1_kernel<<<blocks, THREADBLOCK_SIZE, 0, s>>>(update_point, problem_points, num_points);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    const int THREADBLOCK_SIZE = 32;
    int blocks = ((num_poses - num_fixed_key_frames) + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    update_state_stage_2_kernel<<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        update_pose_w_from_imu_linear, update_pose_other, problem_rig_poses_w_from_imu_linear, problem_rig_poses_other,
        num_poses, num_fixed_key_frames);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  return cudaSuccess;
}

cudaError_t update_model(
    const int* problem_point_ids, const int* problem_pose_ids, const int8_t* problem_camera_ids,
    const float* problem_points, const cuvslam::cuda::Matf33* problem_rig_poses_w_from_imu_linear,
    const float* problem_rig_poses_other, const cuvslam::cuda::Matf33* problem_rig_camera_from_rig_linear,
    const float* problem_rig_camera_from_rig_translation, const float* problem_observation_xys,
    const cuvslam::cuda::Matf22* problem_observation_infos, const cuvslam::cuda::Matf33* problem_rig_poses_preint_JRg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JVg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JVa,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JPg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JPa, const cuvslam::cuda::Matf33* problem_rig_poses_preint_dR,
    const float* problem_rig_poses_preint_gyro_bias_, const float* problem_rig_poses_preint_acc_bias_,
    const float* problem_rig_poses_preint_gyro_bias_diff_, const float* problem_rig_poses_preint_dV,
    const float* problem_rig_poses_preint_dP, const float* problem_rig_poses_preint_dT_s,
    float* model_reprojection_residuals, float* model_repr_robustifier_weights,
    cuvslam::cuda::Matf23* model_repr_jacobians_jt, cuvslam::cuda::Matf23* model_repr_jacobians_jr,
    cuvslam::cuda::Matf23* model_repr_jacobians_jp, float* model_inertial_residuals,
    float* model_random_walk_gyro_residuals, float* model_random_walk_acc_residuals,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jr_left, cuvslam::cuda::Matf93* model_inertial_jacobians_jt_left,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jv_left,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jb_acc_left,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jb_gyro_left,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jr_right, cuvslam::cuda::Matf93* model_inertial_jacobians_jt_right,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jv_right, float threshold, float3 gravity, int num_observations,
    int num_poses, int num_fixed_key_frames, float robustifier_scale,
    const cuvslam::cuda::Matf33& calib_left_from_imu_linear,
    const cuvslam::cuda::Vecf3& calib_left_from_imu_translation, cudaStream_t s) {
  {
    const int THREADBLOCK_SIZE = 32;
    int blocks = (num_observations + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
    update_model_stage_1_kernel<<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        problem_point_ids, problem_pose_ids, problem_camera_ids, problem_points, problem_rig_poses_w_from_imu_linear,
        problem_rig_poses_other, problem_rig_camera_from_rig_linear, problem_rig_camera_from_rig_translation,
        problem_observation_xys, problem_observation_infos, model_reprojection_residuals,
        model_repr_robustifier_weights, model_repr_jacobians_jt, model_repr_jacobians_jr, model_repr_jacobians_jp,
        num_observations, robustifier_scale, calib_left_from_imu_linear, calib_left_from_imu_translation);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    const int WARPS_PER_THREADBLOCK = 1;
    const int THREADBLOCK_SIZE = 32 * WARPS_PER_THREADBLOCK;
    int blocks = ((num_poses - num_fixed_key_frames) + WARPS_PER_THREADBLOCK - 1) / WARPS_PER_THREADBLOCK;
    update_model_stage_2_kernel<THREADBLOCK_SIZE><<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        problem_rig_poses_w_from_imu_linear, problem_rig_poses_other, problem_rig_poses_preint_JRg,
        problem_rig_poses_preint_JVg, problem_rig_poses_preint_JVa, problem_rig_poses_preint_JPg,
        problem_rig_poses_preint_JPa, problem_rig_poses_preint_dR, problem_rig_poses_preint_gyro_bias_,
        problem_rig_poses_preint_acc_bias_, problem_rig_poses_preint_gyro_bias_diff_, problem_rig_poses_preint_dV,
        problem_rig_poses_preint_dP, problem_rig_poses_preint_dT_s, model_inertial_residuals,
        model_random_walk_gyro_residuals, model_random_walk_acc_residuals, model_inertial_jacobians_jr_left,
        model_inertial_jacobians_jt_left, model_inertial_jacobians_jv_left, model_inertial_jacobians_jb_acc_left,
        model_inertial_jacobians_jb_gyro_left, model_inertial_jacobians_jr_right, model_inertial_jacobians_jt_right,
        model_inertial_jacobians_jv_right, threshold, gravity, num_poses, num_fixed_key_frames);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  return cudaSuccess;
}

cudaError_t build_full_system(
    const int* problem_point_num_observations, const int* problem_point_start_observation_id,
    const int* problem_point_observation_ids, const int* problem_pose_num_observations,
    const int* problem_pose_start_observation_id, const int* problem_pose_observation_ids,
    const float* model_repr_robustifier_weights, const cuvslam::cuda::Matf22* problem_observation_infos,
    const cuvslam::cuda::Matf23* model_repr_jacobians_jp, const cuvslam::cuda::Matf23* model_repr_jacobians_jr,
    const cuvslam::cuda::Matf23* model_repr_jacobians_jt, const float* model_reprojection_residuals,
    const int* problem_pose_ids, const cuvslam::cuda::Matf99* problem_rig_poses_preint_info_matrix_,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_acc_random_walk_accum_info_matrix_,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_gyro_random_walk_accum_info_matrix_,
    const float* model_inertial_residuals, const cuvslam::cuda::Matf93* model_inertial_jacobians_jr_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jt_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jv_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jb_acc_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jb_gyro_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jr_right,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jt_right,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jv_right, const float* model_random_walk_gyro_residuals,
    const float* model_random_walk_acc_residuals, const float* problem_rig_poses_other,
    cuvslam::cuda::Matf33* full_system_point_block, float* full_system_point_rhs,
    float* full_system_point_pose_block_transposed, int full_system_point_pose_block_transposed_pitch,
    float* full_system_pose_block, int full_system_pose_block_pitch, float* full_system_pose_rhs, int num_observations,
    int num_points, int num_poses, int num_fixed_key_frames, float robustifier_scale_pose, float imu_penalty,
    float boundary_imu_penalty, float acc_rw_penalty, float prior_gyro, float prior_acc, cudaStream_t s) {
  int num_poses_opt = num_poses - num_fixed_key_frames;

  {
    const int MAX_THREADS_X = 8;
    const int MAX_THREADS_Y = 8;
    dim3 threads_2d(MAX_THREADS_X, MAX_THREADS_Y);
    dim3 blocks_2d((6 * num_poses_opt + MAX_THREADS_X - 1) / MAX_THREADS_X,
                   (3 * num_points + MAX_THREADS_Y - 1) / MAX_THREADS_Y);

    clear_full_system_stage_1_kernel<<<blocks_2d, threads_2d, 0, s>>>(full_system_point_pose_block_transposed,
                                                                      full_system_point_pose_block_transposed_pitch,
                                                                      num_poses_opt, num_points);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    const int MAX_THREADS_X = 8;
    const int MAX_THREADS_Y = 8;
    dim3 threads_2d(MAX_THREADS_X, MAX_THREADS_Y);
    dim3 blocks_2d((15 * num_poses_opt + MAX_THREADS_X - 1) / MAX_THREADS_X,
                   (15 * num_poses_opt + MAX_THREADS_Y - 1) / MAX_THREADS_Y);

    clear_full_system_stage_2_kernel<<<blocks_2d, threads_2d, 0, s>>>(
        full_system_pose_block, full_system_pose_block_pitch, full_system_pose_rhs, num_poses_opt);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    const int WARPS_IN_BLOCK = 1;
    const int THREADBLOCK_SIZE = 32 * WARPS_IN_BLOCK;
    int num_warps = (num_points + 3 - 1) / 3;
    int blocks = (num_warps + WARPS_IN_BLOCK - 1) / WARPS_IN_BLOCK;
    build_full_system_stage_1_kernel<THREADBLOCK_SIZE><<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        problem_point_num_observations, problem_point_start_observation_id, problem_point_observation_ids,
        model_repr_robustifier_weights, problem_observation_infos, model_repr_jacobians_jp, model_repr_jacobians_jr,
        model_repr_jacobians_jt, model_reprojection_residuals, problem_pose_ids, full_system_point_block,
        full_system_point_rhs, full_system_point_pose_block_transposed, full_system_point_pose_block_transposed_pitch,
        num_observations, num_points, num_fixed_key_frames);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    const int WARP_COUNT = 8;
    const int THREADBLOCK_SIZE = WARP_COUNT * 32;
    int blocks = num_poses - num_fixed_key_frames;
    build_full_system_stage_2_kernel<WARP_COUNT><<<blocks, THREADBLOCK_SIZE, 0, s>>>(
        problem_pose_num_observations, problem_pose_start_observation_id, problem_pose_observation_ids,
        model_repr_robustifier_weights, problem_observation_infos, model_repr_jacobians_jp, model_repr_jacobians_jr,
        model_repr_jacobians_jt, model_reprojection_residuals, full_system_pose_block, full_system_pose_block_pitch,
        full_system_pose_rhs, num_poses, num_fixed_key_frames);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  {
    //        printf("num_poses = %d, num_fixed_key_frames = %d\n", num_poses, num_fixed_key_frames);
    dim3 threads_2d(10, 10);
    int blocks = num_poses - num_fixed_key_frames;
    build_full_system_stage_3_kernel<<<blocks, threads_2d, 0, s>>>(
        problem_rig_poses_preint_info_matrix_, model_inertial_residuals, model_inertial_jacobians_jr_left,
        model_inertial_jacobians_jt_left, model_inertial_jacobians_jv_left, model_inertial_jacobians_jb_gyro_left,
        model_inertial_jacobians_jb_acc_left, model_inertial_jacobians_jr_right, model_inertial_jacobians_jt_right,
        model_inertial_jacobians_jv_right, problem_rig_poses_preint_acc_random_walk_accum_info_matrix_,
        problem_rig_poses_preint_gyro_random_walk_accum_info_matrix_, model_random_walk_gyro_residuals,
        model_random_walk_acc_residuals, problem_rig_poses_other, full_system_pose_block, full_system_pose_block_pitch,
        full_system_pose_rhs, num_poses, num_fixed_key_frames, robustifier_scale_pose, imu_penalty,
        boundary_imu_penalty, acc_rw_penalty, prior_gyro, prior_acc);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  return cudaSuccess;
}

cudaError_t init_update(cuvslam::cuda::Matf33* update_pose_w_from_imu_linear, float* update_pose_other,
                        float* update_point, int num_poses_opt, int num_points, cudaStream_t s) {
  {
    const cudaError_t error = cudaMemsetAsync(update_pose_other, 0, num_poses_opt * 12 * sizeof(float), s);
    if (error != cudaSuccess) return error;
  }

  {
    const cudaError_t error = cudaMemsetAsync(update_point, 0, num_points * 3 * sizeof(float), s);
    if (error != cudaSuccess) return error;
  }

  {
    dim3 threads_3d(3, 3, 64 / 9);
    int blocks = (num_poses_opt + threads_3d.z - 1) / threads_3d.z;
    init_update_kernel<<<blocks, threads_3d, 0, s>>>(update_pose_w_from_imu_linear, num_poses_opt);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  return cudaSuccess;
}

}  // namespace cuvslam::cuda::sba_imu
