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

#include <cstdio>

#include <assert.h>

#include "cuda_modules/cuda_kernels/cuda_common.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "cuda_modules/cuda_kernels/cuda_matrix.h"

#define BLOCK_X 16
#define BLOCK_Y 16

#define RESIDUAL_MAX 5e-1f
#define D_MIN 0.1f
#define D_MAX 10.f
#define GRAD_THRESH 1e-1f

namespace cuvslam::cuda::matcher {

namespace {
__device__ __forceinline__ void Transform(const Pose& A, const float3& B, float3& result) {
  result = {A.d_[0][0] * B.x + A.d_[0][1] * B.y + A.d_[0][2] * B.z + A.d_[0][3],
            A.d_[1][0] * B.x + A.d_[1][1] * B.y + A.d_[1][2] * B.z + A.d_[1][3],
            A.d_[2][0] * B.x + A.d_[2][1] * B.y + A.d_[2][2] * B.z + A.d_[2][3]};
}

template <typename T>
__device__ __forceinline__ T* get_ptr(const T* ptr, size_t pitch, int2 xy) {
  return (T*)(((char*)ptr + xy.y * pitch) + xy.x * sizeof(T));
}

__device__ float ComputeHuberLoss(float x_squared, const float delta) {
  const auto delta_squared = delta * delta;
  if (x_squared < delta_squared) {
    return 0.5f * x_squared;
  }
  return delta * sqrtf(x_squared) - 0.5f * delta_squared;
}

// Derivative of Huber loss
__device__ float ComputeDHuberLoss(float x_squared, const float delta) {
  const auto delta_squared = delta * delta;
  if (x_squared < delta_squared) {
    return 0.5f;
  }
  return 0.5f * delta * __fdividef(1.f, sqrtf(x_squared));
}

template <typename T>
__device__ void reduce(T& x) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    x += __shfl_down_sync(0xffffffff, x, offset);
  }
}
}  // namespace

__device__ bool magic_depth(float3 lm_world, const ImgTextures& tex, const Inctinsics& intrinsics,
                            const Extrinsics& extr, Matf16& J, float& residual) {
  float3 lm_cam;
  Transform(extr.cam_from_world, lm_world, lm_cam);

  if (lm_cam.z < D_MIN) {
    return false;
  }

  float inv_z = __fdividef(1.f, lm_cam.z);
  float inv_z_sq = inv_z * inv_z;
  float2 tau{intrinsics.focal_x * lm_cam.x * inv_z + intrinsics.principal_x,
             intrinsics.focal_y * lm_cam.y * inv_z + intrinsics.principal_y};

  if (tau.x < 1 || tau.x >= intrinsics.size_x - 1 || tau.y < 1 || tau.y >= intrinsics.size_y - 1) {
    return false;
  }

  float dpatch[9];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      dpatch[i * 3 + j] = tex2D<float>(tex.curr_depth, tau.x - 1 + i, tau.y - 1 + j);
    }
  }

  float dc = dpatch[3 * 1 + 1];

  if (dc < D_MIN) {
    return false;
  }

  float dgx = (dpatch[2] - dpatch[0] + 2 * (dpatch[5] - dpatch[3]) + dpatch[8] - dpatch[6]) * 0.25f;
  float dgy = (dpatch[6] - dpatch[0] + 2 * (dpatch[7] - dpatch[1]) + dpatch[8] - dpatch[2]) * 0.25f;

  if (abs(dgx) > GRAD_THRESH || abs(dgy) > GRAD_THRESH || abs(lm_cam.z - dc) > RESIDUAL_MAX) {
    return false;
  }

  residual = lm_cam.z - dc;

  float ddc_dx = dgx * intrinsics.focal_x * inv_z;
  float ddc_dy = dgy * intrinsics.focal_y * inv_z;
  float ddc_dz = dgx * (-intrinsics.focal_x * lm_cam.x * inv_z_sq) + dgy * (-intrinsics.focal_y * lm_cam.y * inv_z_sq);

  float a = -ddc_dx;
  float b = -ddc_dy;
  float c = 1.f - ddc_dz;

  const auto& R = extr.cam_from_world;
  J.d_[0][0] = a * (R.d_[0][2] * lm_world.y - R.d_[0][1] * lm_world.z) +
               b * (R.d_[1][2] * lm_world.y - R.d_[1][1] * lm_world.z) +
               c * (R.d_[2][2] * lm_world.y - R.d_[2][1] * lm_world.z);

  J.d_[0][1] = a * (R.d_[0][0] * lm_world.z - R.d_[0][2] * lm_world.x) +
               b * (R.d_[1][0] * lm_world.z - R.d_[1][2] * lm_world.x) +
               c * (R.d_[2][0] * lm_world.z - R.d_[2][2] * lm_world.x);

  J.d_[0][2] = a * (R.d_[0][1] * lm_world.x - R.d_[0][0] * lm_world.y) +
               b * (R.d_[1][1] * lm_world.x - R.d_[1][0] * lm_world.y) +
               c * (R.d_[2][1] * lm_world.x - R.d_[2][0] * lm_world.y);

  J.d_[0][3] = a * R.d_[0][0] + b * R.d_[1][0] + c * R.d_[2][0];
  J.d_[0][4] = a * R.d_[0][1] + b * R.d_[1][1] + c * R.d_[2][1];
  J.d_[0][5] = a * R.d_[0][2] + b * R.d_[1][2] + c * R.d_[2][2];
  return true;
}

__global__ void photometric_kernel(ImgTextures tex, Inctinsics intrinsics, Extrinsics extr, const Track* tracks,
                                   size_t num_tracks, float huber, float* cost, float* num_valid, float* rhs,
                                   float* Hessian) {
  float err = 0;
  float valid = 0;
  float H_h[21];
  float rhs_h[6];

  memset(&rhs_h, 0, 6 * sizeof(float));
  memset(&H_h, 0, 21 * sizeof(float));

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < num_tracks) {
    Track track = tracks[x];
    float3 lm_world = track.lm_xyz;

    Matf16 J;
    float residual;
    bool res;
    res = magic_depth(lm_world, tex, intrinsics, extr, J, residual);

    if (res) {
      err = residual * residual;
      assert(err >= 0);

      float sqrt_w = sqrtf(ComputeDHuberLoss(err, huber));
      err = ComputeHuberLoss(err, huber);

      residual *= sqrt_w;

#pragma unroll
      for (int j = 0; j < 6; j++) {
        J.d_[0][j] *= sqrt_w;
      }

      {
        valid = 1;

        // H = J^T * J (6x6 symmetric, stored as lower triangular)
        H_h[0] = J.d_[0][0] * J.d_[0][0];  // 00

        H_h[1] = J.d_[0][1] * J.d_[0][0];  // 10
        H_h[2] = J.d_[0][1] * J.d_[0][1];  // 11

        H_h[3] = J.d_[0][2] * J.d_[0][0];  // 20
        H_h[4] = J.d_[0][2] * J.d_[0][1];  // 21
        H_h[5] = J.d_[0][2] * J.d_[0][2];  // 22

        H_h[6] = J.d_[0][3] * J.d_[0][0];  // 30
        H_h[7] = J.d_[0][3] * J.d_[0][1];  // 31
        H_h[8] = J.d_[0][3] * J.d_[0][2];  // 32
        H_h[9] = J.d_[0][3] * J.d_[0][3];  // 33

        H_h[10] = J.d_[0][4] * J.d_[0][0];  // 40
        H_h[11] = J.d_[0][4] * J.d_[0][1];  // 41
        H_h[12] = J.d_[0][4] * J.d_[0][2];  // 42
        H_h[13] = J.d_[0][4] * J.d_[0][3];  // 43
        H_h[14] = J.d_[0][4] * J.d_[0][4];  // 44

        H_h[15] = J.d_[0][5] * J.d_[0][0];  // 50
        H_h[16] = J.d_[0][5] * J.d_[0][1];  // 51
        H_h[17] = J.d_[0][5] * J.d_[0][2];  // 52
        H_h[18] = J.d_[0][5] * J.d_[0][3];  // 53
        H_h[19] = J.d_[0][5] * J.d_[0][4];  // 54
        H_h[20] = J.d_[0][5] * J.d_[0][5];  // 55

        // rhs = J^T * r
        rhs_h[0] = J.d_[0][0] * residual;
        rhs_h[1] = J.d_[0][1] * residual;
        rhs_h[2] = J.d_[0][2] * residual;
        rhs_h[3] = J.d_[0][3] * residual;
        rhs_h[4] = J.d_[0][4] * residual;
        rhs_h[5] = J.d_[0][5] * residual;
      }
    }
  }
  {
    // reduction
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    assert(num_warps <= 8);

    // constexpr uint8_t sh_pitch = 2 * sizeof(float) + 27 * sizeof(float)
    static __shared__ float sh_floats[29 * 8];  // Shared mem for 8 partial sum

    reduce(err);
    reduce(valid);

#pragma unroll
    for (uint8_t i = 0; i < 6; i++) {
      reduce(rhs_h[i]);
    }

#pragma unroll
    for (uint8_t i = 0; i < 21; i++) {
      reduce(H_h[i]);
    }

    if (lane == 0) {
      // char* ptr = shared_mem + wid * sh_pitch;
      // printf("wid = %d, sh_pitch = %d\n", wid, sh_pitch);
      sh_floats[wid * 29] = err;
      sh_floats[wid * 29 + 1] = valid;

#pragma unroll
      for (uint8_t i = 0; i < 6; i++) {
        sh_floats[wid * 29 + 2 + i] = rhs_h[i];
      }

#pragma unroll
      for (uint8_t i = 0; i < 21; i++) {
        sh_floats[wid * 29 + 8 + i] = H_h[i];
      }
    }
    __syncthreads();

    if (wid == 0) {
      err = 0;
      valid = 0;
      memset(&rhs_h, 0, 6 * sizeof(float));
      memset(&H_h, 0, 21 * sizeof(float));
      if (lane < num_warps) {
        err = sh_floats[lane * 29];
        valid = sh_floats[lane * 29 + 1];

#pragma unroll
        for (uint8_t i = 0; i < 6; i++) {
          rhs_h[i] = sh_floats[lane * 29 + 2 + i];
        }

#pragma unroll
        for (uint8_t i = 0; i < 21; i++) {
          H_h[i] = sh_floats[lane * 29 + 8 + i];
        }
      }
      __syncwarp();

      reduce(err);
      reduce(valid);

#pragma unroll
      for (uint8_t i = 0; i < 6; i++) {
        reduce(rhs_h[i]);
      }

#pragma unroll
      for (uint8_t i = 0; i < 21; i++) {
        reduce(H_h[i]);
      }
    }

    if (wid == 0 && lane == 0) {
      atomicAdd(cost, err);
      atomicAdd(num_valid, valid);

      uint8_t cumsum = 0;

#pragma unroll
      for (uint8_t r = 0; r < 6; r++) {
        atomicAdd(&rhs[r], rhs_h[r]);
        cumsum += r;

#pragma unroll
        for (uint8_t c = 0; c <= r; c++) {
          atomicAdd(&Hessian[r * 6 + c], H_h[cumsum + c]);
        }
      }
    }
  }
}

__device__ bool magic_points(const Track& track, const ImgTextures& tex, const Inctinsics& intrinsics,
                             const Extrinsics& extr, Matf36& J, float3& residual) {
  float u = intrinsics.focal_x * track.obs_xy.x + intrinsics.principal_x;
  float v = intrinsics.focal_y * track.obs_xy.y + intrinsics.principal_y;

  float z = tex2D<float>(tex.curr_depth, u, v);
  if (z < D_MIN || z > D_MAX) {
    return false;
  }

  float3 obs_cam = {track.obs_xy.x * z, track.obs_xy.y * z, z};

  float3 lm_cam;
  Transform(extr.cam_from_world, track.lm_xyz, lm_cam);
  residual = {lm_cam.x - obs_cam.x, lm_cam.y - obs_cam.y, lm_cam.z - obs_cam.z};
  // J = T_cam_from_world.linear() * point_jacobian(lm_world)
  // where point_jacobian(p) = [-[p]_× | I]
  const auto& M = extr.cam_from_world;    // cam_from_world rotation
  const float3& lm_world = track.lm_xyz;  // landmark in world frame

  // Translation part: J[:, 3:6] = R
#pragma unroll
  for (int i = 0; i < 3; i++) {
#pragma unroll
    for (int j = 0; j < 3; j++) {
      J.d_[i][j + 3] = M.d_[i][j];
    }
  }

  // Rotation part: J[:, 0:3] = R * (-[lm_world]_×)
  J.d_[0][0] = M.d_[0][2] * lm_world.y - M.d_[0][1] * lm_world.z;
  J.d_[1][0] = M.d_[1][2] * lm_world.y - M.d_[1][1] * lm_world.z;
  J.d_[2][0] = M.d_[2][2] * lm_world.y - M.d_[2][1] * lm_world.z;

  J.d_[0][1] = M.d_[0][0] * lm_world.z - M.d_[0][2] * lm_world.x;
  J.d_[1][1] = M.d_[1][0] * lm_world.z - M.d_[1][2] * lm_world.x;
  J.d_[2][1] = M.d_[2][0] * lm_world.z - M.d_[2][2] * lm_world.x;

  J.d_[0][2] = M.d_[0][1] * lm_world.x - M.d_[0][0] * lm_world.y;
  J.d_[1][2] = M.d_[1][1] * lm_world.x - M.d_[1][0] * lm_world.y;
  J.d_[2][2] = M.d_[2][1] * lm_world.x - M.d_[2][0] * lm_world.y;

  return true;
}

__global__ void point_to_point_kernel(ImgTextures tex, Inctinsics intrinsics, Extrinsics extr, const Track* tracks,
                                      size_t num_tracks, float huber, float* cost, float* num_valid, float* rhs,
                                      float* Hessian) {
  float err = 0;
  float valid = 0;
  float H_h[21];
  float rhs_h[6];

  memset(&rhs_h, 0, 6 * sizeof(float));
  memset(&H_h, 0, 21 * sizeof(float));

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < num_tracks) {
    Track track = tracks[x];
    Matf36 J;
    float3 residual;
    if (magic_points(track, tex, intrinsics, extr, J, residual)) {
      err = residual.x * residual.x + residual.y * residual.y + residual.z * residual.z;
      assert(err >= 0);

      float sqrt_w = sqrtf(ComputeDHuberLoss(err, huber));
      err = ComputeHuberLoss(err, huber);

      residual.x *= sqrt_w;
      residual.y *= sqrt_w;
      residual.z *= sqrt_w;

#pragma unroll
      for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 6; j++) {
          J.d_[i][j] *= sqrt_w;
        }
      }

      {
        valid = 1;

        H_h[0] = J.d_[0][0] * J.d_[0][0] + J.d_[1][0] * J.d_[1][0] + J.d_[2][0] * J.d_[2][0];  // 00

        H_h[1] = J.d_[0][1] * J.d_[0][0] + J.d_[1][1] * J.d_[1][0] + J.d_[2][1] * J.d_[2][0];  // 10
        H_h[2] = J.d_[0][1] * J.d_[0][1] + J.d_[1][1] * J.d_[1][1] + J.d_[2][1] * J.d_[2][1];  // 11

        H_h[3] = J.d_[0][2] * J.d_[0][0] + J.d_[1][2] * J.d_[1][0] + J.d_[2][2] * J.d_[2][0];  // 20
        H_h[4] = J.d_[0][2] * J.d_[0][1] + J.d_[1][2] * J.d_[1][1] + J.d_[2][2] * J.d_[2][1];  // 21
        H_h[5] = J.d_[0][2] * J.d_[0][2] + J.d_[1][2] * J.d_[1][2] + J.d_[2][2] * J.d_[2][2];  // 22

        H_h[6] = J.d_[0][3] * J.d_[0][0] + J.d_[1][3] * J.d_[1][0] + J.d_[2][3] * J.d_[2][0];  // 30
        H_h[7] = J.d_[0][3] * J.d_[0][1] + J.d_[1][3] * J.d_[1][1] + J.d_[2][3] * J.d_[2][1];  // 31
        H_h[8] = J.d_[0][3] * J.d_[0][2] + J.d_[1][3] * J.d_[1][2] + J.d_[2][3] * J.d_[2][2];  // 32
        H_h[9] = J.d_[0][3] * J.d_[0][3] + J.d_[1][3] * J.d_[1][3] + J.d_[2][3] * J.d_[2][3];  // 33

        H_h[10] = J.d_[0][4] * J.d_[0][0] + J.d_[1][4] * J.d_[1][0] + J.d_[2][4] * J.d_[2][0];  // 40
        H_h[11] = J.d_[0][4] * J.d_[0][1] + J.d_[1][4] * J.d_[1][1] + J.d_[2][4] * J.d_[2][1];  // 41
        H_h[12] = J.d_[0][4] * J.d_[0][2] + J.d_[1][4] * J.d_[1][2] + J.d_[2][4] * J.d_[2][2];  // 42
        H_h[13] = J.d_[0][4] * J.d_[0][3] + J.d_[1][4] * J.d_[1][3] + J.d_[2][4] * J.d_[2][3];  // 43
        H_h[14] = J.d_[0][4] * J.d_[0][4] + J.d_[1][4] * J.d_[1][4] + J.d_[2][4] * J.d_[2][4];  // 44

        H_h[15] = J.d_[0][5] * J.d_[0][0] + J.d_[1][5] * J.d_[1][0] + J.d_[2][5] * J.d_[2][0];  // 50
        H_h[16] = J.d_[0][5] * J.d_[0][1] + J.d_[1][5] * J.d_[1][1] + J.d_[2][5] * J.d_[2][1];  // 51
        H_h[17] = J.d_[0][5] * J.d_[0][2] + J.d_[1][5] * J.d_[1][2] + J.d_[2][5] * J.d_[2][2];  // 52
        H_h[18] = J.d_[0][5] * J.d_[0][3] + J.d_[1][5] * J.d_[1][3] + J.d_[2][5] * J.d_[2][3];  // 53
        H_h[19] = J.d_[0][5] * J.d_[0][4] + J.d_[1][5] * J.d_[1][4] + J.d_[2][5] * J.d_[2][4];  // 54
        H_h[20] = J.d_[0][5] * J.d_[0][5] + J.d_[1][5] * J.d_[1][5] + J.d_[2][5] * J.d_[2][5];  // 55

        rhs_h[0] = J.d_[0][0] * residual.x + J.d_[1][0] * residual.y + J.d_[2][0] * residual.z;
        rhs_h[1] = J.d_[0][1] * residual.x + J.d_[1][1] * residual.y + J.d_[2][1] * residual.z;
        rhs_h[2] = J.d_[0][2] * residual.x + J.d_[1][2] * residual.y + J.d_[2][2] * residual.z;
        rhs_h[3] = J.d_[0][3] * residual.x + J.d_[1][3] * residual.y + J.d_[2][3] * residual.z;
        rhs_h[4] = J.d_[0][4] * residual.x + J.d_[1][4] * residual.y + J.d_[2][4] * residual.z;
        rhs_h[5] = J.d_[0][5] * residual.x + J.d_[1][5] * residual.y + J.d_[2][5] * residual.z;
      }
    }
  }
  {
    // reduction
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    assert(num_warps <= 8);

    // constexpr uint8_t sh_pitch = 2 * sizeof(float) + 27 * sizeof(float)
    static __shared__ float sh_floats[29 * 8];  // Shared mem for 8 partial sum

    reduce(err);
    reduce(valid);

#pragma unroll
    for (uint8_t i = 0; i < 6; i++) {
      reduce(rhs_h[i]);
    }

#pragma unroll
    for (uint8_t i = 0; i < 21; i++) {
      reduce(H_h[i]);
    }

    if (lane == 0) {
      // char* ptr = shared_mem + wid * sh_pitch;
      // printf("wid = %d, sh_pitch = %d\n", wid, sh_pitch);
      sh_floats[wid * 29] = err;
      sh_floats[wid * 29 + 1] = valid;

#pragma unroll
      for (uint8_t i = 0; i < 6; i++) {
        sh_floats[wid * 29 + 2 + i] = rhs_h[i];
      }

#pragma unroll
      for (uint8_t i = 0; i < 21; i++) {
        sh_floats[wid * 29 + 8 + i] = H_h[i];
      }
    }
    __syncthreads();

    if (wid == 0) {
      err = 0;
      valid = 0;
      memset(&rhs_h, 0, 6 * sizeof(float));
      memset(&H_h, 0, 21 * sizeof(float));
      if (lane < num_warps) {
        err = sh_floats[lane * 29];
        valid = sh_floats[lane * 29 + 1];

#pragma unroll
        for (uint8_t i = 0; i < 6; i++) {
          rhs_h[i] = sh_floats[lane * 29 + 2 + i];
        }

#pragma unroll
        for (uint8_t i = 0; i < 21; i++) {
          H_h[i] = sh_floats[lane * 29 + 8 + i];
        }
      }
      __syncwarp();

      reduce(err);
      reduce(valid);

#pragma unroll
      for (uint8_t i = 0; i < 6; i++) {
        reduce(rhs_h[i]);
      }

#pragma unroll
      for (uint8_t i = 0; i < 21; i++) {
        reduce(H_h[i]);
      }
    }

    if (wid == 0 && lane == 0) {
      atomicAdd(cost, err);
      atomicAdd(num_valid, valid);

      uint8_t cumsum = 0;

#pragma unroll
      for (uint8_t r = 0; r < 6; r++) {
        atomicAdd(&rhs[r], rhs_h[r]);
        cumsum += r;

#pragma unroll
        for (uint8_t c = 0; c <= r; c++) {
          atomicAdd(&Hessian[r * 6 + c], H_h[cumsum + c]);
        }
      }
    }
  }
}

__global__ void lift_kernel(cudaTextureObject_t curr_depth, Inctinsics intrinsics, const Track* tracks,
                            size_t num_tracks, float3* landmarks) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= num_tracks) {
    return;
  }

  Track track = tracks[x];

  float u = intrinsics.focal_x * track.obs_xy.x + intrinsics.principal_x;
  float v = intrinsics.focal_y * track.obs_xy.y + intrinsics.principal_y;

  float z = 0;  // tex2D<float>(curr_depth, u, v);
  float num = 0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      float c = tex2D<float>(curr_depth, u + i, v + j);
      if (c > D_MIN && c < D_MAX) {
        z += c;
        num += 1.f;
      }
    }
  }

  if (num < 7) {
    memset(&landmarks[x], 0, sizeof(float3));
    return;
  }

  z /= num;

  landmarks[x] = {track.obs_xy.x * z, track.obs_xy.y * z, z};
}

}  // namespace cuvslam::cuda::matcher

namespace cuvslam::cuda {

cudaError_t photometric(const ImgTextures& tex, const Inctinsics& intrinsics, const Extrinsics& extr,
                        const Track* tracks, size_t num_tracks, float huber, float* cost, float* num_valid, float* rhs,
                        float* Hessian, cudaStream_t stream) {
  size_t threads = MAX_THREADS;
  size_t blocks = (num_tracks + MAX_THREADS - 1) / MAX_THREADS;
  cuvslam::cuda::matcher::photometric_kernel<<<blocks, threads, 0, stream>>>(tex, intrinsics, extr, tracks, num_tracks,
                                                                             huber, cost, num_valid, rhs, Hessian);
  return cudaGetLastError();
}

cudaError_t point_to_point(const ImgTextures& tex, const Inctinsics& intrinsics, const Extrinsics& extr,
                           const Track* tracks, size_t num_tracks, float huber, float* cost, float* num_valid,
                           float* rhs, float* Hessian, cudaStream_t stream) {
  size_t threads = MAX_THREADS;
  size_t blocks = (num_tracks + MAX_THREADS - 1) / MAX_THREADS;
  cuvslam::cuda::matcher::point_to_point_kernel<<<blocks, threads, 0, stream>>>(
      tex, intrinsics, extr, tracks, num_tracks, huber, cost, num_valid, rhs, Hessian);
  return cudaGetLastError();
}

cudaError_t lift(cudaTextureObject_t curr_depth, const Inctinsics& intrinsics, const Track* tracks, size_t num_tracks,
                 float3* landmarks, cudaStream_t stream) {
  size_t threads = MAX_THREADS;
  size_t blocks = (num_tracks + MAX_THREADS - 1) / MAX_THREADS;
  cuvslam::cuda::matcher::lift_kernel<<<blocks, threads, 0, stream>>>(curr_depth, intrinsics, tracks, num_tracks,
                                                                      landmarks);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
