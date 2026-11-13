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

#include "cuda_modules/cuda_kernels/cuda_common.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

namespace {

constexpr float kDepthMin = 0.1f;
constexpr float kDepthMax = 10.f;

// Relative-depth rejection threshold for the projective ICP residual.
// Correspondences whose |p_cam.z - d_sampled| / p_cam.z exceeds this value are
// discarded (zero residual + zero Jacobian). The threshold scales with depth
// because stereo/RGBD depth noise itself scales with depth. 2 % is a reasonable
// default for typical stereo sensors; tuning this is a local change.
constexpr float kMaxRelDepth = 2e-2f;

__device__ __forceinline__ void TransformPoint(const float* M, float3 p, float3& result) {
  result.x = M[0] * p.x + M[1] * p.y + M[2] * p.z + M[3];
  result.y = M[4] * p.x + M[5] * p.y + M[6] * p.z + M[7];
  result.z = M[8] * p.x + M[9] * p.y + M[10] * p.z + M[11];
}

// Inverse of a row-major SE(3) matrix (upper 3x4 block only): [R t]^-1 = [R^T, -R^T t].
__device__ __forceinline__ void InvertSE3(const float* A, float* C) {
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      C[r * 4 + c] = A[c * 4 + r];
    }
    C[r * 4 + 3] = -(A[0 * 4 + r] * A[3] + A[1 * 4 + r] * A[7] + A[2 * 4 + r] * A[11]);
  }
}

// Point-to-point ICP evaluate kernel (projective / depth-only formulation).
//
// One thread per reference landmark. The landmark is transformed into the
// current camera, projected to a pixel, and the depth texture is sampled once
// at that pixel. The residual is the 1-D depth disagreement r = p_cam.z - d.
// A correspondence is rejected when |r|/p_cam.z exceeds kMaxRelDepth, which
// robustly discards occlusion-edge and depth-discontinuity matches without the
// anisotropic-noise issues of a full 3-D residual.
//
// Both residual (1) and Jacobian (1x6) have the same compact per-factor stride.
//
// landmarks_world: world-frame 3D reference points from DepthPointMap.
// state_ptrs: per-factor pointers to world_from_rig (SE3, row-major 4x4).
// cam_from_rig: device pointer to a single 4x4 SE3 (always provided).
__global__ void point_to_point_icp_evaluate_kernel(const float3* landmarks_world, float const* const* state_ptrs,
                                                   const float* cam_from_rig, cudaTextureObject_t depth_tex,
                                                   float2 focal, float2 principal, int2 image_size, float* residuals,
                                                   float* jacobians, size_t num_factors) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_factors) {
    return;
  }

  auto zero_output = [&]() {
    if (residuals) {
      residuals[tid] = 0.f;
    }
    if (jacobians) {
      for (int j = 0; j < 6; j++) {
        jacobians[tid * 6 + j] = 0.f;
      }
    }
  };

  float rig_from_world[12];
  InvertSE3(state_ptrs[tid], rig_from_world);

  const float3 p_w = landmarks_world[tid];

  float3 p_rig;
  TransformPoint(rig_from_world, p_w, p_rig);
  float3 p_cam;
  TransformPoint(cam_from_rig, p_rig, p_cam);

  if (p_cam.z < kDepthMin) {
    zero_output();
    return;
  }

  const float inv_z = __fdividef(1.f, p_cam.z);
  const float u0 = focal.x * p_cam.x * inv_z + principal.x;
  const float v0 = focal.y * p_cam.y * inv_z + principal.y;

  const int u0i = __float2int_rn(u0);
  const int v0i = __float2int_rn(v0);

  if (u0i < 0 || u0i >= image_size.x || v0i < 0 || v0i >= image_size.y) {
    zero_output();
    return;
  }

  const float d = tex2D<float>(depth_tex, static_cast<float>(u0i) + 0.5f, static_cast<float>(v0i) + 0.5f);
  if (d < kDepthMin || d > kDepthMax) {
    zero_output();
    return;
  }

  const float r = p_cam.z - d;

  // Relative-depth gate: rejects occlusion-edge matches and points that have
  // moved to a different surface since the reference was sampled. The threshold
  // scales with depth because stereo/RGBD depth noise itself scales with z.
  if (fabsf(r) > kMaxRelDepth * p_cam.z) {
    zero_output();
    return;
  }

  if (residuals) {
    residuals[tid] = r;
  }

  if (jacobians) {
    // dr/dxi = d(p_cam.z)/dxi for the right perturbation W' = W * Exp(xi) of W = world_from_rig.
    // p_rig = W^-1 * p_w moves as p_rig' = p_rig - omega x p_rig - t, and p_cam = cam_from_rig * p_rig,
    // so with e = z-row of the cam_from_rig rotation: dr/domega = e x p_rig, dr/dt = -e.
    //
    // Layout: 1x6 row-major, cols = (omega0, omega1, omega2, t0, t1, t2).
    const float3 e = {cam_from_rig[8], cam_from_rig[9], cam_from_rig[10]};

    jacobians[tid * 6 + 0] = e.y * p_rig.z - e.z * p_rig.y;
    jacobians[tid * 6 + 1] = e.z * p_rig.x - e.x * p_rig.z;
    jacobians[tid * 6 + 2] = e.x * p_rig.y - e.y * p_rig.x;
    jacobians[tid * 6 + 3] = -e.x;
    jacobians[tid * 6 + 4] = -e.y;
    jacobians[tid * 6 + 5] = -e.z;
  }
}

}  // namespace

cudaError_t point_to_point_icp_evaluate(const float3* landmarks_world, float const* const* state_ptrs,
                                        const float* cam_from_rig, cudaTextureObject_t depth_tex, float2 focal,
                                        float2 principal, int2 image_size, float* residuals, float* jacobians,
                                        size_t num_factors, cudaStream_t stream) {
  if (num_factors == 0) {
    return cudaSuccess;
  }
  size_t threads = MAX_THREADS;
  size_t blocks = (num_factors + threads - 1) / threads;
  point_to_point_icp_evaluate_kernel<<<blocks, threads, 0, stream>>>(landmarks_world, state_ptrs, cam_from_rig,
                                                                     depth_tex, focal, principal, image_size, residuals,
                                                                     jacobians, num_factors);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
