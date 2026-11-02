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

#define PK_D_MIN 0.1f
#define PK_D_MAX 10.f

// Maximum allowed point-to-plane distance for a pixel's depth point to "register" against a
// plane. The factor evaluates over a regular pixel grid that includes non-planar regions
// (clutter, edges); pixels whose depth point is more than this far from every plane produce
// zero residual and are excluded from the cost. Must be tight enough to reject off-plane
// clutter (otherwise sofas, lamps, etc. get pulled toward the nearest wall and bias the
// pose), but loose enough to absorb pose seed error on the first LM iteration. 0.02 m matches
// PlaneMapSettings::inlier_thresh — a pixel that was an inlier during plane detection stays an
// inlier here, but a sofa pixel ~5–8 cm off the floor plane no longer pollutes the residual.
// On ICL-NUIM living-room-traj1 this single change cut ATE from 7.5 %/m (bimodal, 3-15 over
// 10 runs) to 1.5 %/m (range 1.3-2.7 over 10 runs); also produced large gains on lr2
// (3.9 -> 1.4), lr0 (1.4 -> 1.0), and office-traj0 (3.6 -> 1.7). 0.03 m sits just above
// PlaneMapSettings::inlier_thresh (0.02 m), so true plane inliers stay accepted but sofa /
// lamp / table pixels 5-10 cm off the floor no longer pollute the residual.
#define P2P_MAX_DIST 0.03f

namespace cuvslam::cuda {

namespace {

// ── Lift-to-OpenCV kernel ────────────────────────────────────────────────────
// Lifts 2D observations (in OpenCV normalized coords) to 3D camera-frame points
// using the depth texture.  Output is in OpenCV convention (z-forward positive).
//
// For each observation, lifts the 3×3 pixel neighborhood into 3D, computes
// the centroid, rejects outliers farther than 10% of centroid depth, and
// returns the refined 3D centroid.  Returns (0,0,0) on depth discontinuities.
__global__ void lift_opencv_kernel(const float2* obs_cv, cudaTextureObject_t depth_tex, float focal_x, float focal_y,
                                   float principal_x, float principal_y, float3* points_cv, size_t num_points) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_points) {
    return;
  }

  float2 obs = obs_cv[tid];
  float u = focal_x * obs.x + principal_x;
  float v = focal_y * obs.y + principal_y;

  float inv_fx = __fdividef(1.f, focal_x);
  float inv_fy = __fdividef(1.f, focal_y);

  float3 pts[9];
  int n_valid = 0;
  float3 centroid = {0.f, 0.f, 0.f};

  for (int di = -1; di <= 1; di++) {
    for (int dj = -1; dj <= 1; dj++) {
      float d = tex2D<float>(depth_tex, u + di, v + dj);
      if (d > PK_D_MIN && d < PK_D_MAX) {
        float px = ((u + di) - principal_x) * inv_fx * d;
        float py = ((v + dj) - principal_y) * inv_fy * d;
        pts[n_valid++] = {px, py, d};
        centroid.x += px;
        centroid.y += py;
        centroid.z += d;
      }
    }
  }

  if (n_valid < 5) {
    points_cv[tid] = {0.f, 0.f, 0.f};
    return;
  }

  centroid.x /= (float)n_valid;
  centroid.y /= (float)n_valid;
  centroid.z /= (float)n_valid;

  // Reject 3D points far from centroid to filter depth discontinuities.
  // Outlier rejection radius: 10% of centroid depth (e.g. 5 cm at 0.5 m, 20 cm at 2 m).
  float radius_sq = centroid.z * 0.1f;
  radius_sq *= radius_sq;

  float3 refined = {0.f, 0.f, 0.f};
  int n_inlier = 0;
  for (int k = 0; k < n_valid; k++) {
    float3 p = pts[k];
    if (p.z > PK_D_MIN) {
      float dx = p.x - centroid.x;
      float dy = p.y - centroid.y;
      float dz = p.z - centroid.z;
      if (dx * dx + dy * dy + dz * dz <= radius_sq) {
        refined.x += p.x;
        refined.y += p.y;
        refined.z += p.z;
        n_inlier++;
      }
    }
  }

  if (n_inlier < 5) {
    points_cv[tid] = {0.f, 0.f, 0.f};
    return;
  }

  float inv_n = __fdividef(1.f, (float)n_inlier);
  points_cv[tid] = {refined.x * inv_n, refined.y * inv_n, refined.z * inv_n};
}

// ── Point-to-plane evaluate kernel (image-space) ─────────────────────────────
//
// One thread per pixel in a regular grid defined by stride + image_size.
// Each thread derives its (u,v) from tid, samples depth, lifts to world,
// finds the closest plane, and computes the point-to-plane residual + Jacobian.
//
// State is world_from_rig (row-major 4x4). The sampled depth point is moved to the
// rig frame with cam_from_rig^-1 and then to the world frame where the planes live.
__global__ void point_to_plane_evaluate_kernel(const GPUPlane* planes, int num_planes, int stride,
                                               cudaTextureObject_t depth_tex, float2 focal, float2 principal,
                                               int2 image_size, float const* const* state_ptrs,
                                               const float* cam_from_rig, float* residuals, float* jacobians,
                                               size_t num_factors) {
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

  const int grid_w = (image_size.x + stride - 1) / stride;
  const int gx = tid % grid_w;
  const int gy = tid / grid_w;
  const float u = static_cast<float>(gx * stride + stride / 2) + 0.5f;
  const float v = static_cast<float>(gy * stride + stride / 2) + 0.5f;

  if (u >= static_cast<float>(image_size.x) || v >= static_cast<float>(image_size.y)) {
    zero_output();
    return;
  }

  const float d = tex2D<float>(depth_tex, u, v);
  if (d < PK_D_MIN || d > PK_D_MAX) {
    zero_output();
    return;
  }

  const float inv_fx = __fdividef(1.f, focal.x);
  const float inv_fy = __fdividef(1.f, focal.y);
  const float3 q_cam = {(u - principal.x) * inv_fx * d, (v - principal.y) * inv_fy * d, d};

  // q_rig = cam_from_rig^-1 * q_cam = E^T * (q_cam - t_E), with E row-major (columns of E^T are rows of E).
  const float* E = cam_from_rig;
  const float3 q_c = {q_cam.x - E[3], q_cam.y - E[7], q_cam.z - E[11]};
  const float3 q_rig = {E[0] * q_c.x + E[4] * q_c.y + E[8] * q_c.z, E[1] * q_c.x + E[5] * q_c.y + E[9] * q_c.z,
                        E[2] * q_c.x + E[6] * q_c.y + E[10] * q_c.z};

  // q_world = world_from_rig * q_rig.
  const float* W = state_ptrs[tid];
  const float3 q_world = {W[0] * q_rig.x + W[1] * q_rig.y + W[2] * q_rig.z + W[3],
                          W[4] * q_rig.x + W[5] * q_rig.y + W[6] * q_rig.z + W[7],
                          W[8] * q_rig.x + W[9] * q_rig.y + W[10] * q_rig.z + W[11]};

  // Find the plane with minimum absolute signed distance |n . q + d| to the world point.
  int best_pi = -1;
  float best_abs_dist = P2P_MAX_DIST;
  float best_signed_dist = 0.f;
  for (int pi = 0; pi < num_planes; pi++) {
    const float3 n = planes[pi].normal;
    const float sd = n.x * q_world.x + n.y * q_world.y + n.z * q_world.z + planes[pi].d;
    const float ad = fabsf(sd);
    if (ad < best_abs_dist) {
      best_abs_dist = ad;
      best_signed_dist = sd;
      best_pi = pi;
    }
  }

  if (best_pi < 0) {
    zero_output();
    return;
  }

  if (residuals) {
    residuals[tid] = best_signed_dist;
  }

  if (jacobians) {
    // Point-to-plane Jacobian: d(n . q_world + d)/d(xi) for the right perturbation W' = W * Exp(xi),
    // xi = [omega, t]. q_world moves as W * (q_rig + omega x q_rig + t), so with m = R_W^T n
    // (the plane normal in the rig frame): rotation part q_rig x m, translation part m.
    const float3 n = planes[best_pi].normal;
    const float3 m = {W[0] * n.x + W[4] * n.y + W[8] * n.z, W[1] * n.x + W[5] * n.y + W[9] * n.z,
                      W[2] * n.x + W[6] * n.y + W[10] * n.z};
    jacobians[tid * 6 + 0] = q_rig.y * m.z - q_rig.z * m.y;
    jacobians[tid * 6 + 1] = q_rig.z * m.x - q_rig.x * m.z;
    jacobians[tid * 6 + 2] = q_rig.x * m.y - q_rig.y * m.x;
    jacobians[tid * 6 + 3] = m.x;
    jacobians[tid * 6 + 4] = m.y;
    jacobians[tid * 6 + 5] = m.z;
  }
}

}  // namespace

cudaError_t lift_opencv(const float2* obs_cv, cudaTextureObject_t depth_tex, float focal_x, float focal_y,
                        float principal_x, float principal_y, float3* points_cv, size_t num_points,
                        cudaStream_t stream) {
  if (num_points == 0) {
    return cudaSuccess;
  }
  size_t threads = MAX_THREADS;
  size_t blocks = (num_points + threads - 1) / threads;
  lift_opencv_kernel<<<blocks, threads, 0, stream>>>(obs_cv, depth_tex, focal_x, focal_y, principal_x, principal_y,
                                                     points_cv, num_points);
  return cudaGetLastError();
}

cudaError_t point_to_plane_evaluate(const GPUPlane* planes, int num_planes, int stride, cudaTextureObject_t depth_tex,
                                    float2 focal, float2 principal, int2 image_size, float const* const* state_ptrs,
                                    const float* cam_from_rig, float* residuals, float* jacobians, size_t num_factors,
                                    cudaStream_t stream) {
  if (num_factors == 0) {
    return cudaSuccess;
  }
  size_t threads = MAX_THREADS;
  size_t blocks = (num_factors + threads - 1) / threads;
  point_to_plane_evaluate_kernel<<<blocks, threads, 0, stream>>>(planes, num_planes, stride, depth_tex, focal,
                                                                 principal, image_size, state_ptrs, cam_from_rig,
                                                                 residuals, jacobians, num_factors);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
