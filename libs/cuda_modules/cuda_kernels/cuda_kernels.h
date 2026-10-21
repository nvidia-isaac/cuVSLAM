
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

#include <cstdint>

#include <driver_types.h>
#include <texture_types.h>
#include <vector_types.h>

#include "cuda_modules/cuda_kernels/cuda_common.h"
#include "cuda_modules/cuda_kernels/cuda_matrix.h"

#define PATCH_DIM 9
#define PATCH_HALF 4.5f

#define PYRAMID_LEVELS 10

namespace cuvslam::cuda {

struct Size {
  int width, height;
};

cudaError_t init_conv_kernels();
cudaError_t init_box_prefilter_kernels();
cudaError_t init_gauss_coeffs();
cudaError_t init_gauss_square_kernel();
cudaError_t init_scaler();
cudaError_t init_st_tracker();

cudaError_t conv_grad_x(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, cudaStream_t s);

cudaError_t conv_grad_y(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, cudaStream_t s);

cudaError_t box_blur_x(cudaTextureObject_t src, uint2 srcSize, float* buffer, size_t buffer_pitch, cudaStream_t s);

cudaError_t box_blur_y(cudaTextureObject_t src, uint2 srcSize, float* buffer, size_t buffer_pitch, cudaStream_t s);

cudaError_t copy_border(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, size_t border_size,
                        cudaStream_t s);

cudaError_t gaussian_scaling(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, uint2 dstSize,
                             cudaStream_t stream);

struct Pyramid {
  int levels;
  cudaTextureObject_t level_tex[PYRAMID_LEVELS];  // max 10 levels
  Size level_sizes[PYRAMID_LEVELS];
};

template <typename T, int Dim>
struct PointCacheData {
  T patch[Dim * Dim * PYRAMID_LEVELS];
  T patch_sums[PYRAMID_LEVELS];
  uint32_t level_mask = 0;
};

using PointCacheDataT = PointCacheData<float, PATCH_DIM>;

struct TrackData {
  float2 track;
  float2 offset;
  float info[4];
  bool track_status;
  int cache_index = -1;

  float4 initial_guess_map = {1, 0, 0, 1};
  float search_radius_px = 2048.f;
  float ncc_threshold = 0.8f;
  float ncc;
};

cudaError_t lk_track(Pyramid prevFrameGradXPyramid, Pyramid prevFrameGradYPyramid, Pyramid prevFrameImagePyramid,
                     Pyramid currentFrameImagePyramid, TrackData* track_data, int num_tracks, cudaStream_t stream);

cudaError_t lk_track_horizontal(Pyramid prevFrameGradXPyramid, Pyramid prevFrameImagePyramid,
                                Pyramid currentFrameImagePyramid, TrackData* track_data, int num_tracks,
                                cudaStream_t stream);

cudaError_t st_track(Pyramid currentGradXPyramid, Pyramid currentGradYPyramid, Pyramid prevImagePyramid,
                     Pyramid currentImagePyramid, TrackData* track_data, int num_tracks,
                     unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations, cudaStream_t stream);

cudaError_t st_build_cache(Pyramid previous_image, TrackData* tracks, PointCacheDataT* cache_data, size_t num_tracks,
                           cudaStream_t stream);

cudaError_t st_track_with_cache(Pyramid currentGradXPyramid, Pyramid currentGradYPyramid, Pyramid currentImagePyramid,
                                PointCacheDataT* cache_data, TrackData* track_data, int num_tracks,
                                unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations,
                                cudaStream_t stream);

cudaError_t gftt_values(cudaTextureObject_t gradX, cudaTextureObject_t gradY, float* values, size_t v_pitch,
                        uint2 image_size, cudaStream_t stream);

struct Keypoint {
  float x, y;
  float strength;
};

cudaError_t downsample_gftt_x8(float* in, size_t in_pitch, float* out, uint2 out_size, size_t out_pitch,
                               uint2* out_indices, size_t out_indices_pitch, cudaStream_t s);

cudaError_t non_max_suppression(cudaTextureObject_t gftt, uint2 size, float* measure, size_t measure_pitch,
                                cudaStream_t s);

cudaError_t filter_maximums(float* measure, uint2 size, size_t measure_pitch, const Keypoint* kp, size_t kpCount,
                            cudaStream_t s);

cudaError_t select_features(float* gftt, uint2 gftt_size, size_t gftt_pitch, float* measure, uint2 size,
                            size_t measure_pitch, uint2* indices, size_t indices_pitch, Keypoint* kp, size_t kpCapacity,
                            int* kpCount, int* kpIndex, cudaStream_t s);

// tile_sums: nullptr adds tiles to the bins with float atomics, else scratch for adding them in a fixed order.
cudaError_t accumulateGFTT(cudaTextureObject_t gftt, uint2 size, uint2 bin_size, uint2 num_bins, float* gtff_accum,
                           float* tile_sums, cudaStream_t s);

size_t accumulateGFTT_get_tile_sums_size(uint2 size);

cudaError_t cast_image(const uint8_t* src, size_t spitch, float* dst, size_t dpitch, uint2 size, cudaStream_t s);

cudaError_t cast_depth_u16(const uint16_t* src, size_t spitch, float scale, float* dst, size_t dpitch, uint2 size,
                           cudaStream_t s);

cudaError_t burn_depth_mask(float* dst, size_t dpitch, uint8_t* mask, size_t mpitch, const uint2& size, cudaStream_t s);

cudaError_t cast_image_rgb(const uint8_t* src, size_t spitch, float* dst, size_t dpitch, uint2 size, cudaStream_t s);

cudaError_t resize_mask(const uint8_t* src, uint2 src_size, size_t spitch, uint8_t* dst, uint2 dst_size, size_t dpitch,
                        cudaStream_t s);

// Sorts by strength, strongest first. ties_by_position orders equal strengths by position, not by input order.
void sort_keypoints(Keypoint* kp, size_t size, void* temp_buffer, size_t temp_buffer_size, bool ties_by_position,
                    cudaStream_t s);

size_t sort_keypoints_get_temp_buffer_size(size_t size);

struct Inctinsics {
  float focal_x;
  float focal_y;

  float principal_x;
  float principal_y;

  int size_x;
  int size_y;
};

struct Extrinsics {
  Pose cam_from_world;
  Pose world_from_cam;
};

struct ImgTextures {
  cudaTextureObject_t curr_depth;   // tex with linear interpolation!
  cudaTextureObject_t curr_image;   // tex with linear interpolation!
  cudaTextureObject_t curr_grad_x;  // tex with linear interpolation!
  cudaTextureObject_t curr_grad_y;  // tex with linear interpolation!
};

struct Track {
  float2 obs_xy;  // normalized undistorted coords (OpenCV-style): (u-cx)/fx, (v-cy)/fy
  float3 lm_xyz;  // landmark in world frame
};

cudaError_t photometric(const ImgTextures& tex, const Inctinsics& intrinsics, const Extrinsics& extr,
                        const Track* tracks, size_t num_tracks, float huber, float* cost, float* num_valid, float* rhs,
                        float* Hessian, cudaStream_t stream);

cudaError_t point_to_point(const ImgTextures& tex, const Inctinsics& intrinsics, const Extrinsics& extr,
                           const Track* tracks, size_t num_tracks, float huber, float* cost, float* num_valid,
                           float* rhs, float* Hessian, cudaStream_t stream);

cudaError_t lift(cudaTextureObject_t curr_depth, const Inctinsics& intrinsics, const Track* tracks, size_t num_tracks,
                 float3* landmarks, cudaStream_t stream);

// Point-to-point ICP evaluate kernel for cuNLS integration (no cuNLS dependency).
// Projective formulation: each landmark is transformed into the current camera,
// projected to a single pixel, and compared against that pixel's depth sample.
// Produces a 1-D residual (p_cam.z - d) and 1x6 Jacobian per factor. The kernel
// also applies a fixed relative-depth gate to discard occlusion-edge matches;
// the threshold (kMaxRelDepth) is defined in point_to_point_icp_kernel.cu.
// cam_from_rig: required device pointer to a single 4×4 SE3 (identity when rig == camera).
cudaError_t point_to_point_icp_evaluate(const float3* landmarks_world, float const* const* state_ptrs,
                                        const float* cam_from_rig, cudaTextureObject_t depth_tex, float2 focal,
                                        float2 principal, int2 image_size, float* residuals, float* jacobians,
                                        size_t num_factors, cudaStream_t stream);

// Lift 2D observations (OpenCV normalized coords) to 3D camera-frame points
// using the depth texture.  Output is in OpenCV convention (z-forward positive).
cudaError_t lift_opencv(const float2* obs_cv, cudaTextureObject_t depth_tex, float focal_x, float focal_y,
                        float principal_x, float principal_y, float3* points_cv, size_t num_points,
                        cudaStream_t stream);

// Unproject valid depth pixels to 3D camera-frame points (OpenCV z-forward).
// Rejects pixels outside [depth_min, depth_max]. Outputs compacted point array via atomic counter.
// stride: sample every Nth pixel in each dimension (1 = full resolution).
cudaError_t unproject_depth_points(cudaTextureObject_t depth_tex, float2 focal, float2 principal, int2 image_size,
                                   float depth_min, float depth_max, float3* out_points, int* out_count, int max_output,
                                   int stride, cudaStream_t stream);

// Count (and optionally gather) inlier points for a given plane among active points.
// plane equation: dot(plane_n, p) + plane_d = 0.  Points with |dist| < thresh are inliers.
// If gather_indices is non-null, inlier point indices are written there (up to max_gather).
// If gather_positions is non-null, the corresponding active-set positions (the tid into
// active_indices, not the point id) are written in the same slot order, so callers can
// scatter into per-active-position auxiliary arrays without a CPU-side reverse map.
cudaError_t count_plane_inliers(const float3* points, const int* active_indices, int num_active, float3 plane_n,
                                float plane_d, float thresh, int* out_count, int* gather_indices, int* gather_positions,
                                int max_gather, cudaStream_t stream);

// Batched RANSAC: evaluate multiple plane hypotheses in parallel.
// One thread-block per hypothesis; threads cooperate via warp-shuffle reduction.
// hypotheses: float4 array (nx, ny, nz, d) of length num_hypotheses.
// out_counts: int array of length num_hypotheses receiving per-hypothesis inlier counts.
cudaError_t ransac_batch_evaluate(const float3* points, const int* active_indices, int num_active,
                                  const float4* hypotheses, float thresh, int* out_counts, int num_hypotheses,
                                  cudaStream_t stream);

// Compute centroid (sum) and upper-triangle covariance (sum of products) of inlier points.
// Output layout: out_centroid_cov[0..2] = (sum_x, sum_y, sum_z),
//                out_centroid_cov[3..8] = (xx, xy, xz, yy, yz, zz).
// Caller divides sums by num_inliers to get mean, then cov_ij = raw_ij/N - mean_i*mean_j.
cudaError_t compute_plane_stats(const float3* points, const int* inlier_indices, int num_inliers,
                                float* out_centroid_cov, cudaStream_t stream);

// Generate plane hypotheses from random index triplets on GPU.
// triplets[i] = (a,b,c) indices into active_indices; kernel reads points and computes normals.
cudaError_t generate_hypotheses(const float3* points, const int* active_indices, const int3* triplets,
                                float4* hypotheses, int num_hypotheses, cudaStream_t stream);

// Compact active indices: keep only entries where keep_flags[i] != 0.
// Writes surviving indices to active_out and increments *out_count (must be zeroed by caller).
cudaError_t compact_active_indices(const int* active_in, const uint8_t* keep_flags, int num_active, int* active_out,
                                   int* out_count, cudaStream_t stream);

// Gather points: out[i] = points[indices[i]] for i < num.  One thread per i.
// Used to materialise a tight inlier-point array on the GPU prior to a small DtoH copy.
cudaError_t gather_inlier_points(const float3* points, const int* indices, int num, float3* out, cudaStream_t stream);

// Stride-subsample integer indices: out[i] = in[i * stride] for i < n_out.  Out-of-range
// reads are clamped to the last element.  Deterministic and preserves spatial diversity
// when the input array is unprojected from a regular pixel grid.
cudaError_t stride_subsample_indices(const int* in, int* out, int stride, int n_out, int n_in, cudaStream_t stream);

// Scatter zeros into keep_flags at the given active-set positions.  Used to mark inlier
// active entries for removal in the next compact_active_indices pass without round-tripping
// to CPU.  Callers must pre-fill keep_flags with 1s (e.g. via cudaMemsetAsync) before calling.
cudaError_t mark_positions_for_removal(uint8_t* keep_flags, const int* positions, int num_positions,
                                       cudaStream_t stream);

// Fill out[0..n) with out[i] = i.  Replaces a host-side iota + memcpy when the consumer
// already runs on the GPU (e.g. seeding RANSAC active-index arrays from the depth point cloud).
cudaError_t fill_iota(int* out, int n, cudaStream_t stream);

// GPU-side plane representation: n . x + d = 0, with unit normal n.
struct GPUPlane {
  float3 normal;
  float d;
};

// Point-to-plane factor evaluation (image-space): one thread per pixel in a
// regular grid defined by stride + image_size. Samples depth, lifts to world,
// finds the closest plane, and computes point-to-plane residual + 1x6 Jacobian.
// State is rig_from_world. cam_from_rig is required (device pointer to a single
// 4x4 SE3, identity when the depth camera coincides with the rig origin); the
// kernel composes cam_from_world = cam_from_rig * rig_from_world internally.
cudaError_t point_to_plane_evaluate(const GPUPlane* planes, int num_planes, int stride, cudaTextureObject_t depth_tex,
                                    float2 focal, float2 principal, int2 image_size, float const* const* state_ptrs,
                                    const float* cam_from_rig, float* residuals, float* jacobians, size_t num_factors,
                                    cudaStream_t stream);

}  // namespace cuvslam::cuda
