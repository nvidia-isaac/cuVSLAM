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

#include <stdio.h>

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

#define GFTT_BLOCK_SIZE 16
#define GFFT_ELEMS_PER_THREAD 2
#define GFFT_TILE_SIZE (GFTT_BLOCK_SIZE * GFFT_ELEMS_PER_THREAD)
#define MIN_GFTT_VALUE 0.f
#define GFTT_THRESH 1e-2

namespace cuvslam::cuda {

__device__ __forceinline__ float reduce(float x, int max_iterations = 5) {
  int lane_mask = 1;
#pragma unroll
  for (int i = 0; i < max_iterations; ++i, lane_mask <<= 1) {
    x += __shfl_xor_sync(0xffffffff, x, lane_mask);
  }
  return x;
}

// block size 32x32 = warpSize * warpSize
__launch_bounds__(GFTT_BLOCK_SIZE* GFTT_BLOCK_SIZE) __global__
    void accumulateGFTT_kernel(cudaTextureObject_t gftt, uint2 size, uint2 bin_size, uint2 num_bins, float* bins_array,
                               float* tile_sums) {
  int base_x = blockIdx.x * GFFT_TILE_SIZE + threadIdx.x;
  int base_y = blockIdx.y * GFFT_TILE_SIZE + threadIdx.y;

  // Shared mem for partial sums
  static __shared__ float shared[GFFT_ELEMS_PER_THREAD * GFFT_ELEMS_PER_THREAD][GFTT_BLOCK_SIZE * GFTT_BLOCK_SIZE / 32];
  int block_internal_idx = threadIdx.y * blockDim.x + threadIdx.x;
  int lane = block_internal_idx & 31;
  int wid = block_internal_idx >> 5;

  float val[GFFT_ELEMS_PER_THREAD * GFFT_ELEMS_PER_THREAD];
  for (int i = 0; i < GFFT_ELEMS_PER_THREAD * GFFT_ELEMS_PER_THREAD; ++i) val[i] = 0.f;

  uint2 bin_coords;

  for (int i = 0; i < GFFT_ELEMS_PER_THREAD; ++i) {
    for (int j = 0; j < GFFT_ELEMS_PER_THREAD; ++j) {
      int y = base_y + i * GFTT_BLOCK_SIZE;
      int x = base_x + j * GFTT_BLOCK_SIZE;
      if (x < size.x && y < size.y) {
        val[i * GFFT_ELEMS_PER_THREAD + j] = tex2D<float>(gftt, (float)x, (float)y);
      }
    }
  }
  if ((lane == 0) && (wid < GFFT_ELEMS_PER_THREAD * GFFT_ELEMS_PER_THREAD)) {
    int i = wid / GFFT_ELEMS_PER_THREAD;
    int j = wid - i * GFFT_ELEMS_PER_THREAD;
    bin_coords = {(blockIdx.x * GFFT_TILE_SIZE + GFTT_BLOCK_SIZE / 2 + j * GFTT_BLOCK_SIZE) / bin_size.x,
                  (blockIdx.y * GFFT_TILE_SIZE + GFTT_BLOCK_SIZE / 2 + i * GFTT_BLOCK_SIZE) / bin_size.y};
  }

  for (int i = 0; i < GFFT_ELEMS_PER_THREAD * GFFT_ELEMS_PER_THREAD; ++i)
    val[i] = reduce(val[i]);  // Each warp performs partial reduction

  if (lane == 0) {
    for (int i = 0; i < GFFT_ELEMS_PER_THREAD * GFFT_ELEMS_PER_THREAD; ++i) shared[i][wid] = val[i];
  }
  __syncthreads();

  if (wid < GFFT_ELEMS_PER_THREAD * GFFT_ELEMS_PER_THREAD) {
    float val = (lane < GFTT_BLOCK_SIZE * GFTT_BLOCK_SIZE / 32) ? shared[wid][lane] : 0;

    float block_sum = reduce(val, 3);

    if (lane == 0) {
      if (tile_sums) {
        const unsigned tile_x = blockIdx.x * GFFT_ELEMS_PER_THREAD + wid % GFFT_ELEMS_PER_THREAD;
        const unsigned tile_y = blockIdx.y * GFFT_ELEMS_PER_THREAD + wid / GFFT_ELEMS_PER_THREAD;
        tile_sums[tile_y * gridDim.x * GFFT_ELEMS_PER_THREAD + tile_x] = block_sum;
      } else if (bin_coords.x < num_bins.x && bin_coords.y < num_bins.y) {
        int bin_id = bin_coords.y * num_bins.x + bin_coords.x;
        atomicAdd(&bins_array[bin_id], block_sum);
      }
    }
  }
}

// One thread per bin adds the sums of the tiles centered in it, row by row, so equal images give equal bins.
__global__ void sum_gftt_tiles_kernel(const float* tile_sums, uint2 num_tiles, uint2 bin_size, uint2 num_bins,
                                      float* bins_array) {
  const unsigned bin_x = threadIdx.x;
  const unsigned bin_y = threadIdx.y;
  if (bin_x >= num_bins.x || bin_y >= num_bins.y) {
    return;
  }
  double sum = 0;
  for (unsigned tile_y = 0; tile_y < num_tiles.y; tile_y++) {
    if ((tile_y * GFTT_BLOCK_SIZE + GFTT_BLOCK_SIZE / 2) / bin_size.y != bin_y) {
      continue;
    }
    for (unsigned tile_x = 0; tile_x < num_tiles.x; tile_x++) {
      if ((tile_x * GFTT_BLOCK_SIZE + GFTT_BLOCK_SIZE / 2) / bin_size.x == bin_x) {
        sum += tile_sums[tile_y * num_tiles.x + tile_x];
      }
    }
  }
  bins_array[bin_y * num_bins.x + bin_x] += static_cast<float>(sum);
}

template <typename T>
__device__ __forceinline__ T& pitched_value(T* ptr, int pitch, int x, int y) {
  return *((T*)((char*)ptr + y * pitch) + x);
}

#define DSMP_TIMES 8
#define DSMP_WARPS_X 2
#define DSMP_WARPS_Y 2
#define DSMP_BLOCK_WIDTH (DSMP_WARPS_X * 32)
#define DSMP_BLOCK_HEIGHT DSMP_WARPS_Y
#define DSMP_MAX_ELEMS_PER_THREAD ((DSMP_TIMES * DSMP_TIMES + 31) / 32)
#define DSMP_ELEMS_PER_WARP_X 2

__launch_bounds__(DSMP_BLOCK_WIDTH* DSMP_BLOCK_HEIGHT) __global__
    void downsample_gftt_x8_kernel(const float* __restrict__ in, int in_pitch, float* __restrict__ out, uint2 out_size,
                                   int out_pitch, uint2* __restrict__ out_indices, int out_indices_pitch) {
  int warp_x = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int warp_y = threadIdx.y;
  int base_x = (blockIdx.x * DSMP_WARPS_X + warp_x) * DSMP_ELEMS_PER_WARP_X;
  int y = blockIdx.y * DSMP_WARPS_Y + warp_y;

  if (base_x >= out_size.x || y >= out_size.y) {
    return;
  }

  int x[DSMP_ELEMS_PER_WARP_X];
#pragma unroll
  for (int i = 0; i < DSMP_ELEMS_PER_WARP_X; ++i) x[i] = min(base_x + i, out_size.x - 1);

  unsigned int x_in[DSMP_ELEMS_PER_WARP_X];
#pragma unroll
  for (int i = 0; i < DSMP_ELEMS_PER_WARP_X; ++i) x_in[i] = DSMP_TIMES * x[i];

  unsigned int y_in = DSMP_TIMES * y;

  float maxMeasure[DSMP_ELEMS_PER_WARP_X];
#pragma unroll
  for (int i = 0; i < DSMP_ELEMS_PER_WARP_X; ++i) maxMeasure[i] = MIN_GFTT_VALUE;
  unsigned int indices_packed[DSMP_ELEMS_PER_WARP_X];
#pragma unroll
  for (int i = 0; i < DSMP_ELEMS_PER_WARP_X; ++i) indices_packed[i] = 0;

  int flat_id = lane;
#pragma unroll
  for (int i = 0; i < DSMP_MAX_ELEMS_PER_THREAD; ++i, flat_id += 32) {
    int dy = flat_id / DSMP_TIMES;
    int dx = flat_id - dy * DSMP_TIMES;
    if ((i < (DSMP_MAX_ELEMS_PER_THREAD - 1)) || (dy < DSMP_TIMES)) {
#pragma unroll
      for (int i = 0; i < DSMP_ELEMS_PER_WARP_X; ++i) {
        float measure = pitched_value(in, in_pitch, x_in[i] + dx, y_in + dy);
        if (measure > maxMeasure[i]) {
          maxMeasure[i] = measure;
          indices_packed[i] = dx | (dy << 16);
        }
      }
    }
  }

  int delta = 1;
#pragma unroll
  for (int i = 0; i < 5; ++i, delta <<= 1) {
#pragma unroll
    for (int i = 0; i < DSMP_ELEMS_PER_WARP_X; ++i) {
      float maxMeasure2 = __shfl_down_sync(0xffffffff, maxMeasure[i], delta);
      uint indices_packed2 = __shfl_down_sync(0xffffffff, indices_packed[i], delta);
      if (maxMeasure2 > maxMeasure[i]) {
        maxMeasure[i] = maxMeasure2;
        indices_packed[i] = indices_packed2;
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < DSMP_ELEMS_PER_WARP_X; ++i) {
      if ((i == 0) || (base_x + i < out_size.x)) {
        pitched_value(out, out_pitch, base_x + i, y) = maxMeasure[i];
        uint2 indices = {x_in[i] + (indices_packed[i] & ((1 << 16) - 1)), y_in + (indices_packed[i] >> 16)};
        pitched_value(out_indices, out_indices_pitch, base_x + i, y) = indices;
      }
    }
  }
}

#define NMS_BLOCK_WIDTH 32
#define NMS_BLOCK_HEIGHT 8
#define NMS_ELEMS_PER_THREAD 2
#define NMS_TILE_WIDTH NMS_BLOCK_WIDTH
#define NMS_TILE_HEIGHT (NMS_BLOCK_HEIGHT * NMS_ELEMS_PER_THREAD)
#define NMS_LOAD_ITERATIONS                                                                    \
  (((NMS_TILE_WIDTH + 2) * (NMS_TILE_HEIGHT + 2) + (NMS_BLOCK_WIDTH * NMS_BLOCK_HEIGHT - 1)) / \
   (NMS_BLOCK_WIDTH * NMS_BLOCK_HEIGHT))

__launch_bounds__(NMS_BLOCK_WIDTH* NMS_BLOCK_HEIGHT) __global__
    void non_max_suppression_kernel(cudaTextureObject_t gftt, uint2 size, float* measure, int measure_pitch) {
  int base_x = blockIdx.x * NMS_TILE_WIDTH;
  int base_y = blockIdx.y * NMS_TILE_HEIGHT;

  static __shared__ float shared[NMS_TILE_HEIGHT + 2][NMS_TILE_WIDTH + 2];

  float vals[NMS_LOAD_ITERATIONS];
  int flat_id = threadIdx.y * NMS_BLOCK_WIDTH + threadIdx.x;
  for (int i = 0; i < NMS_LOAD_ITERATIONS; ++i, flat_id += (NMS_BLOCK_WIDTH * NMS_BLOCK_HEIGHT)) {
    int load_row = flat_id / (NMS_TILE_WIDTH + 2);
    int load_col = flat_id - load_row * (NMS_TILE_WIDTH + 2);

    if ((i < NMS_LOAD_ITERATIONS - 1) || (load_row < (NMS_TILE_HEIGHT + 2))) {
      int in_row = load_row + base_y - 1;
      int in_col = load_col + base_x - 1;
      vals[i] = tex2D<float>(gftt, in_col, in_row);
    }
  }

  flat_id = threadIdx.y * NMS_BLOCK_WIDTH + threadIdx.x;
  for (int i = 0; i < NMS_LOAD_ITERATIONS; ++i, flat_id += (NMS_BLOCK_WIDTH * NMS_BLOCK_HEIGHT)) {
    int load_row = flat_id / (NMS_TILE_WIDTH + 2);
    int load_col = flat_id - load_row * (NMS_TILE_WIDTH + 2);

    if ((i < NMS_LOAD_ITERATIONS - 1) || (load_row < (NMS_TILE_HEIGHT + 2))) {
      shared[load_row][load_col] = vals[i];
    }
  }

  __syncthreads();

  int x = base_x + threadIdx.x;
  int y = base_y + threadIdx.y * NMS_ELEMS_PER_THREAD;

  if (x >= size.x || y >= size.y) {
    return;
  }

#pragma unroll
  for (int s = 0; s < NMS_ELEMS_PER_THREAD; ++s) {
    float centerMeasure = shared[threadIdx.y * NMS_ELEMS_PER_THREAD + s + 1][threadIdx.x + 1];
    bool isMax = true;
#pragma unroll
    for (int t = 0; t < 3 * 3; ++t) {
      int i = t / 3;
      int j = t % 3;
      if ((i != 1) || (j != 1)) {
        isMax = isMax & (shared[threadIdx.y * NMS_ELEMS_PER_THREAD + s + i][threadIdx.x + j] <= centerMeasure);
      }
    }
    if ((s == 0) || ((y + s) < size.y)) {
      pitched_value(measure, measure_pitch, x, y + s) = isMax ? centerMeasure : MIN_GFTT_VALUE;
    }
  }
}

#define FLT_BLOCK_WIDTH 32
#define FLT_WINDOW_HALF_WIDTH 4
#define FLT_WINDOW_SIZE (FLT_WINDOW_HALF_WIDTH * 2 + 1)
#define FLT_STORE_ITERATIONS ((FLT_WINDOW_SIZE * FLT_WINDOW_SIZE + FLT_BLOCK_WIDTH - 1) / FLT_BLOCK_WIDTH)

__launch_bounds__(FLT_BLOCK_WIDTH) __global__
    void filter_maximums_kernel(float* __restrict__ measure, uint2 size, int measure_pitch,
                                const Keypoint* __restrict__ kp, int kpCount) {
  int idx = blockIdx.x;

  if (idx >= kpCount) {
    return;
  }

  const Keypoint keypoint = kp[idx];

  int x = __float2int_rn(keypoint.x);
  int y = __float2int_rn(keypoint.y);

  int flat_id = threadIdx.x;
#pragma unroll
  for (int i = 0; i < FLT_STORE_ITERATIONS; ++i, flat_id += FLT_BLOCK_WIDTH) {
    int dy0 = flat_id / FLT_WINDOW_SIZE;
    int dx0 = flat_id - dy0 * FLT_WINDOW_SIZE;
    int dy = dy0 - FLT_WINDOW_HALF_WIDTH;
    int dx = dx0 - FLT_WINDOW_HALF_WIDTH;
    if ((x + dx >= 0) && (y + dy >= 0) && (x + dx < size.x) && (y + dy < size.y) &&
        ((i < (FLT_STORE_ITERATIONS - 1)) || (dy0 < FLT_WINDOW_SIZE))) {
      pitched_value(measure, measure_pitch, x + dx, y + dy) = MIN_GFTT_VALUE;
    }
  }
}

__device__ __forceinline__ float FindMinimum1D(float y1, float y2, float y3) {
  const float k = y1 - 2.f * y2 + y3;

  const float m = fabs(k) <= (10 * kFloatEpsilon) ? 0 : __fdividef(0.5f * (y1 - y3), k);

  // clamp value in case of float err
  if (m >= 0.5f) {
    return 0.5f;
  }
  if (m <= -0.5f) {
    return -0.5f;
  }
  return m;
}

__device__ __forceinline__ float2 refineFeaturePosition(uint2 c, const float* strength, uint2 size, int pitch) {
  if (c.x == 0 || c.x == size.x - 1 || c.y == 0 || c.y == size.y - 1) {
    return {(float)c.x, (float)c.y};
  }

  const float gfttC = pitched_value(strength, pitch, c.x, c.y);
  const float resX = (float)c.x + FindMinimum1D(pitched_value(strength, pitch, c.x - 1, c.y), gfttC,
                                                pitched_value(strength, pitch, c.x + 1, c.y));
  const float resY = (float)c.y + FindMinimum1D(pitched_value(strength, pitch, c.x, c.y - 1), gfttC,
                                                pitched_value(strength, pitch, c.x, c.y + 1));
  return {resX, resY};
}

__global__ void select_features_kernel(const float* __restrict__ gftt, uint2 gftt_size, size_t gftt_pitch,
                                       const float* __restrict__ measure, uint2 size, int measure_pitch,
                                       const uint2* indices, int indices_pitch, Keypoint* __restrict__ kp,
                                       int kpCapacity, int* __restrict__ kpCount, int* __restrict__ kpIndex) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= size.x || y >= size.y) {
    return;
  }

  const float m = pitched_value(measure, measure_pitch, x, y);

  if (m > (float)GFTT_THRESH) {
    const int kpIdx = atomicAdd(kpIndex, 1);
    if (kpIdx < kpCapacity) {
      const uint2 idx = pitched_value(indices, indices_pitch, x, y);
      float2 pos = refineFeaturePosition(idx, gftt, gftt_size, gftt_pitch);
      kp[kpIdx] = {pos.x, pos.y, m};
      atomicAdd(kpCount, 1);
    }
  }
}

cudaError_t downsample_gftt_x8(float* in, size_t in_pitch, float* out, uint2 out_size, size_t out_pitch,
                               uint2* out_indices, size_t out_indices_pitch, cudaStream_t s) {
  dim3 threads(DSMP_BLOCK_WIDTH, DSMP_BLOCK_HEIGHT);
  size_t num_blocks_x =
      (out_size.x + DSMP_WARPS_X * DSMP_ELEMS_PER_WARP_X - 1) / (DSMP_WARPS_X * DSMP_ELEMS_PER_WARP_X);
  size_t num_blocks_y = (out_size.y + DSMP_WARPS_Y - 1) / DSMP_WARPS_Y;

  dim3 blocks(num_blocks_x, num_blocks_y);

  downsample_gftt_x8_kernel<<<blocks, threads, 0, s>>>(in, in_pitch, out, out_size, out_pitch, out_indices,
                                                       out_indices_pitch);
  return cudaGetLastError();
}

cudaError_t non_max_suppression(cudaTextureObject_t gftt, uint2 size, float* measure, size_t measure_pitch,
                                cudaStream_t s) {
  dim3 threads(NMS_BLOCK_WIDTH, NMS_BLOCK_HEIGHT);
  size_t num_blocks_x = (size.x + NMS_TILE_WIDTH - 1) / NMS_TILE_WIDTH;
  size_t num_blocks_y = (size.y + NMS_TILE_HEIGHT - 1) / NMS_TILE_HEIGHT;
  dim3 blocks(num_blocks_x, num_blocks_y);

  non_max_suppression_kernel<<<blocks, threads, 0, s>>>(gftt, size, measure, measure_pitch);
  return cudaGetLastError();
}

cudaError_t filter_maximums(float* measure, uint2 size, size_t measure_pitch, const Keypoint* kp, size_t kpCount,
                            cudaStream_t s) {
  dim3 threads(FLT_BLOCK_WIDTH);
  dim3 blocks(kpCount);

  filter_maximums_kernel<<<blocks, threads, 0, s>>>(measure, size, measure_pitch, kp, kpCount);
  return cudaGetLastError();
}

cudaError_t select_features(float* gftt, uint2 gftt_size, size_t gftt_pitch, float* measure, uint2 size,
                            size_t measure_pitch, uint2* indices, size_t indices_pitch, Keypoint* kp, size_t kpCapacity,
                            int* kpCount, int* kpIndex, cudaStream_t s) {
  dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
  size_t num_blocks_x = (size.x + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
  size_t num_blocks_y = (size.y + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
  dim3 blocks(num_blocks_x, num_blocks_y);

  select_features_kernel<<<blocks, threads, 0, s>>>(gftt, gftt_size, gftt_pitch, measure, size, measure_pitch, indices,
                                                    indices_pitch, kp, kpCapacity, kpCount, kpIndex);
  return cudaGetLastError();
}

cudaError_t accumulateGFTT(cudaTextureObject_t gftt, uint2 size, uint2 bin_size, uint2 num_bins, float* gtff_accum,
                           float* tile_sums, cudaStream_t s) {
  dim3 threads(GFTT_BLOCK_SIZE, GFTT_BLOCK_SIZE);
  size_t num_blocks_x = (size.x + GFFT_TILE_SIZE - 1) / GFFT_TILE_SIZE;
  size_t num_blocks_y = (size.y + GFFT_TILE_SIZE - 1) / GFFT_TILE_SIZE;
  dim3 blocks(num_blocks_x, num_blocks_y);

  accumulateGFTT_kernel<<<blocks, threads, 0, s>>>(gftt, size, bin_size, num_bins, gtff_accum, tile_sums);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess || tile_sums == nullptr) {
    return error;
  }
  const uint2 num_tiles = {static_cast<unsigned>(num_blocks_x * GFFT_ELEMS_PER_THREAD),
                           static_cast<unsigned>(num_blocks_y * GFFT_ELEMS_PER_THREAD)};
  sum_gftt_tiles_kernel<<<1, dim3(num_bins.x, num_bins.y), 0, s>>>(tile_sums, num_tiles, bin_size, num_bins,
                                                                   gtff_accum);
  return cudaGetLastError();
}

size_t accumulateGFTT_get_tile_sums_size(uint2 size) {
  const size_t num_blocks_x = (size.x + GFFT_TILE_SIZE - 1) / GFFT_TILE_SIZE;
  const size_t num_blocks_y = (size.y + GFFT_TILE_SIZE - 1) / GFFT_TILE_SIZE;
  return num_blocks_x * num_blocks_y * GFFT_ELEMS_PER_THREAD * GFFT_ELEMS_PER_THREAD;
}

}  // namespace cuvslam::cuda
