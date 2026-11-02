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

#include <cstdint>
#include <cstdio>

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

__global__ void cast_image_kernel(const uint8_t* __restrict__ src, size_t spitch, float* __restrict__ dst,
                                  size_t dpitch, uint2 size) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= size.x || y >= size.y) {
    return;
  }

  uint8_t value = *((uint8_t*)((char*)src + y * spitch) + x);

  *((float*)((char*)dst + y * dpitch) + x) = (float)value;
}

__global__ void cast_depth_kernel(const uint16_t* __restrict__ src, size_t spitch, float scale, float* __restrict__ dst,
                                  size_t dpitch, uint2 size) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= size.x || y >= size.y) {
    return;
  }

  uint16_t value = *((uint16_t*)((char*)src + y * spitch) + x);

  *((float*)((char*)dst + y * dpitch) + x) = ((float)value) / (scale + 1e-9f);
}

__global__ void cast_image_kernel_rgb(const uint8_t* __restrict__ src, size_t spitch, float* __restrict__ dst,
                                      size_t dpitch, uint2 size) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= size.x || y >= size.y) {
    return;
  }

  int idx = y * spitch + x * 3;

  uint8_t red = src[idx];
  uint8_t green = src[idx + 1];
  uint8_t blue = src[idx + 2];

  float value = 0.299f * red + 0.587f * green + 0.114f * blue;

  *((float*)((char*)dst + y * dpitch) + x) = value;
}

__global__ void burn_depth_mask_kernel(float* __restrict__ dst, size_t dpitch, uint8_t* __restrict__ mask,
                                       size_t mpitch, uint2 size) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= size.x || y >= size.y) {
    return;
  }

  uint8_t m_value = *((uint8_t*)((char*)mask + y * mpitch) + x);

  if (m_value > 0) {
    *((float*)((char*)dst + y * dpitch) + x) = 0;
  }
}

__global__ void nearest_neighbor_resize_kernel(const uint8_t* __restrict__ src, float x_scaling_factor,
                                               float y_scaling_factor, size_t spitch, uint8_t* __restrict__ dst,
                                               uint2 dst_size, size_t dpitch) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dst_size.x && y < dst_size.y) {
    int xIn = x * x_scaling_factor;
    int yIn = y * y_scaling_factor;

    int src_offset = x + y * dpitch;
    int dst_offset = xIn + yIn * spitch;
    dst[src_offset] = src[dst_offset];
  }
}

cudaError_t cast_image_rgb(const uint8_t* src, size_t spitch, float* dst, size_t dpitch, uint2 size, cudaStream_t s) {
  dim3 threads(SQRT_MAX_THREADS, SQRT_MAX_THREADS);
  dim3 blocks((size.x + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS, (size.y + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS);

  cast_image_kernel_rgb<<<blocks, threads, 0, s>>>(src, spitch, dst, dpitch, size);
  return cudaGetLastError();
}

cudaError_t cast_image(const uint8_t* src, size_t spitch, float* dst, size_t dpitch, uint2 size, cudaStream_t s) {
  dim3 threads(SQRT_MAX_THREADS, SQRT_MAX_THREADS);
  dim3 blocks((size.x + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS, (size.y + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS);

  cast_image_kernel<<<blocks, threads, 0, s>>>(src, spitch, dst, dpitch, size);
  return cudaGetLastError();
}

cudaError_t cast_depth_u16(const uint16_t* src, size_t spitch, float scale, float* dst, size_t dpitch, uint2 size,
                           cudaStream_t s) {
  dim3 threads(SQRT_MAX_THREADS, SQRT_MAX_THREADS);
  dim3 blocks((size.x + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS, (size.y + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS);

  cast_depth_kernel<<<blocks, threads, 0, s>>>(src, spitch, scale, dst, dpitch, size);
  return cudaGetLastError();
}

cudaError_t burn_depth_mask(float* dst, size_t dpitch, uint8_t* mask, size_t mpitch, const uint2& size,
                            cudaStream_t s) {
  dim3 threads(SQRT_MAX_THREADS, SQRT_MAX_THREADS);
  dim3 blocks((size.x + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS, (size.y + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS);

  burn_depth_mask_kernel<<<blocks, threads, 0, s>>>(dst, dpitch, mask, mpitch, size);
  return cudaGetLastError();
}

cudaError_t resize_mask(const uint8_t* src, uint2 src_size, size_t spitch, uint8_t* dst, uint2 dst_size, size_t dpitch,
                        cudaStream_t s) {
  dim3 dimBlock(32, 32);
  dim3 dimGrid(dst_size.x / 32 + 1, dst_size.y / 32 + 1);
  float x_scaling_factor = (float)src_size.x / (float)dst_size.x;
  float y_scaling_factor = (float)src_size.y / (float)dst_size.y;
  nearest_neighbor_resize_kernel<<<dimGrid, dimBlock, 0, s>>>(src, x_scaling_factor, y_scaling_factor, spitch, dst,
                                                              dst_size, dpitch);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
