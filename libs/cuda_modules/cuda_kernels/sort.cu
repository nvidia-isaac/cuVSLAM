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

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#if CUDART_VERSION >= 11060
#include <cub/cub.cuh>
#else
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

namespace cuvslam::cuda {

struct KeypointGreater {
  __host__ __device__ bool operator()(const Keypoint& lhs, const Keypoint& rhs) { return lhs.strength > rhs.strength; }
};

// Mixes the position bits, so ties do not favor one part of the image.
__host__ __device__ inline uint32_t PositionHash(const Keypoint& kp) {
  uint32_t x, y;
  memcpy(&x, &kp.x, sizeof(x));
  memcpy(&y, &kp.y, sizeof(y));
  uint32_t h = (x * 0x9E3779B1u) ^ ((y + 0x7F4A7C15u) * 0x85EBCA77u);
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

// Orders equal strengths by position, so the order does not depend on how the keypoints were appended.
struct KeypointGreaterTiesByPosition {
  __host__ __device__ bool operator()(const Keypoint& lhs, const Keypoint& rhs) {
    if (lhs.strength != rhs.strength) {
      return lhs.strength > rhs.strength;
    }
    const uint32_t lhs_hash = PositionHash(lhs);
    const uint32_t rhs_hash = PositionHash(rhs);
    if (lhs_hash != rhs_hash) {
      return lhs_hash < rhs_hash;
    }
    if (lhs.y != rhs.y) {
      return lhs.y < rhs.y;
    }
    return lhs.x < rhs.x;
  }
};

void sort_keypoints(Keypoint* kp, size_t size, void* temp_buffer, size_t temp_buffer_size, bool ties_by_position,
                    cudaStream_t s) {
#if CUDART_VERSION >= 11060
  if (ties_by_position) {
    cub::DeviceMergeSort::SortKeys(temp_buffer, temp_buffer_size, kp, size, KeypointGreaterTiesByPosition(), s);
  } else {
    cub::DeviceMergeSort::SortKeys(temp_buffer, temp_buffer_size, kp, size, KeypointGreater(), s);
  }
#else
  thrust::device_ptr<Keypoint> t_devPtr(kp);
  if (ties_by_position) {
    thrust::stable_sort(t_devPtr, t_devPtr + size, KeypointGreaterTiesByPosition());
  } else {
    thrust::stable_sort(t_devPtr, t_devPtr + size, KeypointGreater());
  }
#endif
}

size_t sort_keypoints_get_temp_buffer_size(size_t size) {
#if CUDART_VERSION >= 11060
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes, (Keypoint*)nullptr, size, KeypointGreater());
  std::size_t ties_by_position_bytes = 0;
  cub::DeviceMergeSort::SortKeys(nullptr, ties_by_position_bytes, (Keypoint*)nullptr, size,
                                 KeypointGreaterTiesByPosition());
  return std::max(temp_storage_bytes, ties_by_position_bytes);
#else
  return 1;
#endif
}

}  // namespace cuvslam::cuda
