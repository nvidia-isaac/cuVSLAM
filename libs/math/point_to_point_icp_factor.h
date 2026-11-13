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

#include <algorithm>

#include <cunls/factor/sized_factor_batch.h>

#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::math {

// Point-to-point ICP cost: residual dim = 1, single SE(3) state block (tangent dim 6).
//
// Projective formulation: the state is world_from_rig (the cuNLS pose convention). cam_from_rig is
// always provided; the kernel composes cam_from_world = cam_from_rig * world_from_rig^-1. Each reference
// landmark is transformed into the current camera, projected to a single pixel, and
// compared against the depth sample there. The residual is the signed depth difference
// r = p_cam.z - d. The kernel also applies a fixed relative-depth gate to robustly
// discard occlusion-edge and wrong-surface matches; the threshold lives alongside
// the kernel (see kMaxRelDepth in point_to_point_icp_kernel.cu).
//
// The batch always reports num_factors active factors. Only the first num_valid_factors landmarks are
// evaluated; the rest get zero residuals and Jacobians, so a persistent batch keeps a fixed size (and the
// solver can reuse the problem structure) while the number of landmarks changes between solves.
class PointToPointICPBatch : public cunls::SizedFactorBatch<1, 6> {
public:
  PointToPointICPBatch(const float3* d_landmarks, const float* d_cam_from_rig, cudaTextureObject_t depth_tex,
                       float2 focal, float2 principal, int2 image_size, size_t num_factors)
      : d_landmarks_(d_landmarks),
        d_cam_from_rig_(d_cam_from_rig),
        depth_tex_(depth_tex),
        focal_(focal),
        principal_(principal),
        image_size_(image_size),
        num_factors_(num_factors),
        num_valid_factors_(num_factors) {}

  // Rebinds the depth image and the number of valid landmarks (at most num_factors) for the next solve.
  void SetFrame(cudaTextureObject_t depth_tex, float2 focal, float2 principal, int2 image_size,
                size_t num_valid_factors) {
    depth_tex_ = depth_tex;
    focal_ = focal;
    principal_ = principal;
    image_size_ = image_size;
    num_valid_factors_ = std::min(num_valid_factors, num_factors_);
  }

  bool Evaluate(float* residuals, float* jacobians, float const* const* state_pointers, cudaStream_t stream,
                const int* factor_ids, size_t num_factor_ids) const final {
    // cuNLS passes the resolved item count; only plain evaluation (item t is factor t) is supported.
    if (factor_ids != nullptr || (num_factor_ids != 0 && num_factor_ids != num_factors_)) {
      return false;
    }
    CUDA_CHECK(cuda::point_to_point_icp_evaluate(d_landmarks_, state_pointers, d_cam_from_rig_, depth_tex_, focal_,
                                                 principal_, image_size_, residuals, jacobians, num_valid_factors_,
                                                 stream));
    const size_t num_padding = num_factors_ - num_valid_factors_;
    if (num_padding > 0) {
      if (residuals != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(residuals + num_valid_factors_, 0, num_padding * sizeof(float), stream));
      }
      if (jacobians != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(jacobians + num_valid_factors_ * 6, 0, num_padding * 6 * sizeof(float), stream));
      }
    }
    return true;
  }

  size_t NumActiveFactors() const final { return num_factors_; }

private:
  const float3* d_landmarks_;
  const float* d_cam_from_rig_;
  cudaTextureObject_t depth_tex_;
  float2 focal_;
  float2 principal_;
  int2 image_size_;
  size_t num_factors_;
  size_t num_valid_factors_;
};

}  // namespace cuvslam::math
