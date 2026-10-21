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

#include <cunls/factor/sized_factor_batch.h>

#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::math {

// Point-to-plane cost (image-space): residual dim = 1, single SE(3) state block (tangent dim 6).
// Each factor corresponds to a pixel in a regular grid (stride + image_size). The kernel
// samples depth, lifts to world, finds the closest plane, and computes the point-to-plane
// signed distance as the residual.
//
// The state is world_from_rig (the cuNLS pose convention). cam_from_rig is always provided (device
// pointer to a single 4x4 SE3, identity when the depth camera coincides with the rig origin); the
// kernel composes world_from_cam = world_from_rig * cam_from_rig^-1 internally to lift the sampled
// depth pixel into the world frame where the planes live.
class PointToPlaneCostFunctionBatch : public cunls::SizedFactorBatch<1, 6> {
public:
  PointToPlaneCostFunctionBatch(const cuda::GPUPlane* d_planes, int num_planes, int stride,
                                cudaTextureObject_t depth_tex, float2 focal, float2 principal, int2 image_size,
                                const float* d_cam_from_rig)
      : d_planes_(d_planes),
        num_planes_(num_planes),
        stride_(stride),
        depth_tex_(depth_tex),
        focal_(focal),
        principal_(principal),
        image_size_(image_size),
        d_cam_from_rig_(d_cam_from_rig) {
    // ceil(image_dimension / stride) gives the number of grid cells in each axis
    const int grid_w = (image_size.x + stride - 1) / stride;
    const int grid_h = (image_size.y + stride - 1) / stride;
    num_factors_ = static_cast<size_t>(grid_w) * grid_h;
  }

  bool Evaluate(float* residuals, float* jacobians, float const* const* state_pointers, cudaStream_t stream,
                const int* factor_ids, size_t num_factor_ids) const final {
    // cuNLS passes the resolved item count; only plain evaluation (item t is factor t) is supported.
    if (factor_ids != nullptr || (num_factor_ids != 0 && num_factor_ids != num_factors_)) {
      return false;
    }
    CUDA_CHECK(cuda::point_to_plane_evaluate(d_planes_, num_planes_, stride_, depth_tex_, focal_, principal_,
                                             image_size_, state_pointers, d_cam_from_rig_, residuals, jacobians,
                                             num_factors_, stream));
    return true;
  }

  size_t NumActiveFactors() const final { return num_factors_; }

private:
  const cuda::GPUPlane* d_planes_;
  int num_planes_;
  int stride_;
  cudaTextureObject_t depth_tex_;
  float2 focal_;
  float2 principal_;
  int2 image_size_;
  const float* d_cam_from_rig_;
  size_t num_factors_;
};

}  // namespace cuvslam::math
