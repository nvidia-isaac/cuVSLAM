
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

// The near plane guard lives in its own header because the SBA CUDA kernels need it too, and
// camera_selection.h pulls in Eigen and libs/common, neither of which belongs in device code.
// Keep this file free of includes so a .cu can take it as-is.
#if defined(__CUDACC__)
#define CUVSLAM_NEAR_PLANE_FN __host__ __device__ inline
#else
#define CUVSLAM_NEAR_PLANE_FN inline
#endif

namespace cuvslam::epipolar {

namespace FrustumProperties {
/// Minimum positive depth (camera +Z forward, OpenCV) for a point to be considered in front of the camera.
constexpr float MINIMUM_HITHER = 0.1f;
}  // namespace FrustumProperties

/// True when a camera-local depth clears the near plane. Prefer this over a hand-written
/// comparison so every guard treats the boundary the same way.
CUVSLAM_NEAR_PLANE_FN bool IsDepthInFront(float z) { return z > FrustumProperties::MINIMUM_HITHER; }

}  // namespace cuvslam::epipolar
