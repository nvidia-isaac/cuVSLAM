
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

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

struct Level {
  const cuda::GPUImageT& curr_depth;
  const cuda::GPUImageT& curr_image;
  const cuda::GPUImageT& curr_grad_x;
  const cuda::GPUImageT& curr_grad_y;
};

class GPUICPTools {
public:
  explicit GPUICPTools();

  using ObsLmPair = Track;

  // Builds the Gauss-Newton system for one step of the depth-ICP pose refinement. Scores `tracks` -
  // each a 2D observation paired with its world-frame landmark - against the depth map in `level` at
  // the pose `cam_from_world`, and reduces them on the GPU into `cost`, `rhs` (J^T r) and `hessian`
  // (J^T J) for the 6-DOF pose. `focal` and `principal` have to match `level`'s resolution, so a
  // caller working on a pyramid level must scale them down to it.
  //
  // Two Huber-robustified terms contribute: a photometric one comparing each landmark's depth to the
  // map, and a point-to-point one lifting each observation to 3D and measuring the 3D gap. Each is
  // averaged over the tracks it accepted, then the two are blended.
  //
  // Returns false when neither term accepted a track. The outputs are unspecified in that case, so
  // the caller must not read them.
  bool match_and_reduce(float& cost, Vector6T& rhs, Matrix6T& hessian, const Vector2T& focal, const Vector2T& principal,
                        const Level& level, const Isometry3T& cam_from_world, const float& huber,
                        const std::vector<ObsLmPair>& tracks) const;

  void lift_points(const cuda::GPUImageT& dst_depth, const Vector2T& focal, const Vector2T& principal,
                   const std::vector<ObsLmPair>& tracks, std::vector<Vector3T>& landmarks) const;

private:
  mutable GPUArrayPinned<float> pinned_photometric_;
  mutable GPUArrayPinned<float> pinned_point_to_point_;
  mutable GPUArrayPinned<Track> pinned_tracks_{2000};
  mutable GPUArrayPinned<float3> pinned_landmarks_{2000};

  mutable cuda::Stream s_;

  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
};
}  // namespace cuvslam::cuda
