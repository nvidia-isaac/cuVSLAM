
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
#include <functional>
#include <optional>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "cuda_modules/gradient_pyramid.h"
#include "cuda_modules/icp_tools.h"
#include "cuda_modules/image_pyramid.h"
#include "pipelines/track.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

namespace cuvslam::pnp {

struct IcpInfo {
  CameraId depth_id;
  const cuda::GaussianGPUImagePyramid& curr_image;
  const cuda::GPUGradientPyramid& curr_grads;
  const cuda::GaussianGPUImagePyramid& curr_depth;

  cuda::Level operator[](int level) const {
    return {curr_depth[level], curr_image[level], curr_grads.gradX()[level], curr_grads.gradY()[level]};
  }
};

struct ICPSettings {
  float lambda = 1e-2;
  float huber_vis = 1e-2;

  float huber_depth = 5e-2;

  int32_t max_iteration = 20;
  bool verbose = false;
  float cost_thresh = 0.6;

  int32_t min_scale_level = 0;
  int32_t max_scale_level = 4;
  int32_t num_iters_per_scale = 20;

  float blending_alpha = 0.8f;
};

class VisualICP {
public:
  explicit VisualICP(const camera::Rig& rig);

  bool solve(Isometry3T& rig_from_world, Matrix6T& static_info_exp,
             const std::vector<camera::Observation>& observations,
             const std::unordered_map<TrackId, Vector3T>& landmarks,  // in map frame!
             const ICPSettings& settings, const IcpInfo* info = nullptr) const;

  void lift_mono_tracks(const IcpInfo& inputs, const Isometry3T& world_from_rig,
                        const std::vector<camera::Observation>& observations,
                        std::vector<cuvslam::pipelines::Landmark>& landmarks) const;

private:
  // Returns false when neither term had a residual, which leaves cost, H and rhs at the empty sum:
  // zero.
  bool total_cost_and_hessian(float& cost, Matrix6T& H, Vector6T& rhs, const Isometry3T& cam_from_world,
                              uint8_t pyramid_level, const ICPSettings& settings,
                              const IcpInfo* depth_info = nullptr) const;

  // Both return false when the term had no residuals. Do not read the outputs in that case.
  bool reprojection_cost_and_hessian(float& cost, Matrix6T& H, Vector6T& rhs, const Isometry3T& cam_from_world,
                                     const ICPSettings& settings) const;

  bool icp_hessian_and_cost(float& cost, Matrix6T& H, Vector6T& rhs, const Isometry3T& cam_from_world,
                            uint8_t pyramid_level, const ICPSettings& settings, const IcpInfo& depth_info) const;

  bool solve_level(Isometry3T& rig_from_world, Matrix6T& static_info_exp, int level, const ICPSettings& settings,
                   const IcpInfo* depth_info, const Isometry3T& cam_from_rig) const;

  camera::Rig rig_;
  cuda::GPUICPTools icp_tools_;

  // mutable std::optional<Isometry3T> prev_delta_;
  mutable Matrix6T motion_prior_;

  mutable std::vector<std::reference_wrapper<const Vector3T>> landmark_for_observation_;
  mutable std::vector<std::reference_wrapper<const camera::Observation>> observations_;

  mutable std::vector<camera::Observation> curr_observations_;

  using ObservationRef = std::reference_wrapper<const camera::Observation>;
  mutable std::vector<std::vector<ObservationRef>> obs_per_camera_;
  mutable std::vector<cuda::GPUICPTools::ObsLmPair> gpu_tracks_;
  mutable std::vector<Vector3T> lifted_landmarks_;

  profiler::PnPProfiler::DomainHelper profiler_domain_ = profiler::PnPProfiler::DomainHelper("VisualICP");
};

}  // namespace cuvslam::pnp
