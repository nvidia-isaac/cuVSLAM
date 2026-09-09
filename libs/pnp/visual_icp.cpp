
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

#include "pnp/visual_icp.h"

#include <iostream>

#include "common/log.h"
#include "common/rotation_utils.h"
#include "math/robust_cost_function.h"
#include "math/twist.h"

namespace cuvslam::pnp {

using Mat36 = Eigen::Matrix<float, 3, 6>;
using Mat23 = Eigen::Matrix<float, 2, 3>;
using Mat26 = Eigen::Matrix<float, 2, 6>;

const float point_z_thresh = 0.01f;
const size_t max_obs_per_camera = 270;

namespace {
Mat36 point_jacobian(const Vector3T& point_3d) {
  Mat36 output;
  output.block<3, 3>(0, 0) = -SkewSymmetric(point_3d);
  output.block<3, 3>(0, 3) = Matrix3T::Identity();
  return output;
}

Mat23 projection_jacobian(const Vector3T& point_3d) {
  float one_over_z = 1.f / point_3d.z();
  Mat23 output;
  output << one_over_z, 0.f, -point_3d.x() * one_over_z * one_over_z, 0.f, one_over_z,
      -point_3d.y() * one_over_z * one_over_z;
  return output;
}
}  // namespace

VisualICP::VisualICP(const camera::Rig& rig) : rig_(rig) {
  motion_prior_ = Matrix6T::Identity() * 5e-2;
  obs_per_camera_.resize(rig_.num_cameras);
}

bool VisualICP::reprojection_cost_and_hessian(float& cost, Matrix6T& H, Vector6T& rhs, const Isometry3T& cam_from_world,
                                              const ICPSettings& settings) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("reprojection_cost_and_hessian");
  cost = 0;

  Vector3T point_cam;

  float count = 0;

  for (size_t obs_id = 0; obs_id < observations_.size(); obs_id++) {
    const auto& obs = observations_[obs_id].get();
    const Vector3T& landmark_world = landmark_for_observation_[obs_id].get();

    point_cam = cam_from_world * landmark_world;

    if (point_cam.z() <= point_z_thresh) continue;

    count++;

    float one_over_z = 1.f / point_cam.z();
    Vector2T r = point_cam.topRows(2) * one_over_z - obs.xy;
    float r_squared_norm = r.dot(obs.xy_info * r);

    float w = math::ComputeDHuberLoss(r_squared_norm, settings.huber_vis);
    Mat26 J = projection_jacobian(point_cam) * cam_from_world.linear() * point_jacobian(landmark_world);

    H += J.transpose() * obs.xy_info * w * J;
    rhs += J.transpose() * obs.xy_info * w * r;

    cost += math::ComputeHuberLoss(r_squared_norm, settings.huber_vis);
  }

  // Nothing projected in front of the camera. H and rhs were zeroed by the caller and never added
  // to, so they already carry the empty sum.
  if (count <= 0) {
    return false;
  }

  H = H / count;
  rhs = rhs / count;
  cost = cost / count;
  return true;
}

bool VisualICP::icp_hessian_and_cost(float& cost, Matrix6T& H, Vector6T& rhs, const Isometry3T& cam_from_world,
                                     uint8_t pyramid_level, const ICPSettings& settings, const IcpInfo& inputs) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("icp_hessian_and_cost");

  const auto& intrinsics = *rig_.intrinsics[inputs.depth_id];
  float scale_x =
      static_cast<float>(inputs.curr_depth[0].cols()) / static_cast<float>(inputs.curr_depth[pyramid_level].cols());
  float scale_y =
      static_cast<float>(inputs.curr_depth[0].rows()) / static_cast<float>(inputs.curr_depth[pyramid_level].rows());

  Vector2T focal = intrinsics.getFocal();
  Vector2T principal = intrinsics.getPrincipal();

  focal = {focal.x() / scale_x, focal.y() / scale_y};
  principal = {principal.x() / scale_x, principal.y() / scale_y};

  gpu_tracks_.clear();
  gpu_tracks_.reserve(observations_.size());
  for (size_t i = 0; i < observations_.size(); i++) {
    const camera::Observation& obs = observations_[i].get();
    const Vector3T& lm = landmark_for_observation_[i].get();
    gpu_tracks_.push_back({{obs.xy.x(), obs.xy.y()}, {lm.x(), lm.y(), lm.z()}});
  }

  const bool have_residuals = icp_tools_.match_and_reduce(cost, rhs, H, focal, principal, inputs[pyramid_level],
                                                          cam_from_world, settings.huber_depth, gpu_tracks_);

  // if (prev_delta_) {
  //   H += motion_prior_;

  //   Vector6T twist_prev;
  //   Vector6T twist_curr;

  //   math::Log(twist_prev, prev_delta_.value());
  //   math::Log(twist_curr, cam_from_world);
  //   rhs -= motion_prior_ * (twist_prev - twist_curr);
  // }
  return have_residuals;
}

bool VisualICP::total_cost_and_hessian(float& cost, Matrix6T& H, Vector6T& rhs, const Isometry3T& rig_from_world,
                                       uint8_t pyramid_level, const ICPSettings& settings,
                                       const IcpInfo* depth_info) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("total_cost_and_hessian");

  H.setZero();
  rhs.setZero();

  cost = 0.f;
  const bool have_reprojection = reprojection_cost_and_hessian(cost, H, rhs, rig_from_world, settings);
  bool have_icp = false;

  if (depth_info) {
    float icp_cost = 0.f;
    Vector6T rhs_icp;
    Matrix6T H_icp;
    have_icp = icp_hessian_and_cost(icp_cost, H_icp, rhs_icp, rig_from_world, pyramid_level, settings, *depth_info);

    // icp_cost / H_icp / rhs_icp carry nothing when the ICP term reports no residuals, so do not
    // read them at all - not even against a zero weight, because 0 * NaN is NaN. H, rhs and cost
    // already hold the reprojection term at full weight, which is what an absent ICP term means.
    if (have_icp) {
      // A reprojection term with no residuals has no average to contribute, so the ICP term takes
      // the whole weight rather than being averaged against a zero cost.
      const float alpha = have_reprojection ? settings.blending_alpha : 0.f;

      rhs = alpha * rhs + (1 - alpha) * rhs_icp;
      H = alpha * H + (1 - alpha) * H_icp;
      cost = alpha * cost + (1 - alpha) * icp_cost;
    }
  }

  return have_reprojection || have_icp;
}

using obs_ref = std::reference_wrapper<const camera::Observation>;

bool VisualICP::solve_level(Isometry3T& rig_from_world, Matrix6T& static_info_exp, int level,
                            const ICPSettings& settings, const IcpInfo* depth_info,
                            const Isometry3T& cam_from_rig) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("solve");

  Matrix6T H;
  Vector6T rhs;

  Isometry3T param_r_from_world = rig_from_world;

  // Compute cam_from_world from rig_from_world
  Isometry3T cam_from_world = cam_from_rig * param_r_from_world;

  float initial_cost = 0.f;
  if (!total_cost_and_hessian(initial_cost, H, rhs, cam_from_world, level, settings, depth_info)) {
    // No term has a residual, so H and rhs are the empty sum. There is no step to solve for and no
    // information to report. Note this is about having no data at all, not about a cost of zero -
    // a set of residuals that happens to fit perfectly still goes through the loop below.
    static_info_exp.setZero();
    return false;
  }
  // if (initial_cost < 5e-3f) {
  //     // Nothing to minimize. We have a "pendulum" effect on still frames otherwise.
  //     static_info_exp.setIdentity();
  //     return true;
  // }
  float current_cost = initial_cost;

  Vector6T scaling = H.diagonal();

  float lambda = settings.lambda;
  const int num_iters = depth_info ? settings.num_iters_per_scale : settings.max_iteration;
  int num_iterations = 0;
  do {
    // A cost of zero means every residual is zero, so rhs is zero and there is no step left to
    // take. Stop before the gain ratio instead of dividing by it, which hands prr and rho a NaN.
    // Huber losses never go negative, so this only catches the perfect fit.
    if (current_cost <= 0.f) {
      break;
    }

    ++num_iterations;
    Matrix6T augmented_system = H + (lambda * scaling).asDiagonal().toDenseMatrix();
    auto decomposition = augmented_system.cast<double>().ldlt();
    Vector6T step = decomposition.solve(-rhs.cast<double>()).cast<float>();

    // Apply perturbation to param_r_from_world (right perturbation)
    Isometry3T delta;
    math::Exp(delta, step);
    Isometry3T rig_guess = param_r_from_world * delta;
    rig_guess.linear() = common::CalculateRotationFromSVD(rig_guess.matrix());

    // Compute cam_from_world for cost evaluation
    Isometry3T cam_guess = cam_from_rig * rig_guess;

    Matrix6T H_guess;
    Vector6T rhs_guess;
    float cost_guess = 0.f;
    if (!total_cost_and_hessian(cost_guess, H_guess, rhs_guess, cam_guess, level, settings, depth_info)) {
      // The step threw away every residual, so its zero cost is not comparable with current_cost
      // and scoring it would read the guess as a perfect fit. Reject it and try a shorter one.
      lambda *= 2.f;
      continue;
    }

    float prr = (step.dot(H * step) + 2.f * lambda * step.dot(scaling.asDiagonal() * step)) / current_cost;

    if (prr < sqrt_epsilon() && step.norm() < sqrt_epsilon()) {
      current_cost = cost_guess;
      break;
    }

    float rho = (1.f - cost_guess / current_cost) / prr;

    if (rho > 0.25f) {
      param_r_from_world = rig_guess;
      cam_from_world = cam_guess;

      if (rho > 0.75f) {
        auto new_lambda = lambda / 2.f;
        if (new_lambda > 0.f) {
          lambda = new_lambda;
        }
      }

      current_cost = cost_guess;
      H = H_guess;
      rhs = rhs_guess;
      scaling = H.diagonal();
    } else {
      lambda *= 2.f;
    }
  } while (num_iterations < num_iters);
  static_info_exp = H;

  bool status = current_cost <= initial_cost;
  if (status) {
    rig_from_world = param_r_from_world;
  }

  return true;
}

bool VisualICP::solve(Isometry3T& rig_from_world, Matrix6T& static_info_exp,
                      const std::vector<camera::Observation>& observations,
                      const std::unordered_map<TrackId, Vector3T>& landmarks,  // in map frame!
                      const ICPSettings& settings, const IcpInfo* depth_info) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("solve");

  if (settings.max_iteration < 0 || settings.min_scale_level < 0 || settings.max_scale_level < 0 ||
      settings.num_iters_per_scale < 0) {
    TraceError("ICPSettings count fields must be non-negative");
    return false;
  }

  // Get cam_from_rig for the depth camera (or camera 0 if no depth)
  CameraId depth_cam_id = depth_info ? depth_info->depth_id : 0;
  Isometry3T cam_from_rig = rig_.camera_from_rig[depth_cam_id];

  if (depth_info) {
    curr_observations_.clear();
    curr_observations_.reserve(observations.size());
    for (const auto& obs : observations) {
      if (obs.cam_id == depth_info->depth_id) {
        curr_observations_.push_back(obs);
      }
    }
    std::sort(curr_observations_.begin(), curr_observations_.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
  }

  landmark_for_observation_.clear();
  landmark_for_observation_.reserve(observations.size());

  observations_.clear();
  observations_.reserve(observations.size());

  {
    obs_per_camera_.resize(rig_.num_cameras);
    for (auto& obs_vec : obs_per_camera_) {
      obs_vec.clear();
    }

    for (const auto& obs : observations) {
      auto it = landmarks.find(obs.id);
      if (it == landmarks.end()) {
        continue;
      }
      obs_per_camera_[obs.cam_id].emplace_back(std::cref(obs));
    }

    for (auto& obs_vec : obs_per_camera_) {
      std::sort(obs_vec.begin(), obs_vec.end(),
                [](const obs_ref& lhs, const obs_ref& rhs) { return lhs.get().id < rhs.get().id; });

      const size_t num_pnp_tracks = std::min(max_obs_per_camera, obs_vec.size());
      obs_vec.erase(obs_vec.begin() + num_pnp_tracks, obs_vec.end());

      std::move(obs_vec.begin(), obs_vec.end(), std::back_inserter(observations_));
    }

    for (const auto& ref : observations_) {
      const camera::Observation& obs = ref.get();
      const Vector3T& point = landmarks.at(obs.id);
      landmark_for_observation_.push_back(std::cref(point));
    }
  }

  int level = 0;
  if (depth_info) {
    level = std::min((int)depth_info->curr_image.getLevelsCount() - 1, settings.max_scale_level);
  }

  bool status = true;

  if (depth_info) {
    while (level >= settings.min_scale_level) {
      status = solve_level(rig_from_world, static_info_exp, level, settings, depth_info, cam_from_rig);
      level--;
    }
  } else {
    status = solve_level(rig_from_world, static_info_exp, 0, settings, nullptr, cam_from_rig);
  }

  // prev_delta_ = std::nullopt;
  // if (status) {
  //   prev_delta_ = dst_from_src;
  // }

  return status;
}

}  // namespace cuvslam::pnp
