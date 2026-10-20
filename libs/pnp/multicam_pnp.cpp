
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

#include "pnp/multicam_pnp.h"

#include "common/log.h"
#include "common/rerun.h"
#include "common/rotation_utils.h"
#include "math/robust_cost_function.h"
#include "math/twist.h"
#include "pnp/visualizer.h"

namespace cuvslam::pnp {

using Mat36 = Eigen::Matrix<float, 3, 6>;
using Mat23 = Eigen::Matrix<float, 2, 3>;
using Mat26 = Eigen::Matrix<float, 2, 6>;

PNPSettings PNPSettings::LCSettings() {
  PNPSettings settings;
  settings.filter_new_observations = false;
  settings.min_observations = 3;
  settings.recalculate_cov = true;
  settings.huber = 5e-2;
  settings.point_z_thresh = 1e-3f;
  return settings;
}

PNPSettings PNPSettings::SLAMRansacSettings() {
  PNPSettings settings;
  settings.filter_new_observations = false;
  settings.min_observations = 3;
  settings.recalculate_cov = false;
  return settings;
}

PNPSettings PNPSettings::InertialSettings() {
  PNPSettings settings;
  settings.recalculate_cov = false;
  settings.huber = 0.1;
  return settings;
}

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

PNPSolver::PNPSolver(const camera::Rig& rig) : rig_(rig) {
  obs_per_camera_.resize(rig_.num_cameras);
  cam_from_w_.reserve(rig_.num_cameras);
}

void PNPSolver::build_camera_from_world(const Isometry3T& rig_from_world) const {
  cam_from_w_.clear();
  cam_from_w_.reserve(rig_.num_cameras);
  for (int i = 0; i < rig_.num_cameras; i++) {
    cam_from_w_.push_back(rig_.camera_from_rig[i] * rig_from_world);
  }
}

float PNPSolver::evaluate_cost(const Isometry3T& rig_from_world, const PNPSettings& settings) const {
  float cost = 0;

  build_camera_from_world(rig_from_world);

  Vector3T point_cam;

  float count = 0;

  for (size_t obs_id = 0; obs_id < observations_.size(); obs_id++) {
    const auto& obs = observations_[obs_id].get();
    const Vector3T& point_w = landmark_for_observation_[obs_id].get();
    const Isometry3T& Tcam_from_w = cam_from_w_[obs.cam_id];

    point_cam = Tcam_from_w * point_w;

    if (point_cam.z() <= settings.point_z_thresh) continue;

    count++;

    float one_over_z = 1.f / point_cam.z();
    Vector2T r = point_cam.topRows(2) * one_over_z - obs.xy;
    float r_squared_norm = r.dot(obs.xy_info * r);

    cost += math::ComputeHuberLoss(r_squared_norm, settings.huber);
  }

  if (count <= 0) {
    count = 1;
  }
  return cost / count;
}

void PNPSolver::build_hessian(const Isometry3T& rig_from_world, Matrix6T& H, Vector6T& rhs,
                              const PNPSettings& settings) const {
  H.setZero();
  rhs.setZero();

  build_camera_from_world(rig_from_world);

  Vector3T point_cam;
  float count = 0;

  for (size_t obs_id = 0; obs_id < observations_.size(); obs_id++) {
    const auto& obs = observations_[obs_id].get();
    const Vector3T& point_w = landmark_for_observation_[obs_id].get();
    const Isometry3T& Tcam_from_w = cam_from_w_[obs.cam_id];

    point_cam = Tcam_from_w * point_w;

    if (point_cam.z() <= settings.point_z_thresh) continue;
    count++;

    float one_over_z = 1.f / point_cam.z();

    Vector2T r = point_cam.topRows(2) * one_over_z - obs.xy;
    float r_squared_norm = r.dot(obs.xy_info * r);

    float w = math::ComputeDHuberLoss(r_squared_norm, settings.huber);
    Mat26 J = projection_jacobian(point_cam) * Tcam_from_w.linear() * point_jacobian(point_w);

    H += J.transpose() * obs.xy_info * w * J;
    rhs += J.transpose() * obs.xy_info * w * r;
  }
  if (count <= 0) {
    count = 1;
  }
  H = H / count;
  rhs = rhs / count;
}

using obs_ref = std::reference_wrapper<const camera::Observation>;

bool PNPSolver::solve(Isometry3T& rig_from_world, Matrix6T& static_info_exp,
                      const std::vector<camera::Observation>& observations,
                      const std::unordered_map<TrackId, Vector3T>& landmarks, const PNPSettings& settings,
                      float* final_cost) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("solve");

  if (settings.max_iteration < 0 || settings.max_obs_per_camera < 0 || settings.min_observations < 0) {
    TraceError("PNPSettings count fields must be non-negative");
    return false;
  }

  const size_t min_observations = static_cast<size_t>(settings.min_observations);
  if (observations.size() < min_observations) {
    TraceMessageIf(settings.verbose, "Num observation = {%d} is less then min (%d)",
                   static_cast<int>(observations.size()), settings.min_observations);
    return false;
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
      if (settings.filter_new_observations) {
        std::sort(obs_vec.begin(), obs_vec.end(),
                  [](const obs_ref& lhs, const obs_ref& rhs) { return lhs.get().id < rhs.get().id; });

        const size_t num_pnp_tracks = std::min(static_cast<size_t>(settings.max_obs_per_camera), obs_vec.size());
        obs_vec.erase(obs_vec.begin() + num_pnp_tracks, obs_vec.end());
      }

      std::move(obs_vec.begin(), obs_vec.end(), std::back_inserter(observations_));
    }

    for (const auto& ref : observations_) {
      const camera::Observation& obs = ref.get();
      const Vector3T& point = landmarks.at(obs.id);
      landmark_for_observation_.push_back(std::cref(point));
    }
  }

  // log observations
  RERUN(logObservations, observations_, rig_, "world/camera_0/images/observations_pnp", Color(255, 0, 0));

  // log landmark_for_observation_ with initial guess in rerun visualizer
  [[maybe_unused]] Isometry3T camera_from_world = rig_.camera_from_rig[0] * rig_from_world;
  RERUN(logLandmarks, landmark_for_observation_, camera_from_world, rig_.intrinsics[0],
        "world/camera_0/images/landmarks_pnp_initial", Color(0, 0, 255));

  Matrix6T H;
  Vector6T rhs;

  auto initial_cost = evaluate_cost(rig_from_world, settings);
  if (initial_cost < 5e-3f) {
    // Nothing to minimize. We have a "pendulum" effect on still frames otherwise.
    static_info_exp.setIdentity();
    if (final_cost) {
      *final_cost = initial_cost;
    }
    // log landmark_for_observation_ with final result in rerun visualizer
    RERUN(logLandmarks, landmark_for_observation_, camera_from_world, rig_.intrinsics[0],
          "world/camera_0/images/landmarks_pnp_final", Color(0, 255, 0));

    return true;
  }
  auto current_cost = initial_cost;

  build_hessian(rig_from_world, H, rhs, settings);

  Vector6T scaling = H.diagonal();

  float lambda = settings.lambda;
  int32_t num_iterations = 0;
  do {
    ++num_iterations;

    Matrix6T augmented_system = H + (lambda * scaling).asDiagonal().toDenseMatrix();
    auto decomposition = augmented_system.ldlt();
    Vector6T step = decomposition.solve(-rhs);

    Isometry3T guess;
    math::Exp(guess, step);
    guess = rig_from_world * guess;
    guess.linear() = common::CalculateRotationFromSVD(guess.matrix());

    float cost = evaluate_cost(guess, settings);

    //     TraceMessageIf(settings.verbose, "curr_cost = {%f}, cost = {%f}", current_cost, cost);

    float predicted_relative_reduction =
        step.dot(H * step) / current_cost + 2.f * lambda * step.dot(scaling.asDiagonal() * step) / current_cost;

    if ((predicted_relative_reduction < sqrt_epsilon()) && (step.norm() < sqrt_epsilon())) {
      current_cost = cost;
      break;
    }

    float rho = (1.f - cost / current_cost) / predicted_relative_reduction;
    if (rho > 0.25f) {
      rig_from_world = guess;

      if (rho > 0.75f) {
        auto new_lambda = lambda / 2.f;
        if (new_lambda > 0.f) {
          lambda = new_lambda;
        }
      }

      current_cost = cost;

      build_hessian(rig_from_world, H, rhs, settings);
      scaling = H.diagonal();
    } else {
      lambda *= 2.f;
    }
  } while (num_iterations < settings.max_iteration);
  static_info_exp = H;

  TraceMessageIf(settings.verbose, "curr_cost = {%f}, init_cost = {%f}", current_cost, initial_cost);

  bool status = current_cost < settings.cost_thresh || current_cost < initial_cost;
  if (final_cost) {
    *final_cost = current_cost;
  }

  if (status && settings.recalculate_cov) {
    static_info_exp.setZero();
    build_camera_from_world(rig_from_world);

    // log landmark_for_observation_ with final result in rerun visualizer
    RERUN(logLandmarks, landmark_for_observation_, cam_from_w_[0], rig_.intrinsics[0],
          "world/camera_0/images/landmarks_pnp_final", Color(0, 255, 0));

    Vector3T point_cam;

    for (size_t obs_id = 0; obs_id < observations_.size(); obs_id++) {
      const auto& obs = observations_[obs_id].get();
      const Vector3T& point_w = landmark_for_observation_[obs_id].get();
      const Isometry3T& Tcam_from_w = cam_from_w_[obs.cam_id];

      point_cam = Tcam_from_w * point_w;

      if (point_cam.z() <= settings.point_z_thresh) continue;

      Mat26 J = projection_jacobian(point_cam) * Tcam_from_w.linear() * point_jacobian(point_w);
      static_info_exp += J.transpose() * obs.xy_info * J;
    }
  } else {
    RERUN(clearViewport, "world/camera_0/images/landmarks_pnp_final");
  }

  return status;
}

}  // namespace cuvslam::pnp
