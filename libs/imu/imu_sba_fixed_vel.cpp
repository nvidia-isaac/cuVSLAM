
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

#include "common/log_types.h"
#include "common/statistic.h"
#include "epipolar/near_plane.h"
#include "math/robust_cost_function.h"
#include "math/twist.h"

#include "imu/imu_sba.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define ROBUST_COST math::ComputeHuberLoss
#define ROBUST_WEIGHT math::ComputeDHuberLoss

namespace {
using namespace cuvslam;

bool is_fixed(const sba_imu::ImuBAProblem& problem, int pid) { return pid < problem.num_fixed_key_frames; }

bool both_fixed(const sba_imu::ImuBAProblem& problem, int pid) {
  return is_fixed(problem, pid) && is_fixed(problem, pid + 1);
}

float effective_imu_penalty(const sba_imu::ImuBAProblem& problem, int i) {
  bool is_boundary = is_fixed(problem, i) && !is_fixed(problem, i + 1);
  return is_boundary ? problem.boundary_imu_penalty : problem.imu_penalty;
}

}  // namespace

namespace cuvslam::sba_imu {

IMUBundlerCpuFixedVel::IMUBundlerCpuFixedVel(const imu::ImuCalibration& calib) : calib_(calib) {}

bool IMUBundlerCpuFixedVel::solve(ImuBAProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU solve", profiler_color_);

  const int num_points = static_cast<int>(problem.points.size());
  const int num_poses = static_cast<int>(problem.rig_poses.size());
  const int num_inertials = num_poses - 1;

  if (num_inertials <= 0) {
    return false;
  }

  internal::ParameterUpdate update;
  update.point.resize(num_points, Vector3T ::Zero());
  update.pose.resize(num_poses - problem.num_fixed_key_frames);

  internal::ModelFunction model;
  UpdateModel(model, problem);

  const float initial_cost = EvaluateCost(problem, update);
  float current_cost = initial_cost;
  problem.initial_cost = initial_cost;

  if (initial_cost < std::numeric_limits<float>::epsilon()) {
    return true;
  }

  if (std::isnan(initial_cost)) {
    return false;
  }

  if (initial_cost == std::numeric_limits<float>::infinity()) {
    return false;
  }

  int iteration = 0;
  const int max_iterations = problem.max_iterations;

  internal::FullSystem full_system;
  internal::ReducedSystem reduced_system;

  BuildFullSystem(full_system, model, problem);

  float lambda = 1e2;

  while (iteration < max_iterations) {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("CPU iteration", profiler_color_);
    ++iteration;
    problem.iterations = iteration;

    BuildReducedSystem(reduced_system, full_system, lambda);
    ComputeUpdate(update, reduced_system, full_system);

    auto cost = EvaluateCost(problem, update);

    if (std::isnan(cost)) {
      // TraceError("std::isnan(cost)");
      return false;
    }

    if (cost == std::numeric_limits<float>::infinity()) {
      // starting point is not feasible
      // TraceError("cost = infinity");
      return false;
    }

    if (current_cost < initial_cost * std::numeric_limits<float>::epsilon()) {
      // std::cout << "init = " << initial_cost << " cost = " << current_cost << std::endl;

      Vector3T gyro_diff, acc_diff;
      for (int i = 0; i < num_poses; i++) {
        if (is_fixed(problem, i)) {
          continue;
        }
        Pose& p = problem.rig_poses[i];
        gyro_diff = p.gyro_bias - p.preintegration.GetOriginalGyroBias();
        acc_diff = p.acc_bias - p.preintegration.GetOriginalAccBias();

        if (gyro_diff.norm() > problem.reintegration_thresh || acc_diff.norm() > problem.reintegration_thresh) {
          p.preintegration.SetNewBias(p.gyro_bias, p.acc_bias);
          p.preintegration.Reintegrate(calib_);
        }
      }
      return true;
    }

    auto predicted_relative_reduction = ComputePredictedRelativeReduction(current_cost, lambda, update, full_system);

    if ((predicted_relative_reduction < epsilon()) && (update.pose_step.lpNorm<Eigen::Infinity>() < epsilon()) &&
        (update.point_step.lpNorm<Eigen::Infinity>() < epsilon())) {
      // std::cout << "init = " << initial_cost << " cost = " << current_cost << std::endl;

      Vector3T gyro_diff, acc_diff;
      for (int i = 0; i < num_poses; i++) {
        if (is_fixed(problem, i)) {
          continue;
        }
        Pose& p = problem.rig_poses[i];
        gyro_diff = p.gyro_bias - p.preintegration.GetOriginalGyroBias();
        acc_diff = p.acc_bias - p.preintegration.GetOriginalAccBias();

        if (gyro_diff.norm() > problem.reintegration_thresh || acc_diff.norm() > problem.reintegration_thresh) {
          p.preintegration.SetNewBias(p.gyro_bias, p.acc_bias);
          p.preintegration.Reintegrate(calib_);
        }
      }
      return true;
    }

    assert(std::isfinite(current_cost));
    assert(std::isfinite(predicted_relative_reduction));
    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    if (rho > 0.25f) {
      if (rho > 0.75f) {
        if (lambda * 0.125f > 0.f) {
          lambda *= 0.125f;
        }
      }

      current_cost = cost;
      UpdateState(problem, update);

      UpdateModel(model, problem);
      BuildFullSystem(full_system, model, problem);
    } else {
      lambda *= 5.f;
    }
  }
  if (current_cost < initial_cost) {
    Vector3T gyro_diff, acc_diff;
    for (int i = 0; i < num_poses; i++) {
      if (is_fixed(problem, i)) {
        continue;
      }
      Pose& p = problem.rig_poses[i];
      gyro_diff = p.gyro_bias - p.preintegration.GetOriginalGyroBias();
      acc_diff = p.acc_bias - p.preintegration.GetOriginalAccBias();

      if (gyro_diff.norm() > problem.reintegration_thresh || acc_diff.norm() > problem.reintegration_thresh) {
        p.preintegration.SetNewBias(p.gyro_bias, p.acc_bias);
        p.preintegration.Reintegrate(calib_);
      }
    }
  }

  return current_cost < initial_cost;
}

float IMUBundlerCpuFixedVel::EvaluateCost(const ImuBAProblem& problem, const internal::ParameterUpdate& update) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU EvaluateCost()", profiler_color_);

  float cost = 0;

  const auto num_observations = static_cast<int32_t>(problem.observation_xys.size());
  const auto num_poses = static_cast<int32_t>(problem.rig_poses.size());
  const auto num_inertials = num_poses - 1;

  // we may "skip" some observations if they are too close to the camera
  int32_t num_skipped = 0;

  const Isometry3T& rig_from_imu = calib_.rig_from_imu();

  Vector3T p_c;
  Vector2T r;
  std::vector<Pose> updated_poses;
  std::vector<Isometry3T> imu_from_w;
  updated_poses.reserve(problem.points.size());
  imu_from_w.reserve(problem.points.size());
  for (size_t pose_idx = 0; pose_idx < problem.rig_poses.size(); pose_idx++) {
    Pose pose = problem.rig_poses[pose_idx];
    if (!is_fixed(problem, pose_idx)) {
      int id = pose_idx - problem.num_fixed_key_frames;

      Isometry3T p = pose.w_from_imu;
      p.translation() += p.linear() * update.pose[id].w_from_imu.translation();
      p.linear() = p.linear() * update.pose[id].w_from_imu.linear();
      p.makeAffine();

      pose.w_from_imu = p;
      pose.velocity += update.pose[id].velocity;
      pose.gyro_bias += update.pose[id].gyro_bias;
      pose.acc_bias += update.pose[id].acc_bias;
    }
    updated_poses.push_back(pose);
    imu_from_w.push_back(pose.w_from_imu.inverse());
  }

  std::vector<Isometry3T> cam_from_imu;
  cam_from_imu.reserve(problem.rig.num_cameras);
  for (int i = 0; i < problem.rig.num_cameras; i++) {
    cam_from_imu.push_back(problem.rig.camera_from_rig[i] * rig_from_imu);
  }

  for (int obs = 0; obs < num_observations; ++obs) {
    const auto point_idx = problem.point_ids[obs];
    const auto pose_idx = problem.pose_ids[obs];
    const auto camera_idx = problem.camera_ids[obs];
    auto p_w = problem.points[point_idx] + update.point[point_idx];

    p_c = cam_from_imu[camera_idx] * imu_from_w[pose_idx] * p_w;

    if (epipolar::IsDepthInFront(p_c.z())) {
      r = p_c.topRows(2) / p_c.z() - problem.observation_xys[obs];
      cost += ROBUST_COST(r.dot(problem.observation_infos[obs] * r), problem.robustifier_scale);
    } else {
      ++num_skipped;
    }
  }

  if (num_skipped == num_observations) {
    return std::numeric_limits<float>::infinity();
  }

  Matrix9T info;
  Matrix3T info_gyro_rw, info_acc_rw;

  Vector3T rot_error;
  Vector3T velocity_error_term, trans_error_term;

  Vector9T inertial_error;
  Vector3T random_walk_gyro_error, random_walk_acc_error;

  for (int i = 0; i < num_inertials; i++) {
    if (both_fixed(problem, i)) {
      continue;
    }
    const auto& preint = problem.rig_poses[i].preintegration;

    preint.InfoMatrix(info);
    preint.InfoGyroRWMatrix(info_gyro_rw);
    preint.InfoAccRWMatrix(info_acc_rw);

    const float eff_penalty = effective_imu_penalty(problem, i);
    info *= eff_penalty;
    info_gyro_rw *= eff_penalty;
    info_acc_rw *= (problem.acc_rw_penalty >= 0 ? problem.acc_rw_penalty : eff_penalty);

    const Pose& pu1 = updated_poses[i];
    const Pose& pu2 = updated_poses[i + 1];

    Matrix3T R1T = pu1.w_from_imu.linear().transpose();
    Matrix3T R2 = pu2.w_from_imu.linear();

    const Matrix3T dR = preint.GetDeltaRotation(pu1.gyro_bias);
    const Vector3T dV = preint.GetDeltaVelocity(pu1.gyro_bias, pu1.acc_bias);
    const Vector3T dP = preint.GetDeltaPosition(pu1.gyro_bias, pu1.acc_bias);
    const float dT = preint.GetDeltaT_s();

    math::Log(rot_error, dR.transpose() * R1T * R2);

    const Vector3T& v1 = pu1.velocity;
    const Vector3T& v2 = pu2.velocity;
    const Vector3T& p1 = pu1.w_from_imu.translation();
    const Vector3T& p2 = pu2.w_from_imu.translation();

    velocity_error_term = R1T * (v2 - v1 - problem.gravity * dT);
    trans_error_term = R1T * (p2 - p1 - v1 * dT - 0.5 * problem.gravity * dT * dT);

    inertial_error.segment<3>(0) = rot_error;
    inertial_error.segment<3>(3) = velocity_error_term - dV;
    inertial_error.segment<3>(6) = trans_error_term - dP;

    random_walk_gyro_error = pu1.gyro_bias - pu2.gyro_bias;
    random_walk_acc_error = pu1.acc_bias - pu2.acc_bias;

    cost += random_walk_gyro_error.dot(info_gyro_rw * random_walk_gyro_error);
    cost += random_walk_acc_error.dot(info_acc_rw * random_walk_acc_error);

    cost += inertial_error.dot(info * inertial_error);
  }

  Matrix3T prior_gyro_info = Matrix3T::Identity() * problem.prior_gyro;
  Matrix3T prior_acc_info = Matrix3T::Identity() * problem.prior_acc;

  // bias priors on oldest non-fixed pose only; random walk propagates to newer poses
  if (problem.num_fixed_key_frames < num_poses) {
    const Pose& pose = updated_poses[problem.num_fixed_key_frames];
    cost += pose.gyro_bias.dot(prior_gyro_info * pose.gyro_bias);
    cost += pose.acc_bias.dot(prior_acc_info * pose.acc_bias);
  }

  return cost;
}

void IMUBundlerCpuFixedVel::ComputeUpdate(internal::ParameterUpdate& update,
                                          const internal::ReducedSystem& reduced_system,
                                          const internal::FullSystem& full_system) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU ComputeUpdate()", profiler_color_);
  const auto num_points = static_cast<int>(full_system.point_block.size());
  const auto num_poses = static_cast<int>(full_system.pose_block.cols() / 15);

  Eigen::ColPivHouseholderQR<Eigen::MatrixXf> solver(reduced_system.pose_block);
  Eigen::VectorXf dpose = solver.solve(reduced_system.pose_rhs);

  Eigen::VectorXf dpoint = reduced_system.point_rhs;

  const Eigen::MatrixXf& Yt = reduced_system.camera_backsub_block;  // 3 * num_points x 6 * num_poses
  for (int i = 0; i < num_poses; ++i) {
    dpoint -= Yt.block(0, 6 * i, Yt.rows(), 6) * dpose.segment<6>(15 * i);
  }

  for (int i = 0; i < num_points; ++i) {
    update.point[i] = dpoint.segment<3>(3 * i);
  }

  Matrix3T dR;
  for (int i = 0; i < num_poses; ++i) {
    math::Exp(dR, dpose.segment<3>(15 * i));
    update.pose[i].w_from_imu.linear() = dR;
    update.pose[i].w_from_imu.translation() = dpose.segment<3>(15 * i + 3);
    update.pose[i].velocity = dpose.segment<3>(15 * i + 6);
    update.pose[i].gyro_bias = dpose.segment<3>(15 * i + 9);
    update.pose[i].acc_bias = dpose.segment<3>(15 * i + 12);
  }

  update.pose_step = std::move(dpose);
  update.point_step = std::move(dpoint);
}

void IMUBundlerCpuFixedVel::UpdateState(ImuBAProblem& problem, const internal::ParameterUpdate& update) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU UpdateState()", profiler_color_);
  const auto num_points = static_cast<int>(problem.points.size());
  const auto num_poses = static_cast<int>(problem.rig_poses.size());

  for (int i = 0; i < num_points; ++i) {
    problem.points[i] += update.point[i];
  }

  for (int i = 0; i < num_poses; ++i) {
    if (!is_fixed(problem, i)) {
      Pose& p = problem.rig_poses[i];
      int id = i - problem.num_fixed_key_frames;

      p.w_from_imu.translation() += p.w_from_imu.linear() * update.pose[id].w_from_imu.translation();
      p.w_from_imu.linear() = p.w_from_imu.linear() * update.pose[id].w_from_imu.linear();
      p.w_from_imu.makeAffine();

      p.velocity += update.pose[id].velocity;
      p.gyro_bias += update.pose[id].gyro_bias;
      p.acc_bias += update.pose[id].acc_bias;

      // TODO maybe reintegrate measurements here
      p.preintegration.SetNewBias(p.gyro_bias, p.acc_bias);

      Vector3T gyro_bias_diff, acc_bias_diff;
      p.preintegration.GetDeltaBias(gyro_bias_diff, acc_bias_diff);
      if (gyro_bias_diff.norm() > problem.reintegration_thresh || acc_bias_diff.norm() > problem.reintegration_thresh) {
        p.preintegration.Reintegrate(calib_);
      }
    }
  }
}

float IMUBundlerCpuFixedVel::ComputePredictedRelativeReduction(float current_cost, float lambda,
                                                               const internal::ParameterUpdate& update,
                                                               const internal::FullSystem& system) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU ComputePredictedRelativeReduction()", profiler_color_);
  assert(current_cost > std::numeric_limits<float>::epsilon());

  const int num_points = static_cast<int>(system.point_block.size());
  const int num_poses = static_cast<int>(system.pose_block.cols() / 15);

  float hessian_term = 0;

  {
    float pose_term = update.pose_step.dot(system.pose_block * update.pose_step);
    float point_term = 0;
    for (int i = 0; i < num_poses; i++) {
      point_term += update.point_step.dot(system.point_pose_block.block(0, 6 * i, update.point_step.size(), 6) *
                                          update.pose_step.segment<6>(15 * i));
    }
    pose_term += point_term;

    for (int i = 0; i < num_points; ++i) {
      point_term +=
          update.point_step.segment<3>(3 * i).dot(system.point_block[i] * update.point_step.segment<3>(3 * i));
    }

    hessian_term = (pose_term + point_term) / current_cost;
  }
  assert(std::isfinite(hessian_term));

  float scaling_term = 0;

  {
    float pose_term = update.pose_step.dot(system.pose_block.diagonal().asDiagonal() * update.pose_step);
    float point_term = 0;

    Vector3T dp;
    for (int i = 0; i < num_points; ++i) {
      dp = update.point_step.segment<3>(3 * i);
      point_term += dp.dot(system.point_block[i].diagonal().asDiagonal() * dp);
    }
    scaling_term = ((pose_term + point_term) / current_cost) * 2.f * lambda;
  }
  assert(std::isfinite(scaling_term));
  const float prediction = hessian_term + scaling_term;

  assert(std::isfinite(prediction));

  return prediction;
}

void IMUBundlerCpuFixedVel::UpdateModel(internal::ModelFunction& model, ImuBAProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU UpdateModel()", profiler_color_);
  const Isometry3T& rig_from_imu = calib_.rig_from_imu();

  const int num_observations = static_cast<int>(problem.observation_xys.size());
  const int num_poses = static_cast<int>(problem.rig_poses.size());
  const int num_inertials = num_poses - 1;
  model.repr_jacobians.resize(num_observations);
  model.repr_robustifier_weights.resize(num_observations);

  model.reprojection_residuals.resize(num_observations);

  model.random_walk_gyro_residuals.resize(num_inertials);
  model.random_walk_acc_residuals.resize(num_inertials);

  model.inertial_jacobians.resize(num_inertials);
  model.inertial_residuals.resize(num_inertials);

  // we may "skip" some observations if they are too close to the camera
  int32_t num_skipped = 0;

  // TODO: check adjoints!
  Isometry3T cam_from_imu[camera::Rig::kMaxCameras];
  for (int i = 0; i < problem.rig.num_cameras; ++i) {
    cam_from_imu[i] = problem.rig.camera_from_rig[i] * rig_from_imu;
  }

  std::vector<Isometry3T> inv_poses;
  inv_poses.reserve(num_poses);
  for (int i = 0; i < num_poses; i++) {
    const Isometry3T& w_from_imu = problem.rig_poses[i].w_from_imu;
    inv_poses.push_back(w_from_imu.inverse());
  }

  internal::Mat23 dproj, M;
  for (int obs = 0; obs < num_observations; ++obs) {
    const int point_idx = problem.point_ids[obs];
    const int pose_idx = problem.pose_ids[obs];
    const int8_t camera_idx = problem.camera_ids[obs];

    const Vector3T& p_w = problem.points[point_idx];
    const Isometry3T& imu_from_w = inv_poses[pose_idx];
    const Vector3T p_imu = imu_from_w * p_w;
    const Vector3T p_c = cam_from_imu[camera_idx] * p_imu;

    Matrix3T Rimu_from_w = imu_from_w.linear();
    Matrix3T Rcam_from_imu = cam_from_imu[camera_idx].linear();

    if (epipolar::IsDepthInFront(p_c.z())) {
      Vector2T prediction = p_c.topRows(2) / p_c.z();
      Vector2T r = prediction - problem.observation_xys[obs];

      float inv_z = 1.f / p_c.z();
      dproj << inv_z, 0.f, -prediction.x() * inv_z, 0.f, inv_z, -prediction.y() * inv_z;

      internal::Mat23& JR = model.repr_jacobians[obs].JR;
      internal::Mat23& Jp = model.repr_jacobians[obs].Jpoint;

      // Mat23 M = dproj * Rcam_from_imu;
      M(0, 0) = dproj(0, 0) * Rcam_from_imu(0, 0) + dproj(0, 2) * Rcam_from_imu(2, 0);
      M(0, 1) = dproj(0, 0) * Rcam_from_imu(0, 1) + dproj(0, 2) * Rcam_from_imu(2, 1);
      M(0, 2) = dproj(0, 0) * Rcam_from_imu(0, 2) + dproj(0, 2) * Rcam_from_imu(2, 2);

      M(1, 0) = dproj(1, 1) * Rcam_from_imu(1, 0) + dproj(1, 2) * Rcam_from_imu(2, 0);
      M(1, 1) = dproj(1, 1) * Rcam_from_imu(1, 1) + dproj(1, 2) * Rcam_from_imu(2, 1);
      M(1, 2) = dproj(1, 1) * Rcam_from_imu(1, 2) + dproj(1, 2) * Rcam_from_imu(2, 2);

      model.reprojection_residuals[obs] = r;

      // Jr = M * Skew(p_imu)
      JR(0, 0) = M(0, 1) * p_imu.z() - M(0, 2) * p_imu.y();
      JR(0, 1) = -M(0, 0) * p_imu.z() + M(0, 2) * p_imu.x();
      JR(0, 2) = M(0, 0) * p_imu.y() - M(0, 1) * p_imu.x();

      JR(1, 0) = M(1, 1) * p_imu.z() - M(1, 2) * p_imu.y();
      JR(1, 1) = -M(1, 0) * p_imu.z() + M(1, 2) * p_imu.x();
      JR(1, 2) = M(1, 0) * p_imu.y() - M(1, 1) * p_imu.x();

      // Jr = M * Rimu_from_w
      Jp(0, 0) = M(0, 0) * Rimu_from_w(0, 0) + M(0, 1) * Rimu_from_w(1, 0) + M(0, 2) * Rimu_from_w(2, 0);
      Jp(0, 1) = M(0, 0) * Rimu_from_w(0, 1) + M(0, 1) * Rimu_from_w(1, 1) + M(0, 2) * Rimu_from_w(2, 1);
      Jp(0, 2) = M(0, 0) * Rimu_from_w(0, 2) + M(0, 1) * Rimu_from_w(1, 2) + M(0, 2) * Rimu_from_w(2, 2);

      Jp(1, 0) = M(1, 0) * Rimu_from_w(0, 0) + M(1, 1) * Rimu_from_w(1, 0) + M(1, 2) * Rimu_from_w(2, 0);
      Jp(1, 1) = M(1, 0) * Rimu_from_w(0, 1) + M(1, 1) * Rimu_from_w(1, 1) + M(1, 2) * Rimu_from_w(2, 1);
      Jp(1, 2) = M(1, 0) * Rimu_from_w(0, 2) + M(1, 1) * Rimu_from_w(1, 2) + M(1, 2) * Rimu_from_w(2, 2);
      model.repr_jacobians[obs].Jt = -M;

      float err = r.dot(problem.observation_infos[obs] * r);
      model.repr_robustifier_weights[obs] = ROBUST_WEIGHT(err, problem.robustifier_scale);
    } else {
      model.reprojection_residuals[obs].setZero();
      model.repr_jacobians[obs].JR.setZero();
      model.repr_jacobians[obs].Jt.setZero();
      model.repr_jacobians[obs].Jpoint.setZero();
      model.repr_robustifier_weights[obs] = 0;
      ++num_skipped;
    }
  }

  //    CalculateInformationMatrices(model, problem);

  Vector3T rot_error;
  Vector3T vel_error_term, trans_error_term;

  Vector3T gyro_bias_diff, acc_bias_diff;

  // https://arxiv.org/pdf/1512.02363.pdf
  for (int i = 0; i < num_inertials; i++) {
    if (both_fixed(problem, i)) {
      continue;
    }
    const Pose& pose_left = problem.rig_poses[i];
    const Pose& pose_right = problem.rig_poses[i + 1];
    const auto& preint = pose_left.preintegration;
    const Isometry3T& w_from_imu1 = pose_left.w_from_imu;
    const Isometry3T& w_from_imu2 = pose_right.w_from_imu;

    Matrix3T R1T = w_from_imu1.linear().transpose();
    Matrix3T R2 = w_from_imu2.linear();

    const Matrix3T dR = preint.GetDeltaRotation(pose_left.gyro_bias);
    const Vector3T dV = preint.GetDeltaVelocity(pose_left.gyro_bias, pose_left.acc_bias);
    const Vector3T dP = preint.GetDeltaPosition(pose_left.gyro_bias, pose_left.acc_bias);
    const float dT = preint.GetDeltaT_s();

    math::Log(rot_error, dR.transpose() * R1T * R2);

    vel_error_term = R1T * (pose_right.velocity - pose_left.velocity - problem.gravity * dT);

    trans_error_term = R1T * (w_from_imu2.translation() - w_from_imu1.translation() - pose_left.velocity * dT -
                              0.5 * problem.gravity * dT * dT);

    model.inertial_residuals[i].segment<3>(0) = rot_error;
    model.inertial_residuals[i].segment<3>(3) = vel_error_term - dV;
    model.inertial_residuals[i].segment<3>(6) = trans_error_term - dP;

    {
      // rot
      model.inertial_jacobians[i].JR_left.block<3, 3>(0, 0) =
          -math::twist_right_inverse_jacobian(rot_error) * R2.transpose() * R1T.transpose();
      // vel
      model.inertial_jacobians[i].JR_left.block<3, 3>(3, 0) = SkewSymmetric(vel_error_term);
      // tr
      model.inertial_jacobians[i].JR_left.block<3, 3>(6, 0) = SkewSymmetric(trans_error_term);
    }

    {
      model.inertial_jacobians[i].Jt_left.setZero();
      // tr
      model.inertial_jacobians[i].Jt_left.block<3, 3>(6, 0) = -Matrix3T::Identity();
    }

    {
      model.inertial_jacobians[i].Jv_left.setZero();
      // vel
      model.inertial_jacobians[i].Jv_left.block<3, 3>(3, 0) = -R1T;
      // tr
      model.inertial_jacobians[i].Jv_left.block<3, 3>(6, 0) = -R1T * dT;
    }

    {
      model.inertial_jacobians[i].Jb_acc_left.setZero();
      // vel
      model.inertial_jacobians[i].Jb_acc_left.block<3, 3>(3, 0) = -preint.JVa;
      // tr
      model.inertial_jacobians[i].Jb_acc_left.block<3, 3>(6, 0) = -preint.JPa;
    }

    {
      preint.GetDeltaBias(gyro_bias_diff, acc_bias_diff);

      // rot
      model.inertial_jacobians[i].Jb_gyro_left.block<3, 3>(0, 0) =
          -math::twist_left_inverse_jacobian(rot_error) * math::twist_right_jacobian(preint.JRg * gyro_bias_diff) *
          preint.JRg;
      // vel
      model.inertial_jacobians[i].Jb_gyro_left.block<3, 3>(3, 0) = -preint.JVg;
      // tr
      model.inertial_jacobians[i].Jb_gyro_left.block<3, 3>(6, 0) = -preint.JPg;
    }

    {
      model.inertial_jacobians[i].JR_right.setZero();
      // rot
      model.inertial_jacobians[i].JR_right.block<3, 3>(0, 0) = math::twist_right_inverse_jacobian(rot_error);
    }

    {
      model.inertial_jacobians[i].Jt_right.setZero();
      // tr
      model.inertial_jacobians[i].Jt_right.block<3, 3>(6, 0) = R1T * R2;
    }

    {
      model.inertial_jacobians[i].Jv_right.setZero();
      // vel
      model.inertial_jacobians[i].Jv_right.block<3, 3>(3, 0) = R1T;
    }

    model.random_walk_gyro_residuals[i] = pose_left.gyro_bias - pose_right.gyro_bias;
    model.random_walk_acc_residuals[i] = pose_left.acc_bias - pose_right.acc_bias;
  }
}

void IMUBundlerCpuFixedVel::BuildFullSystem(internal::FullSystem& system, const internal::ModelFunction& model,
                                            const ImuBAProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU BuildFullSystem()", profiler_color_);
  const auto num_points = static_cast<int>(problem.points.size());
  const auto num_poses = static_cast<int>(problem.rig_poses.size());
  const auto num_poses_opt = num_poses - problem.num_fixed_key_frames;
  const auto num_inertials = num_poses - 1;
  const auto num_observations = static_cast<int>(problem.observation_xys.size());

  // HZ: V
  system.point_block.clear();
  system.point_block.resize(num_points, Matrix3T::Zero());

  // HZ: e_a

  // 3 - rotation, 3 - translation, 3 - velocity, 3 - gyro bias, 3 - acc bias
  system.pose_rhs.resize(15 * num_poses_opt);
  system.pose_rhs.setZero();

  // HZ: U
  system.pose_block.resize(15 * num_poses_opt, 15 * num_poses_opt);
  system.pose_block.setZero();

  // HZ: transp(W)
  // 6 since all other jacobians except R and t are zero
  system.point_pose_block.resize(3 * num_points, 6 * num_poses_opt);
  system.point_pose_block.setZero();

  // HZ: e_b
  system.point_rhs.resize(3 * num_points);
  system.point_rhs.setZero();

  Matrix3T hpp, hrr, htt, hrt, hpr, hpt;

  internal::Mat32 K;

  // TODO: loop over cameras instead, then loop over points.
  // This way we will accumulate in local variable instead of scattering.
  for (int obs = 0; obs < num_observations; ++obs) {
    const float weight = model.repr_robustifier_weights[obs];
    const int point_idx = problem.point_ids[obs];
    const int pose_idx = problem.pose_ids[obs];
    const int id = pose_idx - problem.num_fixed_key_frames;
    const internal::ReprJacobians& jacobians = model.repr_jacobians[obs];

    K.noalias() = jacobians.Jpoint.transpose() * problem.observation_infos[obs];
    hpp.noalias() = K * jacobians.Jpoint;
    hpr.noalias() = K * jacobians.JR;
    hpt.noalias() = K * jacobians.Jt;

    K.noalias() = jacobians.JR.transpose() * problem.observation_infos[obs];
    hrr.noalias() = K * jacobians.JR;
    hrt.noalias() = K * jacobians.Jt;

    htt.noalias() = jacobians.Jt.transpose() * problem.observation_infos[obs] * jacobians.Jt;

    // both the point and the camera are relaxed
    if (!is_fixed(problem, pose_idx)) {
      // HZ: transp(W) = sum_ij(transp(W_ij))
      system.point_pose_block.block<3, 3>(3 * point_idx, 6 * id) += hpr * weight;
      system.point_pose_block.block<3, 3>(3 * point_idx, 6 * id + 3) += hpt * weight;
    }

    Vector2T m = problem.observation_infos[obs] * weight * model.reprojection_residuals[obs];

    // camera is relaxed
    if (!is_fixed(problem, pose_idx)) {
      // HZ: U_j = sum_i(U_ij)
      system.pose_block.block<3, 3>(15 * id, 15 * id) += hrr * weight;
      system.pose_block.block<3, 3>(15 * id, 15 * id + 3) += hrt * weight;
      system.pose_block.block<3, 3>(15 * id + 3, 15 * id) += hrt.transpose() * weight;
      system.pose_block.block<3, 3>(15 * id + 3, 15 * id + 3) += htt * weight;

      system.pose_rhs.segment<3>(15 * id) -= jacobians.JR.transpose() * m;
      system.pose_rhs.segment<3>(15 * id + 3) -= jacobians.Jt.transpose() * m;
    }

    // point is relaxed
    system.point_block[point_idx] += hpp * weight;
    system.point_rhs.segment<3>(3 * point_idx) -= jacobians.Jpoint.transpose() * m;
  }

  Matrix3T h_r1r1, h_r1t1, h_r1v1, h_r1bg, h_r1ba, h_r1r2, h_r1t2, h_r1v2, h_t1t1, h_t1v1, h_t1bg, h_t1ba, h_t1r2,
      h_t1t2, h_t1v2, h_v1v1, h_v1bg, h_v1ba, h_v1r2, h_v1t2, h_v1v2, h_bgbg, h_bgba, h_bgr2, h_bgt2, h_bgv2, h_baba,
      h_bar2, h_bat2, h_bav2, h_r2r2, h_r2t2, h_r2v2, h_t2t2, h_t2v2, h_v2v2;

  // inertial part
  Matrix9T info;
  Eigen::Matrix<float, 3, 9> temp;
  Eigen::Matrix<float, 9, 1> info_r;

  for (int i = 0; i < num_inertials; i++) {
    if (both_fixed(problem, i)) {
      continue;
    }

    const auto& preint = problem.rig_poses[i].preintegration;
    const auto& jacobians = model.inertial_jacobians[i];
    preint.InfoMatrix(info);

    info *= effective_imu_penalty(problem, i);

    temp = jacobians.JR_left.transpose() * info;

    h_r1r1 = temp * jacobians.JR_left;
    h_r1t1 = temp * jacobians.Jt_left;
    h_r1v1 = temp * jacobians.Jv_left;
    h_r1bg = temp * jacobians.Jb_gyro_left;
    h_r1ba = temp * jacobians.Jb_acc_left;
    h_r1r2 = temp * jacobians.JR_right;
    h_r1t2 = temp * jacobians.Jt_right;
    h_r1v2 = temp * jacobians.Jv_right;

    temp = jacobians.Jt_left.transpose() * info;
    h_t1t1 = temp * jacobians.Jt_left;
    h_t1v1 = temp * jacobians.Jv_left;
    h_t1bg = temp * jacobians.Jb_gyro_left;
    h_t1ba = temp * jacobians.Jb_acc_left;
    h_t1r2 = temp * jacobians.JR_right;
    h_t1t2 = temp * jacobians.Jt_right;
    h_t1v2 = temp * jacobians.Jv_right;

    temp = jacobians.Jv_left.transpose() * info;
    h_v1v1 = temp * jacobians.Jv_left;
    h_v1bg = temp * jacobians.Jb_gyro_left;
    h_v1ba = temp * jacobians.Jb_acc_left;
    h_v1r2 = temp * jacobians.JR_right;
    h_v1t2 = temp * jacobians.Jt_right;
    h_v1v2 = temp * jacobians.Jv_right;

    temp = jacobians.Jb_gyro_left.transpose() * info;
    h_bgbg = temp * jacobians.Jb_gyro_left;
    h_bgba = temp * jacobians.Jb_acc_left;
    h_bgr2 = temp * jacobians.JR_right;
    h_bgt2 = temp * jacobians.Jt_right;
    h_bgv2 = temp * jacobians.Jv_right;

    temp = jacobians.Jb_acc_left.transpose() * info;
    h_baba = temp * jacobians.Jb_acc_left;
    h_bar2 = temp * jacobians.JR_right;
    h_bat2 = temp * jacobians.Jt_right;
    h_bav2 = temp * jacobians.Jv_right;

    temp = jacobians.JR_right.transpose() * info;
    h_r2r2 = temp * jacobians.JR_right;
    h_r2t2 = temp * jacobians.Jt_right;
    h_r2v2 = temp * jacobians.Jv_right;

    temp = jacobians.Jt_right.transpose() * info;
    h_t2t2 = temp * jacobians.Jt_right;
    h_t2v2 = temp * jacobians.Jv_right;

    h_v2v2 = jacobians.Jv_right.transpose() * info * jacobians.Jv_right;

    int id = i - problem.num_fixed_key_frames;

    {
      if (!is_fixed(problem, i)) {
        system.pose_block.block<3, 3>(15 * id, 15 * id) += h_r1r1;
        system.pose_block.block<3, 3>(15 * id, 15 * id + 3) += h_r1t1;
        system.pose_block.block<3, 3>(15 * id, 15 * id + 6) += h_r1v1;
        system.pose_block.block<3, 3>(15 * id, 15 * id + 9) += h_r1bg;
        system.pose_block.block<3, 3>(15 * id, 15 * id + 12) += h_r1ba;
        if (!is_fixed(problem, i + 1)) {
          system.pose_block.block<3, 3>(15 * id, 15 * id + 15) += h_r1r2;
          system.pose_block.block<3, 3>(15 * id, 15 * id + 18) += h_r1t2;
          system.pose_block.block<3, 3>(15 * id, 15 * id + 21) += h_r1v2;
        }
      }
    }

    {
      if (!is_fixed(problem, i)) {
        system.pose_block.block<3, 3>(15 * id + 3, 15 * id) += h_r1t1.transpose();
        system.pose_block.block<3, 3>(15 * id + 3, 15 * id + 3) += h_t1t1;
        system.pose_block.block<3, 3>(15 * id + 3, 15 * id + 6) += h_t1v1;
        system.pose_block.block<3, 3>(15 * id + 3, 15 * id + 9) += h_t1bg;
        system.pose_block.block<3, 3>(15 * id + 3, 15 * id + 12) += h_t1ba;
        if (!is_fixed(problem, i + 1)) {
          system.pose_block.block<3, 3>(15 * id + 3, 15 * id + 15) += h_t1r2;
          system.pose_block.block<3, 3>(15 * id + 3, 15 * id + 18) += h_t1t2;
          system.pose_block.block<3, 3>(15 * id + 3, 15 * id + 21) += h_t1v2;
        }
      }
    }

    {
      if (!is_fixed(problem, i)) {
        system.pose_block.block<3, 3>(15 * id + 6, 15 * id) += h_r1v1.transpose();
        system.pose_block.block<3, 3>(15 * id + 6, 15 * id + 3) += h_t1v1.transpose();
        system.pose_block.block<3, 3>(15 * id + 6, 15 * id + 6) += h_v1v1;
        system.pose_block.block<3, 3>(15 * id + 6, 15 * id + 9) += h_v1bg;
        system.pose_block.block<3, 3>(15 * id + 6, 15 * id + 12) += h_v1ba;
        if (!is_fixed(problem, i + 1)) {
          system.pose_block.block<3, 3>(15 * id + 6, 15 * id + 15) += h_v1r2;
          system.pose_block.block<3, 3>(15 * id + 6, 15 * id + 18) += h_v1t2;
          system.pose_block.block<3, 3>(15 * id + 6, 15 * id + 21) += h_v1v2;
        }
      }
    }

    {
      if (!is_fixed(problem, i)) {
        system.pose_block.block<3, 3>(15 * id + 9, 15 * id) += h_r1bg.transpose();
        system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 3) += h_t1bg.transpose();
        system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 6) += h_v1bg.transpose();
        system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 9) += h_bgbg;
        system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 12) += h_bgba;
        if (!is_fixed(problem, i + 1)) {
          system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 15) += h_bgr2;
          system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 18) += h_bgt2;
          system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 21) += h_bgv2;
        }
      }
    }

    {
      if (!is_fixed(problem, i)) {
        system.pose_block.block<3, 3>(15 * id + 12, 15 * id) += h_r1ba.transpose();
        system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 3) += h_t1ba.transpose();
        system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 6) += h_v1ba.transpose();
        system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 9) += h_bgba.transpose();
        system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 12) += h_baba;
        if (!is_fixed(problem, i + 1)) {
          system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 15) += h_bar2;
          system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 18) += h_bat2;
          system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 21) += h_bav2;
        }
      }
    }

    {
      if (!is_fixed(problem, i + 1)) {
        if (!is_fixed(problem, i)) {
          system.pose_block.block<3, 3>(15 * id + 15, 15 * id) += h_r1r2.transpose();
          system.pose_block.block<3, 3>(15 * id + 15, 15 * id + 3) += h_t1r2.transpose();
          system.pose_block.block<3, 3>(15 * id + 15, 15 * id + 6) += h_v1r2.transpose();
          system.pose_block.block<3, 3>(15 * id + 15, 15 * id + 9) += h_bgr2.transpose();
          system.pose_block.block<3, 3>(15 * id + 15, 15 * id + 12) += h_bar2.transpose();
        }
        system.pose_block.block<3, 3>(15 * id + 15, 15 * id + 15) += h_r2r2;
        system.pose_block.block<3, 3>(15 * id + 15, 15 * id + 18) += h_r2t2;
        system.pose_block.block<3, 3>(15 * id + 15, 15 * id + 21) += h_r2v2;
      }
    }

    {
      if (!is_fixed(problem, i + 1)) {
        if (!is_fixed(problem, i)) {
          system.pose_block.block<3, 3>(15 * id + 18, 15 * id) += h_r1t2.transpose();
          system.pose_block.block<3, 3>(15 * id + 18, 15 * id + 3) += h_t1t2.transpose();
          system.pose_block.block<3, 3>(15 * id + 18, 15 * id + 6) += h_v1t2.transpose();
          system.pose_block.block<3, 3>(15 * id + 18, 15 * id + 9) += h_bgt2.transpose();
          system.pose_block.block<3, 3>(15 * id + 18, 15 * id + 12) += h_bat2.transpose();
        }
        system.pose_block.block<3, 3>(15 * id + 18, 15 * id + 15) += h_r2t2.transpose();
        system.pose_block.block<3, 3>(15 * id + 18, 15 * id + 18) += h_t2t2;
        system.pose_block.block<3, 3>(15 * id + 18, 15 * id + 21) += h_t2v2;
      }
    }

    {
      if (!is_fixed(problem, i + 1)) {
        if (!is_fixed(problem, i)) {
          system.pose_block.block<3, 3>(15 * id + 21, 15 * id) += h_r1v2.transpose();
          system.pose_block.block<3, 3>(15 * id + 21, 15 * id + 3) += h_t1v2.transpose();
          system.pose_block.block<3, 3>(15 * id + 21, 15 * id + 6) += h_v1v2.transpose();
          system.pose_block.block<3, 3>(15 * id + 21, 15 * id + 9) += h_bgv2.transpose();
          system.pose_block.block<3, 3>(15 * id + 21, 15 * id + 12) += h_bav2.transpose();
        }
        system.pose_block.block<3, 3>(15 * id + 21, 15 * id + 15) += h_r2v2.transpose();
        system.pose_block.block<3, 3>(15 * id + 21, 15 * id + 18) += h_t2v2.transpose();
        system.pose_block.block<3, 3>(15 * id + 21, 15 * id + 21) += h_v2v2;
      }
    }

    info_r = info * model.inertial_residuals[i];

    if (!is_fixed(problem, i)) {
      system.pose_rhs.segment<3>(15 * id) -= model.inertial_jacobians[i].JR_left.transpose() * info_r;

      system.pose_rhs.segment<3>(15 * id + 3) -= model.inertial_jacobians[i].Jt_left.transpose() * info_r;

      system.pose_rhs.segment<3>(15 * id + 6) -= model.inertial_jacobians[i].Jv_left.transpose() * info_r;

      system.pose_rhs.segment<3>(15 * id + 9) -= model.inertial_jacobians[i].Jb_gyro_left.transpose() * info_r;

      system.pose_rhs.segment<3>(15 * id + 12) -= model.inertial_jacobians[i].Jb_acc_left.transpose() * info_r;
    }

    if (!is_fixed(problem, i + 1)) {
      system.pose_rhs.segment<3>(15 * id + 15) -= model.inertial_jacobians[i].JR_right.transpose() * info_r;

      system.pose_rhs.segment<3>(15 * id + 18) -= model.inertial_jacobians[i].Jt_right.transpose() * info_r;

      system.pose_rhs.segment<3>(15 * id + 21) -= model.inertial_jacobians[i].Jv_right.transpose() * info_r;
    }
  }

  Matrix3T info_gyro_rw, info_acc_rw;
  // random walk
  for (int i = 0; i < num_inertials; i++) {
    if (both_fixed(problem, i)) {
      continue;
    }

    const IMUPreintegration& preint = problem.rig_poses[i].preintegration;
    preint.InfoGyroRWMatrix(info_gyro_rw);
    preint.InfoAccRWMatrix(info_acc_rw);

    const float eff_penalty = effective_imu_penalty(problem, i);
    info_gyro_rw *= eff_penalty;
    info_acc_rw *= (problem.acc_rw_penalty >= 0 ? problem.acc_rw_penalty : eff_penalty);

    int id = i - problem.num_fixed_key_frames;

    if (!is_fixed(problem, i)) {
      system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 9) += info_gyro_rw;
      system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 12) += info_acc_rw;

      system.pose_rhs.segment<3>(15 * id + 9) -= info_gyro_rw * model.random_walk_gyro_residuals[i];

      system.pose_rhs.segment<3>(15 * id + 12) -= info_acc_rw * model.random_walk_acc_residuals[i];
    }

    if (!is_fixed(problem, i) && !is_fixed(problem, i + 1)) {
      system.pose_block.block<3, 3>(15 * id + 9, 15 * id + 24) -= info_gyro_rw;
      system.pose_block.block<3, 3>(15 * id + 24, 15 * id + 9) -= info_gyro_rw;
      system.pose_block.block<3, 3>(15 * id + 12, 15 * id + 27) -= info_acc_rw;
      system.pose_block.block<3, 3>(15 * id + 27, 15 * id + 12) -= info_acc_rw;
    }

    if (!is_fixed(problem, i + 1)) {
      system.pose_block.block<3, 3>(15 * id + 24, 15 * id + 24) += info_gyro_rw;
      system.pose_block.block<3, 3>(15 * id + 27, 15 * id + 27) += info_acc_rw;

      system.pose_rhs.segment<3>(15 * id + 24) += info_gyro_rw * model.random_walk_gyro_residuals[i];

      system.pose_rhs.segment<3>(15 * id + 27) += info_acc_rw * model.random_walk_acc_residuals[i];
    }
  }

  // bias priors on oldest non-fixed pose only; random walk propagates to newer poses
  if (num_poses_opt > 0) {
    Matrix3T prior_gyro_info = Matrix3T::Identity() * problem.prior_gyro;
    Matrix3T prior_acc_info = Matrix3T::Identity() * problem.prior_acc;
    const int i = problem.num_fixed_key_frames;
    system.pose_block.block<3, 3>(9, 9) += prior_gyro_info;
    system.pose_block.block<3, 3>(12, 12) += prior_acc_info;
    system.pose_rhs.segment<3>(9) -= prior_gyro_info * problem.rig_poses[i].gyro_bias;
    system.pose_rhs.segment<3>(12) -= prior_acc_info * problem.rig_poses[i].acc_bias;
  }
}

void IMUBundlerCpuFixedVel::BuildReducedSystem(internal::ReducedSystem& reduced_system,
                                               const internal::FullSystem& full_system, const float lambda) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU BuildReducedSystem()", profiler_color_);
  const auto num_points = static_cast<int>(full_system.point_block.size());
  const auto num_poses = static_cast<int>(full_system.pose_block.cols() / 15);

  reduced_system.inverse_point_block.resize(num_points);
  reduced_system.point_rhs.resize(num_points * 3);

  Matrix3T m, m_inv;
  Vector3T s;
  for (int i = 0; i < num_points; ++i) {
    // HZ: m = V* = V + lambda * diag(V)
    // dampening
    m = full_system.point_block[i] + (full_system.point_block[i].diagonal() * lambda).asDiagonal().toDenseMatrix();

    m_inv = m.ldlt().solve(Matrix3T::Identity());
    reduced_system.inverse_point_block[i] = m_inv;

    // HZ: inv(V*) * e_b_i
    reduced_system.point_rhs.segment<3>(3 * i) = m_inv * full_system.point_rhs.segment<3>(3 * i);
  }

  // HZ: W = transp(transp(W))
  // cam_point * inverse(point_point)
  Eigen::MatrixXf Y = full_system.point_pose_block.transpose();        // 6 num_poses x 3 num points
  reduced_system.camera_backsub_block = full_system.point_pose_block;  // 3 num points x 6 num_poses

  for (int i = 0; i < num_points; ++i) {
    // HZ: Y_ij = W * inv(V*)
    Y.block(0, i * 3, 6 * num_poses, 3) *= reduced_system.inverse_point_block[i];
  }

  // HZ: S_jj = U_j - Y_ij * transp(W)

  Eigen::MatrixXf YWt = Y * full_system.point_pose_block;  // 6 num_poses x 6 num poses
  Eigen::VectorXf Ypoint_rhs = Y * full_system.point_rhs;  // 6 num_poses
  reduced_system.pose_rhs = full_system.pose_rhs;
  reduced_system.pose_block = full_system.pose_block;

  for (int i = 0; i < num_poses; i++) {
    for (int j = 0; j < num_poses; j++) {
      reduced_system.pose_block.block<6, 6>(15 * i, 15 * j) -= YWt.block<6, 6>(6 * i, 6 * j);
    }
    reduced_system.pose_rhs.segment<6>(15 * i) -= Ypoint_rhs.segment<6>(6 * i);
  }
  // HZ: Y_ij
  reduced_system.camera_backsub_block = Y.transpose();

  // dampening
  for (int i = 0; i < reduced_system.pose_block.cols(); ++i) {
    // HZ: S_jj* = S_jj + lambda * U_j
    reduced_system.pose_block(i, i) += lambda * full_system.pose_block(i, i);
  }
}

}  // namespace cuvslam::sba_imu
