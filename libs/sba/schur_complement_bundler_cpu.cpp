
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

#include "sba/schur_complement_bundler_cpu.h"

#include "common/isometry.h"
#include "common/log_types.h"
#include "common/rotation_utils.h"
#include "common/statistic.h"
#include "epipolar/near_plane.h"
#include "math/robust_cost_function.h"
#include "math/twist.h"

namespace cuvslam::sba {

using namespace schur_complement_bundler_cpu_internal;

// TODO: (msmirnov) check performance cost, cost and weight computation
// can be vectorized.
// Note: Student-t loss is expensive to compute because of log.
// There is still work to do to pick the best loss function.
// I've tried both Student-t and Huber and both give similar
// results with Student-t giving slightly better metrics on KITTI
// with default parameters.
#define ROBUST_COST math::ComputeStudentLoss
#define ROBUST_WEIGHT math::ComputeDStudentLoss

namespace cuvslam_schur_complement_bundler_cpu_internal {

template <class Derived>
Eigen::Matrix3f Skew(const Eigen::MatrixBase<Derived>& x) {
  Eigen::Matrix3f s;
  s << 0, -x(2), x(1), x(2), 0, -x(0), -x(1), x(0), 0;
  return s;
}

template <class Derived>
void Adjoint(Eigen::MatrixBase<Derived>& ad, const Isometry3T& transformation) {
  const Matrix3T& rotation = transformation.linear();
  ad.template block<3, 3>(0, 0) = rotation;
  ad.template block<3, 3>(0, 3).setZero();
  ad.template block<3, 3>(3, 0) = Skew(transformation.translation()) * rotation;
  ad.template block<3, 3>(3, 3) = rotation;
}
}  // namespace cuvslam_schur_complement_bundler_cpu_internal

using namespace cuvslam_schur_complement_bundler_cpu_internal;

// TODO: compute robustifier weights in a separate loop
void schur_complement_bundler_cpu_internal::UpdateModel(ModelFunction& model, BundleAdjustmentProblem& problem) {
  const auto num_observations = static_cast<int>(problem.observation_xys.size());
  model.point_jacobians.resize(num_observations);
  model.pose_jacobians.resize(num_observations);
  model.residuals.resize(num_observations);
  model.robustifier_weights.resize(num_observations);

  using Vec3 = Eigen::Vector3f;
  using Vec2 = Eigen::Vector2f;

  Eigen::Matrix<float, 6, 6> ads[camera::Rig::kMaxCameras];
  for (int i = 0; i < problem.rig.num_cameras; ++i) {
    cuvslam_schur_complement_bundler_cpu_internal::Adjoint(ads[i], problem.rig.camera_from_rig[i]);
  }

  for (int observation = 0; observation < num_observations; ++observation) {
    const auto point_idx = problem.point_ids[observation];
    assert(point_idx < static_cast<int>(problem.points.size()));

    const auto pose_idx = problem.pose_ids[observation];
    assert(pose_idx < static_cast<int>(problem.rig_from_world.size()));

    const auto camera_idx = problem.camera_ids[observation];
    assert(camera_idx < problem.rig.num_cameras);

    auto& p_w = problem.points[point_idx];

    const Isometry3T camera_from_world = problem.rig.camera_from_rig[camera_idx] * problem.rig_from_world[pose_idx];

    // TODO: we don't need to store camera ids, they are available
    // from observations_start indirectly
    Vec3 p_c = camera_from_world * p_w;

    if (epipolar::IsDepthInFront(p_c.z())) {
      Vec2 prediction = p_c.topRows(2) / p_c.z();
      Vec2 r = problem.observation_xys[observation] - prediction;

      auto inv_z = 1.f / p_c.z();

      Eigen::Matrix<float, 2, 3> dproj;
      dproj << inv_z, 0.f, -prediction.x() * inv_z, 0.f, inv_z, -prediction.y() * inv_z;

      // camera_from_world.rotation = camera_from_world.matrix().block<3, 3>(0, 0)
      Eigen::Matrix<float, 2, 3> dp = dproj * camera_from_world.matrix().block<3, 3>(0, 0);
      Eigen::Matrix<float, 2, 6> dc;
      dc.block<2, 3>(0, 0) = dproj * cuvslam_schur_complement_bundler_cpu_internal::Skew(-p_c);
      dc.block<2, 3>(0, 3) = dproj;

      // we have computed jacobian for the camera update,
      // transform back to rig update
      dc = dc * ads[camera_idx];

      model.residuals[observation] = r;
      model.point_jacobians[observation] = dp;
      model.pose_jacobians[observation] = dc;
      // as robustifier_weights will be recalculated in CalculateInformationMatrices
      // we just set them to non 0 to indicate that we want them to be calculated

      auto r_sq_norm = r.dot(problem.observation_infos[observation] * r);
      model.robustifier_weights[observation] = ROBUST_WEIGHT(r_sq_norm, problem.robustifier_scale);
    } else {
      model.residuals[observation].setZero();
      model.point_jacobians[observation].setZero();
      model.pose_jacobians[observation].setZero();
      model.robustifier_weights[observation] = 0.f;
    }
  }
}

// TODO: premultiply everything by covariance
// HZ references to Hartley-Zisserman book "Multiple view geometry in computer vision" second edition
// start from Algorithm A6.4 (p. 613)
void schur_complement_bundler_cpu_internal::BuildFullSystem(FullSystem& system, const ModelFunction& model,
                                                            const BundleAdjustmentProblem& problem) {
  const auto num_points = static_cast<int>(problem.points.size()) - problem.num_fixed_points;
  const auto num_poses = static_cast<int>(problem.rig_from_world.size()) - problem.num_fixed_key_frames;
  const auto num_observations = static_cast<int>(model.residuals.size());

  assert(num_points >= 0);
  assert(num_poses >= 0);

  // HZ: V
  system.point_block.clear();
  system.point_block.resize(num_points, Matrix3T::Zero());

  // HZ: e_a
  system.pose_rhs.resize(6 * num_poses);
  system.pose_rhs.setZero();

  // HZ: U
  system.pose_block.resize(6 * num_poses, 6 * num_poses);
  system.pose_block.setZero();

  // HZ: transp(W)
  system.point_pose_block.resize(3 * num_points, 6 * num_poses);
  system.point_pose_block.setZero();

  // HZ: e_b
  system.point_rhs.resize(3 * num_points);
  system.point_rhs.setZero();

  // TODO: loop over cameras instead, then loop over points.
  // This way we will accumulate in local variable instead of scattering.
  for (int obs = 0; obs < num_observations; ++obs) {
    const float weight = model.robustifier_weights[obs];
    const Matrix2T& info = problem.observation_infos[obs];

    Eigen::Matrix3f hpp = model.point_jacobians[obs].transpose() * info * model.point_jacobians[obs];

    // HZ: transp(W_ij) = transp(B_ij) * inv(E_x_ij) * A_ij
    Eigen::Matrix<float, 3, 6> hpc = model.point_jacobians[obs].transpose() * info * model.pose_jacobians[obs];

    // HZ: U_ij = transp(A_ij) * inv(E_x_ij) * A_ij
    Eigen::Matrix<float, 6, 6> hcc = model.pose_jacobians[obs].transpose() * info * model.pose_jacobians[obs];

    // HZ: c -> j
    auto c = problem.pose_ids[obs];

    // HZ: p -> i
    auto p = problem.point_ids[obs];

    // both the point and the camera are relaxed
    if ((c < num_poses) && (p < num_points)) {
      // HZ: transp(W) = sum_ij(transp(W_ij))
      system.point_pose_block.block<3, 6>(3 * p, 6 * c) += hpc * weight;
    }

    // camera is relaxed
    if (c < num_poses) {
      // HZ: U_j = sum_i(U_ij)
      system.pose_block.block<6, 6>(6 * c, 6 * c) += hcc * weight;
      // HZ: e_a_j = sum_i(transp(A_ij) * inv(E_x_ij) * e_ij)
      system.pose_rhs.segment<6>(6 * c) +=
          model.pose_jacobians[obs].transpose() * (info * weight * model.residuals[obs]);
    }

    // point is relaxed
    if (p < num_points) {
      // HZ: V_i = sum_j(V_ij)
      system.point_block[p] += hpp * weight;
      // HZ: e_b_i = sum_j(transp(B_ij) * inv(E_x_ij) * e_ij)
      system.point_rhs.segment<3>(3 * p) +=
          model.point_jacobians[obs].transpose() * (info * weight * model.residuals[obs]);
    }
  }
}

void schur_complement_bundler_cpu_internal::BuildReducedSystem(ReducedSystem& reduced_system,
                                                               const FullSystem& full_system, const float lambda) {
  const auto num_points = static_cast<int>(full_system.point_block.size());

  reduced_system.inverse_point_block.resize(num_points);
  reduced_system.point_rhs.resize(num_points * 3);

  for (int i = 0; i < num_points; ++i) {
    // HZ: m = V* = V + lambda * diag(V)
    // dampening
    Eigen::Matrix3f m =
        full_system.point_block[i] + (full_system.point_block[i].diagonal() * lambda).asDiagonal().toDenseMatrix();

    // rank(m) >= 2 by construction
    // If m is close to being rank deficient we estimate rank to be 2,
    // thus avoiding large values in the solution.
    // If under-estimated rank will lead to an increase in the cost,
    // next iteration of Levenberg-Marquardt will increase dampening
    // and the dampened matrix will be of rank 3.

    Eigen::Matrix3f inv;
    inv.setZero();

    // TODO: (msmirnov) check if SVD is too costly here.
    // We don't need preconditioner because the matrix is square
    Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::NoQRPreconditioner> usv(m, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // TODO: (msmirnov) optimize bounds
    // This threshold effectively bounds condition number of the matrix.
    usv.setThreshold(1e-6f);

    if (usv.rank() == 3) {
      inv = usv.solve(Eigen::Matrix3f::Identity());
    } else if (usv.rank() == 2) {
      // zero out singular direction (low curvature)
      Eigen::Vector3f s = usv.singularValues();
      assert(s(0) >= s(1) && s(1) >= s(2));
      s(0) = 1.f / s(0);
      s(1) = 1.f / s(1);
      s(2) = 0.f;
      inv = usv.matrixV() * s.asDiagonal() * usv.matrixU().transpose();
    } else {
      // point is behind all cameras => nothing to do
    }

    // HZ: ? = inv(V*)
    reduced_system.inverse_point_block[i] = inv;

    // HZ: inv(V*) * e_b_i
    Eigen::Vector3f r = full_system.point_rhs.segment<3>(3 * i);
    reduced_system.point_rhs.segment<3>(3 * i) = inv * r;
  }

  // HZ: W = transp(transp(W))
  // cam_point * inverse(point_point)
  Eigen::MatrixXf temp = full_system.point_pose_block.transpose();

  for (int i = 0; i < num_points; ++i) {
    // HZ: Y_ij = W * inv(V*)
    temp.block(0, i * 3, temp.rows(), 3) *= reduced_system.inverse_point_block[i];
  }

  // HZ: S_jj = U_j - Y_ij * transp(W)
  reduced_system.pose_block = full_system.pose_block - temp * full_system.point_pose_block;

  // HZ: e_j = e_a_j - Y_ij * e_b_i
  reduced_system.pose_rhs = full_system.pose_rhs - temp * full_system.point_rhs;

  // HZ: Y_ij
  reduced_system.camera_backsub_block = temp;

  // dampening
  for (int i = 0; i < reduced_system.pose_block.cols(); ++i) {
    // HZ: S_jj* = S_jj + lambda * U_j
    reduced_system.pose_block(i, i) += lambda * full_system.pose_block(i, i);
  }
}

float schur_complement_bundler_cpu_internal::EvaluateCost(const BundleAdjustmentProblem& problem,
                                                          const ParameterUpdate& update) {
  using Vec3 = Eigen::Vector3f;
  using Vec2 = Eigen::Vector2f;

  float cost{0};

  const auto num_observations = static_cast<int32_t>(problem.observation_xys.size());

  // we may "skip" some observations if they are too close to the camera
  int32_t num_skipped{0};

  for (int observation = 0; observation < num_observations; ++observation) {
    const auto point_idx = problem.point_ids[observation];
    assert(point_idx < static_cast<int>(problem.points.size()));
    assert(point_idx < static_cast<int>(update.point.size()));

    const auto pose_idx = problem.pose_ids[observation];
    assert(pose_idx < static_cast<int>(problem.rig_from_world.size()));
    assert(pose_idx < static_cast<int>(update.pose.size()));

    const auto camera_idx = problem.camera_ids[observation];
    assert(camera_idx < static_cast<int>(problem.rig.num_cameras));

    auto& p_w = problem.points[point_idx] + update.point[point_idx];
    Vec3 p_r = update.pose[pose_idx] * (problem.rig_from_world[pose_idx] * p_w);

    // TODO: we don't need to store camera ids, they are available
    // from observations_start indirectly
    Vec3 p_c = problem.rig.camera_from_rig[camera_idx] * p_r;

    if (epipolar::IsDepthInFront(p_c.z())) {
      Vec2 r = problem.observation_xys[observation] - p_c.topRows(2) / p_c.z();

      const Matrix2T& info = problem.observation_infos[observation];
      cost += ROBUST_COST(r.dot(info * r), problem.robustifier_scale);
    } else {
      ++num_skipped;
    }
  }

  if (num_skipped == num_observations) {
    return std::numeric_limits<float>::infinity();
  }

  return cost;
}

void schur_complement_bundler_cpu_internal::ComputeUpdate(ParameterUpdate& update, const ReducedSystem& reduced_system,
                                                          const FullSystem& full_system) {
  // Try LDLT instead.
  Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::NoQRPreconditioner> usv(reduced_system.pose_block,
                                                                   Eigen::ComputeFullU | Eigen::ComputeFullV);

  // control condition number
  usv.setThreshold(1e-6f);

  Eigen::VectorXf dpose = usv.solve(reduced_system.pose_rhs);

  Eigen::VectorXf dpoint = reduced_system.point_rhs - reduced_system.camera_backsub_block.transpose() * dpose;

  const auto num_points = static_cast<int>(full_system.point_block.size());
  const auto num_poses = static_cast<int>(full_system.pose_block.cols() / 6);

  for (int i = 0; i < num_points; ++i) {
    update.point[i] = dpoint.segment<3>(3 * i);
  }

  for (int i = 0; i < num_poses; ++i) {
    cuvslam::Isometry3T d;
    math::Exp(d, cuvslam::Vector6T(dpose.segment<6>(6 * i)));
    update.pose[i] = d;
  }

  update.pose_step = std::move(dpose);
  update.point_step = std::move(dpoint);
}

void schur_complement_bundler_cpu_internal::UpdateState(BundleAdjustmentProblem& problem,
                                                        const ParameterUpdate& update) {
  const auto num_points = static_cast<int32_t>(problem.points.size()) - problem.num_fixed_points;
  const auto num_poses = static_cast<int32_t>(problem.rig_from_world.size()) - problem.num_fixed_key_frames;

  for (int32_t i = 0; i < num_points; ++i) {
    problem.points[i] += update.point[i];
  }

  for (int32_t i = 0; i < num_poses; ++i) {
    Isometry3T m = problem.rig_from_world[i];
    m = update.pose[i] * m;
    m.linear() = common::CalculateRotationFromSVD(m.matrix());
    m.makeAffine();
    problem.rig_from_world[i] = m;
  }
}

float schur_complement_bundler_cpu_internal::ComputePredictedRelativeReduction(const float current_cost,
                                                                               const float lambda,
                                                                               const ParameterUpdate& update,
                                                                               const FullSystem& system) {
  assert(current_cost > std::numeric_limits<float>::epsilon());

  const int num_points = static_cast<int>(system.point_block.size());

  float hessian_term{0};

  {
    float pose_term = (update.pose_step.dot(system.pose_block * update.pose_step) +
                       update.pose_step.dot(system.point_pose_block.transpose() * update.point_step)) /
                      current_cost;

    float point_term = (update.point_step.dot(system.point_pose_block * update.pose_step));

    for (int i = 0; i < num_points; ++i) {
      point_term +=
          update.point_step.segment<3>(3 * i).dot(system.point_block[i] * update.point_step.segment<3>(3 * i));
    }

    point_term /= current_cost;

    hessian_term = pose_term + point_term;
  }
  assert(std::isfinite(hessian_term));

  float scaling_term{0};

  {
    float pose_term = update.pose_step.dot(system.pose_block.diagonal().asDiagonal() * update.pose_step);
    float point_term{0};

    for (int i = 0; i < num_points; ++i) {
      Eigen::Vector3f dp = update.point_step.segment<3>(3 * i);
      point_term += dp.dot(system.point_block[i].diagonal().asDiagonal() * dp);
    }

    scaling_term = ((pose_term + point_term) / current_cost) * 2.f * lambda;
  }
  assert(std::isfinite(scaling_term));

  const float prediction = hessian_term + scaling_term;

  assert(std::isfinite(prediction));

  return prediction;
}

bool SchurComplementBundlerCpu::solve(BundleAdjustmentProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU solve", profiler_color_);

  const auto num_points = static_cast<int32_t>(problem.points.size());
  const auto num_poses = static_cast<int32_t>(problem.rig_from_world.size());

  ParameterUpdate update;
  update.point.resize(num_points, Vector3T::Zero());
  update.pose.resize(num_poses, Isometry3T::Identity());

  ModelFunction model;
  UpdateModel_(model, problem);

  const float initial_cost = EvaluateCost_(problem, update);
  float current_cost = initial_cost;
  problem.initial_cost = initial_cost;
  problem.last_cost = initial_cost;

  if (initial_cost < std::numeric_limits<float>::epsilon()) {
    return true;
  }

  if (initial_cost == std::numeric_limits<float>::infinity()) {
    // starting point is not feasible
    return false;
  }
  if (num_points < problem.num_fixed_points) {
    return false;
  }
  if (num_poses < problem.num_fixed_key_frames) {
    return false;
  }

  int iteration{0};
  const int max_iterations = problem.max_iterations;

  FullSystem full_system;
  ReducedSystem reduced_system;

  BuildFullSystem_(full_system, model, problem);

  float lambda = 0.001f;

  while (iteration < max_iterations) {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("CPU iteration", profiler_color_);
    ++iteration;
    problem.iterations = iteration;

    BuildReducedSystem_(reduced_system, full_system, lambda);
    ComputeUpdate_(update, reduced_system, full_system);

    auto cost = EvaluateCost_(problem, update);

    if (current_cost < initial_cost * std::numeric_limits<float>::epsilon()) {
      return true;
    }

    auto predicted_relative_reduction = ComputePredictedRelativeReduction_(current_cost, lambda, update, full_system);

    if ((predicted_relative_reduction < sqrt_epsilon()) &&
        (update.pose_step.lpNorm<Eigen::Infinity>() < sqrt_epsilon()) &&
        (update.point_step.lpNorm<Eigen::Infinity>() < sqrt_epsilon())) {
      return true;
    }

    // We guarantee that initial cost is not infinity and we will
    // never accept a step that leads to an infinite cost.
    // This mean that we will never have a situation when
    // cost = inf and current_cost = inf.
    // The only possible case is cost = inf (step leads outside of feasible region).
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
      problem.last_cost = cost;
      UpdateState_(problem, update);

      UpdateModel_(model, problem);
      BuildFullSystem_(full_system, model, problem);
    } else {
      lambda *= 5.f;
    }
  }

  log::Value<LogSba>("SBA iterations", iteration);
  return true;
}

float SchurComplementBundlerCpu::EvaluateCost_(const BundleAdjustmentProblem& problem, const ParameterUpdate& update) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU EvaluateCost()", profiler_color_);
  return EvaluateCost(problem, update);
}

void SchurComplementBundlerCpu::ComputeUpdate_(ParameterUpdate& update, const ReducedSystem& reduced_system,
                                               const FullSystem& full_system) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU ComputeUpdate()", profiler_color_);
  ComputeUpdate(update, reduced_system, full_system);
}

void SchurComplementBundlerCpu::UpdateState_(BundleAdjustmentProblem& problem, const ParameterUpdate& update) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU UpdateState()", profiler_color_);
  UpdateState(problem, update);
}

float SchurComplementBundlerCpu::ComputePredictedRelativeReduction_(const float current_cost, const float lambda,
                                                                    const ParameterUpdate& update,
                                                                    const FullSystem& system) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU ComputePredictedRelativeReduction()", profiler_color_);
  return ComputePredictedRelativeReduction(current_cost, lambda, update, system);
}

void SchurComplementBundlerCpu::UpdateModel_(ModelFunction& model, BundleAdjustmentProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU UpdateModel()", profiler_color_);
  UpdateModel(model, problem);
}

void SchurComplementBundlerCpu::BuildFullSystem_(FullSystem& system, const ModelFunction& model,
                                                 const BundleAdjustmentProblem& problem) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU BuildFullSystem()", profiler_color_);
  BuildFullSystem(system, model, problem);
}

void SchurComplementBundlerCpu::BuildReducedSystem_(ReducedSystem& reduced_system, const FullSystem& full_system,
                                                    const float lambda) {
  TRACE_EVENT ev = profiler_domain_.trace_event("CPU BuildReducedSystem()", profiler_color_);
  BuildReducedSystem(reduced_system, full_system, lambda);
}

}  // namespace cuvslam::sba
