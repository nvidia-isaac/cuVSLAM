
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

#include "math/pgo.h"

#include "Eigen/Sparse"

#include "common/types.h"
#include "common/vector_3t.h"

#include "math/robust_cost_function.h"
#include "math/twist.h"

namespace cuvslam::math {

constexpr int num_reserve_floats = 200;

namespace {

Isometry3T update(const Isometry3T& pose, const Vector6T& step) {
  Isometry3T R;
  Exp(R, step);
  return pose * R;
}

Isometry3T update(const Isometry3T& pose, const Vector3T& step) {
  Isometry3T R;
  Vector6T step_ = Vector6T::Zero();
  step_.segment<3>(3) = step;
  Exp(R, step_);
  return pose * R;
}

struct ConstrainedSystem {
  Matrix3T H;
  Vector3T rhs;
};

struct Index {
  std::vector<int> pose_to_var;  // pose_to_var[pose_id] -> var_id or -1 if no variable
  std::vector<int> var_to_pose;  // var_to_pose[var_id] -> pose_id
};

bool is_pose_constrained(const PGOInput& inputs, int pose_id) {
  return inputs.constrained_pose_ids.find(pose_id) != inputs.constrained_pose_ids.end();
}

void insert6T(Eigen::SparseMatrix<float>& HS, int row, int col, const Matrix6T& m) {
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      HS.insert(i + row, j + col) = m(i, j);
    }
  }
}

struct pair_hash {
  inline std::size_t operator()(const std::pair<int, int>& v) const {
    std::hash<int> int_hasher;
    return int_hasher(v.first) ^ int_hasher(v.second);
  }
};

// Precondition: inputs.poses.size() > inputs.constrained_pose_ids.size())
void fill_sparse(const PGOInput& inputs, const std::vector<std::pair<int, int>>& blocks, const Eigen::MatrixXf& H,
                 Eigen::SparseMatrix<float>& HS) {
  const size_t order = H.rows();
  HS.resize(order, order);
  HS.reserve(Eigen::VectorXi::Constant(order, num_reserve_floats));

  assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
  const size_t num_constraints = inputs.constrained_pose_ids.size();
  const size_t num_non_constraints = inputs.poses.size() - num_constraints;

  for (size_t i = 0; i < num_non_constraints; i++) {
    insert6T(HS, 6 * i, 6 * i, H.block<6, 6>(6 * i, 6 * i));
  }

  for (const auto& [id1, id2] : blocks) {
    insert6T(HS, 6 * id1, 6 * id2, H.block<6, 6>(6 * id1, 6 * id2));
    insert6T(HS, 6 * id2, 6 * id1, H.block<6, 6>(6 * id2, 6 * id1));
  }
}

void calc_blocks(const PGOInput& inputs, const Index& index, std::vector<std::pair<int, int>>& blocks) {
  blocks.clear();
  std::unordered_set<std::pair<int, int>, pair_hash> seen_nodes;
  for (const auto& [pose1_id, pose2_id, _, __] : inputs.deltas) {
    bool pose1_constrained = is_pose_constrained(inputs, pose1_id);
    bool pose2_constrained = is_pose_constrained(inputs, pose2_id);

    if (!pose1_constrained && !pose2_constrained) {
      int var1_id = index.pose_to_var[pose1_id];
      int var2_id = index.pose_to_var[pose2_id];

      if (var1_id == var2_id) {
        continue;
      }
      bool seen = seen_nodes.find({var1_id, var2_id}) != seen_nodes.end();

      if (!seen) {
        seen_nodes.insert({var1_id, var2_id});
        seen_nodes.insert({var2_id, var1_id});
        blocks.push_back({var1_id, var2_id});
      }
    }
  }
}

}  // namespace

float evaluate_cost(const PGOInput& inputs, const Index& index, const std::vector<Isometry3T>& variables,
                    const Eigen::VectorXf& updates) {
  float cost = 0;

  std::vector<Isometry3T> updated_vars;
  updated_vars.reserve(variables.size());
  for (size_t i = 0; i < variables.size(); i++) {
    const Isometry3T& var = variables[i];
    const Vector6T& s = updates.segment<6>(6 * i);
    updated_vars.push_back(update(var, s));
  }

  for (const auto& [pose1_id, pose2_id, pose1_from_pose2, info] : inputs.deltas) {
    bool pose1_constrained = is_pose_constrained(inputs, pose1_id);
    bool pose2_constrained = is_pose_constrained(inputs, pose2_id);
    if (!pose1_constrained || !pose2_constrained) {
      Isometry3T p1 = inputs.poses[pose1_id];
      if (!pose1_constrained) {
        p1 = updated_vars[index.pose_to_var[pose1_id]];
      }
      Isometry3T p2 = inputs.poses[pose2_id];
      if (!pose2_constrained) {
        p2 = updated_vars[index.pose_to_var[pose2_id]];
      }

      Vector6T err;
      Log(err, pose1_from_pose2.inverse() * p1.inverse() * p2);

      float x_squared = err.dot(info * err);
      cost += ComputeHuberLoss(x_squared, inputs.robustifier);
    }
  }

  return cost;
}

// Precondition: inputs.poses.size() > inputs.constrained_pose_ids.size())
float evaluate_cost_planar(const PGOInput& inputs, const Index& index, const std::vector<Isometry3T>& variables,
                           const Eigen::VectorXf& updates, const std::vector<Vector3T>& constraint_steps) {
  float cost = 0;

  std::vector<Isometry3T> updated_vars;
  updated_vars.reserve(variables.size());

  assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
  const size_t num_constraints = inputs.constrained_pose_ids.size();
  const size_t num_non_constraints = inputs.poses.size() - num_constraints;

  for (size_t i = 0; i < num_non_constraints; i++) {
    const Isometry3T& var = variables[i];
    const Vector6T& s = updates.segment<6>(6 * i);
    updated_vars.push_back(update(var, s));
  }

  for (size_t i = 0; i < num_constraints; i++) {
    const Isometry3T& var = variables[num_non_constraints + i];
    const Vector3T& s = constraint_steps[i];
    updated_vars.push_back(update(var, s));
  }

  Eigen::Matrix<float, 1, 3> n_T = inputs.plane_normal.segment<3>(0).transpose();
  for (const Isometry3T& v : updated_vars) {
    float e = n_T * v.translation() + inputs.plane_normal[3];
    cost += e * inputs.planar_weight * e;
  }

  for (const auto& [pose1_id, pose2_id, pose1_from_pose2, info] : inputs.deltas) {
    bool pose1_constrained = is_pose_constrained(inputs, pose1_id);
    bool pose2_constrained = is_pose_constrained(inputs, pose2_id);
    if (!pose1_constrained || !pose2_constrained) {
      Isometry3T p1 = inputs.poses[pose1_id];
      if (!pose1_constrained) {
        p1 = updated_vars[index.pose_to_var[pose1_id]];
      }
      Isometry3T p2 = inputs.poses[pose2_id];
      if (!pose2_constrained) {
        p2 = updated_vars[index.pose_to_var[pose2_id]];
      }

      Vector6T err;
      Log(err, pose1_from_pose2.inverse() * p1.inverse() * p2);

      float x_squared = err.dot(info * err);
      cost += ComputeHuberLoss(x_squared, inputs.robustifier);
    }
  }

  return cost;
}

// Precondition: inputs.poses.size() > inputs.constrained_pose_ids.size())
void build_hessian_planar(const PGOInput& inputs, const Index& index, const std::vector<Isometry3T>& variables,
                          Eigen::MatrixXf& H, Eigen::VectorXf& rhs, std::vector<ConstrainedSystem>& systems) {
  assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
  const size_t num_constraints = inputs.constrained_pose_ids.size();
  const size_t num_non_constraints = inputs.poses.size() - num_constraints;

  for (size_t i = 0; i < num_non_constraints; i++) {
    H.block<6, 6>(6 * i, 6 * i).setZero();
  }

  for (const auto& [pose1_id, pose2_id, pose1_from_pose2, info] : inputs.deltas) {
    bool pose1_constrained = is_pose_constrained(inputs, pose1_id);
    bool pose2_constrained = is_pose_constrained(inputs, pose2_id);

    if (!pose1_constrained && !pose2_constrained) {
      int var1_id = index.pose_to_var[pose1_id];
      int var2_id = index.pose_to_var[pose2_id];

      H.block<6, 6>(var1_id * 6, var2_id * 6).setZero();
      H.block<6, 6>(var2_id * 6, var1_id * 6).setZero();
    }
  }

  // H.setZero();
  rhs.setZero();

  systems.clear();
  systems.reserve(num_constraints);

  Eigen::Matrix<float, 1, 3> J_planar;
  Eigen::Matrix<float, 1, 3> n_T = inputs.plane_normal.segment<3>(0).transpose();

  for (size_t i = 0; i < num_non_constraints; i++) {
    const Isometry3T& v = variables[i];
    J_planar = n_T * v.linear();
    float e = n_T * v.translation() + inputs.plane_normal[3];

    H.block<3, 3>(6 * i + 3, 6 * i + 3) += J_planar.transpose() * inputs.planar_weight * J_planar;
    rhs.segment<3>(6 * i + 3) -= J_planar.transpose() * inputs.planar_weight * e;
  }

  for (size_t i = 0; i < num_constraints; i++) {
    const Isometry3T& v = variables[num_non_constraints + i];
    J_planar = n_T * v.linear();
    float e = n_T * v.translation() + inputs.plane_normal[3];

    systems.push_back(
        {J_planar.transpose() * inputs.planar_weight * J_planar, -J_planar.transpose() * inputs.planar_weight * e});
  }

  for (const auto& [pose1_id, pose2_id, pose1_from_pose2, info] : inputs.deltas) {
    bool pose1_constrained = is_pose_constrained(inputs, pose1_id);
    bool pose2_constrained = is_pose_constrained(inputs, pose2_id);

    if (pose1_constrained && pose2_constrained) {
      continue;
    }

    Isometry3T p1 = inputs.poses[pose1_id];
    if (!pose1_constrained) {
      p1 = variables[index.pose_to_var[pose1_id]];
    }
    Isometry3T p2 = inputs.poses[pose2_id];
    if (!pose2_constrained) {
      p2 = variables[index.pose_to_var[pose2_id]];
    }

    Vector6T err;
    Log(err, pose1_from_pose2.inverse() * p1.inverse() * p2);

    float x_squared = err.dot(info * err);
    float w = ComputeDHuberLoss(x_squared, inputs.robustifier);

    Matrix6T info_w = info * w;

    Matrix6T Jp2 = se3_twist_right_inverse_jacobian(err);
    Matrix6T Jp1 = -Jp2 * adjoint(p2.inverse() * p1);

    Matrix6T Jp1T_info = Jp1.transpose() * info_w;

    if (!pose1_constrained) {
      int var_id = index.pose_to_var[pose1_id];

      H.block<6, 6>(var_id * 6, var_id * 6) += Jp1T_info * Jp1;
      rhs.segment<6>(var_id * 6) -= Jp1T_info * err;
    }

    if (!pose2_constrained) {
      int var_id = index.pose_to_var[pose2_id];

      Matrix6T Jp2T_info = Jp2.transpose() * info_w;
      H.block<6, 6>(var_id * 6, var_id * 6) += Jp2T_info * Jp2;
      rhs.segment<6>(var_id * 6) -= Jp2T_info * err;
    }

    if (!pose1_constrained && !pose2_constrained) {
      int var1_id = index.pose_to_var[pose1_id];
      int var2_id = index.pose_to_var[pose2_id];

      Matrix6T hp1p2 = Jp1.transpose() * info_w * Jp2;

      H.block<6, 6>(var1_id * 6, var2_id * 6) += hp1p2;
      H.block<6, 6>(var2_id * 6, var1_id * 6) += hp1p2.transpose();
    }
  }
}

void build_hessian(const PGOInput& inputs, const Index& index, const std::vector<Isometry3T>& variables,
                   Eigen::MatrixXf& H, Eigen::VectorXf& rhs) {
  H.setZero();
  rhs.setZero();

  for (const auto& [pose1_id, pose2_id, pose1_from_pose2, info] : inputs.deltas) {
    bool pose1_constrained = is_pose_constrained(inputs, pose1_id);
    bool pose2_constrained = is_pose_constrained(inputs, pose2_id);

    if (pose1_constrained && pose2_constrained) {
      continue;
    }

    Isometry3T p1 = inputs.poses[pose1_id];
    if (!pose1_constrained) {
      p1 = variables[index.pose_to_var[pose1_id]];
    }
    Isometry3T p2 = inputs.poses[pose2_id];
    if (!pose2_constrained) {
      p2 = variables[index.pose_to_var[pose2_id]];
    }

    Vector6T err;
    Log(err, pose1_from_pose2.inverse() * p1.inverse() * p2);

    float x_squared = err.dot(info * err);
    float w = ComputeDHuberLoss(x_squared, inputs.robustifier);

    Matrix6T info_w = info * w;

    Matrix6T Jp2 = se3_twist_right_inverse_jacobian(err);
    Matrix6T Jp1 = -Jp2 * adjoint(p2.inverse() * p1);

    Matrix6T Jp1T_info = Jp1.transpose() * info_w;

    if (!pose1_constrained) {
      int var_id = index.pose_to_var[pose1_id];

      H.block<6, 6>(var_id * 6, var_id * 6) += Jp1T_info * Jp1;
      rhs.segment<6>(var_id * 6) -= Jp1T_info * err;
    }

    if (!pose2_constrained) {
      int var_id = index.pose_to_var[pose2_id];

      Matrix6T Jp2T_info = Jp2.transpose() * info_w;
      H.block<6, 6>(var_id * 6, var_id * 6) += Jp2T_info * Jp2;
      rhs.segment<6>(var_id * 6) -= Jp2T_info * err;
    }

    if (!pose1_constrained && !pose2_constrained) {
      int var1_id = index.pose_to_var[pose1_id];
      int var2_id = index.pose_to_var[pose2_id];

      Matrix6T hp1p2 = Jp1.transpose() * info_w * Jp2;

      H.block<6, 6>(var1_id * 6, var2_id * 6) += hp1p2;
      H.block<6, 6>(var2_id * 6, var1_id * 6) += hp1p2.transpose();
    }
  }
}

void constraint_solve(float lambda, const std::vector<ConstrainedSystem>& systems, std::vector<Vector3T>& steps) {
  steps.clear();
  steps.reserve(systems.size());

  for (size_t i = 0; i < systems.size(); i++) {
    Matrix3T aug_h = systems[i].H + lambda * Matrix3T::Identity();

    // std::cout << "aug_h = " << std::endl << aug_h << std::endl;
    // std::cout << "rhs = " << std::endl << systems[i].rhs << std::endl;

    // Vector3T step = aug_h.ldlt().solve(systems[i].rhs);
    Vector3T step = aug_h.ldlt().solve(systems[i].rhs);

    // std::cout << "step = " << std::endl << step << std::endl;

    steps.push_back(std::move(step));
  }
}

void PGOInput::clear() {
  deltas.clear();
  poses.clear();
  constrained_pose_ids.clear();
}

// Precondition: inputs.poses.size() > inputs.constrained_pose_ids.size())
bool PGO::pgo_planar(PGOInput& inputs, int max_iterations) const {
  assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
  const size_t num_constraints = inputs.constrained_pose_ids.size();
  const size_t num_poses = inputs.poses.size();
  const size_t num_non_constraints = num_poses - num_constraints;

  std::vector<Isometry3T> variables;
  Index index;

  index.pose_to_var.reserve(num_poses);
  index.var_to_pose.reserve(num_poses);
  variables.reserve(num_poses);

  for (size_t i = 0; i < num_poses; i++) {
    int var_id = -1;
    if (!is_pose_constrained(inputs, i)) {
      var_id = static_cast<int>(variables.size());
      variables.push_back(inputs.poses[i]);
      index.var_to_pose.push_back(i);
    }
    index.pose_to_var.push_back(var_id);
  }

  for (size_t i = 0; i < num_poses; i++) {
    int var_id = -1;
    if (is_pose_constrained(inputs, i)) {
      var_id = static_cast<int>(variables.size());
      variables.push_back(inputs.poses[i]);
      index.var_to_pose.push_back(i);
    }
    index.pose_to_var.push_back(var_id);
  }

  std::vector<std::pair<int, int>> block_coords;
  calc_blocks(inputs, index, block_coords);

  size_t order = num_non_constraints * 6;

  step_.setZero(order);
  rhs_.resize(order);
  // H_.resize(order, order);
  H_.setZero(order, order);

  std::vector<ConstrainedSystem> systems;
  std::vector<Vector3T> constrained_steps;
  systems.reserve(num_constraints);
  constrained_steps.resize(num_constraints, Vector3T::Zero());

  assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
  auto initial_cost = evaluate_cost_planar(inputs, index, variables, step_, constrained_steps);
  // std::cout << "initial_cost = " << initial_cost << std::endl;
  auto current_cost = initial_cost;

  assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
  build_hessian_planar(inputs, index, variables, H_, rhs_, systems);

  float lambda = 1e-3f;

  int32_t num_iterations = 0;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>, Eigen::Upper> solver;

  do {
    ++num_iterations;

    // std::cout << "H = " << std::endl << H_ << std::endl;
    // std::cout << "rhs = " << std::endl << rhs_ << std::endl;

    assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
    fill_sparse(inputs, block_coords, H_, HS_);
    HS_.diagonal() += lambda * HS_.diagonal();

    solver.compute(HS_);
    step_ = solver.solve(rhs_);

    constraint_solve(lambda, systems, constrained_steps);

    assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
    auto cost = evaluate_cost_planar(inputs, index, variables, step_, constrained_steps);
    // std::cout << "curr = " << current_cost << ", cost = " << cost << std::endl;

    // float prr1 = step_.dot(H_ * step_);
    float prr1 = step_.transpose() * H_ * step_;
    float prr2 = step_.dot(step_);
    for (size_t i = 0; i < systems.size(); i++) {
      const Matrix3T& hess_ = systems[i].H;
      const Vector3T& s_ = constrained_steps[i];
      prr1 += s_.dot(hess_ * s_);
      prr2 += s_.dot(s_);
    }

    auto predicted_relative_reduction = prr1 / current_cost + 2.f * lambda * prr2 / current_cost;

    if ((predicted_relative_reduction < 1e-6) && (step_.norm() < 1e-6)) {
      current_cost = cost;
      break;
    }

    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    // we have achieved sufficient decrease
    if (rho > 0.25f) {
      // accept step
      {
        for (size_t i = 0; i < num_non_constraints; i++) {
          Isometry3T& var = variables[i];
          const Vector6T& s = step_.segment<6>(6 * i);
          var = update(var, s);
        }

        for (size_t i = 0; i < num_constraints; i++) {
          Isometry3T& var = variables[num_non_constraints + i];
          const Vector3T& s = constrained_steps[i];
          var = update(var, s);
        }
      }

      // our model is good
      if (rho > 0.75f) {
        lambda *= 0.5f;
      }

      current_cost = cost;
      assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
      build_hessian_planar(inputs, index, variables, H_, rhs_, systems);
    } else {
      lambda *= 2.f;
    }
  } while (num_iterations < max_iterations);
  // std::cout << "curr = " << current_cost << ", init = " << initial_cost << std::endl;

  bool status = current_cost < initial_cost || num_iterations == max_iterations;

  if (status) {
    for (size_t i = 0; i < variables.size(); i++) {
      int pose_id = index.var_to_pose[i];
      inputs.poses[pose_id] = variables[i];
    }
  }
  return status;
}

// Precondition: inputs.poses.size() > inputs.constrained_pose_ids.size())
bool PGO::pgo_regular(PGOInput& inputs, int max_iterations) const {
  assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
  const size_t num_poses = inputs.poses.size();

  std::vector<Isometry3T> variables;
  Index index;

  index.pose_to_var.reserve(num_poses);
  index.var_to_pose.reserve(num_poses);
  variables.reserve(num_poses);

  for (size_t i = 0; i < num_poses; i++) {
    int var_id = -1;
    if (!is_pose_constrained(inputs, i)) {
      var_id = static_cast<int>(variables.size());
      variables.push_back(inputs.poses[i]);
      index.var_to_pose.push_back(i);
    }
    index.pose_to_var.push_back(var_id);
  }
  size_t order = variables.size() * 6;

  std::vector<std::pair<int, int>> block_coords;
  calc_blocks(inputs, index, block_coords);

  step_.setZero(order);
  rhs_.resize(order);
  H_.resize(order, order);

  auto initial_cost = evaluate_cost(inputs, index, variables, step_);
  // std::cout << "initial_cost = " << initial_cost << std::endl;
  auto current_cost = initial_cost;

  build_hessian(inputs, index, variables, H_, rhs_);

  float lambda = 1e-3f;

  int32_t num_iterations = 0;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>, Eigen::Upper> solver;

  do {
    ++num_iterations;

    assert(inputs.poses.size() > inputs.constrained_pose_ids.size());
    fill_sparse(inputs, block_coords, H_, HS_);
    HS_.diagonal() += lambda * HS_.diagonal();

    solver.compute(HS_);
    step_ = solver.solve(rhs_);

    auto cost = evaluate_cost(inputs, index, variables, step_);

    auto predicted_relative_reduction = step_.dot(H_ * step_) / current_cost +
                                        2.f * lambda * step_.dot(H_.diagonal().asDiagonal() * step_) / current_cost;

    if ((predicted_relative_reduction < 1e-4) && (step_.norm() < 1e-4)) {
      current_cost = cost;
      break;
    }

    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    // std::cout << "curr = " << current_cost << ", cost = " << cost << " rho = " << rho << std::endl;
    // we have achieved sufficient decrease
    if (rho > 0.25f) {
      // accept step
      {
        for (size_t i = 0; i < variables.size(); i++) {
          Isometry3T& var = variables[i];
          const Vector6T& s = step_.segment<6>(6 * i);
          var = update(var, s);
        }
      }

      // our model is good
      if (rho > 0.75f) {
        lambda *= 0.5f;
      }

      current_cost = cost;
      build_hessian(inputs, index, variables, H_, rhs_);
    } else {
      lambda *= 2.f;
    }
  } while (num_iterations < max_iterations);
  // std::cout << "curr = " << current_cost << ", init = " << initial_cost << std::endl;

  bool status = current_cost < initial_cost || num_iterations == max_iterations;

  if (status) {
    for (size_t i = 0; i < variables.size(); i++) {
      int pose_id = index.var_to_pose[i];
      inputs.poses[pose_id] = variables[i];
    }
  }
  return status;
}

bool PGO::run(PGOInput& inputs, int max_iterations) const {
  if (inputs.constrained_pose_ids.size() >= inputs.poses.size()) {
    return false;  // wrong input
  }

  if (inputs.use_planar_constraint) {
    return pgo_planar(inputs, max_iterations);
  } else {
    return pgo_regular(inputs, max_iterations);
  }
}

}  // namespace cuvslam::math
