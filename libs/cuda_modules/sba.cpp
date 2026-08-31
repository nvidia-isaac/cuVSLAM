
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

#include "cuda_modules/sba.h"

#include "common/vector_3t.h"
#include "cuda_modules/culib_helper.h"

namespace cuvslam::cuda::sba {

GPUModelFunction::GPUModelFunction(int max_observations)
    : max_observations_(max_observations),
      point_jacobians_(max_observations),
      pose_jacobians_(max_observations),
      residuals_(max_observations),
      robustifier_weights_(max_observations) {
  meta_.point_jacobians = point_jacobians_.ptr();
  meta_.pose_jacobians = pose_jacobians_.ptr();
  meta_.residuals = residuals_.ptr();
  meta_.robustifier_weights = robustifier_weights_.ptr();
}

const GPUModelFunctionMeta& GPUModelFunction::meta() const { return meta_; }

bool GPUModelFunction::get(int num_observations,
                           cuvslam::sba::schur_complement_bundler_cpu_internal::ModelFunction& model_function,
                           cudaStream_t s) const {
  if (num_observations > max_observations_) {
    return false;
  }

  point_jacobians_.copy_top_n(GPUCopyDirection::ToCPU, num_observations, s);
  pose_jacobians_.copy_top_n(GPUCopyDirection::ToCPU, num_observations, s);
  residuals_.copy_top_n(GPUCopyDirection::ToCPU, num_observations, s);
  robustifier_weights_.copy_top_n(GPUCopyDirection::ToCPU, num_observations, s);

  cudaStreamSynchronize(s);

  model_function.point_jacobians.resize(num_observations);
  model_function.pose_jacobians.resize(num_observations);
  model_function.residuals.resize(num_observations);
  model_function.robustifier_weights.resize(num_observations);
  for (int i = 0; i < num_observations; i++) {
    const Mat<float, 2, 3>& point_jacobian = point_jacobians_.operator[](i);
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 3; k++) {
        model_function.point_jacobians[i](j, k) = point_jacobian[j][k];
      }
    }

    const Mat<float, 2, 6>& pose_jacobian = pose_jacobians_.operator[](i);
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 6; k++) {
        model_function.pose_jacobians[i](j, k) = pose_jacobian[j][k];
      }
    }

    model_function.residuals[i] = {residuals_.operator[](i).x, residuals_.operator[](i).y};
    model_function.robustifier_weights[i] = robustifier_weights_.operator[](i);
  }
  return true;
}

GPULinearSystem::GPULinearSystem(int max_points, int max_poses)
    : max_points_(max_points),
      max_poses_(max_poses),
      point_block_(max_points),
      point_rhs_(3 * max_points),
      point_pose_block_transposed_(3 * max_points, 6 * max_poses),
      pose_block_(6 * max_poses, 6 * max_poses),
      pose_rhs_(6 * max_poses) {
  meta_.point_block = point_block_.ptr();
  meta_.point_rhs = point_rhs_.ptr();

  meta_.point_pose_block_transposed = point_pose_block_transposed_.ptr();
  meta_.point_pose_block_transposed_pitch = point_pose_block_transposed_.pitch();

  meta_.pose_block = pose_block_.ptr();
  meta_.pose_block_pitch = pose_block_.pitch();
  meta_.pose_rhs = pose_rhs_.ptr();
}

const GPULinearSystemMeta& GPULinearSystem::meta() const { return meta_; }

bool GPULinearSystem::get(int num_points, int num_poses, temporary::FullSystem& cpu_system, cudaStream_t s) const {
  cpu_system.point_block.clear();
  cpu_system.point_block.resize(num_points, Matrix3T::Zero());

  cpu_system.point_rhs.resize(3 * num_points);
  cpu_system.point_rhs.setZero();

  cpu_system.point_pose_block.resize(3 * num_points, 6 * num_poses);
  cpu_system.point_pose_block.setZero();

  cpu_system.pose_block.resize(6 * num_poses, 6 * num_poses);
  cpu_system.pose_block.setZero();

  cpu_system.pose_rhs.resize(6 * num_poses);
  cpu_system.pose_rhs.setZero();

  assert(static_cast<size_t>(3 * num_points) <= point_rhs_.size());
  assert(static_cast<size_t>(6 * num_poses) <= pose_rhs_.size());

  point_block_.copy_top_n(GPUCopyDirection::ToCPU, num_points, s);
  point_rhs_.copy_top_n(GPUCopyDirection::ToCPU, 3 * num_points, s);
  pose_rhs_.copy_top_n(GPUCopyDirection::ToCPU, 6 * num_poses, s);

  Eigen::Matrix<float, -1, -1, Eigen::DontAlign | Eigen::RowMajor> point_pose_block_transposed_temp{
      point_pose_block_transposed_.rows(), point_pose_block_transposed_.cols()};
  point_pose_block_transposed_.copy(GPUCopyDirection::ToCPU, point_pose_block_transposed_temp.data(), s);

  Eigen::MatrixXf pose_block_temp{pose_block_.rows(), pose_block_.cols()};
  pose_block_.copy(GPUCopyDirection::ToCPU, pose_block_temp.data(), s);

  cudaStreamSynchronize(s);

  for (int i = 0; i < num_points; i++) {
    const Matf33& mat = point_block_.operator[](i);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        cpu_system.point_block[i](j, k) = mat[j][k];
      }
    }
  }

  for (int i = 0; i < 3 * num_points; i++) {
    cpu_system.point_rhs(i) = point_rhs_.operator[](i);
  }
  for (int i = 0; i < 6 * num_poses; i++) {
    cpu_system.pose_rhs(i) = pose_rhs_.operator[](i);
  }

  cpu_system.point_pose_block = point_pose_block_transposed_temp.transpose().block(0, 0, 3 * num_points, 6 * num_poses);

  cpu_system.pose_block = pose_block_temp.block(0, 0, 6 * num_poses, 6 * num_poses);
  return true;
}

bool GPULinearSystem::get(int num_points, int num_poses, temporary::ReducedSystem& cpu_system, cudaStream_t s) const {
  cpu_system.inverse_point_block.clear();
  cpu_system.inverse_point_block.resize(num_points, Matrix3T::Zero());

  cpu_system.point_rhs.resize(3 * num_points);
  cpu_system.point_rhs.setZero();

  cpu_system.camera_backsub_block.resize(6 * num_poses, 3 * num_points);
  cpu_system.camera_backsub_block.setZero();

  cpu_system.pose_block.resize(6 * num_poses, 6 * num_poses);
  cpu_system.pose_block.setZero();

  cpu_system.pose_rhs.resize(6 * num_poses);
  cpu_system.pose_rhs.setZero();

  assert(static_cast<size_t>(3 * num_points) <= point_rhs_.size());
  assert(static_cast<size_t>(6 * num_poses) <= pose_rhs_.size());

  point_block_.copy_top_n(GPUCopyDirection::ToCPU, num_points, s);
  point_rhs_.copy_top_n(GPUCopyDirection::ToCPU, 3 * num_points, s);
  pose_rhs_.copy_top_n(GPUCopyDirection::ToCPU, 6 * num_poses, s);

  Eigen::Matrix<float, -1, -1, Eigen::DontAlign | Eigen::RowMajor> point_pose_block_transposed_temp{
      point_pose_block_transposed_.rows(), point_pose_block_transposed_.cols()};
  point_pose_block_transposed_.copy(GPUCopyDirection::ToCPU, point_pose_block_transposed_temp.data(), s);

  Eigen::MatrixXf pose_block_temp{pose_block_.rows(), pose_block_.cols()};
  pose_block_.copy(GPUCopyDirection::ToCPU, pose_block_temp.data(), s);

  cudaStreamSynchronize(s);

  for (int i = 0; i < num_points; i++) {
    const Matf33& mat = point_block_.operator[](i);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        cpu_system.inverse_point_block[i](j, k) = mat[j][k];
      }
    }
  }

  for (int i = 0; i < 3 * num_points; i++) {
    cpu_system.point_rhs(i) = point_rhs_.operator[](i);
  }
  for (int i = 0; i < 6 * num_poses; i++) {
    cpu_system.pose_rhs(i) = pose_rhs_.operator[](i);
  }

  cpu_system.camera_backsub_block = point_pose_block_transposed_temp.block(0, 0, 6 * num_poses, 3 * num_points);

  cpu_system.pose_block = pose_block_temp.block(0, 0, 6 * num_poses, 6 * num_poses);
  return true;
}

bool GPULinearSystem::set(const temporary::FullSystem& cpu_system, cudaStream_t s) {
  int num_points = cpu_system.point_block.size();
  int num_poses = cpu_system.pose_rhs.size() / 6;
  if (max_points_ < num_points || max_poses_ < num_poses) {
    return false;
  }

  for (int i = 0; i < num_points; i++) {
    Matf33& mat = point_block_.operator[](i);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        mat[j][k] = cpu_system.point_block[i](j, k);
      }
    }
  }
  point_block_.copy_top_n(GPUCopyDirection::ToGPU, num_points, s);

  for (int i = 0; i < 3 * num_points; i++) {
    point_rhs_.operator[](i) = cpu_system.point_rhs(i);
  }
  point_rhs_.copy_top_n(GPUCopyDirection::ToGPU, 3 * num_points, s);

  for (int i = 0; i < 6 * num_poses; i++) {
    pose_rhs_.operator[](i) = cpu_system.pose_rhs(i);
  }
  pose_rhs_.copy_top_n(GPUCopyDirection::ToGPU, 6 * num_poses, s);

  Eigen::Matrix<float, -1, -1, Eigen::DontAlign | Eigen::RowMajor> point_pose_block_transposed_temp{6 * max_poses_,
                                                                                                    3 * max_points_};
  point_pose_block_transposed_temp.setZero();

  Eigen::Matrix<float, -1, -1, Eigen::DontAlign | Eigen::RowMajor> pose_block_temp{6 * max_poses_, 6 * max_poses_};
  pose_block_temp.setZero();

  point_pose_block_transposed_temp.block(0, 0, 6 * num_poses, 3 * num_points) = cpu_system.point_pose_block.transpose();

  pose_block_temp.block(0, 0, 6 * num_poses, 6 * num_poses) = cpu_system.pose_block;

  point_pose_block_transposed_.copy(GPUCopyDirection::ToGPU, point_pose_block_transposed_temp.data(), s);
  pose_block_.copy(GPUCopyDirection::ToGPU, pose_block_temp.data(), s);
  return true;
}

bool GPULinearSystem::set(const temporary::ReducedSystem& cpu_system, cudaStream_t s) {
  int num_points = cpu_system.inverse_point_block.size();
  int num_poses = cpu_system.pose_rhs.size() / 6;
  if (max_points_ < num_points || max_poses_ < num_poses) {
    return false;
  }

  for (int i = 0; i < num_points; i++) {
    Matf33& mat = point_block_.operator[](i);
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        mat[j][k] = cpu_system.inverse_point_block[i](j, k);
      }
    }
  }
  point_block_.copy_top_n(GPUCopyDirection::ToGPU, num_points, s);

  for (int i = 0; i < 3 * num_points; i++) {
    point_rhs_.operator[](i) = cpu_system.point_rhs(i);
  }
  point_rhs_.copy_top_n(GPUCopyDirection::ToGPU, 3 * num_points, s);

  for (int i = 0; i < 6 * num_poses; i++) {
    pose_rhs_.operator[](i) = cpu_system.pose_rhs(i);
  }
  pose_rhs_.copy_top_n(GPUCopyDirection::ToGPU, 6 * num_poses, s);

  Eigen::Matrix<float, -1, -1, Eigen::DontAlign | Eigen::RowMajor> point_pose_block_transposed_temp{6 * max_poses_,
                                                                                                    3 * max_points_};
  point_pose_block_transposed_temp.setZero();

  Eigen::MatrixXf pose_block_temp{6 * max_poses_, 6 * max_poses_};
  pose_block_temp.setZero();

  point_pose_block_transposed_temp.block(0, 0, 6 * num_poses, 3 * num_points) = cpu_system.camera_backsub_block;

  pose_block_temp.block(0, 0, 6 * num_poses, 6 * num_poses) = cpu_system.pose_block;

  point_pose_block_transposed_.copy(GPUCopyDirection::ToGPU, point_pose_block_transposed_temp.data(), s);
  pose_block_.copy(GPUCopyDirection::ToGPU, pose_block_temp.data(), s);
  return true;
}

int GPULinearSystem::max_points() const { return max_points_; }

int GPULinearSystem::max_poses() const { return max_poses_; }

GPUParameterUpdate::GPUParameterUpdate(int max_points, int max_poses)
    : max_points_(max_points),
      max_poses_(max_poses),
      poses_(max_poses),
      points_(max_points),
      pose_steps_(max_poses),
      point_steps_(max_points) {
  meta_.pose = poses_.ptr();
  meta_.point = points_.ptr();
  meta_.points_step = point_steps_.ptr();
  meta_.pose_step = pose_steps_.ptr();
}

const GPUParameterUpdateMeta& GPUParameterUpdate::meta() const { return meta_; }

bool GPUParameterUpdate::get(int num_points, int num_poses, temporary::ParameterUpdate& cpu_update,
                             cudaStream_t s) const {
  if (num_points > max_points_ || num_poses > max_poses_) {
    return false;
  }

  poses_.copy_top_n(GPUCopyDirection::ToCPU, num_poses, s);
  pose_steps_.copy_top_n(GPUCopyDirection::ToCPU, num_poses, s);

  points_.copy_top_n(GPUCopyDirection::ToCPU, num_points, s);
  point_steps_.copy_top_n(GPUCopyDirection::ToCPU, num_points, s);

  cpu_update.point.resize(num_points, Vector3T::Zero());
  cpu_update.pose.resize(num_poses, Isometry3T::Identity());

  cpu_update.point_step.resize(3 * num_points);
  cpu_update.pose_step.resize(6 * num_poses);

  cudaStreamSynchronize(s);

  for (int i = 0; i < num_points; i++) {
    const float3& point = points_.operator[](i);
    const float3& step = point_steps_.operator[](i);
    Vector3T vec_3 = {point.x, point.y, point.z};
    Vector3T vector_3_t = {step.x, step.y, step.z};
    cpu_update.point[i] = vec_3;
    cpu_update.point_step.segment<3>(3 * i) = vector_3_t;
  }

  for (int i = 0; i < num_poses; i++) {
    const Pose& pose = poses_.operator[](i);
    const float6& step = pose_steps_.operator[](i);
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        cpu_update.pose[i].matrix()(j, k) = pose[j][k];
      }
    }
    cuvslam::Vector6T vector_6_t;
    vector_6_t << step.x1, step.x2, step.x3, step.x4, step.x5, step.x6;
    cpu_update.pose_step.segment<6>(6 * i) = vector_6_t;
  }
  return true;
}

bool GPUParameterUpdate::set(const temporary::ParameterUpdate& cpu_update, cudaStream_t s) {
  int num_points = cpu_update.point.size();
  int num_poses = cpu_update.pose.size();
  if (max_points_ < num_points || max_poses_ < num_poses) {
    return false;
  }

  for (int i = 0; i < num_points; i++) {
    float3& point = points_.operator[](i);

    point = {cpu_update.point[i].x(), cpu_update.point[i].y(), cpu_update.point[i].z()};

    if (static_cast<size_t>(i) < point_steps_.size()) {
      if (3 * i + 2 < cpu_update.point_step.size()) {
        point_steps_.operator[](i) = {
            cpu_update.point_step[3 * i],
            cpu_update.point_step[3 * i + 1],
            cpu_update.point_step[3 * i + 2],
        };
      } else {
        point_steps_.operator[](i) = {0, 0, 0};
      }
    }
  }
  points_.copy_top_n(GPUCopyDirection::ToGPU, num_points, s);
  point_steps_.copy_top_n(GPUCopyDirection::ToGPU, num_points, s);

  for (int i = 0; i < num_poses; i++) {
    Pose& pose = poses_.operator[](i);
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        pose[j][k] = cpu_update.pose[i].matrix()(j, k);
      }
    }
    if (static_cast<size_t>(i) < pose_steps_.size()) {
      float6& step = pose_steps_.operator[](i);
      if (6 * i + 5 < cpu_update.pose_step.size()) {
        step.x1 = cpu_update.pose_step[6 * i];
        step.x2 = cpu_update.pose_step[6 * i + 1];
        step.x3 = cpu_update.pose_step[6 * i + 2];
        step.x4 = cpu_update.pose_step[6 * i + 3];
        step.x5 = cpu_update.pose_step[6 * i + 4];
        step.x6 = cpu_update.pose_step[6 * i + 5];
      } else {
        step.x1 = 0;
        step.x2 = 0;
        step.x3 = 0;
        step.x4 = 0;
        step.x5 = 0;
        step.x6 = 0;
      }
    }
  }
  poses_.copy_top_n(GPUCopyDirection::ToGPU, num_poses, s);
  pose_steps_.copy_top_n(GPUCopyDirection::ToGPU, num_poses, s);
  return true;
}

GPUBundleAdjustmentProblem::GPUBundleAdjustmentProblem(int max_points, int max_poses, int max_observations)
    : points_(max_points), observations_(max_observations), poses_(max_poses) {
  prepare_meta();
  CUDA_CHECK(cudaMalloc((void**)&rig_, sizeof(Rig)));
  meta_.rig = rig_;
}

void GPUBundleAdjustmentProblem::set_rig(const camera::Rig& rig) {
  Rig cpu_rig;
  meta_.num_cameras = rig.num_cameras;
  cpu_rig.num_cameras = rig.num_cameras;
  for (int cam = 0; cam < rig.num_cameras; cam++) {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        cpu_rig.camera_from_rig[cam][i][j] = rig.camera_from_rig[cam](i, j);
      }
    }
  }

  // automatically invokes cuda synchronize
  CUDA_CHECK(cudaMemcpy((void*)rig_, (void*)&cpu_rig, sizeof(Rig), cudaMemcpyHostToDevice));
}

void GPUBundleAdjustmentProblem::prepare_meta() {
  meta_.points = points_.ptr();
  meta_.rig_from_world = poses_.ptr();
  meta_.observations = observations_.ptr();

  cpu_original_observation_indices_.resize(observations_.size());
}

GPUBundleAdjustmentProblem::~GPUBundleAdjustmentProblem() {
  if (rig_) {
    CUDA_CHECK_NOTHROW(cudaFree(rig_));
  }
}

bool GPUBundleAdjustmentProblem::set(const cuvslam::sba::BundleAdjustmentProblem& problem, cudaStream_t s) {
  // calc factors
  if (problem.observation_xys.size() > observations_.size()) {
    observations_.resize(2 * problem.observation_xys.size());
  }
  if (problem.points.size() > points_.size()) {
    points_.resize(2 * problem.points.size());
  }
  if (problem.rig_from_world.size() > poses_.size()) {
    poses_.resize(2 * problem.rig_from_world.size());
  }

  prepare_meta();

  num_observations_ = problem.observation_xys.size();
  num_points_ = problem.points.size();
  num_poses_ = problem.rig_from_world.size();

  auto cpu_points_ = &points_[0];
  for (int i = 0; i < num_points_; i++) {
    cpu_points_[i].coords = {problem.points[i].x(), problem.points[i].y(), problem.points[i].z()};
    cpu_points_[i].num_observations = 0;
  }

  for (int j = 0; j < num_observations_; j++) {
    const int i = problem.point_ids[j];
    ++cpu_points_[i].num_observations;
  }

  int k = 0;
  for (int i = 0; i < num_points_; i++) {
    cpu_points_[i].first_observation_id = k;
    k += cpu_points_[i].num_observations;
  }

  for (int j = 0; j < num_observations_; j++) {
    const int i = problem.point_ids[j];
    cpu_original_observation_indices_[cpu_points_[i].first_observation_id++] = j;
  }

  for (int i = 0; i < num_points_; i++) {
    cpu_points_[i].first_observation_id -= cpu_points_[i].num_observations;
  }

  points_.copy_top_n(GPUCopyDirection::ToGPU, num_points_, s);

  auto cpu_poses_ = &poses_[0];
  for (int i = 0; i < num_poses_; i++) {
    const Eigen::Matrix<float, 4, 4, Eigen::DontAlign | Eigen::ColMajor>& transform =
        problem.rig_from_world[i].matrix();

    Pose& pose = cpu_poses_[i];

    for (int j = 0; j < 4; j++) {
      for (int m = 0; m < 4; m++) {
        pose[j][m] = transform(j, m);
      }
    }
  }

  poses_.copy_top_n(GPUCopyDirection::ToGPU, num_poses_, s);

  auto cpu_observations_ = &observations_[0];
  for (int jj = 0; jj < num_observations_; jj++) {
    const int j = cpu_original_observation_indices_[jj];

    Observation& obs = cpu_observations_[jj];
    obs.xy = {
        problem.observation_xys[j].x(),
        problem.observation_xys[j].y(),
    };
    obs.info[0][0] = problem.observation_infos[j](0, 0);
    obs.info[0][1] = problem.observation_infos[j](0, 1);
    obs.info[1][0] = problem.observation_infos[j](1, 0);
    obs.info[1][1] = problem.observation_infos[j](1, 1);

    obs.point_id = problem.point_ids[j];
    obs.pose_id = problem.pose_ids[j];
    obs.camera_id = problem.camera_ids[j];
  }

  observations_.copy_top_n(GPUCopyDirection::ToGPU, num_observations_, s);

  return true;
}

bool GPUBundleAdjustmentProblem::get(cuvslam::sba::BundleAdjustmentProblem& problem, cudaStream_t s) const {
  points_.copy_top_n(GPUCopyDirection::ToCPU, num_points_, s);
  poses_.copy_top_n(GPUCopyDirection::ToCPU, num_poses_, s);
  observations_.copy_top_n(GPUCopyDirection::ToCPU, num_observations_, s);

  cudaStreamSynchronize(s);

  for (int i = 0; i < num_points_; i++) {
    const Point& point = points_[i];
    problem.points[i] = {point.coords.x, point.coords.y, point.coords.z};
  }

  for (int jj = 0; jj < num_observations_; jj++) {
    const int j = cpu_original_observation_indices_[jj];

    const Observation& obs = observations_[jj];
    problem.observation_xys[j] = {obs.xy.x, obs.xy.y};
    problem.observation_infos[j] << obs.info[0][0], obs.info[0][1], obs.info[1][0], obs.info[1][1];
  }

  for (int i = 0; i < num_poses_; i++) {
    const Pose& pose = poses_[i];

    Eigen::Matrix<float, 4, 4>& problem_pose = problem.rig_from_world[i].matrix();

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        problem_pose(j, k) = pose[j][k];
      }
    }
  }

  return true;
}

int GPUBundleAdjustmentProblem::num_points() const { return num_points_; }

int GPUBundleAdjustmentProblem::num_poses() const { return num_poses_; }

int GPUBundleAdjustmentProblem::num_observations() const { return num_observations_; }

int GPUBundleAdjustmentProblem::original_observation_index(int j) const { return cpu_original_observation_indices_[j]; }

const GPUBundleAdjustmentProblemMeta& GPUBundleAdjustmentProblem::meta() const {
  meta_.num_points = num_points_;
  meta_.num_observations = num_observations_;
  meta_.num_poses = num_poses_;

  return meta_;
}

GPUCholeskySolver::GPUCholeskySolver(int max_system_order)
    : max_system_order_(max_system_order), workspace_(workspace_size_), factor_(max_system_order_ * max_system_order_) {
  // Nothing has been solved yet, so report failure instead of whatever the pinned buffer came with.
  factorization_status_[0] = -1;

  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
}

GPUCholeskySolver::~GPUCholeskySolver() {
  if (cusolver_handle_) {
    CUSOLVER_CHECK_NOTHROW(cusolverDnDestroy(cusolver_handle_));
  }
  if (cublas_handle_) {
    CUBLAS_CHECK_NOTHROW(cublasDestroy(cublas_handle_));
  }
}

void GPUCholeskySolver::solve(const float* A, size_t A_pitch, const float* b, float* x, int system_order,
                              cudaStream_t s) {
  CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle_, s));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, s));
  if (system_order > max_system_order_) {
    max_system_order_ = 2 * system_order;
    factor_.resize(max_system_order_ * max_system_order_);
  }

  CUDA_CHECK(cudaMemcpy2DAsync(factor_.ptr(), system_order * sizeof(float), A, A_pitch, system_order * sizeof(float),
                               system_order, cudaMemcpyDeviceToDevice, s));

  int new_workspace_size;
  CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, system_order, factor_.ptr(),
                                             system_order, &new_workspace_size));

  if (workspace_size_ < new_workspace_size) {
    workspace_size_ = 2 * new_workspace_size;
    workspace_.resize(workspace_size_);
  }

  CUDA_CHECK(cudaMemsetAsync((void*)factorization_status_.ptr(), 0, sizeof(int), s));

  // cholesky factorization of lower triangular part
  CUSOLVER_CHECK(cusolverDnSpotrf(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, system_order, factor_.ptr(), system_order,
                                  workspace_.ptr(), workspace_size_, factorization_status_.ptr()));

  // Read the factorization status back before the triangular solve below overwrites it with its
  // own. The caller checks it after synchronizing the stream.
  factorization_status_.copy(GPUCopyDirection::ToCPU, s);

  CUDA_CHECK(cudaMemcpyAsync(x, b, sizeof(float) * system_order, cudaMemcpyDeviceToDevice, s));
  CUSOLVER_CHECK(cusolverDnSpotrs(cusolver_handle_, CUBLAS_FILL_MODE_LOWER, system_order, 1, factor_.ptr(),
                                  system_order, x, system_order, factorization_status_.ptr()));
}

bool GPUCholeskySolver::solve_succeeded() const { return factorization_status_[0] == 0; }

GPULevenbergMarquardtStep::GPULevenbergMarquardtStep(int max_points, int max_poses)
    : max_points_(max_points), max_poses_(max_poses), solver_(6 * max_poses) {}

bool GPULevenbergMarquardtStep::compute_update(const GPUParameterUpdateMeta& update,
                                               const GPULinearSystemMeta& reduced_system, int num_points, int num_poses,
                                               GPUArrayPinned<float>& points_poses_update_max,  // must contain 2 floats
                                               cudaStream_t s) {
  int system_order = 6 * num_poses;
  if (max_poses_ < num_poses || max_points_ < num_points) {
    return false;
  }

  // Either count at zero leaves a kernel below with a zero grid or block dimension, which CUDA
  // rejects as an invalid configuration rather than running as a no-op. Say so instead.
  if (num_points <= 0 || num_poses <= 0) {
    return false;
  }

  solver_.solve(reduced_system.pose_block, reduced_system.pose_block_pitch, reduced_system.pose_rhs,
                (float*)update.pose_step, system_order, s);

  CUDA_CHECK(calc_update((float*)update.point, reduced_system.point_pose_block_transposed,
                         reduced_system.point_pose_block_transposed_pitch, reduced_system.point_rhs, update.pose,
                         update.pose_step, num_points, num_poses, s));

  CUDA_CHECK(
      cudaMemcpyAsync(update.points_step, update.point, num_points * sizeof(float3), cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(reduce_abs_max((float*)update.points_step, num_points * 3, points_poses_update_max.ptr(), s));

  CUDA_CHECK(reduce_abs_max((float*)update.pose_step, num_poses * 6, points_poses_update_max.ptr() + 1, s));

  return true;
}

bool GPULevenbergMarquardtStep::apply_update(const GPUBundleAdjustmentProblemMeta& problem,
                                             const GPUParameterUpdateMeta& update, int num_points, int num_poses,
                                             cudaStream_t s) {
  CUDA_CHECK(
      update_parameters(problem.points, update.point, problem.rig_from_world, update.pose, num_points, num_poses, s));
  return true;
}

bool GPULevenbergMarquardtStep::solve_succeeded() const { return solver_.solve_succeeded(); }

bool GPULevenbergMarquardtStep::predict_reduction(float current_cost, float lambda,
                                                  const GPUParameterUpdateMeta& update,
                                                  const GPULinearSystemMeta& full_system, int num_points, int num_poses,
                                                  float* prediction, cudaStream_t s) {
  float* point_hessian_term_ = hessian_and_scaling_terms_.ptr();
  float* point_scaling_term_ = point_hessian_term_ + 1;
  float* pose_scaling_term_ = point_hessian_term_ + 2;

  CUDA_CHECK(cudaMemsetAsync(point_hessian_term_, 0, 3 * sizeof(float), s));

  CUDA_CHECK(v1T_x_M_x_v2((float*)update.points_step, full_system.point_pose_block_transposed,
                          full_system.point_pose_block_transposed_pitch, 6 * num_poses, 3 * num_points,
                          (float*)update.pose_step, point_hessian_term_, true, s));

  {
    CUDA_CHECK(
        cudaMemcpyAsync(pose_hessian_term_.ptr(), point_hessian_term_, sizeof(float), cudaMemcpyDeviceToDevice, s));
    // update.pose_step.dot(system.pose_block * update.pose_step)
    CUDA_CHECK(v1T_x_M_x_v2((float*)update.pose_step, full_system.pose_block, full_system.pose_block_pitch,
                            6 * num_poses, 6 * num_poses, (float*)update.pose_step, pose_hessian_term_.ptr(), false,
                            s));

    CUDA_CHECK(point_term(update.points_step, full_system.point_block, num_points, false, point_hessian_term_, s));
  }

  {
    // float pose_term = update.pose_step.dot(system.pose_block.diagonal().asDiagonal() * update.pose_step);
    CUDA_CHECK(pose_scaling_term((float*)update.pose_step, full_system.pose_block, full_system.pose_block_pitch,
                                 6 * num_poses, pose_scaling_term_, s));
  }

  { CUDA_CHECK(point_term(update.points_step, full_system.point_block, num_points, true, point_scaling_term_, s)); }

  {
    CUDA_CHECK(make_prediction(current_cost, lambda, pose_hessian_term_.ptr(), point_hessian_term_, pose_scaling_term_,
                               point_scaling_term_, prediction, s));
  }

  return true;
}

}  // namespace cuvslam::cuda::sba
