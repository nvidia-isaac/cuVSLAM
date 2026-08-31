
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

#include "imu/soft_inertial_pnp.h"

#include "common/imu_calibration.h"
#include "common/log.h"
#include "epipolar/near_plane.h"
#include "math/robust_cost_function.h"
#include "math/twist.h"

#include "imu/positive_matrix.h"

#define ROBUST_COST math::ComputeHuberLoss
#define ROBUST_WEIGHT math::ComputeDHuberLoss

namespace cuvslam::sba_imu {

using Mat93 = Eigen::Matrix<float, 9, 3>;
using Mat39 = Eigen::Matrix<float, 3, 9>;
using Mat23 = Eigen::Matrix<float, 2, 3>;

using Mat30 = Eigen::Matrix<float, 30, 30>;
using Vec30 = Eigen::Matrix<float, 30, 1>;
using Vec15 = Eigen::Matrix<float, 15, 1>;

// marginalize previous frame states and generate new prior for frame
// This is as Schur Complement algorighm
Matrix15T Marginalize(const Mat30& H) {
  // Goal
  // a  | ab | ac       a*  | 0 | ac*
  // ba | b  | bc  -->  0   | 0 | 0
  // ca | cb | c        ca* | 0 | c*

  // Reorder as follows:
  // a  | ab | ac       a  | ac | ab
  // ba | b  | bc  -->  ca | c  | cb
  // ca | cb | c        ba | bc | b

  Mat30 Hn = Mat30::Zero();
  Hn.block<15, 15>(0, 0) = H.block<15, 15>(15, 15);
  Hn.block<15, 15>(0, 15) = H.block<15, 15>(15, 0);
  Hn.block<15, 15>(15, 0) = H.block<15, 15>(0, 15);
  Hn.block<15, 15>(15, 15) = H.block<15, 15>(0, 0);

  // Perform marginalization (Schur complement)
  Eigen::JacobiSVD<Matrix15T> svd(Hn.block<15, 15>(15, 15), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::JacobiSVD<Matrix15T>::SingularValuesType singularValues_inv = svd.singularValues();
  for (int i = 0; i < 15; ++i) {
    if (singularValues_inv(i) > 1e-6) {
      singularValues_inv(i) = 1.0 / singularValues_inv(i);
    } else {
      singularValues_inv(i) = 0;
    }
  }
  Matrix15T invHb = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();

  Matrix15T res = Hn.block<15, 15>(0, 0) - Hn.block<15, 15>(0, 15) * invHb * Hn.block<15, 15>(15, 0);
  return res;
}

void CalcOutliers(const imu::ImuCalibration& imu_calibration, const StereoPnPInput& input, const Pose& pose_right,
                  float thresh, std::vector<bool>& is_obs_outlier) {
  const auto num_observations = static_cast<int>(input.observation_xys.size());

  const auto& calib = imu_calibration;
  const Isometry3T rig_from_w = calib.rig_from_imu() * pose_right.w_from_imu.inverse();

  std::vector<Isometry3T> cam_from_w;
  for (int i = 0; i < input.rig.num_cameras; i++) {
    cam_from_w.push_back(input.rig.camera_from_rig[i] * rig_from_w);
  }

  Vector3T p_c;
  Vector2T r;
  for (int obs = 0; obs < num_observations; ++obs) {
    const auto point_idx = input.point_ids[obs];
    const auto camera_idx = input.camera_ids[obs];
    const auto& p_w = input.points[point_idx];

    p_c = cam_from_w[camera_idx] * p_w;

    if (epipolar::IsDepthInFront(p_c.z())) {
      r = p_c.topRows(2) / p_c.z() - input.observation_xys[obs];
      float err = r.dot(input.observation_infos[obs] * r);
      is_obs_outlier[obs] = err > thresh;
    } else {
      is_obs_outlier[obs] = true;
    }
  }
}

float EvaluateCost(const imu::ImuCalibration& imu_calibration, const StereoPnPInput& input,
                   const std::vector<bool>& is_obs_outlier, const Pose& original_pose_left, const Pose& pose_left,
                   const Pose& pose_right, const Pose& update_left, const Pose& update_right, float imu_info_penalty) {
  float cost = 0;

  const auto num_observations = static_cast<int>(input.observation_xys.size());

  const auto& calib = imu_calibration;
  const Isometry3T& rig_from_imu = calib.rig_from_imu();

  const sba_imu::IMUPreintegration& preint = pose_left.preintegration;

  Pose pu1, pu2;
  {
    pu1.w_from_imu = pose_left.w_from_imu;
    pu1.w_from_imu.translation() += pose_left.w_from_imu.linear() * update_left.w_from_imu.translation();
    pu1.w_from_imu.linear() = pose_left.w_from_imu.linear() * update_left.w_from_imu.linear();
    pu1.w_from_imu.makeAffine();

    pu1.velocity = pose_left.velocity + update_left.velocity;
    pu1.gyro_bias = pose_left.gyro_bias + update_left.gyro_bias;
    pu1.acc_bias = pose_left.acc_bias + update_left.acc_bias;
  }

  {
    pu2.w_from_imu = pose_right.w_from_imu;
    pu2.w_from_imu.translation() += pose_right.w_from_imu.linear() * update_right.w_from_imu.translation();
    pu2.w_from_imu.linear() = pose_right.w_from_imu.linear() * update_right.w_from_imu.linear();
    pu2.w_from_imu.makeAffine();

    pu2.velocity = pose_right.velocity + update_right.velocity;
    pu2.gyro_bias = pose_right.gyro_bias + update_right.gyro_bias;
    pu2.acc_bias = pose_right.acc_bias + update_right.acc_bias;
  }

  Vector3T p_c;
  Vector2T r;
  for (int obs = 0; obs < num_observations; ++obs) {
    if (is_obs_outlier[obs]) {
      continue;
    }
    const auto point_idx = input.point_ids[obs];
    const auto camera_idx = input.camera_ids[obs];
    const auto& p_w = input.points[point_idx];

    p_c = input.rig.camera_from_rig[camera_idx] * rig_from_imu * pu2.w_from_imu.inverse() * p_w;

    if (epipolar::IsDepthInFront(p_c.z())) {
      r = p_c.topRows(2) / p_c.z() - input.observation_xys[obs];
      cost += ROBUST_COST(r.dot(input.observation_infos[obs] * r), input.robustifier_scale);
    }
  }

  Matrix9T info;
  Matrix3T info_gyro_rw, info_acc_rw;

  Vector3T rot_error;
  Vector3T velocity_error_term, trans_error_term;

  Vector9T inertial_error;
  Vector3T random_walk_gyro_error, random_walk_acc_error;

  {
    preint.InfoMatrix(info);
    preint.InfoGyroRWMatrix(info_gyro_rw);
    preint.InfoAccRWMatrix(info_acc_rw);

    info *= imu_info_penalty;
    info_gyro_rw *= imu_info_penalty;
    info_acc_rw *= imu_info_penalty;

    const Matrix3T dR = preint.GetDeltaRotation(pu1.gyro_bias);
    const Vector3T dV = preint.GetDeltaVelocity(pu1.gyro_bias, pu1.acc_bias);
    const Vector3T dP = preint.GetDeltaPosition(pu1.gyro_bias, pu1.acc_bias);
    const float dT = preint.GetDeltaT_s();

    math::Log(rot_error, dR.transpose() * pu1.w_from_imu.linear().transpose() * pu2.w_from_imu.linear());

    const Vector3T& v1 = pu1.velocity;
    const Vector3T& v2 = pu2.velocity;

    const Vector3T& p1 = pu1.w_from_imu.translation();
    const Vector3T& p2 = pu2.w_from_imu.translation();

    velocity_error_term = pu1.w_from_imu.linear().transpose() * (v2 - v1 - input.gravity * dT);

    trans_error_term = pu1.w_from_imu.linear().transpose() * (p2 - p1 - v1 * dT - 0.5 * input.gravity * dT * dT);

    inertial_error.segment<3>(0) = rot_error;
    inertial_error.segment<3>(3) = velocity_error_term - dV;
    inertial_error.segment<3>(6) = trans_error_term - dP;

    random_walk_gyro_error = pu1.gyro_bias - pu2.gyro_bias;
    random_walk_acc_error = pu1.acc_bias - pu2.acc_bias;

    cost += inertial_error.dot(info * inertial_error);
    cost += random_walk_gyro_error.dot(info_gyro_rw * random_walk_gyro_error);
    cost += random_walk_acc_error.dot(info_acc_rw * random_walk_acc_error);
  }

  Matrix3T prior_gyro_info = Matrix3T::Identity() * input.prior_gyro;
  Matrix3T prior_acc_info = Matrix3T::Identity() * input.prior_acc;
  Matrix3T tr_constraint_info = Matrix3T::Identity() * input.translation_constraint;

  Vec15 left_pose_err;
  const Matrix3T& Ro = original_pose_left.w_from_imu.linear();
  const Matrix3T& Rl = pu1.w_from_imu.linear();
  math::Log(rot_error, Ro.transpose() * Rl);
  left_pose_err.segment<3>(0) = rot_error;
  left_pose_err.segment<3>(3) =
      Ro.transpose() * (pu1.w_from_imu.translation() - original_pose_left.w_from_imu.translation());
  left_pose_err.segment<3>(6) = pu1.velocity - original_pose_left.velocity;
  left_pose_err.segment<3>(9) = pu1.gyro_bias - original_pose_left.gyro_bias;
  left_pose_err.segment<3>(12) = pu1.acc_bias - original_pose_left.acc_bias;

  Vector3T tr_constraint = pu1.w_from_imu.translation() - pu2.w_from_imu.translation();

  // priors
  {
    cost += pu1.gyro_bias.dot(prior_gyro_info * pu1.gyro_bias);
    cost += pu2.gyro_bias.dot(prior_gyro_info * pu2.gyro_bias);

    cost += pu1.acc_bias.dot(prior_acc_info * pu1.acc_bias);
    cost += pu2.acc_bias.dot(prior_acc_info * pu2.acc_bias);

    cost += ROBUST_COST(tr_constraint.dot(tr_constraint_info * tr_constraint), input.robustifier_scale_tr);
    cost += ROBUST_COST(left_pose_err.dot(original_pose_left.info * left_pose_err), input.robustifier_scale_pose);
  }

  return cost;
}

void build_hessian(const imu::ImuCalibration& imu_calibration, const StereoPnPInput& problem,
                   const std::vector<bool>& is_obs_outlier, const Pose& original_pose_left, const Pose& pose_left,
                   const Pose& pose_right, float imu_info_penalty, Mat30& H, Vec30& rhs) {
  H.setZero();
  rhs.setZero();

  const auto& calib = imu_calibration;
  const Isometry3T& rig_from_imu = calib.rig_from_imu();

  Isometry3T cam_from_imu[camera::Rig::kMaxCameras];
  for (int i = 0; i < problem.rig.num_cameras; ++i) {
    cam_from_imu[i] = problem.rig.camera_from_rig[i] * rig_from_imu;
  }

  const auto num_observations = static_cast<int32_t>(problem.observation_xys.size());
  Mat23 dproj, repr_JR, repr_Jt;
  for (int obs = 0; obs < num_observations; ++obs) {
    if (is_obs_outlier[obs]) {
      continue;
    }
    const auto point_idx = problem.point_ids[obs];
    const auto camera_idx = problem.camera_ids[obs];
    const auto& p_w = problem.points[point_idx];

    const Isometry3T& w_from_imu = pose_right.w_from_imu;
    Vector3T p_c = problem.rig.camera_from_rig[camera_idx] * rig_from_imu * w_from_imu.inverse() * p_w;

    if (epipolar::IsDepthInFront(p_c.z())) {
      Vector2T prediction = p_c.topRows(2) / p_c.z();
      Vector2T r = prediction - problem.observation_xys[obs];

      float inv_z = 1.f / p_c.z();
      dproj << inv_z, 0.f, -prediction.x() * inv_z, 0.f, inv_z, -prediction.y() * inv_z;

      dproj = dproj * cam_from_imu[camera_idx].linear();

      repr_JR = dproj * SkewSymmetric(w_from_imu.linear().transpose() * (p_w - w_from_imu.translation()));
      repr_Jt = -dproj;

      float weight = ROBUST_WEIGHT(r.dot(problem.observation_infos[obs] * r), problem.robustifier_scale);

      H.block<3, 3>(15, 15) += repr_JR.transpose() * problem.observation_infos[obs] * repr_JR * weight;
      H.block<3, 3>(15, 18) += repr_JR.transpose() * problem.observation_infos[obs] * repr_Jt * weight;
      H.block<3, 3>(18, 15) += repr_Jt.transpose() * problem.observation_infos[obs] * repr_JR * weight;
      H.block<3, 3>(18, 18) += repr_Jt.transpose() * problem.observation_infos[obs] * repr_Jt * weight;

      rhs.segment<3>(15) -= repr_JR.transpose() * problem.observation_infos[obs] * weight * r;
      rhs.segment<3>(18) -= repr_Jt.transpose() * problem.observation_infos[obs] * weight * r;
    }
  }

  const auto& preint = pose_left.preintegration;

  Matrix9T info;
  Matrix3T gyro_rw_info, acc_rw_info;

  preint.InfoMatrix(info);
  preint.InfoGyroRWMatrix(gyro_rw_info);
  preint.InfoAccRWMatrix(acc_rw_info);

  info *= imu_info_penalty;
  gyro_rw_info *= imu_info_penalty;
  acc_rw_info *= imu_info_penalty;

  Vector3T rot_error;
  Vector3T vel_error_term, trans_error_term;
  Vector3T gyro_bias_diff, acc_bias_diff;

  Vector9T inertial_residual;

  Mat93 JR_left, Jt_left, Jv_left, Jbg_left, Jba_left, JR_right, Jt_right, Jv_right;

  Mat39 temp;
  // https://arxiv.org/pdf/1512.02363.pdf
  {
    const Isometry3T& w_from_imu1 = pose_left.w_from_imu;
    const Isometry3T& w_from_imu2 = pose_right.w_from_imu;

    const Matrix3T dR = preint.GetDeltaRotation(pose_left.gyro_bias);
    const Vector3T dV = preint.GetDeltaVelocity(pose_left.gyro_bias, pose_left.acc_bias);
    const Vector3T dP = preint.GetDeltaPosition(pose_left.gyro_bias, pose_left.acc_bias);
    const float dT = preint.GetDeltaT_s();

    math::Log(rot_error, dR.transpose() * w_from_imu1.linear().transpose() * w_from_imu2.linear());

    vel_error_term =
        w_from_imu1.linear().transpose() * (pose_right.velocity - pose_left.velocity - problem.gravity * dT);

    trans_error_term = w_from_imu1.linear().transpose() * (w_from_imu2.translation() - w_from_imu1.translation() -
                                                           pose_left.velocity * dT - 0.5 * problem.gravity * dT * dT);

    inertial_residual.segment<3>(0) = rot_error;
    inertial_residual.segment<3>(3) = vel_error_term - dV;
    inertial_residual.segment<3>(6) = trans_error_term - dP;

    {
      {
        // rot
        JR_left.block<3, 3>(0, 0) =
            -math::twist_right_inverse_jacobian(rot_error) * w_from_imu2.linear().transpose() * w_from_imu1.linear();
        // vel
        JR_left.block<3, 3>(3, 0) = SkewSymmetric(vel_error_term);
        // tr
        JR_left.block<3, 3>(6, 0) = SkewSymmetric(trans_error_term);
      }

      {
        Jt_left.setZero();
        // tr
        Jt_left.block<3, 3>(6, 0) = -Matrix3T::Identity();
      }

      {
        Jv_left.setZero();
        // vel
        Jv_left.block<3, 3>(3, 0) = -w_from_imu1.linear().transpose();
        // tr
        Jv_left.block<3, 3>(6, 0) = -w_from_imu1.linear().transpose() * dT;
      }

      {
        Jba_left.setZero();
        // vel
        Jba_left.block<3, 3>(3, 0) = -preint.JVa;
        // tr
        Jba_left.block<3, 3>(6, 0) = -preint.JPa;
      }

      {
        preint.GetDeltaBias(gyro_bias_diff, acc_bias_diff);
        // rot
        Jbg_left.block<3, 3>(0, 0) = -math::twist_left_inverse_jacobian(rot_error) *
                                     math::twist_right_jacobian(preint.JRg * gyro_bias_diff) * preint.JRg;
        // vel
        Jbg_left.block<3, 3>(3, 0) = -preint.JVg;
        // tr
        Jbg_left.block<3, 3>(6, 0) = -preint.JPg;
      }

      {
        JR_right.setZero();
        // rot
        JR_right.block<3, 3>(0, 0) = math::twist_right_inverse_jacobian(rot_error);
      }

      {
        Jt_right.setZero();
        // tr
        Jt_right.block<3, 3>(6, 0) = w_from_imu1.linear().transpose() * w_from_imu2.linear();
      }

      {
        Jv_right.setZero();
        // vel
        Jv_right.block<3, 3>(3, 0) = w_from_imu1.linear().transpose();
      }
    }
  }

  Matrix3T h_r1r1, h_r1t1, h_r1v1, h_r1bg, h_r1ba, h_r1r2, h_r1t2, h_r1v2, h_t1t1, h_t1v1, h_t1bg, h_t1ba, h_t1r2,
      h_t1t2, h_t1v2, h_v1v1, h_v1bg, h_v1ba, h_v1r2, h_v1t2, h_v1v2, h_bgbg, h_bgba, h_bgr2, h_bgt2, h_bgv2, h_baba,
      h_bar2, h_bat2, h_bav2, h_r2r2, h_r2t2, h_r2v2, h_t2t2, h_t2v2, h_v2v2;

  temp = JR_left.transpose() * info;
  h_r1r1 = temp * JR_left;
  h_r1t1 = temp * Jt_left;
  h_r1v1 = temp * Jv_left;
  h_r1bg = temp * Jbg_left;
  h_r1ba = temp * Jba_left;
  h_r1r2 = temp * JR_right;
  h_r1t2 = temp * Jt_right;
  h_r1v2 = temp * Jv_right;

  temp = Jt_left.transpose() * info;
  h_t1t1 = temp * Jt_left;
  h_t1v1 = temp * Jv_left;
  h_t1bg = temp * Jbg_left;
  h_t1ba = temp * Jba_left;
  h_t1r2 = temp * JR_right;
  h_t1t2 = temp * Jt_right;
  h_t1v2 = temp * Jv_right;

  temp = Jv_left.transpose() * info;
  h_v1v1 = temp * Jv_left;
  h_v1bg = temp * Jbg_left;
  h_v1ba = temp * Jba_left;
  h_v1r2 = temp * JR_right;
  h_v1t2 = temp * Jt_right;
  h_v1v2 = temp * Jv_right;

  temp = Jbg_left.transpose() * info;
  h_bgbg = temp * Jbg_left;
  h_bgba = temp * Jba_left;
  h_bgr2 = temp * JR_right;
  h_bgt2 = temp * Jt_right;
  h_bgv2 = temp * Jv_right;

  temp = Jba_left.transpose() * info;
  h_baba = temp * Jba_left;
  h_bar2 = temp * JR_right;
  h_bat2 = temp * Jt_right;
  h_bav2 = temp * Jv_right;

  temp = JR_right.transpose() * info;
  h_r2r2 = temp * JR_right;
  h_r2t2 = temp * Jt_right;
  h_r2v2 = temp * Jv_right;

  temp = Jt_right.transpose() * info;
  h_t2t2 = temp * Jt_right;
  h_t2v2 = temp * Jv_right;

  h_v2v2 = Jv_right.transpose() * info * Jv_right;

  H.block<3, 3>(0, 0) += h_r1r1;
  H.block<3, 3>(0, 3) += h_r1t1;
  H.block<3, 3>(0, 6) += h_r1v1;
  H.block<3, 3>(0, 9) += h_r1bg;
  H.block<3, 3>(0, 12) += h_r1ba;
  H.block<3, 3>(0, 15) += h_r1r2;
  H.block<3, 3>(0, 18) += h_r1t2;
  H.block<3, 3>(0, 21) += h_r1v2;

  H.block<3, 3>(3, 0) += h_r1t1.transpose();
  H.block<3, 3>(3, 3) += h_t1t1;
  H.block<3, 3>(3, 6) += h_t1v1;
  H.block<3, 3>(3, 9) += h_t1bg;
  H.block<3, 3>(3, 12) += h_t1ba;
  H.block<3, 3>(3, 15) += h_t1r2;
  H.block<3, 3>(3, 18) += h_t1t2;
  H.block<3, 3>(3, 21) += h_t1v2;

  H.block<3, 3>(6, 0) += h_r1v1.transpose();
  H.block<3, 3>(6, 3) += h_t1v1.transpose();
  H.block<3, 3>(6, 6) += h_v1v1;
  H.block<3, 3>(6, 9) += h_v1bg;
  H.block<3, 3>(6, 12) += h_v1ba;
  H.block<3, 3>(6, 15) += h_v1r2;
  H.block<3, 3>(6, 18) += h_v1t2;
  H.block<3, 3>(6, 21) += h_v1v2;

  H.block<3, 3>(9, 0) += h_r1bg.transpose();
  H.block<3, 3>(9, 3) += h_t1bg.transpose();
  H.block<3, 3>(9, 6) += h_v1bg.transpose();
  H.block<3, 3>(9, 9) += h_bgbg;
  H.block<3, 3>(9, 12) += h_bgba;
  H.block<3, 3>(9, 15) += h_bgr2;
  H.block<3, 3>(9, 18) += h_bgt2;
  H.block<3, 3>(9, 21) += h_bgv2;

  H.block<3, 3>(12, 0) += h_r1ba.transpose();
  H.block<3, 3>(12, 3) += h_t1ba.transpose();
  H.block<3, 3>(12, 6) += h_v1ba.transpose();
  H.block<3, 3>(12, 9) += h_bgba.transpose();
  H.block<3, 3>(12, 12) += h_baba;
  H.block<3, 3>(12, 15) += h_bar2;
  H.block<3, 3>(12, 18) += h_bat2;
  H.block<3, 3>(12, 21) += h_bav2;

  H.block<3, 3>(15, 0) += h_r1r2.transpose();
  H.block<3, 3>(15, 3) += h_t1r2.transpose();
  H.block<3, 3>(15, 6) += h_v1r2.transpose();
  H.block<3, 3>(15, 9) += h_bgr2.transpose();
  H.block<3, 3>(15, 12) += h_bar2.transpose();
  H.block<3, 3>(15, 15) += h_r2r2;
  H.block<3, 3>(15, 18) += h_r2t2;
  H.block<3, 3>(15, 21) += h_r2v2;

  H.block<3, 3>(18, 0) += h_r1t2.transpose();
  H.block<3, 3>(18, 3) += h_t1t2.transpose();
  H.block<3, 3>(18, 6) += h_v1t2.transpose();
  H.block<3, 3>(18, 9) += h_bgt2.transpose();
  H.block<3, 3>(18, 12) += h_bat2.transpose();
  H.block<3, 3>(18, 15) += h_r2t2.transpose();
  H.block<3, 3>(18, 18) += h_t2t2;
  H.block<3, 3>(18, 21) += h_t2v2;

  H.block<3, 3>(21, 0) += h_r1v2.transpose();
  H.block<3, 3>(21, 3) += h_t1v2.transpose();
  H.block<3, 3>(21, 6) += h_v1v2.transpose();
  H.block<3, 3>(21, 9) += h_bgv2.transpose();
  H.block<3, 3>(21, 12) += h_bav2.transpose();
  H.block<3, 3>(21, 15) += h_r2v2.transpose();
  H.block<3, 3>(21, 18) += h_t2v2.transpose();
  H.block<3, 3>(21, 21) += h_v2v2;

  inertial_residual = info * inertial_residual;
  rhs.segment<3>(0) -= JR_left.transpose() * inertial_residual;
  rhs.segment<3>(3) -= Jt_left.transpose() * inertial_residual;
  rhs.segment<3>(6) -= Jv_left.transpose() * inertial_residual;
  rhs.segment<3>(9) -= Jbg_left.transpose() * inertial_residual;
  rhs.segment<3>(12) -= Jba_left.transpose() * inertial_residual;
  rhs.segment<3>(15) -= JR_right.transpose() * inertial_residual;
  rhs.segment<3>(18) -= Jt_right.transpose() * inertial_residual;
  rhs.segment<3>(21) -= Jv_right.transpose() * inertial_residual;

  Vector3T gyro_rw_residual = pose_left.gyro_bias - pose_right.gyro_bias;
  Vector3T acc_rw_residual = pose_left.acc_bias - pose_right.acc_bias;

  H.block<3, 3>(9, 9) += gyro_rw_info;
  H.block<3, 3>(12, 12) += acc_rw_info;

  H.block<3, 3>(24, 24) += gyro_rw_info;
  H.block<3, 3>(27, 27) += acc_rw_info;

  H.block<3, 3>(9, 24) -= gyro_rw_info;
  H.block<3, 3>(24, 9) -= gyro_rw_info;

  H.block<3, 3>(12, 27) -= acc_rw_info;
  H.block<3, 3>(27, 12) -= acc_rw_info;

  rhs.segment<3>(9) -= gyro_rw_info * gyro_rw_residual;
  rhs.segment<3>(12) -= acc_rw_info * acc_rw_residual;

  rhs.segment<3>(24) += gyro_rw_info * gyro_rw_residual;
  rhs.segment<3>(27) += acc_rw_info * acc_rw_residual;

  Matrix3T prior_gyro_info = Matrix3T::Identity() * problem.prior_gyro;
  Matrix3T prior_acc_info = Matrix3T::Identity() * problem.prior_acc;
  {
    H.block<3, 3>(9, 9) += prior_gyro_info;
    H.block<3, 3>(24, 24) += prior_gyro_info;

    H.block<3, 3>(12, 12) += prior_acc_info;
    H.block<3, 3>(27, 27) += prior_acc_info;

    rhs.segment<3>(9) -= prior_gyro_info * pose_left.gyro_bias;
    rhs.segment<3>(24) -= prior_gyro_info * pose_right.gyro_bias;

    rhs.segment<3>(12) -= prior_acc_info * pose_left.acc_bias;
    rhs.segment<3>(27) -= prior_acc_info * pose_right.acc_bias;
  }

  Vec15 left_pose_err;

  const Matrix3T& Ro = original_pose_left.w_from_imu.linear();
  const Matrix3T& Rl = pose_left.w_from_imu.linear();
  math::Log(rot_error, Ro.transpose() * Rl);
  left_pose_err.segment<3>(0) = rot_error;
  left_pose_err.segment<3>(3) =
      Ro.transpose() * (pose_left.w_from_imu.translation() - original_pose_left.w_from_imu.translation());
  left_pose_err.segment<3>(6) = pose_left.velocity - original_pose_left.velocity;
  left_pose_err.segment<3>(9) = pose_left.gyro_bias - original_pose_left.gyro_bias;
  left_pose_err.segment<3>(12) = pose_left.acc_bias - original_pose_left.acc_bias;

  float c = left_pose_err.dot(original_pose_left.info * left_pose_err);
  float w = ROBUST_WEIGHT(c, problem.robustifier_scale_pose);

  Matrix3T JR = math::twist_right_inverse_jacobian(rot_error);
  Matrix3T JR_t = JR.transpose();
  Matrix3T JT = Ro.transpose() * Rl;
  Matrix3T JT_t = JT.transpose();

  const Matrix15T& prior_info = original_pose_left.info * w;

  H.block<3, 3>(0, 0) += JR_t * prior_info.block<3, 3>(0, 0) * JR;
  H.block<3, 3>(0, 3) += JR_t * prior_info.block<3, 3>(0, 3) * JT;
  H.block<3, 3>(0, 6) += JR_t * prior_info.block<3, 3>(0, 6);
  H.block<3, 3>(0, 9) += JR_t * prior_info.block<3, 3>(0, 9);
  H.block<3, 3>(0, 12) += JR_t * prior_info.block<3, 3>(0, 12);

  H.block<3, 3>(3, 0) += JT_t * prior_info.block<3, 3>(3, 0) * JR;
  H.block<3, 3>(3, 3) += JT_t * prior_info.block<3, 3>(3, 3) * JT;
  H.block<3, 3>(3, 6) += JT_t * prior_info.block<3, 3>(3, 6);
  H.block<3, 3>(3, 9) += JT_t * prior_info.block<3, 3>(3, 9);
  H.block<3, 3>(3, 12) += JT_t * prior_info.block<3, 3>(3, 12);

  H.block<3, 3>(6, 0) += prior_info.block<3, 3>(6, 0) * JR;
  H.block<3, 3>(6, 3) += prior_info.block<3, 3>(6, 3) * JT;
  H.block<3, 3>(6, 6) += prior_info.block<3, 3>(6, 6);
  H.block<3, 3>(6, 9) += prior_info.block<3, 3>(6, 9);
  H.block<3, 3>(6, 12) += prior_info.block<3, 3>(6, 12);

  H.block<3, 3>(9, 0) += prior_info.block<3, 3>(9, 0) * JR;
  H.block<3, 3>(9, 3) += prior_info.block<3, 3>(9, 3) * JT;
  H.block<3, 3>(9, 6) += prior_info.block<3, 3>(9, 6);
  H.block<3, 3>(9, 9) += prior_info.block<3, 3>(9, 9);
  H.block<3, 3>(9, 12) += prior_info.block<3, 3>(9, 12);

  H.block<3, 3>(12, 0) += prior_info.block<3, 3>(12, 0) * JR;
  H.block<3, 3>(12, 3) += prior_info.block<3, 3>(12, 3) * JT;
  H.block<3, 3>(12, 6) += prior_info.block<3, 3>(12, 6);
  H.block<3, 3>(12, 9) += prior_info.block<3, 3>(12, 9);
  H.block<3, 3>(12, 12) += prior_info.block<3, 3>(12, 12);

  left_pose_err = prior_info * left_pose_err;

  rhs.segment<3>(0) -= JR_t * left_pose_err.segment<3>(0);
  rhs.segment<3>(3) -= JT_t * left_pose_err.segment<3>(3);
  rhs.segment<3>(6) -= left_pose_err.segment<3>(6);
  rhs.segment<3>(9) -= left_pose_err.segment<3>(9);
  rhs.segment<3>(12) -= left_pose_err.segment<3>(12);

  {
    Vector3T tr_constraint = pose_left.w_from_imu.translation() - pose_right.w_from_imu.translation();
    Matrix3T Jt1 = pose_left.w_from_imu.linear();
    Matrix3T Jt2 = -pose_right.w_from_imu.linear();

    Matrix3T pose_constr_info = Matrix3T::Identity() * problem.translation_constraint;

    float w = ROBUST_WEIGHT(tr_constraint.dot(pose_constr_info * tr_constraint), problem.robustifier_scale_tr);

    h_t1t1 = Jt1.transpose() * pose_constr_info * Jt1;
    h_t1t2 = Jt1.transpose() * pose_constr_info * Jt2;
    h_t2t2 = Jt2.transpose() * pose_constr_info * Jt2;

    H.block<3, 3>(3, 3) += h_t1t1 * w;
    H.block<3, 3>(3, 18) += h_t1t2 * w;
    H.block<3, 3>(18, 3) += h_t1t2.transpose() * w;
    H.block<3, 3>(18, 18) += h_t2t2 * w;

    rhs.segment<3>(3) -= Jt1.transpose() * pose_constr_info * w * tr_constraint;
    rhs.segment<3>(18) -= Jt2.transpose() * pose_constr_info * w * tr_constraint;
  }
}

void CalcUpdate(const Vec30& step, Pose& left_update, Pose& right_update) {
  Matrix3T dR;
  math::Exp(dR, step.segment<3>(0));

  left_update.w_from_imu.linear() = dR;
  left_update.w_from_imu.translation() = step.segment<3>(3);
  left_update.velocity = step.segment<3>(6);
  left_update.gyro_bias = step.segment<3>(9);
  left_update.acc_bias = step.segment<3>(12);

  math::Exp(dR, step.segment<3>(15));
  right_update.w_from_imu.linear() = dR;
  right_update.w_from_imu.translation() = step.segment<3>(18);
  right_update.velocity = step.segment<3>(21);
  right_update.gyro_bias = step.segment<3>(24);
  right_update.acc_bias = step.segment<3>(27);
}

// OUT: pose, prev_pose - updated poses if success, otherwise it's guaranteed to be unchanged
bool solve_inertial_pnp(const imu::ImuCalibration& imu_calibration, const StereoPnPInput& problem,
                        const std::vector<bool>& is_obs_outlier,
                        Pose& prev_pose,  // non-const because of velocities updates
                        Pose& pose, float imu_info_penalty) {
  Mat30 hessian;
  hessian.setZero();

  Vec30 negative_gradient;
  negative_gradient.setZero();

  const Pose original_prev_pose = prev_pose;  // TODO: copy std::vector<imu::ImuMeasurement> inside
  const Pose original_pose = pose;            // TODO: copy std::vector<imu::ImuMeasurement> inside

  Pose left_update, right_update;
  CalcUpdate(negative_gradient, left_update, right_update);

  auto initial_cost = EvaluateCost(imu_calibration, problem, is_obs_outlier, original_prev_pose, prev_pose, pose,
                                   left_update, right_update, imu_info_penalty);

  TraceMessage("PnP: initial_cost=%.2f", initial_cost);

  if (initial_cost < 10.f) {
    // Nothing to minimize. We have a "pendulum" effect on still frames otherwise.
    prev_pose = original_prev_pose;
    pose = original_pose;
    return true;
  }

  auto current_cost = initial_cost;

  build_hessian(imu_calibration, problem, is_obs_outlier, original_prev_pose, prev_pose, pose, imu_info_penalty,
                hessian, negative_gradient);

  // Freeze bias DOFs: zero off-diagonal coupling and set large diagonal to lock them
  if (problem.freeze_bias) {
    // Bias indices: left gyro[9-11], left acc[12-14], right gyro[24-26], right acc[27-29]
    const int bias_indices[] = {9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29};
    for (int idx : bias_indices) {
      hessian.row(idx).setZero();
      hessian.col(idx).setZero();
      hessian(idx, idx) = 1e10f;
      negative_gradient(idx) = 0.f;
    }
  }

  Vec30 scaling = hessian.diagonal();

  float lambda = 1.f;

  int32_t num_iterations = 0;

  Vec30 step;

  const int max_iterations = problem.max_iterations;
  do {
    ++num_iterations;
    Mat30 augmented_system = hessian + (lambda * scaling).asDiagonal().toDenseMatrix();

    auto decomposition = augmented_system.ldlt();
    if (!decomposition.isPositive()) {
      lambda = (hessian - hessian.diagonal().asDiagonal().toDenseMatrix()).colwise().sum().maxCoeff();
      continue;
    }

    step = decomposition.solve(negative_gradient);

    CalcUpdate(step, left_update, right_update);
    auto cost = EvaluateCost(imu_calibration, problem, is_obs_outlier, original_prev_pose, prev_pose, pose, left_update,
                             right_update, imu_info_penalty);

    // std::cout << "curr = " << current_cost << ", cost = " << cost << std::endl;

    if (cost == std::numeric_limits<float>::infinity()) {
      prev_pose = original_prev_pose;
      pose = original_pose;
      return false;
    }

    auto predicted_relative_reduction =
        step.dot(hessian * step) / current_cost + 2.f * lambda * step.dot(step) / current_cost;

    if ((predicted_relative_reduction < 1e-10) && (step.norm() < 1e-10)) {
      current_cost = cost;
      break;
    }

    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    // we have achieved sufficient decrease
    if (rho > 0.25f) {
      // accept step
      {
        prev_pose.w_from_imu.translation() += prev_pose.w_from_imu.linear() * left_update.w_from_imu.translation();
        prev_pose.w_from_imu.linear() = prev_pose.w_from_imu.linear() * left_update.w_from_imu.linear();
        prev_pose.w_from_imu.makeAffine();

        prev_pose.velocity += left_update.velocity;
        prev_pose.gyro_bias += left_update.gyro_bias;
        prev_pose.acc_bias += left_update.acc_bias;

        pose.w_from_imu.translation() += pose.w_from_imu.linear() * right_update.w_from_imu.translation();
        pose.w_from_imu.linear() = pose.w_from_imu.linear() * right_update.w_from_imu.linear();
        pose.w_from_imu.makeAffine();

        pose.velocity += right_update.velocity;
        pose.gyro_bias += right_update.gyro_bias;
        pose.acc_bias += right_update.acc_bias;
      }

      // our model is good
      if (rho > 0.75f) {
        lambda *= 0.2f;
      }

      current_cost = cost;
      build_hessian(imu_calibration, problem, is_obs_outlier, original_prev_pose, prev_pose, pose, imu_info_penalty,
                    hessian, negative_gradient);
      if (problem.freeze_bias) {
        const int bias_indices[] = {9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29};
        for (int idx : bias_indices) {
          hessian.row(idx).setZero();
          hessian.col(idx).setZero();
          hessian(idx, idx) = 1e10f;
          negative_gradient(idx) = 0.f;
        }
      }
      scaling = scaling.cwiseMax(hessian.diagonal());

      pose.info = Marginalize(hessian);

    } else {
      lambda *= 2.f;
    }
  } while (num_iterations < max_iterations);
  auto dt_prev = (prev_pose.w_from_imu.translation() - original_prev_pose.w_from_imu.translation()).eval();
  auto dt_curr = (pose.w_from_imu.translation() - original_pose.w_from_imu.translation()).eval();
  TraceMessage("PnP: iters=%d initial=%.2f final=%.2f dt_prev=[%.4f,%.4f,%.4f] dt_curr=[%.4f,%.4f,%.4f]",
               num_iterations, initial_cost, current_cost, dt_prev.x(), dt_prev.y(), dt_prev.z(), dt_curr.x(),
               dt_curr.y(), dt_curr.z());
  const bool result = current_cost < initial_cost;
  if (!result) {
    prev_pose = original_prev_pose;
    pose = original_pose;
  }
  return result;
}

// OUT: pose, prev_pose - updated poses if success, otherwise it's guaranteed to be unchanged
bool SoftInertialPnP(const imu::ImuCalibration& imu_calibration, const StereoPnPInput& problem,
                     Pose& prev_pose,  // non-const because of velocities updates
                     Pose& pose, float imu_info_penalty) {
  std::vector<bool> is_outlier;
  is_outlier.resize(problem.observation_xys.size(), false);
  if (!MakeMatrixPositive(prev_pose.info)) {
    return false;
  }
  return solve_inertial_pnp(imu_calibration, problem, is_outlier, prev_pose, pose, imu_info_penalty);
}

bool SoftInertialPnPWithOutliers(const imu::ImuCalibration& imu_calibration, const StereoPnPInput& problem,
                                 Pose& prev_pose, Pose& pose, float imu_info_penalty) {
  std::vector<bool> is_outlier;
  is_outlier.resize(problem.observation_xys.size(), false);
  if (!MakeMatrixPositive(prev_pose.info)) {
    return false;
  }
  bool out = false;
  for (float thresh : problem.outlier_thresh) {
    out = solve_inertial_pnp(imu_calibration, problem, is_outlier, prev_pose, pose, imu_info_penalty);
    CalcOutliers(imu_calibration, problem, pose, thresh, is_outlier);

    // size_t a = std::accumulate(is_outlier.begin(), is_outlier.end(), 0);
    // std::cout << "num outliers ratio = " << static_cast<float>(a) / is_outlier.size() << std::endl;
  }
  // std::cout << "-------------------------" << std::endl;
  return out;
}

}  // namespace cuvslam::sba_imu
