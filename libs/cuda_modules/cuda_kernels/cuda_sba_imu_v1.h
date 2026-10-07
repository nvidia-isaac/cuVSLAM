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

#include <driver_types.h>
#include <vector_types.h>

#include "cuda_modules/cuda_kernels/cuda_common.h"
#include "cuda_modules/cuda_kernels/cuda_matrix.h"

namespace cuvslam::cuda::sba_imu {

cudaError_t build_reduced_system(const cuvslam::cuda::Matf33* full_system_point_block,
                                 const float* full_system_point_rhs,
                                 const float* full_system_point_pose_block_transposed,
                                 int full_system_point_pose_block_transposed_pitch, const float* full_system_pose_block,
                                 int full_system_pose_block_pitch, const float* full_system_pose_rhs,
                                 float* reduced_system_point_rhs, float* reduced_system_camera_backsub_block_transposed,
                                 int reduced_system_camera_backsub_block_transposed_pitch,
                                 float* reduced_system_pose_block, int reduced_system_pose_block_pitch,
                                 float* reduced_system_pose_rhs, const float* lambda, float threshold, int num_points,
                                 int num_poses, cudaStream_t s);

cudaError_t calc_update(const float* reduced_system_point_rhs, const float* update_pose_step,
                        const float* reduced_system_camera_backsub_block_transposed,
                        int reduced_system_camera_backsub_block_transposed_pitch, float* update_point_step,
                        float* update_point, cuvslam::cuda::Matf33* update_pose_w_from_imu_linear,
                        float* update_pose_other, int num_points, int num_poses, cudaStream_t s);

cudaError_t evaluate_cost(
    const cuvslam::cuda::Matf33* problem_rig_poses_w_from_imu_linear, const float* problem_rig_poses_other,
    const cuvslam::cuda::Matf33* update_pose_w_from_imu_linear, const float* update_pose_other,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JRg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JVg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JVa,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JPg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JPa, const cuvslam::cuda::Matf33* problem_rig_poses_preint_dR,
    const float* problem_rig_poses_preint_gyro_bias_, const float* problem_rig_poses_preint_acc_bias_,
    const float* problem_rig_poses_preint_dV, const float* problem_rig_poses_preint_dP,
    const float* problem_rig_poses_preint_dT_s, const cuvslam::cuda::Matf99* problem_rig_poses_preint_info_matrix_,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_acc_random_walk_accum_info_matrix_,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_gyro_random_walk_accum_info_matrix_,
    const int* problem_point_ids, const int* problem_pose_ids, const int8_t* problem_camera_ids,
    const float* problem_points, const float* update_point,
    const cuvslam::cuda::Matf33* problem_rig_camera_from_rig_linear,
    const float* problem_rig_camera_from_rig_translation, const float* problem_observation_xys,
    const cuvslam::cuda::Matf22* problem_observation_infos, cuvslam::cuda::Matf33* imu_from_w_linear,
    float* imu_from_w_translation, float* cost, int* num_skipped, float* partial_costs, float threshold, int num_poses,
    int num_observations, int num_fixed_key_frames, float prior_gyro, float prior_acc, float3 gravity,
    float imu_penalty, float boundary_imu_penalty, float acc_rw_penalty, float robustifier_scale,
    const cuvslam::cuda::Matf33& calib_left_from_imu_linear,
    const cuvslam::cuda::Vecf3& calib_left_from_imu_translation, cudaStream_t s);

cudaError_t compute_predicted_reduction(cuvslam::cuda::Matf33* full_system_point_blocks, const float* update_point_step,
                                        const float* update_pose_step,
                                        const float* full_system_point_pose_block_transposed,
                                        int full_system_point_pose_block_transposed_pitch,
                                        const float* full_system_pose_block, int full_system_pose_block_pitch,
                                        float* prediction, int* working_update_point_and_pose_step_significant,
                                        int num_points, int num_poses, const float* lambda,
                                        float max_abs_update_epsilon, cudaStream_t s);

cudaError_t update_state(const float* update_point, const cuvslam::cuda::Matf33* update_pose_w_from_imu_linear,
                         const float* update_pose_other, float* problem_points,
                         cuvslam::cuda::Matf33* problem_rig_poses_w_from_imu_linear, float* problem_rig_poses_other,
                         int num_points, int num_poses, int num_fixed_key_frames, cudaStream_t s);

cudaError_t update_model(
    const int* problem_point_ids, const int* problem_pose_ids, const int8_t* problem_camera_ids,
    const float* problem_points, const cuvslam::cuda::Matf33* problem_rig_poses_w_from_imu_linear,
    const float* problem_rig_poses_other, const cuvslam::cuda::Matf33* problem_rig_camera_from_rig_linear,
    const float* problem_rig_camera_from_rig_translation, const float* problem_observation_xys,
    const cuvslam::cuda::Matf22* problem_observation_infos, const cuvslam::cuda::Matf33* problem_rig_poses_preint_JRg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JVg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JVa,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JPg,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_JPa, const cuvslam::cuda::Matf33* problem_rig_poses_preint_dR,
    const float* problem_rig_poses_preint_gyro_bias_, const float* problem_rig_poses_preint_acc_bias_,
    const float* problem_rig_poses_preint_gyro_bias_diff_, const float* problem_rig_poses_preint_dV,
    const float* problem_rig_poses_preint_dP, const float* problem_rig_poses_preint_dT_s,
    float* model_reprojection_residuals, float* model_repr_robustifier_weights,
    cuvslam::cuda::Matf23* model_repr_jacobians_jt, cuvslam::cuda::Matf23* model_repr_jacobians_jr,
    cuvslam::cuda::Matf23* model_repr_jacobians_jp, float* model_inertial_residuals,
    float* model_random_walk_gyro_residuals, float* model_random_walk_acc_residuals,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jr_left, cuvslam::cuda::Matf93* model_inertial_jacobians_jt_left,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jv_left,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jb_acc_left,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jb_gyro_left,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jr_right, cuvslam::cuda::Matf93* model_inertial_jacobians_jt_right,
    cuvslam::cuda::Matf93* model_inertial_jacobians_jv_right, float threshold, float3 gravity, int num_observations,
    int num_poses, int num_fixed_key_frames, float robustifier_scale,
    const cuvslam::cuda::Matf33& calib_left_from_imu_linear,
    const cuvslam::cuda::Vecf3& calib_left_from_imu_translation, cudaStream_t s);

cudaError_t build_full_system(
    const int* problem_point_num_observations, const int* problem_point_start_observation_id,
    const int* problem_point_observation_ids, const int* problem_pose_num_observations,
    const int* problem_pose_start_observation_id, const int* problem_pose_observation_ids,
    const float* model_repr_robustifier_weights, const cuvslam::cuda::Matf22* problem_observation_infos,
    const cuvslam::cuda::Matf23* model_repr_jacobians_jp, const cuvslam::cuda::Matf23* model_repr_jacobians_jr,
    const cuvslam::cuda::Matf23* model_repr_jacobians_jt, const float* model_reprojection_residuals,
    const int* problem_pose_ids, const cuvslam::cuda::Matf99* problem_rig_poses_preint_info_matrix_,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_acc_random_walk_accum_info_matrix_,
    const cuvslam::cuda::Matf33* problem_rig_poses_preint_gyro_random_walk_accum_info_matrix_,
    const float* model_inertial_residuals, const cuvslam::cuda::Matf93* model_inertial_jacobians_jr_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jt_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jv_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jb_gyro_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jb_acc_left,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jr_right,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jt_right,
    const cuvslam::cuda::Matf93* model_inertial_jacobians_jv_right, const float* model_random_walk_gyro_residuals,
    const float* model_random_walk_acc_residuals, const float* problem_rig_poses_other,
    cuvslam::cuda::Matf33* full_system_point_block, float* full_system_point_rhs,
    float* full_system_point_pose_block_transposed, int full_system_point_pose_block_transposed_pitch,
    float* full_system_pose_block, int full_system_pose_block_pitch, float* full_system_pose_rhs, int num_observations,
    int num_points, int num_poses, int num_fixed_key_frames, float imu_penalty, float boundary_imu_penalty,
    float acc_rw_penalty, float prior_gyro, float prior_acc, cudaStream_t s);

cudaError_t init_update(cuvslam::cuda::Matf33* update_pose_w_from_imu_linear, float* update_pose_other,
                        float* update_point, int num_poses_opt, int num_points, cudaStream_t s);

}  // namespace cuvslam::cuda::sba_imu
