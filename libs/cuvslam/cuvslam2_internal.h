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
#include <optional>
#include <string_view>

/**
 * @file cuvslam2_internal.h
 *
 * The parameters in this file expose low-level solver knobs that are tuned for the default
 * hardware and dataset configurations shipped with cuVSLAM. They are subject to change or
 * removal in any release without notice and are not covered by API stability guarantees.
 *
 * Do NOT adjust these values.
 *
 * Normal users should call Odometry::Track() without passing Internals — the defaults
 * are correct for virtually all use cases.
 */

namespace cuvslam {
namespace internal {

/**
 * @brief Per-frame low-level solver parameters for Odometry::Track() (stateless).
 *
 * All fields carry carefully tuned defaults. Instantiate with Internals{} and
 * override only the fields you understand and have validated experimentally.
 * Each Track() call is independent; changes do not persist to the next call.
 */
struct Internals {
  // ============================================================
  // Feature Selection Settings (from sof::Settings)
  // ============================================================

  /// Number of desired feature tracks. Default: 550
  int32_t num_desired_tracks = 550;

  /// Image border regions to ignore (in pixels). Default: 0
  int32_t border_top = 0;
  int32_t border_bottom = 0;
  int32_t border_left = 0;
  int32_t border_right = 0;

  /// Enable box filter preprocessing for this frame. Default: false
  bool box3_prefilter = false;

  /// Enable RANSAC filtering for this frame. Default: false
  bool ransac_filter = false;

  /// All modes but Mono: new features are detected if the percent of tracks surviving
  /// from the last detection falls below this value (0-100). Default: 48.f
  float feature_survivor_from_last = 48.f;

  // ============================================================
  // Keyframe Settings (from KeyFrameSettings)
  // ============================================================

  /// Keyframe selection threshold: frame becomes keyframe if survivor tracks
  /// from last keyframe falls below this percentage (0-100). Default: 48.f
  float kf_survivor_from_last = 48.f;

  /// Maximum time delta between consecutive keyframes (in seconds). Default: 60
  int64_t kf_max_timedelta_between_kfs_s = 60;

  /// Optional keyframe decision. std::nullopt uses automatic keyframe selection (default),
  /// true forces a keyframe, and false forces a non-keyframe.
  std::optional<bool> kf_override_frame_selection;

  // ============================================================
  // PNP Solver Settings - visual PNP tracking (stereo / multi-camera modes)
  // ============================================================

  /// Levenberg-Marquardt damping factor. Default: 1e-3
  float vo_pnp_lambda = 1e-3f;

  /// Huber robustifier scale for reprojection residuals. Default: 2e-2
  float vo_pnp_huber = 2e-2f;

  /// Maximum LM solver iterations. Default: 13
  int32_t vo_pnp_max_iteration = 13;

  /// Whether to recompute the covariance matrix after a successful solve. Default: true
  bool vo_pnp_recalculate_cov = true;

  /// Whether to filter/sort observations to max_obs_per_camera oldest tracks first. Default: true
  bool vo_pnp_filter_new_observations = true;

  /// Maximum observations fed to the solver per camera. Default: 270
  int32_t vo_pnp_max_obs_per_camera = 270;

  /// Minimum z-depth for a landmark to be used (in camera frame). Default: 0.01
  float vo_pnp_point_z_thresh = 0.01f;

  /// Minimum number of observations required to attempt a solve. Default: 13
  int32_t vo_pnp_min_observations = 13;

  /// Absolute convergence threshold for the PnP solver.
  /// Solve succeeds when: (current_cost < vo_pnp_cost_thresh) || (current_cost < initial_cost).
  /// The OR means either condition alone is sufficient: a low final cost always passes, and
  /// any net improvement over the initial cost also passes regardless of its absolute value.
  /// Because of the OR, setting vo_pnp_cost_thresh very large (e.g. FLT_MAX) makes the
  /// absolute-threshold branch always true, rendering the relative-drop check irrelevant.
  /// Default: 0.6
  float vo_pnp_cost_thresh = 0.6f;

  // ============================================================
  // PNP Solver Settings - stereo fallback in inertial mode
  // Defaults match pnp::PNPSettings::InertialSettings().
  // ============================================================

  /// Levenberg-Marquardt damping factor. Default: 1e-3
  float inertial_stereo_pnp_lambda = 1e-3f;

  /// Huber robustifier scale for reprojection residuals. Default: 0.1
  float inertial_stereo_pnp_huber = 0.1f;

  /// Maximum LM solver iterations. Default: 13
  int32_t inertial_stereo_pnp_max_iteration = 13;

  /// Whether to recompute the covariance matrix after a successful solve. Default: false
  bool inertial_stereo_pnp_recalculate_cov = false;

  /// Whether to filter/sort observations to max_obs_per_camera oldest tracks first. Default: true
  bool inertial_stereo_pnp_filter_new_observations = true;

  /// Maximum observations fed to the solver per camera. Default: 270
  int32_t inertial_stereo_pnp_max_obs_per_camera = 270;

  /// Minimum z-depth for a landmark to be used (in camera frame). Default: 0.01
  float inertial_stereo_pnp_point_z_thresh = 0.01f;

  /// Minimum number of observations required to attempt a solve. Default: 13
  int32_t inertial_stereo_pnp_min_observations = 13;

  /// Absolute convergence threshold for the inertial-mode stereo fallback PnP solver.
  /// Default: 0.6
  float inertial_stereo_pnp_cost_thresh = 0.6f;

  // ============================================================
  // Inertial PNP Solver Settings - IMU-fused tracking path
  // ============================================================

  /// Robustifier scale for inertial PNP reprojection residuals. Default: 0.4
  float imu_pnp_robustifier_scale = 0.4f;

  /// Maximum inertial PNP solver iterations. Default: 20
  int32_t imu_pnp_max_iteration = 20;

  /// Minimum number of observations required to attempt inertial PNP. Default: 25
  int32_t imu_pnp_min_observations = 25;

  // ============================================================
  // ICP Solver Settings - mono-depth (RGB-D) mode only
  // ============================================================

  /// Levenberg-Marquardt damping factor. Default: 1e-2
  float icp_lambda = 1e-2f;

  /// Huber robustifier scale for visual reprojection residuals. Default: 1e-2
  float icp_huber_vis = 1e-2f;

  /// Huber robustifier scale for depth ICP residuals. Default: 5e-2
  float icp_huber_depth = 5e-2f;

  /// Maximum LM solver iterations (used when no depth pyramid, i.e. pure visual). Default: 20
  int32_t icp_max_iteration = 20;

  /// Convergence threshold (cost must not exceed initial cost). Default: 0.6
  float icp_cost_thresh = 0.6f;

  /// Finest pyramid level to process (0 = full resolution / finest). Default: 0
  int32_t icp_min_scale_level = 0;

  /// Coarsest pyramid level to start from (higher = coarser / lower resolution). Default: 4
  /// Processing proceeds from icp_max_scale_level down to icp_min_scale_level (coarse to fine).
  int32_t icp_max_scale_level = 4;

  /// LM iterations per pyramid level when depth is available. Default: 20
  int32_t icp_num_iters_per_scale = 20;

  /// Blending weight between visual (alpha) and depth ICP (1-alpha) residuals. Default: 0.8
  float icp_blending_alpha = 0.8f;
};

// TODO(vikuznetsov): remove when https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165 is fixed
inline Internals GetDefaultInternals() { return Internals{}; }

/**
 * @brief Key/value pair for Odometry::ApplyPersistentInternalParameters.
 * @see Odometry::ApplyPersistentInternalParameters for the list of supported keys.
 */
struct InternalParameter {
  std::string_view key;    ///< Parameter name (e.g. `sba.num_sba_iterations`).
  std::string_view value;  ///< Parameter value as a string.
};

}  // namespace internal
}  // namespace cuvslam
