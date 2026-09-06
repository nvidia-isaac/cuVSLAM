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

#include "cuvslam/cuvslam2.h"
#include "params/params.h"

/**
 * @file cuvslam_params.h
 *
 * Parameter descriptors for the public configuration structs.
 *
 * Every other settings struct declares its fields in the header that defines it. These two cannot:
 * Odometry::Config and Slam::Config live in cuvslam2.h, which is the public API boundary and one of
 * the three headers shipped to C++ consumers, so it must not include anything internal. The
 * descriptors therefore live here, in a header that is compiled into libcuvslam but never shipped.
 *
 * Keeping public config in the same registry as the solver knobs is what lets a report capture the
 * whole configuration of a run rather than just its expert tuning.
 */

// Spellings accepted for the public odometry enums. These duplicate no logic: the string forms the
// YAML loader used to accept by hand now come from here.
CUVSLAM_PARAM_ENUM_BEGIN(cuvslam::Odometry::MulticameraMode)
CUVSLAM_PARAM_ENUM_VALUE("performance", Performance)
CUVSLAM_PARAM_ENUM_VALUE("precision", Precision)
CUVSLAM_PARAM_ENUM_VALUE("moderate", Moderate)
CUVSLAM_PARAM_ENUM_END()

CUVSLAM_PARAM_ENUM_BEGIN(cuvslam::Odometry::OdometryMode)
CUVSLAM_PARAM_ENUM_VALUE("multicamera", Multicamera)
CUVSLAM_PARAM_ENUM_VALUE("inertial", Inertial)
CUVSLAM_PARAM_ENUM_VALUE("rgbd", RGBD)
CUVSLAM_PARAM_ENUM_VALUE("mono", Mono)
CUVSLAM_PARAM_ENUM_VALUE("multisensor", Multisensor)
CUVSLAM_PARAM_ENUM_END()

// Addressed as `odometry.<name>`. rgbd_settings and multisensor_settings are registered separately
// under `odometry.rgbd` and `odometry.multisensor`.
CUVSLAM_PARAMS_BEGIN(cuvslam::Odometry::Config)
CUVSLAM_PARAM(multicam_mode, "Multicamera primary/secondary topology")
CUVSLAM_PARAM(odometry_mode, "Which sensors the tracker expects and how it fuses them")
CUVSLAM_PARAM(use_gpu, "Track on the GPU")
CUVSLAM_PARAM(async_sba, "Run SBA asynchronously")
CUVSLAM_PARAM(use_motion_model, "Enable the internal pose prediction mechanism")
CUVSLAM_PARAM(use_denoising, "Denoise input images; disable if they are already filtered")
CUVSLAM_PARAM(rectified_stereo_camera, "Assume rectified cameras with principal points on a horizontal line")
CUVSLAM_PARAM(enable_observations_export, "Enable GetLastObservations(); costs time and memory")
CUVSLAM_PARAM(enable_landmarks_export, "Enable GetLastLandmarks(); costs time and memory")
CUVSLAM_PARAM(enable_final_landmarks_export, "Enable GetFinalLandmarks(); implies the other two export flags")
CUVSLAM_PARAM_BOUNDED(max_frame_delta_s, "Warn when the gap between frames exceeds this, seconds", NonNegative())
CUVSLAM_PARAM(debug_dump_directory, "Directory to dump input data to in edex format; empty disables dumping")
CUVSLAM_PARAM(debug_imu_mode, "Integrate IMU only, with no visual tracking")
CUVSLAM_PARAM(min_depth, "Nearest scene depth sampled for L2R initial guesses, meters; <0 auto-detects")
CUVSLAM_PARAM(max_depth, "Farthest scene depth sampled for L2R initial guesses, meters; <0 auto-detects")
CUVSLAM_PARAMS_END()

// Addressed as `odometry.rgbd.<name>`. depth_camera_id is unbounded: -1 means no depth camera.
CUVSLAM_PARAMS_BEGIN(cuvslam::Odometry::RGBDSettings)
CUVSLAM_PARAM(depth_scale_factor, "Divisor converting raw depth values to meters")
CUVSLAM_PARAM(depth_camera_id, "Camera the depth image is pixel-aligned with; -1 for none")
CUVSLAM_PARAM(enable_depth_stereo_tracking, "Allow 2D tracking between the depth-aligned camera and others")
CUVSLAM_PARAMS_END()

// Addressed as `odometry.multisensor.<name>`.
CUVSLAM_PARAMS_BEGIN(cuvslam::Odometry::MultisensorSettings)
CUVSLAM_PARAM(depth_camera_ids, "Camera ids that supply depth images; empty means no depth")
CUVSLAM_PARAM(depth_scale_factor, "Divisor converting raw depth values to meters, applied to every depth camera")
CUVSLAM_PARAM(enable_depth_stereo_tracking, "Allow 2D tracking between depth-aligned cameras and others")
CUVSLAM_PARAMS_END()

// Addressed as `slam.<name>`. max_map_size and throttling_time_ms are unsigned, so a negative value
// is rejected as unparseable rather than wrapping.
CUVSLAM_PARAMS_BEGIN(cuvslam::Slam::Config)
CUVSLAM_PARAM(map_cache_path, "Path to sync the map to on disk (LMDB); empty keeps it in memory only")
CUVSLAM_PARAM(use_gpu, "Run SLAM on the GPU")
CUVSLAM_PARAM(sync_mode, "Run SLAM on the calling thread instead of a worker thread")
CUVSLAM_PARAM(enable_reading_internals, "Allow reading pose graph, loop closures and landmarks from SLAM")
CUVSLAM_PARAM(planar_constraints, "Constrain SLAM poses to a horizontal plane")
CUVSLAM_PARAM(gt_align_mode, "Ground-truth map-building mode; not realtime, no loop closure or global optimization")
CUVSLAM_PARAM_BOUNDED(map_cell_size, "Map cell size, meters; 0 derives it from the camera baseline", NonNegative())
CUVSLAM_PARAM_BOUNDED(max_landmarks_distance, "Farthest landmark to include in the map, meters", NonNegative())
CUVSLAM_PARAM(max_map_size, "Maximum poses in the pose graph; 0 means unlimited")
CUVSLAM_PARAM(throttling_time_ms, "Minimum interval between loop closure events, ms")
CUVSLAM_PARAM(retention_time_ms, "How long odometry delta history is kept for past-timestamp localization, ms")
CUVSLAM_PARAM(delay_warning_queue_size, "Queue length at which cuVSLAM warns that SLAM is falling behind odometry")
CUVSLAM_PARAMS_END()
