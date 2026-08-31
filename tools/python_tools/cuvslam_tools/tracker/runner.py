#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""Single-sequence tracking pipeline used by the cuvslam_tracker CLI.

Example usage:
    pip install rerun-sdk
    cuvslam_tracker \
        --odometry_mode=mono \
        --dataset=/path/to/video \
        --config_path=/path/to/config \
        --visualize_rerun

Note: If you don't set config_path, it will try to find stereo.edex in the same directory as the video.
"""

import argparse
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Any, Optional
import numpy as np
import cuvslam as vslam
from cuvslam_tools.tracker import conversions as conv
from cuvslam_tools.tracker.edex_reader import EdexReader
from cuvslam_tools.tracker.filters import BlackoutFilter
from cuvslam_tools.tracker.kitti_benchmark import export_kitti_benchmark_artifacts
from cuvslam_tools.tracker.video_reader import VideoReader
from cuvslam_tools.tracker.visualizer import RerunVisualizer, plot_trajectory
from cuvslam_tools.tracker.metrics import calculate_sequence_errors


def get_fps(tracking_time, n_frames):
    """Return frames per second, or -1 when tracking time is not positive."""
    if tracking_time > 0:
        return n_frames / tracking_time
    return -1


@dataclass
class Stat:
    """Summary statistics and report paths from one tracking run."""

    sequence_title: str = ""
    n_frames: int = 0
    tracking_time: float = 0
    average_fps: float = -1
    bird_view_with_errors_path: str = ""
    gt_av_translation_error: float = 0
    gt_av_rotation_error: float = 0
    gt_n_error_segments: int = 0
    gt_simple_error: float = 0
    num_tracking_losts: int = 0
    odometry_mode: str = ""
    # per-instance list of dicts {length, t_pct, r_deg_per_m}, populated by
    # metrics.calculate_sequence_errors in the segment branch.
    seg_err_points: list = field(default_factory=list)


class TrackerResults:
    """Store tracking outputs for downstream reporting and export."""

    def __init__(self):
        """Initialize empty tracking result containers."""
        self.rig: Optional[vslam.Rig] = None
        self.frame_metadata: Dict[int, Dict[str, Any]] = {}
        self.world_from_rig: Dict[int, vslam.Pose] = {}
        self.final_landmarks: Dict[int, vslam.Landmark] = {}
        self.tracks2D: Dict[int, List[vslam.Observation]] = {}
        self.loop_closures: Dict[int, vslam.Pose] = {}
        self.stat: Stat = Stat()


class Tracker:
    """Thin stateful wrapper around the cuVSLAM Python Tracker binding."""

    def __init__(self, rig: vslam.Rig, args: argparse.Namespace):
        """Configure cuVSLAM tracker, optional SLAM, exports, and visualization."""
        self.visualizer = None

        # Configure tracker with args if they're in dict format
        self.odom_cfg = vslam.Odometry.Config()
        self.configure_tracker(args)

        # Configure RGBD settings if in RGBD mode
        rgbd_settings = self._initialize_rgbd_settings(args)
        if rgbd_settings:
            self.odom_cfg.rgbd_settings = rgbd_settings

        # Configure SLAM if needed
        self.slam_cfg = None
        if getattr(args, 'use_slam', False):
            self.odom_cfg.enable_observations_export = True
            self.odom_cfg.enable_landmarks_export = True
            self.slam_cfg = vslam.Slam.Config()
            self.slam_cfg.use_gpu = self.odom_cfg.use_gpu
            self.slam_cfg.sync_mode = args.sync_slam
            if args.visualize_rerun:
                self.slam_cfg.enable_reading_internals = True

        if args.print_config:
            print(
                f"cuVSLAM version: {vslam.get_version()}\n"
                f"Odometry config:\n{conv.to_str(self.odom_cfg)}"
            )
            if self.slam_cfg:
                print(f"SLAM config:\n{conv.to_str(self.slam_cfg)}")

        self.stat = Stat()
        self.stat.odometry_mode = str(self.odom_cfg.odometry_mode)
        self.tracker = vslam.Tracker(rig, self.odom_cfg, self.slam_cfg)

        self.frame_id_from_ts = {}
        self.world_from_rig = {}
        self.final_landmarks = {}
        self.tracks2D = {}
        self.landmarks = {}
        self.frame_metadata = {}  # Store loop and direction info for each frame
        self.loop_closures = {}
        self.num_loops = args.num_loops
        self.processed_frame_count = 0

        if args.visualize_rerun:
            self.visualizer = RerunVisualizer()
            if (not self.odom_cfg.enable_observations_export and
                not self.odom_cfg.enable_landmarks_export and
                not self.odom_cfg.enable_final_landmarks_export ):
                print("Exporting landmarks or observations is disabled, skipping visualization in Rerun")

    def configure_tracker(self, args: argparse.Namespace) -> None:
        """Configure tracker with default and user settings"""

        # Configure tracker with safe argument checking
        config_attrs = [
            'odometry_mode', 'multicam_mode', 'use_gpu', 'async_sba',
            'use_motion_model', 'use_denoising', 'rectified_stereo_camera',
            'enable_observations_export', 'enable_landmarks_export',
            'enable_final_landmarks_export', 'max_frame_delta_s',
            'debug_dump_directory', 'debug_imu_mode'
        ]

        for attr in config_attrs:
            if hasattr(args, attr):
                setattr(self.odom_cfg, attr, getattr(args, attr))

    def _initialize_rgbd_settings(self, args: argparse.Namespace) -> Optional[vslam.Odometry.RGBDSettings]:
        """Initialize RGBD settings based on args and default values.

        Args:
            args: Command-line arguments containing RGBD configuration

        Returns:
            Odometry.RGBDSettings if in RGBD mode, None otherwise

        Raises:
            ValueError: If depth_camera_id is not provided when in RGBD mode
        """
        # Only initialize RGBD settings if in RGBD mode
        if not hasattr(args, 'odometry_mode') or args.odometry_mode != vslam.Odometry.OdometryMode.RGBD:
            return None

        # Create RGBD settings
        rgbd_settings = vslam.Odometry.RGBDSettings()

        # depth_camera_id is REQUIRED - must be provided in stereo.edex file
        depth_camera_id = getattr(args, 'depth_camera_id', None)
        if depth_camera_id is None:
            raise ValueError(
                "RGBD mode is enabled but 'depth_camera_id' is not provided. "
                "This parameter must be specified in the stereo.edex configuration file."
            )
        rgbd_settings.depth_camera_id = depth_camera_id

        # Set depth_scale_factor (default: 1.0)
        rgbd_settings.depth_scale_factor = getattr(args, 'depth_scale_factor', 1.0)

        # Set enable_depth_stereo_tracking (default: False)
        rgbd_settings.enable_depth_stereo_tracking = getattr(args, 'enable_depth_stereo_tracking', False)

        if args.print_config:
            print(f"RGBD settings initialized: depth_camera_id={rgbd_settings.depth_camera_id}, "
                  f"depth_scale_factor={rgbd_settings.depth_scale_factor}, "
                  f"enable_depth_stereo_tracking={rgbd_settings.enable_depth_stereo_tracking}")

        return rgbd_settings

    def process_images(self, frame_id: int, timestamps: Sequence[int],
                       images: Sequence, masks: Sequence,
                       depths: Optional[Sequence] = None):
        """Track one synchronized image frame and store resulting poses and exports."""
        timestamp = max(timestamps)
        self.processed_frame_count += 1
        self.start_time = time.perf_counter()
        odom_pose, slam_pose = self.tracker.track(timestamp, images, masks, depths)
        odom_world_from_rig = odom_pose.world_from_rig.pose if odom_pose.world_from_rig else None
        if odom_world_from_rig is None:
            self.stat.num_tracking_losts += 1

        # Use SLAM pose if available, otherwise use odometry pose.
        if slam_pose is not None:
            self.world_from_rig[frame_id] = slam_pose
        elif odom_world_from_rig is not None:
            self.world_from_rig[frame_id] = odom_world_from_rig
        elif self.world_from_rig:
            previous_frame_id = max(self.world_from_rig)
            self.world_from_rig[frame_id] = self.world_from_rig[previous_frame_id]
            warnings.warn(
                f"Tracking returned no pose for frame {frame_id}; "
                f"reusing pose from frame {previous_frame_id}.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            warnings.warn(
                f"Tracking returned no pose for frame {frame_id}; skipping pose export.",
                RuntimeWarning, stacklevel=2,
            )

        self.end_time = time.perf_counter()
        self.stat.tracking_time += self.end_time - self.start_time

        self.frame_id_from_ts[timestamp] = frame_id
        observations_0 = []
        landmarks = []
        if self.odom_cfg.enable_observations_export:
            # Get last observations for the main camera
            observations_0 = self.tracker.odometry.get_last_observations(0)
            self.tracks2D[frame_id] = observations_0
        if self.odom_cfg.enable_landmarks_export:
            # Get last landmarks for the main camera
            landmarks = self.tracker.odometry.get_last_landmarks()
            self.landmarks[frame_id] = landmarks

        # Get loop closures if SLAM is enabled
        if self.slam_cfg:
            last_loop_closures = self.tracker.slam.get_loop_closure_poses()
            if last_loop_closures:
                for lc in last_loop_closures:
                    self.loop_closures[lc.timestamp_ns] = lc.pose

        if self.visualizer and (odom_world_from_rig is not None or slam_pose is not None):
            gravity = None
            if self.odom_cfg.odometry_mode == vslam.Odometry.OdometryMode.Inertial:
                # Gravity estimation requires collecting sufficient number of keyframes
                # with motion diversity
                gravity_raw = self.tracker.odometry.get_last_gravity()
                gravity = np.array(gravity_raw) if gravity_raw is not None else None
            if self.odom_cfg.enable_final_landmarks_export:
                self.final_landmarks = self.tracker.odometry.get_final_landmarks()
            # SLAM data
            pose_graph = None
            map_landmarks = None
            lc_landmarks = None
            if self.slam_cfg:
                slam = self.tracker.slam
                pose_graph = slam.get_pose_graph()
                map_landmarks = slam.get_landmarks(vslam.Slam.DataLayer.Map)
                lc_landmarks = slam.get_landmarks(vslam.Slam.DataLayer.LoopClosure)
            self.visualizer.visualize_frame(
                frame_id=frame_id,
                images=images,
                odom_pose=odom_world_from_rig,
                slam_pose=slam_pose,
                observations_0=observations_0,
                last_landmarks=landmarks,
                loop_closures=self.loop_closures,
                final_landmarks=self.final_landmarks,
                pose_graph=pose_graph,
                map_landmarks=map_landmarks,
                lc_landmarks=lc_landmarks,
                timestamp=timestamp,
                gravity=gravity,
            )

    def process_imu(self, timestamp: int, linear_accelerations: Sequence[float],
                    angular_velocities: Sequence[float]):
        """Register one IMU sample when inertial odometry is active."""
        if self.odom_cfg.odometry_mode == vslam.Odometry.OdometryMode.Inertial:
            imu_measurement = vslam.ImuMeasurement()
            imu_measurement.timestamp_ns = timestamp
            imu_measurement.linear_accelerations = linear_accelerations
            imu_measurement.angular_velocities = angular_velocities
            self.tracker.register_imu_measurement(0, imu_measurement)

    def get_camera_pose(self, frame_id: int):
        """Return the pose stored for a frame id, if any."""
        return self.world_from_rig.get(frame_id, None)

    def set_frame_metadata(self, frame_id: int, metadata: Dict[str, Any]):
        """Store frame metadata emitted by the dataset reader."""
        self.frame_metadata[frame_id] = metadata

    def run_tracking_and_measure_performance(self, dataset, tracker_results: TrackerResults,
                                              processor=None):
        """Replay a dataset through a processor and finalize tracker results."""
        dataset.replay(processor if processor is not None else self)

        # if slam is enabled, overwrite all slam poses in the end after LCs and PGOs
        if self.slam_cfg:
            slam_poses = self.tracker.slam.get_all_slam_poses()
            if slam_poses:
                for pose in slam_poses:
                    frame_id = self.frame_id_from_ts[pose.timestamp_ns]
                    self.world_from_rig[frame_id] = pose.pose

        self.stat.n_frames = self.processed_frame_count
        self.stat.average_fps = get_fps(self.stat.tracking_time, self.stat.n_frames)

        if self.odom_cfg.enable_final_landmarks_export:
            self.final_landmarks = self.tracker.odometry.get_final_landmarks()
        tracker_results.frame_metadata = self.frame_metadata
        tracker_results.world_from_rig = self.world_from_rig
        tracker_results.loop_closures = self.loop_closures
        tracker_results.final_landmarks = self.final_landmarks
        tracker_results.tracks2D = self.tracks2D
        tracker_results.stat = self.stat


def save_result_to_edex(world_from_rig: Dict[int, vslam.Pose],
                        final_landmarks: Dict[int, vslam.Landmark],
                        tracks2D: Dict[int, List[vslam.Observation]],
                        output_data_file: str):
    """Save poses, landmarks, and 2D tracks to a numpy data file."""

    # TODO: add get_internal_rig in python API to get camera intrinsics and save yaml output config
    output_data = {
        'camera_poses': {},
        'landmarks_3d': {},
        'tracks_2d': {}
    }

    # Store camera poses
    for frame_id, pose in world_from_rig.items():
        output_data['camera_poses'][frame_id] = {
            'rotation': pose.rotation,
            'translation': pose.translation
        }

    # Store 3D landmarks
    for landmark_id, landmark in final_landmarks.items():
        output_data['landmarks_3d'][landmark_id] = landmark

    # Store 2D tracks
    for frame_id, observations in tracks2D.items():
        output_data['tracks_2d'][frame_id] = [
            {
                'id': obs.id,
                'u': obs.u,
                'v': obs.v,
                'camera_index': obs.camera_index
            }
            for obs in observations
        ]

    # Save all data to a single .npy file
    np.save(output_data_file, output_data)


def _warn_unsupported_config_fields(config: Dict[str, Any]) -> None:
    """Warn about legacy reporter config fields that the Python tool ignores."""
    if config.get("write_cache"):
        warnings.warn(
            "Ignoring reporter config field write_cache=true; EDEX result cache export is not supported.",
            stacklevel=2,
        )
    if config.get("use_icp_scaling"):
        warnings.warn(
            "Ignoring reporter config field use_icp_scaling=true; ICP scale correction is not supported.",
            stacklevel=2,
        )
    if config.get("use_cuda"):
        warnings.warn(
            "Ignoring reporter config field use_cuda=true; use --use_gpu for the Python API.",
            stacklevel=2,
        )
    if "edex_folder" in config:
        warnings.warn(
            "Ignoring unsupported legacy config field edex_folder; use dataset_folder + sequence_folder.",
            stacklevel=2,
        )


def _warn_unsupported_sequence_fields(sequence: Dict[str, Any]) -> None:
    """Warn about legacy sequence fields that the Python tracker does not use."""
    if sequence.get("use_gt_scale"):
        warnings.warn(
            f"Ignoring use_gt_scale for {sequence.get('sequence_title', '<unnamed>')}; not supported.",
            stacklevel=2,
        )
    if sequence.get("start_end_residual"):
        warnings.warn(
            f"Ignoring start_end_residual for {sequence.get('sequence_title', '<unnamed>')}; not supported.",
            stacklevel=2,
        )
    if sequence.get("precompute_2d_tracks"):
        warnings.warn(
            f"Ignoring precompute_2d_tracks for {sequence.get('sequence_title', '<unnamed>')}.",
            stacklevel=2,
        )
    if sequence.get("precompute_key_frames"):
        warnings.warn(
            f"Ignoring precompute_key_frames for {sequence.get('sequence_title', '<unnamed>')}.",
            stacklevel=2,
        )
    if "multicam_mode" in sequence or "multicam_setup" in sequence:
        warnings.warn(
            "Ignoring per-sequence multicam_mode/multicam_setup; use CLI --multicam_mode.",
            stacklevel=2,
        )


def _resolve_sequence_gt_path(sequence: Dict[str, Any]) -> Optional[str]:
    """Return a usable ground-truth path from a reporter sequence entry."""
    gt_file_path = sequence.get("gt_file_path")
    if not gt_file_path:
        if gt_file_path == "":
            warnings.warn(
                f"Ignoring empty gt_file_path for {sequence.get('sequence_title', '<unnamed>')}.",
                stacklevel=2,
            )
        return None
    return gt_file_path


def track(args: argparse.Namespace,
          refined_focal: Optional[tuple[float, float]] = None,
          refined_principal: Optional[tuple[float, float]] = None) -> TrackerResults:
    """Run tracking for one dataset or video and return collected results."""
    if isinstance(args.multicam_mode, str):
        args.multicam_mode = conv.str2multicam_mode(args.multicam_mode)
    if isinstance(args.odometry_mode, str):
        args.odometry_mode = conv.str2odometry_mode(args.odometry_mode)

    # Normalize and apply backward-compat default: bare --num_loops without
    # --repeat_type now means Repeat; implicit Shuttle is deprecated.
    args.repeat_type = (args.repeat_type or 'none').lower()
    if args.num_loops > 0 and args.repeat_type == 'none':
        warnings.warn(
            "--num_loops without --repeat_type now defaults to 'repeat'; "
            "pass --repeat_type shuttle to keep the old shuttle semantics.",
            FutureWarning, stacklevel=2,
        )
        args.repeat_type = 'repeat'

    if args.repeat_type == 'shuttle' and args.odometry_mode == vslam.Odometry.OdometryMode.Inertial:
        raise ValueError("Inertial mode is not supported for shuttle mode")

    if args.dataset.endswith('.mp4'):
        dataset = VideoReader(args.dataset, stereo_edex=args.config_path,
                              num_loops=args.num_loops, repeat_type=args.repeat_type,
                              gt_path=getattr(args, 'gt_path', None))
    else:
        rgbd_mode = args.odometry_mode == vslam.Odometry.OdometryMode.RGBD
        dataset = EdexReader(args.dataset, stereo_edex=args.config_path,
                             num_loops=args.num_loops, rgbd_mode=rgbd_mode,
                             repeat_type=args.repeat_type,
                             cache_uncompressed=getattr(args, 'cache_uncompressed', False),
                             gt_path=getattr(args, 'gt_path', None),
                             camera_ids=getattr(args, 'camera_ids', None))

    tracker_results = TrackerResults()
    if args.sequence_title:
        tracker_results.stat.sequence_title = args.sequence_title
    if not dataset.validate_rig():
        print("Rig parameters are invalid")
        return tracker_results

    assert dataset.rig is not None
    if refined_focal is not None and refined_principal is not None:
        dataset.rig.cameras[0].focal = refined_focal
        dataset.rig.cameras[0].principal = refined_principal

    # If dataset has RGBD settings, transfer them to args
    # depth_camera_id MUST come from stereo.edex
    if hasattr(dataset, 'rgbd_settings') and dataset.rgbd_settings:
        # depth_camera_id always comes from stereo.edex
        args.depth_camera_id = dataset.rgbd_settings.depth_camera_id

        # Use dataset settings if args don't have custom values (command-line can override these)
        if not hasattr(args, 'depth_scale_factor') or args.depth_scale_factor == 1.0:
            args.depth_scale_factor = dataset.rgbd_settings.depth_scale_factor
        if not hasattr(args, 'enable_depth_stereo_tracking') or not args.enable_depth_stereo_tracking:
            args.enable_depth_stereo_tracking = dataset.rgbd_settings.enable_depth_stereo_tracking

    tracker = Tracker(dataset.rig, args)
    tracker_results.rig = dataset.rig
    if args.sequence_title:
        tracker.stat.sequence_title = args.sequence_title

    # Compose Processing decorator chain (outermost first). Add future fault-injection
    # filters here; each implements Processing and forwards to its inner.
    processor = tracker
    if args.blackout_period > 0:
        processor = BlackoutFilter(processor, args.blackout_period, args.blackout_duration)

    tracker.run_tracking_and_measure_performance(dataset, tracker_results, processor=processor)

    if dataset.gt_transforms:
        calculate_sequence_errors(tracker.world_from_rig, dataset.gt_transforms, tracker.stat,
                                  tracker.frame_metadata, args.use_segments, args.segment_lengths,
                                  args.num_loops, args.repeat_type)

    suffix = "_refined" if refined_focal is not None and refined_principal is not None else ""
    plot_trajectory_path = None
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
        plot_trajectory_path = os.path.join(args.output_dir, "plots", f"{args.sequence_title}{suffix}.png")
        tracker.stat.bird_view_with_errors_path = os.path.abspath(plot_trajectory_path)

    plot_trajectory(
        tracker.world_from_rig,
        tracker.loop_closures,
        dataset.gt_transforms,
        args.visualize_plot,
        plot_trajectory_path,
        dataset.gt_from_shuttle)

    export_kitti_benchmark_artifacts(
        tracker_results.world_from_rig,
        tracker_results.loop_closures,
        args.output_dir,
        args.sequence_title,
        use_slam=getattr(args, "use_slam", False),
        suffix=suffix,
    )

    if args.save_output_tracker_data:
        if not args.output_dir:
            print("No output directory provided, skipping output tracker data saving")
            return tracker_results
        if tracker.odom_cfg.enable_final_landmarks_export:
            os.makedirs(os.path.join(args.output_dir, "output_tracker_data"), exist_ok=True)
            output_data_path = os.path.join(args.output_dir, "output_tracker_data",
                                            f"{args.sequence_title}{suffix}.npy")
            save_result_to_edex(
                tracker.world_from_rig,
                tracker.final_landmarks,
                tracker.tracks2D,
                output_data_path)
        else:
            print("Final landmarks export is disabled, skipping output tracker data saving")

    return tracker_results
