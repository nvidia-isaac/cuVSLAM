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

import os
import sys
from typing import List, Optional

import numpy as np

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBStreamType, OBFormat,
    AlignFilter, OBPropertyID, OBPermissionType
)

import cuvslam as vslam
from camera_utils import get_rgbd_calibration, get_orbbec_rgbd_rig, frame_to_rgb_image

# Add path for visualizer import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'realsense'))
from visualizer import RerunVisualizer

# Constants
RESOLUTION = (640, 480)
FPS = 30
WARMUP_FRAMES = 60
NUM_VIZ_CAMERAS = 2  # Color + Depth visualization

# Calculate jitter threshold based on FPS + buffer
FRAME_PERIOD_MS = 1000 / FPS
IMAGE_JITTER_THRESHOLD_NS = (FRAME_PERIOD_MS + 5) * 1e6  # Convert to nanoseconds


def enable_ir_emitter(pipeline: Pipeline) -> None:
    """Enable the IR laser emitter for depth sensing.

    The IR emitter projects patterns needed for active depth sensing.

    Args:
        pipeline: Orbbec pipeline
    """
    device = pipeline.get_device()
    emitter_enabled = False

    # Try OB_PROP_LASER_BOOL (most common)
    try:
        if device.is_property_supported(
            OBPropertyID.OB_PROP_LASER_BOOL, OBPermissionType.PERMISSION_READ_WRITE
        ):
            device.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, True)
            current = device.get_bool_property(OBPropertyID.OB_PROP_LASER_BOOL)
            if current:
                print("IR emitter enabled (OB_PROP_LASER_BOOL)")
                emitter_enabled = True
            else:
                print("Warning: Failed to enable laser via OB_PROP_LASER_BOOL")
    except Exception as e:
        print(f"OB_PROP_LASER_BOOL not available: {e}")

    # Try OB_PROP_LASER_CONTROL_INT (0=off, 1=on, 2=auto)
    if not emitter_enabled:
        try:
            if device.is_property_supported(
                OBPropertyID.OB_PROP_LASER_CONTROL_INT, OBPermissionType.PERMISSION_READ_WRITE
            ):
                device.set_int_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, 1)
                print("IR emitter enabled (OB_PROP_LASER_CONTROL_INT)")
                emitter_enabled = True
        except Exception as e:
            print(f"OB_PROP_LASER_CONTROL_INT not available: {e}")

    if not emitter_enabled:
        print("Warning: Could not enable IR emitter - device may not support this feature")


def main() -> None:
    """Main function for Orbbec RGBD tracking."""
    # Create pipeline and config
    pipeline = Pipeline()
    config = Config()

    # Enable IR emitter for active depth sensing
    enable_ir_emitter(pipeline)

    # Configure color stream
    print(f"Configuring streams: {RESOLUTION[0]}x{RESOLUTION[1]} @ {FPS} FPS")

    try:
        # Configure color stream with specified resolution
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profile_list.get_video_stream_profile(
            RESOLUTION[0], RESOLUTION[1], OBFormat.RGB, FPS
        )
        config.enable_stream(color_profile)
    except Exception as e:
        print(f"Warning: Requested color resolution not available: {e}, using default")
        config.enable_video_stream(OBSensorType.COLOR_SENSOR)

    try:
        # Configure depth stream with specified resolution
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profile_list.get_video_stream_profile(
            RESOLUTION[0], RESOLUTION[1], OBFormat.Y16, FPS
        )
        config.enable_stream(depth_profile)
    except Exception as e:
        print(f"Warning: Requested depth resolution not available: {e}, using default")
        config.enable_video_stream(OBSensorType.DEPTH_SENSOR)

    # Enable frame synchronization
    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print(f"Warning: Could not enable frame sync: {e}")

    # Start pipeline
    pipeline.start(config)

    # Create align filter to align depth to color
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    # Get RGBD calibration data
    print("Getting RGBD calibration...")
    rgbd_params = get_rgbd_calibration(pipeline)

    # Create rig
    rig = get_orbbec_rgbd_rig(rgbd_params)

    # cuVSLAM depth_scale_factor: value to divide depth pixels by to get meters
    depth_scale_factor = 1000.0  # Orbbec depth is in mm, so divide by 1000 to get meters

    print(f"Depth scale factor for cuVSLAM: {depth_scale_factor}")

    # Configure RGBD settings
    rgbd_settings = vslam.Odometry.RGBDSettings()
    rgbd_settings.depth_scale_factor = depth_scale_factor
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = False

    # Configure tracker
    odom_cfg = vslam.Odometry.Config(
        async_sba=True,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        odometry_mode=vslam.Odometry.OdometryMode.RGBD,
        rgbd_settings=rgbd_settings
    )

    # Initialize tracker and visualizer
    tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryOnlyRealtime, odom_cfg)
    visualizer = RerunVisualizer(num_viz_cameras=NUM_VIZ_CAMERAS)

    frame_id = 0
    prev_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []

    print("Starting RGBD tracking with cuvslam...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # Wait for synchronized frames
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            # Align depth to color
            aligned_frames = align_filter.process(frames)
            if aligned_frames is None:
                continue
            aligned_frames = aligned_frames.as_frame_set()

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            # Get timestamp (in microseconds, convert to nanoseconds)
            timestamp = int(color_frame.get_timestamp_us() * 1000)

            # Check timestamp difference with previous frame
            if prev_timestamp is not None:
                timestamp_diff = timestamp - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms"
                    )

            # Store current timestamp for next iteration
            prev_timestamp = timestamp

            # Convert color frame to RGB numpy array
            color_rgb = frame_to_rgb_image(color_frame)
            if color_rgb is None:
                print("Warning: Failed to convert color frame; skipping frame")
                continue

            # Get depth data as uint16
            try:
                depth_data = np.frombuffer(
                    depth_frame.get_data(), dtype=np.uint16
                ).reshape((depth_frame.get_height(), depth_frame.get_width()))
            except ValueError:
                print("Warning: Failed to reshape depth data")
                continue

            frame_id += 1

            # Warmup for specified number of frames
            if frame_id > WARMUP_FRAMES:
                # Track frame
                odom_pose_estimate, _ = tracker.track(
                    timestamp, images=[color_rgb], depths=[depth_data]
                )

                if odom_pose_estimate.world_from_rig is None:
                    print("Warning: Pose tracking not valid")
                    continue

                odom_pose = odom_pose_estimate.world_from_rig.pose
                trajectory.append(odom_pose.translation)

                # Visualize results for color and depth cameras
                observations = tracker.odometry.get_last_observations(0)
                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=[color_rgb, depth_data],
                    pose=odom_pose,
                    observations_main_cam=[observations, observations],
                    trajectory=trajectory,
                    timestamp=timestamp
                )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        pipeline.stop()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
