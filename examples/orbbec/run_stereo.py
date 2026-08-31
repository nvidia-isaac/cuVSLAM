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
    Pipeline, Config, OBSensorType, OBFrameType, OBFormat,
    OBPropertyID, OBPermissionType
)

import cuvslam as vslam
from camera_utils import get_stereo_calibration, get_orbbec_stereo_rig, process_ir_frame

# Add path for visualizer import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'realsense'))
from visualizer import RerunVisualizer

# Constants
# Available IR resolutions: (640, 480), (848, 480), (1280, 800)
RESOLUTION = (640, 480)
FPS = 30

WARMUP_FRAMES = 60
# Calculate jitter threshold based on FPS + buffer
FRAME_PERIOD_MS = 1000 / FPS
IMAGE_JITTER_THRESHOLD_NS = (FRAME_PERIOD_MS + 5) * 1e6  # Convert to nanoseconds

# Sensor types required for stereo tracking
STEREO_SENSORS = {OBSensorType.LEFT_IR_SENSOR, OBSensorType.RIGHT_IR_SENSOR}


def disable_ir_emitter(pipeline: Pipeline) -> None:
    """Disable the IR laser emitter for stereo tracking.

    The IR emitter creates patterns that can interfere with stereo matching.
    For passive stereo tracking, it should be disabled.

    Args:
        pipeline: Orbbec pipeline
    """
    device = pipeline.get_device()
    emitter_disabled = False

    # Try OB_PROP_LASER_BOOL (most common)
    try:
        if device.is_property_supported(
            OBPropertyID.OB_PROP_LASER_BOOL, OBPermissionType.PERMISSION_READ_WRITE
        ):
            device.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, False)
            # Verify it was set
            current = device.get_bool_property(OBPropertyID.OB_PROP_LASER_BOOL)
            if not current:
                print("IR emitter disabled (OB_PROP_LASER_BOOL)")
                emitter_disabled = True
            else:
                print("Warning: Failed to disable laser via OB_PROP_LASER_BOOL")
    except Exception as e:
        print(f"OB_PROP_LASER_BOOL not available: {e}")

    # Try OB_PROP_LASER_CONTROL_INT (0=off, 1=on, 2=auto)
    if not emitter_disabled:
        try:
            if device.is_property_supported(
                OBPropertyID.OB_PROP_LASER_CONTROL_INT, OBPermissionType.PERMISSION_READ_WRITE
            ):
                device.set_int_property(OBPropertyID.OB_PROP_LASER_CONTROL_INT, 0)
                print("IR emitter disabled (OB_PROP_LASER_CONTROL_INT)")
                emitter_disabled = True
        except Exception as e:
            print(f"OB_PROP_LASER_CONTROL_INT not available: {e}")

    # Try OB_PROP_LDP_BOOL (Laser Diode Power)
    if not emitter_disabled:
        try:
            if device.is_property_supported(
                OBPropertyID.OB_PROP_LDP_BOOL, OBPermissionType.PERMISSION_READ_WRITE
            ):
                device.set_bool_property(OBPropertyID.OB_PROP_LDP_BOOL, False)
                print("IR emitter disabled (OB_PROP_LDP_BOOL)")
                emitter_disabled = True
        except Exception as e:
            print(f"OB_PROP_LDP_BOOL not available: {e}")

    if not emitter_disabled:
        print("Warning: Could not disable IR emitter - device may not support this feature")


def main() -> None:
    """Main function for Orbbec stereo tracking."""
    # Create pipeline and config
    pipeline = Pipeline()
    config = Config()

    # Check for dual IR support
    sensor_types = {pipeline.get_device().get_sensor_list().get_type_by_index(i)
                    for i in range(pipeline.get_device().get_sensor_list().get_count())}
    if not STEREO_SENSORS.issubset(sensor_types):
        raise RuntimeError("Device does not support dual IR sensors (LEFT_IR + RIGHT_IR required)")

    # Disable IR emitter for passive stereo tracking
    disable_ir_emitter(pipeline)

    # Configure left and right IR streams with specified resolution
    print(f"Configuring streams: {RESOLUTION[0]}x{RESOLUTION[1]} @ {FPS} FPS")

    for sensor_type in [OBSensorType.LEFT_IR_SENSOR, OBSensorType.RIGHT_IR_SENSOR]:
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        profile = None

        # Try specified resolution with different formats
        for fmt in [OBFormat.Y8, OBFormat.Y16]:
            try:
                profile = profile_list.get_video_stream_profile(
                    RESOLUTION[0], RESOLUTION[1], fmt, FPS
                )
                break
            except Exception as e:
                print(f"Warning: Failed to get stream profile for {sensor_type}: {e}")
                continue

        if profile is None:
            # Fall back to default profile
            profile = profile_list.get_default_video_stream_profile()
            print(f"Using default profile for {sensor_type}: "
                  f"{profile.get_width()}x{profile.get_height()} @ {profile.get_fps()} FPS")

        config.enable_stream(profile)

    # Enable frame synchronization
    try:
        pipeline.enable_frame_sync()
    except Exception as e:
        print(f"Warning: Could not enable frame sync: {e}")

    # Start pipeline
    pipeline.start(config)

    # Get stereo calibration data
    print("Getting stereo calibration...")
    stereo_params = get_stereo_calibration(pipeline)

    # Create rig
    rig = get_orbbec_stereo_rig(stereo_params)

    # Configure tracker
    odom_cfg = vslam.Odometry.Config(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        rectified_stereo_camera=True  # Orbbec IR images are typically rectified
    )

    # Initialize tracker and visualizer
    tracker = vslam.Tracker(rig, odom_cfg)
    visualizer = RerunVisualizer()

    frame_id = 0
    prev_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []

    print("Starting stereo tracking with cuvslam...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # Wait for synchronized frames
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            # Get stereo IR frames
            left_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
            right_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)

            if left_frame is None or right_frame is None:
                continue

            # Get timestamp (in microseconds, convert to nanoseconds)
            timestamp = int(left_frame.get_timestamp_us() * 1000)

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

            # Process IR frames
            left_img = process_ir_frame(left_frame)
            right_img = process_ir_frame(right_frame)

            if left_img is None or right_img is None:
                print("Warning: Failed to convert IR frames; skipping frame pair")
                continue

            frame_id += 1

            # Warmup for specified number of frames
            if frame_id > WARMUP_FRAMES:
                # Track frame
                odom_pose_estimate, _ = tracker.track(
                    timestamp, images=(left_img, right_img)
                )

                if odom_pose_estimate.world_from_rig is None:
                    print("Warning: Pose tracking not valid")
                    continue

                odom_pose = odom_pose_estimate.world_from_rig.pose
                trajectory.append(odom_pose.translation)

                # Visualize results for left camera
                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=[left_img],
                    pose=odom_pose,
                    observations_main_cam=[tracker.odometry.get_last_observations(0)],
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
