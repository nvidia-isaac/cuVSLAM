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

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs
import yaml
from copy import deepcopy

import cuvslam as vslam
from camera_utils import (
    configure_device,
    get_camera_intrinsics,
    get_rs_multi_rig,
    setup_pipeline,
)
from visualizer import RerunVisualizer

# Constants
CONFIG_FILE = "frame_agx_rig.yaml"
WARMUP_FRAMES = 500
SYNC_MATCHING_THRESHOLD_MS = 5 * 1e6  # 5ms in nanoseconds
IMAGE_JITTER_THRESHOLD_NS = 35 * 1e6  # 35ms in nanoseconds


def load_camera_configuration() -> Dict:
    """Load camera configuration from YAML file.

    Returns:
        Dictionary containing camera configuration
    """
    print("Loading camera configuration...")
    with open(CONFIG_FILE, 'r') as file:
        return yaml.safe_load(file)


def setup_camera_pipelines(serial_numbers: List[str]) -> Tuple[
    List[rs.pipeline], List[rs.config]
]:
    """Set up pipelines for all cameras.

    Args:
        serial_numbers: List of camera serial numbers

    Returns:
        Tuple of (pipelines, configs)
    """
    print("Setting up camera pipelines...")
    pipelines = []
    configs = []

    for i, serial in enumerate(serial_numbers):
        pipeline, config = setup_pipeline(serial)
        pipelines.append(pipeline)
        configs.append(config)
        print(f"Camera {i+1} ({serial}) pipeline configured")

    return pipelines, configs


def extract_camera_parameters(
    pipelines: List[rs.pipeline],
    configs: List[rs.config],
    stereo_cameras: List[Dict]
) -> Dict[str, Dict]:
    """Extract intrinsics for all cameras.

    Args:
        pipelines: List of RealSense pipelines
        configs: List of RealSense configs
        stereo_cameras: List of stereo camera configurations

    Returns:
        Dictionary containing camera parameters
    """
    print("Extracting camera intrinsics...")
    camera_params = {}

    for i, (pipeline, config, stereo_cam) in enumerate(
        zip(pipelines, configs, stereo_cameras)
    ):
        left_intrinsics, right_intrinsics = get_camera_intrinsics(
            pipeline, config
        )
        camera_idx = f"camera_{i+1}"
        camera_params[camera_idx] = {
            "left": {
                "intrinsics": left_intrinsics,
                "extrinsics": stereo_cam["left_camera"]["transform"]
            },
            "right": {
                "intrinsics": right_intrinsics,
                "extrinsics": stereo_cam["right_camera"]["transform"]
            }
        }
        print(f"Camera {i+1} intrinsics extracted")

    return camera_params


def configure_all_devices(
    pipelines: List[rs.pipeline], configs: List[rs.config]
) -> None:
    """Configure all devices.

    Args:
        pipelines: List of RealSense pipelines
        configs: List of RealSense configs
    """
    print("Configuring devices...")
    for i, (pipeline, config) in enumerate(zip(pipelines, configs)):
        configure_device(pipeline, config, is_master=(i == 0))


def start_all_cameras(
    pipelines: List[rs.pipeline], configs: List[rs.config]
) -> None:
    """Start streaming from all cameras.

    Args:
        pipelines: List of RealSense pipelines
        configs: List of RealSense configs
    """
    print("Starting camera streams...")
    for pipeline, config in zip(pipelines, configs):
        pipeline.start(config)


def get_synchronized_frames(
    pipelines: List[rs.pipeline], frame_id: int
) -> Tuple[List[int], List[np.ndarray]]:
    """Get frames from all cameras.

    Args:
        pipelines: List of RealSense pipelines
        frame_id: Current frame ID

    Returns:
        Tuple of (timestamps, images)
    """
    all_timestamps = []
    all_images = []

    for i, pipeline in enumerate(pipelines):
        frames = pipeline.wait_for_frames()
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)

        if not left_frame or not right_frame:
            print(f"Skip frames from camera {i+1}, frame_id: {frame_id}")
            continue

        all_timestamps.append(int(left_frame.timestamp * 1e6))
        all_images.extend([
            np.asanyarray(left_frame.get_data()),
            np.asanyarray(right_frame.get_data())
        ])

    return all_timestamps, all_images


def check_timestamp_synchronization(timestamps: List[int]) -> None:
    """Check timestamp synchronization between cameras.

    Args:
        timestamps: List of timestamps from all cameras
    """
    max_timestamp_diff = 0
    for i in range(len(timestamps)):
        for j in range(i+1, len(timestamps)):
            timestamp_diff = abs(timestamps[i] - timestamps[j])
            max_timestamp_diff = max(max_timestamp_diff, timestamp_diff)

    if max_timestamp_diff > SYNC_MATCHING_THRESHOLD_MS:
        print(
            f"Warning: Timestamp synchronization failed, "
            f"max_timestamp_diff: {max_timestamp_diff/1e6:.2f} ms exceeds "
            f"threshold {SYNC_MATCHING_THRESHOLD_MS/1e6:.2f} ms"
        )


def main() -> None:
    """Main function for multi-camera tracking."""
    # Load configuration
    config_data = load_camera_configuration()
    stereo_cameras = config_data['stereo_cameras']
    serial_numbers = [cam['serial'] for cam in stereo_cameras]

    # Setup pipelines
    pipelines, configs = setup_camera_pipelines(serial_numbers)

    # Extract camera parameters
    camera_params = extract_camera_parameters(
        pipelines, configs, stereo_cameras
    )

    # Create rig with all cameras
    print("Creating camera rig...")
    rig = get_rs_multi_rig(camera_params)

    # Initialize tracker
    odom_cfg = vslam.Odometry.Config(
        async_sba=False,
        enable_final_landmarks_export=True,
        rectified_stereo_camera=True
    )
    tracker = vslam.Tracker(rig, odom_cfg)

    # Configure all devices
    configure_all_devices(pipelines, configs)

    # Start all cameras
    start_all_cameras(pipelines, configs)

    # Initialize visualization
    visualizer = RerunVisualizer(num_viz_cameras=len(serial_numbers))
    trajectory: List[np.ndarray] = []
    frame_id = 0
    prev_timestamp: Optional[int] = None

    try:
        print("Start running")
        while True:
            frame_id += 1

            # Get frames from all cameras
            all_timestamps, all_images = get_synchronized_frames(
                pipelines, frame_id
            )

            # Warmup period for camera synchronization
            if frame_id < WARMUP_FRAMES:
                continue

            # Skip if any camera frames are missing
            if len(all_timestamps) < len(pipelines):
                continue

            # Check for frame drops
            if prev_timestamp is not None:
                timestamp_diff = all_timestamps[0] - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms"
                    )

            # Check timestamp synchronization
            check_timestamp_synchronization(all_timestamps)

            # Track frame using the first timestamp
            odom_pose_estimate, _ = tracker.track(all_timestamps[0], tuple(all_images))

            odom_pose_with_cov = odom_pose_estimate.world_from_rig
            if odom_pose_with_cov is None:
                print(f"Tracking failed at frame {frame_id}")
                continue
            trajectory.append(odom_pose_with_cov.pose.translation)

            # Visualize results (showing only left cameras)
            left_images = [all_images[i] for i in range(0, len(all_images), 2)]
            left_observations = [
                tracker.odometry.get_last_observations(i)
                for i in range(0, len(all_images), 2)
            ]

            visualizer.visualize_frame(
                frame_id=frame_id,
                images=left_images,
                pose=odom_pose_with_cov.pose,
                observations_main_cam=left_observations,
                trajectory=trajectory,
                timestamp=all_timestamps[0]
            )

            # Store current timestamp for next iteration
            prev_timestamp = deepcopy(all_timestamps[0])

    finally:
        # Stop streaming from all cameras
        for pipeline in pipelines:
            pipeline.stop()


if __name__ == "__main__":
    main()
