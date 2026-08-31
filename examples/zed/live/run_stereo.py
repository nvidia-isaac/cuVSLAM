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
import pyzed.sl as sl

import cuvslam as vslam
from camera_utils import get_zed_stereo_rig, setup_zed_camera

# Add path for visualizer import
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'realsense'))
from visualizer import RerunVisualizer

# Constants
# sl.RESOLUTION.AUTO picks the camera's native mode (HD720 on USB ZED,
# HD1200 on ZED X). See README for the per-camera compatibility table.
RESOLUTION = sl.RESOLUTION.AUTO
FPS = 60
RAW = False


def main():
    """Main function for stereo tracking with ZED camera."""
    # Initialize ZED camera
    zed, camera_info = setup_zed_camera(RESOLUTION, FPS)

    # Derive jitter threshold from the camera's actual FPS; the ZED SDK clamps
    # the requested FPS to the highest supported value for the chosen mode.
    actual_fps = camera_info.camera_configuration.fps
    image_jitter_threshold_ns = (1000 / actual_fps + 2) * 1e6  # ms -> ns, +2 ms buffer

    # Configure tracker
    odom_cfg = vslam.Odometry.Config(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True
    )

    if RAW:
        cfg.rectified_stereo_camera = False
    else:
        cfg.rectified_stereo_camera = True

    # Create rig using utility function
    rig = get_zed_stereo_rig(camera_info, raw = RAW)

    # Initialize tracker
    tracker = vslam.Tracker(rig, odom_cfg)
    visualizer = RerunVisualizer()

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()

    # Create image containers
    image_left = sl.Mat()
    image_right = sl.Mat()

    frame_id = 0
    prev_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []

    print("Starting stereo tracking with cuvslam...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # A new image is available if grab() returns SUCCESS
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Get timestamp
                timestamp = int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

                # Check timestamp difference with previous frame
                if prev_timestamp is not None:
                    timestamp_diff = timestamp - prev_timestamp
                    if timestamp_diff > image_jitter_threshold_ns:
                        print(
                            f"Warning: Camera stream message drop: timestamp gap "
                            f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                            f"{image_jitter_threshold_ns/1e6:.2f} ms"
                        )

                # Store current timestamp for next iteration
                prev_timestamp = timestamp

                # Get images
                if RAW:
                    zed.retrieve_image(image_left, sl.VIEW.LEFT_UNRECTIFIED)
                    zed.retrieve_image(image_right, sl.VIEW.RIGHT_UNRECTIFIED)
                else:
                    zed.retrieve_image(image_left, sl.VIEW.LEFT)
                    zed.retrieve_image(image_right, sl.VIEW.RIGHT)

                # Convert to numpy arrays and ensure proper RGB format
                left_data = image_left.get_data()
                right_data = image_right.get_data()

                # Ensure contiguous arrays with proper RGB format from BGRA
                left_rgb = np.ascontiguousarray(left_data[:,:,[2,1,0]])
                right_rgb = np.ascontiguousarray(right_data[:,:,[2,1,0]])

                frame_id += 1

                odom_pose_estimate, _ = tracker.track(timestamp, images=[left_rgb, right_rgb])

                if odom_pose_estimate.world_from_rig is None:
                    print("Warning: Pose tracking not valid")
                    continue

                odom_pose = odom_pose_estimate.world_from_rig.pose
                trajectory.append(odom_pose.translation)

                # Visualize results for left camera
                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=[left_rgb],
                    pose=odom_pose,
                    observations_main_cam=[tracker.odometry.get_last_observations(0)],
                    trajectory=trajectory,
                    timestamp=timestamp
                )

    finally:
        # Clean up
        zed.close()


if __name__ == "__main__":
    main()
