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
from camera_utils import get_zed_rgbd_rig, setup_zed_camera

# Add path for visualizer import
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'realsense'))
from visualizer import RerunVisualizer

# Constants
# sl.RESOLUTION.AUTO picks the camera's native mode (HD720 on USB ZED,
# HD1200 on ZED X). See README for the per-camera compatibility table.
RESOLUTION = sl.RESOLUTION.AUTO
FPS = 30
RUN_STEREO = False


def main():
    """Main function for RGBD tracking with ZED camera."""
    # Initialize ZED camera
    zed, camera_info = setup_zed_camera(RESOLUTION, FPS, sl.DEPTH_MODE.PERFORMANCE, sl.UNIT.MILLIMETER)

    # Derive jitter threshold from the camera's actual FPS; the ZED SDK clamps
    # the requested FPS to the highest supported value for the chosen mode.
    actual_fps = camera_info.camera_configuration.fps
    image_jitter_threshold_ns = (1000 / actual_fps + 2) * 1e6  # ms -> ns, +2 ms buffer

    # Configure RGBD settings
    rgbd_settings = vslam.Odometry.RGBDSettings()
    rgbd_settings.depth_scale_factor = 1000
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = RUN_STEREO

    # Configure tracker
    odom_cfg = vslam.Odometry.Config(
        async_sba=True,
        enable_final_landmarks_export=True,
        odometry_mode=vslam.Odometry.OdometryMode.RGBD,
        rgbd_settings=rgbd_settings,
        rectified_stereo_camera=RUN_STEREO
    )

    # Create rig using utility function
    rig = get_zed_rgbd_rig(camera_info, RUN_STEREO)

    # Initialize tracker and visualizer
    tracker = vslam.Tracker(rig, odom_cfg)
    visualizer = RerunVisualizer(num_viz_cameras=2+RUN_STEREO)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()

    # Create image containers
    image_left = sl.Mat()
    depth_left = sl.Mat()
    if RUN_STEREO:
        image_right = sl.Mat()

    frame_id = 0
    prev_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []

    print("Starting RGBD tracking with cuvslam...")
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

                frame_id += 1

                # Get images
                zed.retrieve_image(image_left, sl.VIEW.LEFT)
                zed.retrieve_measure(depth_left, sl.MEASURE.DEPTH)
                left_data = image_left.get_data()
                left_rgb = np.ascontiguousarray(left_data[:,:,[2,1,0]])
                depth_data = np.round(depth_left.get_data()).astype(np.uint16)

                if RUN_STEREO:
                    zed.retrieve_image(image_right, sl.VIEW.RIGHT)
                    right_data = image_right.get_data()
                    right_rgb = np.ascontiguousarray(right_data[:,:,[2,1,0]])

                    # Track frame with both color and depth
                    odom_pose_estimate, _ = tracker.track(
                        timestamp, images=[left_rgb, right_rgb], depths=[depth_data]
                    )
                else:
                    odom_pose_estimate, _ = tracker.track(
                        timestamp, images=[left_rgb], depths=[depth_data]
                    )

                if odom_pose_estimate.world_from_rig is None:
                    print("Warning: Pose tracking not valid")
                    continue

                odom_pose = odom_pose_estimate.world_from_rig.pose
                trajectory.append(odom_pose.translation)

                # Store current timestamp for next iteration
                prev_timestamp = timestamp

                # Visualize results for color and depth cameras
                # Same observations for both, since we only have one image
                observations = tracker.odometry.get_last_observations(0)
                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=[left_rgb, depth_data, right_rgb] if RUN_STEREO else [left_rgb, depth_data],
                    pose=odom_pose,
                    observations_main_cam=[observations]*(2+RUN_STEREO), # 2 for stereo, 1 for mono
                    trajectory=trajectory,
                    timestamp=timestamp
                )

    finally:
        # Clean up
        zed.close()


if __name__ == "__main__":
    main()
