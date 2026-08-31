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
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

import cuvslam as vslam

# Add the realsense folder to the system path to import visualizer
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../realsense'))
)

from visualizer import RerunVisualizer

# Constants
FPS = 30
RESOLUTION = (1280, 720)
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_NS = 35 * 1e6  # 35ms in nanoseconds

# Camera border margins to exclude features near image edges
# This helps avoid using features from highly distorted regions in unrectified OAK-D images
# Features detected within these margins will not be processed
BORDER_TOP = 50
BORDER_BOTTOM = 0
BORDER_LEFT = 70
BORDER_RIGHT = 70

# Conversion factor from cm to meters
CM_TO_METERS = 100

def oak_transform_to_pose(oak_extrinsics: List[List[float]]) -> vslam.Pose:
    """Convert 4x4 transformation matrix to cuVSLAM pose.

    Args:
        oak_extrinsics: 4x4 transformation matrix from OAK calibration

    Returns:
        vslam.Pose object
    """
    extrinsics_array = np.array(oak_extrinsics)
    rotation_matrix = extrinsics_array[:3, :3]
    translation_vector = extrinsics_array[:3, 3] / CM_TO_METERS  # Convert to meters

    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_vector)


def set_cuvslam_camera(oak_params: Dict[str, Any]) -> vslam.Camera:
    """Create a Camera object from OAK camera parameters.

    Args:
        oak_params: Dictionary containing camera parameters

    Returns:
        vslam.Camera object
    """
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(
        vslam.Distortion.Model.Polynomial, oak_params['distortion']
    )

    cam.focal = (
        oak_params['intrinsics'][0][0],
        oak_params['intrinsics'][1][1]
    )
    cam.principal = (
        oak_params['intrinsics'][0][2],
        oak_params['intrinsics'][1][2]
    )
    cam.size = oak_params['resolution']
    cam.rig_from_camera = oak_transform_to_pose(oak_params['extrinsics'])

    # Features within these outer frames will be ignored by cuVSLAM
    cam.border_top = BORDER_TOP
    cam.border_bottom = BORDER_BOTTOM
    cam.border_left = BORDER_LEFT
    cam.border_right = BORDER_RIGHT

    return cam


def get_stereo_calibration(
    calib_data: dai.CalibrationHandler, resolution: Tuple[int, int]
) -> Dict[str, Dict[str, Any]]:
    """Get calibration data from the OAK-D calibration handler.

    Args:
        calib_data: Calibration handler from the OAK-D device
        resolution: Camera resolution as (width, height)

    Returns:
        Dictionary containing stereo calibration parameters
    """
    stereo_camera = {'left': {}, 'right': {}}

    # Set image size
    stereo_camera['left']['resolution'] = resolution
    stereo_camera['right']['resolution'] = resolution

    # Get intrinsics for left and right cameras (scaled to the requested resolution)
    stereo_camera['left']['intrinsics'] = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_B, resolution[0], resolution[1]
    )
    stereo_camera['right']['intrinsics'] = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_C, resolution[0], resolution[1]
    )

    # Get extrinsics (transformation of left and right cameras relative to center RGB camera)
    stereo_camera['left']['extrinsics'] = calib_data.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A
    )
    stereo_camera['right']['extrinsics'] = calib_data.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A
    )

    # Get distortion coefficients for left and right cameras (first 8 coefficients)
    stereo_camera['left']['distortion'] = calib_data.getDistortionCoefficients(
        dai.CameraBoardSocket.CAM_B
    )[:8]
    stereo_camera['right']['distortion'] = calib_data.getDistortionCoefficients(
        dai.CameraBoardSocket.CAM_C
    )[:8]

    return stereo_camera


def main() -> None:
    """Main function for OAK-D stereo tracking."""
    # Create device and read calibration before pipeline creation
    device = dai.Device()
    calib_data = device.readCalibration()
    stereo_camera = get_stereo_calibration(calib_data, RESOLUTION)

    cameras = [
        set_cuvslam_camera(stereo_camera['left']),
        set_cuvslam_camera(stereo_camera['right'])
    ]

    # Create rig and tracker
    odom_cfg = vslam.Odometry.Config(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        rectified_stereo_camera=False
    )
    tracker = vslam.Tracker(vslam.Rig(cameras), odom_cfg)

    # Create pipeline with the device
    pipeline = dai.Pipeline(device)

    # Create stereo pair using new Camera node API
    left_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorFps=FPS
    )
    right_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C, sensorFps=FPS
    )

    # Use Sync node to synchronize stereo frames
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5 / FPS))

    # Request grayscale outputs at specified resolution
    left_output = left_camera.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    right_output = right_camera.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)

    # Link camera outputs to sync node
    left_output.link(sync.inputs["left"])
    right_output.link(sync.inputs["right"])

    # Create output queue from sync node
    sync_queue = sync.out.createOutputQueue()

    # Initialize visualization and tracking variables
    visualizer = RerunVisualizer()
    frame_id = 0
    prev_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []

    # Start the pipeline
    pipeline.start()

    # Capture and process stereo frames
    while pipeline.isRunning():
        message_group: dai.MessageGroup = sync_queue.get()
        left_frame = message_group["left"]
        right_frame = message_group["right"]

        # Get synchronized timestamp from message group (convert timedelta to ns)
        timestamp_ns = int(message_group.getTimestamp().total_seconds() * 1e9)


        # Check timestamp difference with previous frame
        if prev_timestamp is not None:
            timestamp_diff = timestamp_ns - prev_timestamp
            if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                print(
                    f"Warning: Camera stream message drop: timestamp gap "
                    f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                    f"{IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms"
                )

        frame_id += 1

        # Warmup for specified number of frames
        if frame_id > WARMUP_FRAMES:
            left_img = left_frame.getCvFrame()
            right_img = right_frame.getCvFrame()

            # Track frame
            odom_pose_estimate, _ = tracker.track(timestamp_ns, (left_img, right_img))
            odom_pose = odom_pose_estimate.world_from_rig
            if odom_pose is None:
                print(f"Tracking failed at frame {frame_id}")
                continue

            trajectory.append(odom_pose.pose.translation)

            # Visualize results
            visualizer.visualize_frame(
                frame_id=frame_id,
                images=[left_img],
                pose=odom_pose.pose,
                observations_main_cam=[tracker.odometry.get_last_observations(0)],
                trajectory=trajectory,
                timestamp=timestamp_ns
            )

        prev_timestamp = timestamp_ns


if __name__ == "__main__":
    main()
