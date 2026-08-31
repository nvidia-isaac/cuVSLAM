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

import csv
import os
from typing import List, Tuple

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

import cuvslam


def load_frame(image_path: str) -> np.ndarray:
    """Load an image from a file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path)
    frame = np.array(image)

    if image.mode == 'L':
        # mono8
        if len(frame.shape) != 2:
            raise ValueError("Expected mono8 image to have 2 dimensions [H W].")
    elif image.mode == 'RGB':
        # rgb8 - convert to BGR for cuvslam compatibility
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected rgb8 image to have 3 dimensions with 3 channels [H W C].")
        # Convert RGB to BGR by reversing the channel order and ensure contiguous
        frame = np.ascontiguousarray(frame[:, :, ::-1])
    elif image.mode == 'I;16':
        # uint16 depth image
        if len(frame.shape) != 2:
            raise ValueError("Expected uint16 depth image to have 2 dimensions [H W].")
        frame = frame.astype(np.uint16)
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")

    return frame


def transform_to_pose(transform_16: List[float]) -> cuvslam.Pose:
    """Convert a 4x4 transformation matrix to a cuvslam.Pose object."""
    transform = np.array(transform_16).reshape(4, 4)
    rotation_quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return cuvslam.Pose(rotation=rotation_quat, translation=transform[:3, 3])


def _find_transform_key(config: dict) -> str:
    """Find transformation key that starts with 'T_' in config."""
    for key in config:
        if key.startswith('T_'):
            return key
    raise ValueError("No transformation found in config")


def get_transform_from_config(config: dict) -> np.ndarray:
    """Extract transformation matrix from config dictionary."""
    transform_key = _find_transform_key(config)
    return np.array(config[transform_key]['data']).reshape(4, 4)


def transform_to_cam0_reference(cam0_transform: np.ndarray,
                               sensor_transform: np.ndarray) -> np.ndarray:
    """Transform sensor pose to be relative to cam0 (cam0 becomes identity)."""

    cam0_body = np.linalg.inv(cam0_transform)
    return cam0_body @ sensor_transform


def _load_yaml_config(yaml_path: str) -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Sensor YAML not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def _get_sensor_paths(euroc_path: str) -> Tuple[str, str, str, bool]:
    """Get sensor file paths and calibration type."""
    if not os.path.exists(euroc_path):
        raise FileNotFoundError(f"EuRoC dataset path does not exist: {euroc_path}")

    # Check for recalibrated sensors
    cam0_recalibrated = os.path.join(euroc_path, 'cam0', 'sensor_recalibrated.yaml')
    if os.path.exists(cam0_recalibrated):
        return (
            cam0_recalibrated,
            os.path.join(euroc_path, 'cam1', 'sensor_recalibrated.yaml'),
            os.path.join(euroc_path, 'imu0', 'sensor_recalibrated.yaml'),
            False  # Not default calibration
        )

    return (
        os.path.join(euroc_path, 'cam0', 'sensor.yaml'),
        os.path.join(euroc_path, 'cam1', 'sensor.yaml'),
        os.path.join(euroc_path, 'imu0', 'sensor.yaml'),
        True  # Default calibration
    )


def _create_camera_from_config(config: dict, is_default: bool) -> cuvslam.Camera:
    """Create camera object from config."""
    cam = cuvslam.Camera()
    cam.focal = config['intrinsics'][0:2]
    cam.principal = config['intrinsics'][2:4]
    cam.size = config['resolution']

    if is_default:
        # Brown-Conrady distortion model requires only k1, k2, k3, p1, p2
        # but only k1, k2, p1, p2 are provided in the default calibration
        cam.distortion = cuvslam.Distortion(
            cuvslam.Distortion.Model.Brown,
            config['distortion_coefficients'][:2] + [0] + config['distortion_coefficients'][2:]
        )
    else:
        cam.distortion = cuvslam.Distortion(
            cuvslam.Distortion.Model.Fisheye,
            config['distortion_coefficients']
        )

    return cam


def get_rig(euroc_path: str) -> cuvslam.Rig:
    """Get a Rig object from EuRoC dataset path with transformations relative to cam0."""
    cam0_path, cam1_path, imu_path, is_default = _get_sensor_paths(euroc_path)

    # Load configurations
    cam0_config = _load_yaml_config(cam0_path)
    cam1_config = _load_yaml_config(cam1_path)
    imu_config = _load_yaml_config(imu_path)

    # Get transformation matrices
    cam0_transform = get_transform_from_config(cam0_config)
    cam1_transform = get_transform_from_config(cam1_config)
    imu_transform = get_transform_from_config(imu_config)

    # Create cameras
    cam0 = _create_camera_from_config(cam0_config, is_default)
    cam1 = _create_camera_from_config(cam1_config, is_default)

    # Cam0 becomes identity
    cam0.rig_from_camera = cuvslam.Pose(
        rotation=[0, 0, 0, 1],  # Identity quaternion
        translation=[0, 0, 0]    # Zero translation
    )

    if is_default:
        cam1_cam0_transform = transform_to_cam0_reference(cam0_transform, cam1_transform)
        # Handle default IMU calibration frame
        imu_cam0_transform = transform_to_cam0_reference(cam0_transform, imu_transform)
    else:
        cam1_cam0_transform = cam1_transform
        imu_cam0_transform = imu_transform

    # Cam1 relative to cam0
    cam1.rig_from_camera = transform_to_pose(cam1_cam0_transform.flatten().tolist())

    # Create IMU
    imu = cuvslam.ImuCalibration()
    imu.rig_from_imu = transform_to_pose(imu_cam0_transform.flatten().tolist())
    imu.gyroscope_noise_density = imu_config['gyroscope_noise_density']
    imu.gyroscope_random_walk = imu_config['gyroscope_random_walk']
    imu.accelerometer_noise_density = imu_config['accelerometer_noise_density']
    imu.accelerometer_random_walk = imu_config['accelerometer_random_walk']
    imu.frequency = imu_config['rate_hz']

    rig = cuvslam.Rig()
    rig.cameras = [cam0, cam1]
    rig.imus = [imu]

    return rig


def read_csv_data(csv_path: str, sensor_type: str) -> List[dict]:
    """Read timestamp and sensor data from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    valid_sensors = {'camera', 'imu'}
    if sensor_type not in valid_sensors:
        raise ValueError(f"Unknown sensor type: {sensor_type}. Must be one of {valid_sensors}")

    data = []
    with open(csv_path, 'r') as f:
        next(f)  # Skip header line
        reader = csv.reader(f)
        for row in reader:
            if not row:  # Skip empty rows
                continue
            if sensor_type == 'camera':
                if len(row) < 2:
                    raise ValueError(f"Invalid camera data format. Expected timestamp and filename, got: {row}")
                data.append({
                    'timestamp': int(row[0]),
                    'filename': row[1]
                })
            else:  # imu
                if len(row) < 7:
                    raise ValueError(f"Invalid IMU data format. Expected timestamp and 6 values, got: {row}")
                data.append({
                    'timestamp': int(row[0]),
                    'gyro': [float(x) for x in row[1:4]],
                    'accel': [float(x) for x in row[4:7]]
                })

    return data


def prepare_frame_metadata_euroc(euroc_path: str,
                                odometry_mode: cuvslam.Odometry.OdometryMode) -> List[dict]:
    """Process EuRoC dataset camera files and generate synchronized frame metadata."""
    if not os.path.exists(euroc_path):
        raise ValueError(f"EuRoC dataset path does not exist: {euroc_path}")

    left_cam_csv = os.path.join(euroc_path, 'cam0', 'data.csv')
    right_cam_csv = os.path.join(euroc_path, 'cam1', 'data.csv')

    if not os.path.exists(left_cam_csv) or not os.path.exists(right_cam_csv):
        raise ValueError(f"Camera data CSV files not found in {left_cam_csv} or {right_cam_csv}")

    left_cam_data = read_csv_data(left_cam_csv, 'camera')
    right_cam_data = read_csv_data(right_cam_csv, 'camera')

    # Verify timestamps match between cameras
    if len(left_cam_data) != len(right_cam_data):
        raise ValueError("Number of frames differs between left and right cameras")

    for i, (left_cam, right_cam) in enumerate(zip(left_cam_data, right_cam_data)):
        if left_cam['timestamp'] != right_cam['timestamp']:
            raise ValueError(f"Timestamp mismatch at frame {i}")

    # Generate stereo frame metadata
    frames_metadata = [{
        'type': 'stereo',
        'timestamp': left_cam['timestamp'],
        'images_paths': [
            os.path.join(euroc_path, 'cam0', 'data', left_cam['filename']),
            os.path.join(euroc_path, 'cam1', 'data', right_cam['filename']),
        ],
    } for left_cam, right_cam in zip(left_cam_data, right_cam_data)]

    # Add IMU data if in inertial mode
    if odometry_mode == cuvslam.Odometry.OdometryMode.Inertial:
        imu_csv = os.path.join(euroc_path, 'imu0', 'data.csv')
        if not os.path.exists(imu_csv):
            raise ValueError(f"IMU data CSV file not found in {imu_csv}")

        imu_data = read_csv_data(imu_csv, 'imu')
        frames_metadata.extend({
            'type': 'imu',
            'timestamp': measurement['timestamp'],
            'gyro': measurement['gyro'],
            'accel': measurement['accel']
        } for measurement in imu_data)

    # Sort frames by timestamp
    frames_metadata.sort(key=lambda x: x['timestamp'])

    return frames_metadata
