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

"""Utility functions for converting between different data formats and types."""

from typing import List, Union, Any
import numpy as np
import argparse
from scipy.spatial.transform import Rotation
import cuvslam as vslam


def to_str(obj: Any, prefix: str = "  ") -> str:
    """Convert object attributes to formatted string.

    Args:
        obj: Object to convert
        prefix: String prefix for each line

    Returns:
        Formatted string of object attributes
    """
    attributes = [
        attr for attr in dir(obj)
        if not callable(getattr(obj, attr)) and not attr.startswith("__")
    ]
    return "\n".join(f"{prefix}{attr}: {getattr(obj, attr)}" for attr in attributes)


def str2bool(value: Union[str, bool]) -> bool:
    """Convert string to boolean.

    Args:
        value: String or boolean value to convert

    Returns:
        Boolean value

    Raises:
        ArgumentTypeError: If value cannot be converted to boolean
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        "Boolean value expected (true/false, 1/0, yes/no, y/n)."
    )


def str2multicam_mode(value: str) -> vslam.Odometry.MulticameraMode:
    """Convert string to multicamera mode enum."""
    str2multicam_mode_map = {
        'performance': vslam.Odometry.MulticameraMode.Performance,
        'precision': vslam.Odometry.MulticameraMode.Precision,
        'moderate': vslam.Odometry.MulticameraMode.Moderate
    }
    return str2multicam_mode_map[value]


def str2odometry_mode(value: str) -> vslam.Odometry.OdometryMode:
    """Convert string to odometry mode enum."""
    str2odometry_mode_map = {
        'mono': vslam.Odometry.OdometryMode.Mono,
        'multicamera': vslam.Odometry.OdometryMode.Multicamera,
        'inertial': vslam.Odometry.OdometryMode.Inertial,
        'rgbd': vslam.Odometry.OdometryMode.RGBD
    }
    return str2odometry_mode_map[value]


def to_distortion_model(distortion: str) -> vslam.Distortion.Model:
    """Convert string to distortion model enum."""
    distortion = distortion.lower()
    if distortion == 'pinhole':
        return vslam.Distortion.Model.Pinhole
    if distortion == 'fisheye':
        return vslam.Distortion.Model.Fisheye
    if distortion == 'brown':
        return vslam.Distortion.Model.Brown
    if distortion == 'brown5k':
        return vslam.Distortion.Model.Brown
    if distortion == 'polynomial':
        return vslam.Distortion.Model.Polynomial
    raise ValueError(f"Unknown distortion model: {distortion}")


def rotation_matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion."""
    return Rotation.from_matrix(rotation).as_quat()


def rotation_quat_to_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    return Rotation.from_quat(rotation).as_matrix()


def pose_to_transform(pose: vslam.Pose) -> np.ndarray:
    """Convert pose to transformation matrix."""
    rotation = rotation_quat_to_matrix(pose.rotation)
    translation = pose.translation.reshape(3, 1)
    transform = np.vstack((np.column_stack((rotation, translation)), [0, 0, 0, 1]))
    return transform

def transform_to_pose(
    transform_3x4: List[List[float]],
    opengl_to_opencv: bool = True
) -> vslam.Pose:
    """Convert transformation matrix to pose."""
    transform = np.array(transform_3x4)
    if transform.shape != (3, 4) and transform.shape != (4, 4):
        raise ValueError(f"Transform should be 3x4 or 4x4, got {transform.shape}")

    if opengl_to_opencv:
        transform = opengl_to_opencv_transform(transform)

    return vslam.Pose(
        rotation=rotation_matrix_to_quat(transform[:3, :3]),
        translation=transform[:3, 3]
    )


# Coordinate system conversions
def opengl_to_opencv_transform(transform_matrix: np.ndarray) -> np.ndarray:
    """Convert transformation matrix from OpenGL to OpenCV coordinates.

    Args:
        transform_matrix: 3x4 or 4x4 transformation matrix in OpenGL coordinates

    Returns:
        Transformed matrix in OpenCV coordinates
    """
    conversion = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    transform_4x4 = (
        np.vstack([transform_matrix, [0, 0, 0, 1]])
        if transform_matrix.shape == (3, 4)
        else transform_matrix
    )

    result = conversion @ transform_4x4 @ np.linalg.inv(conversion)
    return result[:3, :] if transform_matrix.shape == (3, 4) else result


def opencv_to_opengl_transform(transform_matrix: np.ndarray) -> np.ndarray:
    """Convert transformation matrix from OpenCV to OpenGL coordinates."""
    # The conversion is symmetric, so we can reuse the same function
    return opengl_to_opencv_transform(transform_matrix)


def pose_opengl_to_opencv(pose: vslam.Pose) -> vslam.Pose:
    """Convert pose from OpenGL to OpenCV coordinates."""
    rotation_matrix = rotation_quat_to_matrix(pose.rotation)
    transform_3x4 = np.hstack([rotation_matrix, pose.translation.reshape(3, 1)])
    result_3x4 = opengl_to_opencv_transform(transform_3x4)

    return vslam.Pose(
        rotation=rotation_matrix_to_quat(result_3x4[:3, :3]),
        translation=result_3x4[:3, 3]
    )


def pose_opencv_to_opengl(pose: vslam.Pose) -> vslam.Pose:
    """Convert pose from OpenCV to OpenGL coordinates."""
    rotation_matrix = rotation_quat_to_matrix(pose.rotation)
    transform_3x4 = np.hstack([rotation_matrix, pose.translation.reshape(3, 1)])
    result_3x4 = opencv_to_opengl_transform(transform_3x4)

    return vslam.Pose(
        rotation=rotation_matrix_to_quat(result_3x4[:3, :3]),
        translation=result_3x4[:3, 3]
    )
