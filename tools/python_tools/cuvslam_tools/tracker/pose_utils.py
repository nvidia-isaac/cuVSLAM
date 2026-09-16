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

"""Pose conversion helpers that do not import the cuVSLAM Python binding."""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_transform(pose: Any) -> np.ndarray:
    """Convert a cuVSLAM-like pose or an existing matrix to a 4x4 transform."""
    if isinstance(pose, np.ndarray):
        transform = np.asarray(pose, dtype=float)
        if transform.shape == (3, 4):
            transform = np.vstack((transform, [0.0, 0.0, 0.0, 1.0]))
        if transform.shape != (4, 4):
            raise ValueError(f"Pose transform must be 3x4 or 4x4, got {transform.shape}")
        return transform

    rotation = np.asarray(pose.rotation, dtype=float)
    translation = np.asarray(pose.translation, dtype=float)
    if rotation.shape != (4,) or translation.shape != (3,):
        raise ValueError("Pose must contain an xyzw quaternion and a 3-vector translation")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(rotation).as_matrix()
    transform[:3, 3] = translation
    return transform


def json_pose_to_transform(value: dict) -> np.ndarray:
    """Convert a launcher's JSON pose object into a finite 4x4 transform."""
    class _Pose:
        rotation = value.get("rotation")
        translation = value.get("translation")

    transform = pose_to_transform(_Pose())
    if not np.isfinite(transform).all():
        raise ValueError("Launcher pose contains a non-finite value")
    return transform
