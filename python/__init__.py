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

"""cuVSLAM Python bindings."""

import warnings

from ._backend import load_backend

_core = load_backend()

# Import select bindings for the main namespace
get_version = _core.get_version
set_verbosity = _core.set_verbosity
warm_up_gpu = _core.warm_up_gpu
Pose = _core.Pose
Distortion = _core.Distortion
Camera = _core.Camera
ImuCalibration = _core.ImuCalibration
ImuMeasurement = _core.ImuMeasurement
Rig = _core.Rig
PoseStamped = _core.PoseStamped
PoseWithCovariance = _core.PoseWithCovariance
PoseEstimate = _core.PoseEstimate
Observation = _core.Observation
Landmark = _core.Landmark
Odometry = _core.Odometry
Slam = _core.Slam
Tracker = _core.Tracker

# Python helper functions for file-based config loading
from . import utils

# # Explicit exports for better IntelliSense
__all__ = [
    'Odometry',
    'Slam',
    'Tracker',
    'utils',
    'get_version',
    'set_verbosity',
    'warm_up_gpu',
    'Pose',
    'Distortion',
    'Camera',
    'ImuCalibration',
    'ImuMeasurement',
    'Rig',
    'PoseStamped',
    'PoseWithCovariance',
    'PoseEstimate',
    'Observation',
    'Landmark']

__version__ = get_version()[0].split('+')[0]


def __getattr__(name):
    if name == 'core':
        warnings.warn(
            'cuvslam.core is deprecated; use cuvslam.Odometry and cuvslam.Slam directly.',
            DeprecationWarning,
            stacklevel=2)
        return _core
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
