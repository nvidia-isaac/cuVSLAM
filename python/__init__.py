# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Open Software License is intended to be used permissively and enable the
# further development of AI technologies. Subject to the terms of this License, NVIDIA confirms that you are free to
# commercially use, modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership to any
# outputs generated using the software or derivative works thereof. By using, reproducing, modifying, distributing,
# performing or displaying any portion or element of the software or derivative works thereof, you agree to be bound by
# this License.

"""cuVSLAM Python bindings."""

import warnings

# Import select bindings for the main namespace
from .pycuvslam import (
    get_version,
    set_verbosity,
    warm_up_gpu,
    Pose,
    Distortion,
    Camera,
    ImuCalibration,
    ImuMeasurement,
    Rig,
    PoseStamped,
    PoseWithCovariance,
    PoseEstimate,
    Observation,
    Landmark,
    Odometry,
    Slam,
    Tracker)
from . import pycuvslam as _core

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
