# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Open Software License is intended to be used permissively and enable the
# further development of AI technologies. Subject to the terms of this License, NVIDIA confirms that you are free to
# commercially use, modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership to any
# outputs generated using the software or derivative works thereof. By using, reproducing, modifying, distributing,
# performing or displaying any portion or element of the software or derivative works thereof, you agree to be bound by
# this License.

"""cuVSLAM Python bindings."""

from . import _cuda_libs

# Make the CUDA math libraries this package links against, but does not bundle, loadable when they come from the
# nvidia-* pip packages instead of a CUDA Toolkit installation. No-op otherwise.
_cuda_libs.preload()

try:
    # Import all bindings under core namespace
    from . import pycuvslam as core
except ImportError as error:
    if 'cannot open shared object file' not in str(error):
        raise
    raise ImportError('{}\n\n{}'.format(error, _cuda_libs.MISSING_LIBRARY_HINT)) from error

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
    Landmark)

# Import the wrapper class
from .tracker import Tracker

# Python helper functions for file-based config loading
from . import utils

# # Explicit exports for better IntelliSense
__all__ = [
    'Tracker',
    'core',
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
