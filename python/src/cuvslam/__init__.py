# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Open Software License is intended to be used permissively and enable the
# further development of AI technologies. Subject to the terms of this License, NVIDIA confirms that you are free to
# commercially use, modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership to any
# outputs generated using the software or derivative works thereof. By using, reproducing, modifying, distributing,
# performing or displaying any portion or element of the software or derivative works thereof, you agree to be bound by
# this License.

"""cuVSLAM Python bindings."""

import contextlib
import warnings
from importlib import metadata as _metadata


def _warn_if_multiple_distributions():
    installed = []
    for name in ('cuvslam', 'cuvslam_cu12', 'cuvslam_cu13'):
        try:
            installed.append(_metadata.distribution(name).metadata['Name'])
        except _metadata.PackageNotFoundError:
            pass
    if len(installed) > 1:
        warnings.warn(
            f'Several cuVSLAM distributions are installed ({", ".join(installed)}). They install the same files, '
            'so uninstalling one breaks the others. Uninstall all of them, then install only one.',
            RuntimeWarning,
            stacklevel=2)


def _load_cuda_libraries():
    """Load the CUDA libraries from NVIDIA wheels (the [cuda] extra), which the dynamic loader does not search."""
    try:
        from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib
    except ImportError:
        return
    for name in ('cublas', 'cusparse', 'cusolver'):
        # cuda-pathfinder raises RuntimeError when it finds a library but cannot load it; the normal loader may still
        # find a usable system copy.
        with contextlib.suppress(DynamicLibNotFoundError, RuntimeError):
            load_nvidia_dynamic_lib(name)


_warn_if_multiple_distributions()
_load_cuda_libraries()

# Import select bindings for the main namespace
from ._cuvslam import (
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
from . import _cuvslam as _core

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
