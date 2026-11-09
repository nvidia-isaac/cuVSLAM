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
import ctypes
import importlib.util
import os
import subprocess
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


def _load_tensorrt():
    """Load libnvinfer from the NVIDIA TensorRT wheel (the [tensorrt] extra), which VPR later opens by name."""
    # find_spec does not import tensorrt_libs, whose __init__ loads several GB of TensorRT builder libraries.
    spec = importlib.util.find_spec('tensorrt_libs')
    if spec is None:
        return
    for directory in spec.submodule_search_locations or []:
        with contextlib.suppress(OSError):
            ctypes.CDLL(os.path.join(directory, 'libnvinfer.so.10'))
            return


_warn_if_multiple_distributions()
_load_cuda_libraries()
_load_tensorrt()

# Import select bindings for the main namespace
from ._cuvslam import (
    get_version,
    is_cuda_build,
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
    Tracker,
    VprVocabularyBuilder)
from . import _cuvslam as _core

# Python helper functions for file-based config loading
from . import utils


def build_anyloc_engine(engine_path):
    """Build the AnyLoc TensorRT engine for this GPU and TensorRT with the builder and model this package ships."""
    anyloc = os.path.join(os.path.dirname(__file__), 'anyloc')
    model = os.path.join(anyloc, 'anyloc_vits14_l9_value_322')
    env = dict(os.environ)
    spec = importlib.util.find_spec('tensorrt_libs')
    if spec is not None:
        # The TensorRT of the [tensorrt] extra, which VPR runs with too
        env['LD_LIBRARY_PATH'] = os.pathsep.join(
            [*(spec.submodule_search_locations or []), *filter(None, [env.get('LD_LIBRARY_PATH')])])
    subprocess.run([os.path.join(anyloc, 'anyloc_engine_builder'), f'--onnx={model}.onnx', f'--engine={engine_path}',
                    f'--reference_input={model}_reference_input.npy',
                    f'--reference_output={model}_reference_output.npy'], check=True, env=env)


# # Explicit exports for better IntelliSense
__all__ = [
    'Odometry',
    'Slam',
    'Tracker',
    'VprVocabularyBuilder',
    'utils',
    'build_anyloc_engine',
    'get_version',
    'is_cuda_build',
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
