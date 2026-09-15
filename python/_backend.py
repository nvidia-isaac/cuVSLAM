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

"""Select the installed CUDA-specific cuVSLAM backend."""

from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
import os
from typing import Optional

_BACKEND_ENVIRONMENT_VARIABLE = "CUVSLAM_CUDA_MAJOR"
_CUDA_MAJORS = ("12", "13")


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False


def _available_backends() -> dict[str, str]:
    package = __package__ or "cuvslam"
    backends = {}

    # The direct module is used by source/editable installs. Published wheels put
    # native modules in CUDA-major-specific subpackages instead.
    direct_module = f"{package}.pycuvslam"
    if _module_exists(direct_module):
        backends["local"] = direct_module

    for major in _CUDA_MAJORS:
        module_name = f"{package}.cu{major}.pycuvslam"
        if _module_exists(module_name):
            backends[major] = module_name
    return backends


def _missing_backend_message(major: Optional[str] = None) -> str:
    if major is not None:
        return (
            f"cuVSLAM CUDA {major} backend is not installed. "
            f"Install it with `python -m pip install cuvslam-cu{major}`."
        )
    return (
        "No cuVSLAM CUDA backend is installed. Install `cuvslam-cu12` or "
        "`cuvslam-cu13`; `python -m pip install cuvslam` selects CUDA 13."
    )


@lru_cache(maxsize=1)
def load_backend():
    """Import and return the selected native binding module."""
    requested = os.environ.get(_BACKEND_ENVIRONMENT_VARIABLE)
    if requested is not None:
        requested = requested.removeprefix("cu")
        if requested not in _CUDA_MAJORS:
            raise ImportError(
                f"{_BACKEND_ENVIRONMENT_VARIABLE} must be 12 or 13, got {requested!r}."
            )

    backends = _available_backends()
    if requested is not None:
        module_name = backends.get(requested)
        if module_name is None:
            raise ImportError(_missing_backend_message(requested))
        return import_module(module_name)

    if "local" in backends:
        return import_module(backends["local"])

    installed = [major for major in _CUDA_MAJORS if major in backends]
    if not installed:
        raise ImportError(_missing_backend_message())
    if len(installed) > 1:
        raise ImportError(
            "Both cuVSLAM CUDA backends are installed. Set "
            f"{_BACKEND_ENVIRONMENT_VARIABLE}=12 or "
            f"{_BACKEND_ENVIRONMENT_VARIABLE}=13 before importing cuvslam."
        )
    return import_module(backends[installed[0]])
