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

"""Resolution of the CUDA math libraries that PyCuVSLAM links against but does not bundle.

``libcuvslam.so`` links against cuBLAS, cuSOLVER, cuSPARSE and their dependencies. Wheels deliberately exclude them
during ``auditwheel repair`` (see ``scripts/build_pycuvslam_in_docker.sh``) because they are large, so they have to be
provided by the environment: either by a CUDA Toolkit installation, or by the ``nvidia-*`` pip packages that the
``cu12``/``cu13`` extras of this package declare.

Pip installs those packages into ``<site-packages>/nvidia/<component>/lib``, a location the dynamic loader does not
search. Loading them here by absolute path, before the extension module is imported, registers their sonames with the
loader so ``libcuvslam.so`` resolves against them without the caller having to set ``LD_LIBRARY_PATH``.
"""

import ctypes
import os
from glob import glob

# Components of the pip CUDA layout that PyCuVSLAM needs, ordered so that a library is loaded after the libraries it
# depends on (cuSOLVER needs cuBLAS and cuSPARSE, which in turn need nvJitLink). preload() does not rely on the order
# being exactly right, it retries, but a good order keeps the common case to a single pass.
CUDA_COMPONENTS = ('nvjitlink', 'cublas', 'cusparse', 'cusolver')

# Appended to the ImportError raised when the extension module cannot be loaded because of a missing library.
MISSING_LIBRARY_HINT = (
    "PyCuVSLAM links against the CUDA math libraries (cuBLAS, cuSOLVER, cuSPARSE) but does not bundle them.\n"
    "Provide them either by installing a CUDA Toolkit whose major version matches this wheel's cu12/cu13 tag, or by\n"
    "installing the matching pip packages:\n"
    "    pip install 'cuvslam[cu12]'   # CUDA 12 wheels\n"
    "    pip install 'cuvslam[cu13]'   # CUDA 13 wheels")


def nvidia_root(package_root=None):
    """Return the directory the nvidia-* pip packages install into, next to this package."""
    if package_root is None:
        package_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(package_root), 'nvidia')


def candidate_libraries(package_root=None):
    """Return the pip-provided CUDA libraries to preload, in dependency order."""
    root = nvidia_root(package_root)
    if not os.path.isdir(root):
        return []
    candidates = []
    for component in CUDA_COMPONENTS:
        candidates.extend(sorted(glob(os.path.join(root, component, 'lib', 'lib*.so.*'))))
    return candidates


def preload(package_root=None):
    """Load the pip-provided CUDA math libraries, if any, and return the paths that were loaded.

    Does nothing when the nvidia-* packages are not installed: the libraries are then expected to come from a system
    CUDA Toolkit, which the loader finds on its own.
    """
    pending = candidate_libraries(package_root)
    loaded = []
    # A library whose own dependencies are not loaded yet fails to load, so sweep the list until a full pass makes no
    # progress. Whatever is left unloaded is not reported here; the extension import below produces the actionable
    # error message.
    while pending:
        remaining = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                remaining.append(path)
            else:
                loaded.append(path)
        if len(remaining) == len(pending):
            break
        pending = remaining
    return loaded
