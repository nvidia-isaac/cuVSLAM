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

# Build cuNLS (CUDA nonlinear least squares) from source via FetchContent, like the other
# externals. BUILD_SHARED_LIBS is OFF in this build (see FindExt.cmake), so cuNLS produces a
# static archive with spdlog bundled in; consumers only need the CUDA toolkit math libraries at
# link time. cuNLS reuses cuVSLAM's already-declared spdlog and googletest FetchContent entries
# (declared earlier in FindExt.cmake). It downloads a prebuilt cuDSS archive at configure time for
# its headers only: libcudss.so is dlopen()ed at runtime, and only if the cuDSS solver is selected.

include(FetchContent)

if(NOT USE_CUDA)
    message(FATAL_ERROR "USE_CUNLS requires USE_CUDA")
endif()

if(CUDAToolkit_VERSION VERSION_LESS "12.6")
    message(FATAL_ERROR "USE_CUNLS requires CUDA >= 12.6 (found ${CUDAToolkit_VERSION}). "
                        "Either install CUDA 12.6+ or set -DUSE_CUNLS=OFF.")
endif()

set(CUNLS_VERSION "nightly-2026-10-06")

# Keep cuNLS's own tests and Python bindings out of this build.
set(BUILD_TESTING OFF)
set(BUILD_PYTHON_BINDINGS OFF)

# No DOWNLOAD_EXTRACT_TIMESTAMP here: the keyword does not exist on the cmake 3.22 shipped in the
# Jetson Orin base images (it gets parsed as part of the URL_HASH value and breaks the configure).
# On cmake >= 3.24 its absence only triggers a harmless one-time CMP0135 dev warning.
FetchContent_Declare(
    cunls
    URL https://github.com/nvidia-isaac/cuNLS/archive/refs/tags/${CUNLS_VERSION}.tar.gz
    URL_HASH SHA256=434f9a4f9affd96933bcd16591a9b7724e9226436a29381684b06198b8c080d0
)
FetchContent_MakeAvailable(cunls)

if(NOT TARGET cunls::cunls)
    message(FATAL_ERROR "cuNLS source build did not define the 'cunls::cunls' target")
endif()

message(STATUS "cuNLS: building from source (${CUNLS_VERSION})")
