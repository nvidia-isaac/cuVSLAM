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
# static archive with its own dependencies (spdlog, cuDSS) bundled in; consumers only need the
# CUDA toolkit math libraries at link time. cuNLS reuses cuVSLAM's already-declared spdlog and
# googletest FetchContent entries (declared earlier in FindExt.cmake) and downloads a prebuilt
# cuDSS archive at configure time.

include(FetchContent)

if(NOT USE_CUDA)
    message(FATAL_ERROR "USE_CUNLS requires USE_CUDA")
endif()

if(CUDAToolkit_VERSION VERSION_LESS "12.6")
    message(FATAL_ERROR "USE_CUNLS requires CUDA >= 12.6 (found ${CUDAToolkit_VERSION}). "
                        "Either install CUDA 12.6+ or set -DUSE_CUNLS=OFF.")
endif()

set(CUNLS_VERSION "Release_07_13_2026")

# Keep cuNLS's own tests and Python bindings out of this build.
set(BUILD_TESTING OFF)
set(BUILD_PYTHON_BINDINGS OFF)

# cuNLS declares cmake_minimum_required(VERSION 3.24), but the Ubuntu 22.04 base images used for
# Jetson Orin ship cmake 3.22.1. cuNLS does not actually use any 3.24-only feature (it sets an
# explicit CMAKE_CUDA_ARCHITECTURES list, not 'native'/'all-major', and only DOWNLOAD_EXTRACT_TIMESTAMP
# carries 3.24 semantics, which degrade harmlessly on 3.22), so we lower its floor to 3.22 at populate
# time. The sed is idempotent: re-running it on an already-patched tree is a no-op. Revisit this when
# bumping CUNLS_VERSION in case upstream starts relying on a genuine 3.24 feature.
# No DOWNLOAD_EXTRACT_TIMESTAMP here: the keyword does not exist on the cmake 3.22 shipped in the
# Jetson Orin base images (it gets parsed as part of the URL_HASH value and breaks the configure).
# On cmake >= 3.24 its absence only triggers a harmless one-time CMP0135 dev warning.
FetchContent_Declare(
    cunls
    URL https://github.com/nvidia-isaac/cuNLS/archive/refs/tags/${CUNLS_VERSION}.tar.gz
    URL_HASH SHA256=23b2917ae3903e6a688edb1652e40202d314527cd7fa9db68c762f0429375f77
    PATCH_COMMAND sed -i "s/cmake_minimum_required(VERSION 3.24)/cmake_minimum_required(VERSION 3.22)/" CMakeLists.txt
)
FetchContent_MakeAvailable(cunls)

if(NOT TARGET cunls)
    message(FATAL_ERROR "cuNLS source build did not define the 'cunls' target")
endif()

# cuNLS uses directory-scoped include_directories() and does not attach the include root to the
# 'cunls' target, so expose it here for consumers that #include <cunls/...>.
target_include_directories(cunls INTERFACE $<BUILD_INTERFACE:${cunls_SOURCE_DIR}>)

# cuNLS headers (pulled in transitively by e.g. libs/pnp) include CCCL/libcudacxx headers such as
# <cuda/std/array>. On some toolkit layouts (e.g. the Jetson Orin NGC image) these live under
# <toolkit-include>/cccl rather than directly under <toolkit-include>. cuNLS adds that path to its
# own directory-scoped include_directories() but does not propagate it on the 'cunls' target, so
# consumers compiling host (.cpp) translation units that pull in cuNLS headers cannot find them.
# Propagate the cccl path on the INTERFACE where it exists (no-op on flat toolkit layouts).
foreach(_cunls_cuda_inc ${CUDAToolkit_INCLUDE_DIRS})
    if(EXISTS "${_cunls_cuda_inc}/cccl")
        target_include_directories(cunls INTERFACE $<BUILD_INTERFACE:${_cunls_cuda_inc}/cccl>)
    endif()
endforeach()

# The rest of cuVSLAM links against the namespaced target; cuNLS only exports that name on install.
if(NOT TARGET cunls::cunls)
    add_library(cunls::cunls ALIAS cunls)
endif()

message(STATUS "cuNLS: building from source (${CUNLS_VERSION})")
