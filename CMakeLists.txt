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

cmake_minimum_required(VERSION 3.19)

# Default to building for all CUDA architectures.
# On CMake < 3.23, "all" is not a valid CUDA_ARCHITECTURES value, so we fall back to
# passing -arch=all directly to nvcc via CUVSLAM_CUDA_ARCH_ALL.
# User override: -DCMAKE_CUDA_ARCHITECTURES=<arch> automatically disables the fallback.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "all" CACHE STRING "CUDA architectures to compile for (e.g. 87, 86, all)")
  endif()
else()
  # CMake < 3.23: detect whether user explicitly passed CMAKE_CUDA_ARCHITECTURES.
  # On first configure it won't be defined at all. On reconfigures it's in the cache
  # (type STRING, set by cmake). User -D flags arrive with type UNINITIALIZED.
  get_property(_cuda_arch_type CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY TYPE)
  if(_cuda_arch_type STREQUAL "UNINITIALIZED")
    # User explicitly passed -DCMAKE_CUDA_ARCHITECTURES=<arch>
    set(CUVSLAM_CUDA_ARCH_ALL FALSE)
  else()
    # Not set by user — use -arch=all fallback
    set(CUVSLAM_CUDA_ARCH_ALL TRUE)
  endif()
endif()

project(cuVSLAM)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CUVSLAM_MASTER_PROJECT ON)
else()
  set(CUVSLAM_MASTER_PROJECT OFF)
endif()

if (CUVSLAM_MASTER_PROJECT AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "cuVSLAM: CMAKE_BUILD_TYPE is not set, setting it to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Set output executable output path
# We want to do this so that all executable are in the same place and we can copy the dlls to that place
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(CUVSLAM_TOOLS_PYENV ${CMAKE_SOURCE_DIR}/.venv CACHE PATH "Path to python virtual env for tools")
set(CUVSLAM_DATASETS $ENV{HOME}/datasets CACHE PATH "Path to image datasets folder")
set(CUVSLAM_OUTPUT ${CMAKE_BINARY_DIR}/output CACHE PATH "Path to output folder")
set(CUDAToolkit_ROOT /usr/local/cuda CACHE PATH "Path to installed CUDA toolkit")
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc CACHE PATH "Path to installed CUDA compiler")

option(TREAT_WARNINGS_AS_ERRORS "Treat warnings as error" OFF)
option(USE_CUNLS "Build with cunls support." ON)
option(USE_CERES "Build with ceres support." OFF)
option(USE_NVTX "Build with nvtx tracing support." OFF)
option(USE_SLAM_OUTPUT "Build with option to enable debug output of SLAM" OFF)
option(USE_LMDB "Use lmdb library for the database of landmarks (map)" ON)
option(USE_CUDA "Use CUDA Toolkit" ON)
option(ENFORCE_GPU "Disable the usage of pure CPU version of cuVSLAM" ON)
option(CUVSLAM_LOG_ENABLE "Enable root cuvslam::Log." OFF)
option(SOF_USE_SMALLER_NCC "Use 3-pixel area for NCC. Possible it is useful for fisheye. By default, NCC uses 5 pixels." OFF)
option(CUVSLAM_BUILD_SHARED_LIB "Build shared library version of cuVSLAM" TRUE)
option(USE_RERUN "Use Rerun for visualization" OFF)

include(cmake/cuVSLAMUtils.cmake)
setup_cuvslam_settings()

include(cmake/FindExt.cmake)

# Add testing functionalities so that we can take advantage of GTest and CTest
enable_testing()

# CMake < 3.23 fallback (see CUVSLAM_CUDA_ARCH_ALL at the top of this file): "all" is not a valid
# CUDA_ARCHITECTURES value before 3.23, so build for every architecture by passing -arch=all to nvcc
# and disabling the per-target architecture list. Applied here, after FindExt(), so it does not
# affect cuNLS or other FetchContent dependencies (they set their own architectures), and before
# add_subdirectory() so every cuVSLAM CUDA target inherits it. Without this, targets other than
# cuda_kernels are left with an empty CUDA_ARCHITECTURES and CMake errors at generate time
# (CMP0104) on CMake < 3.23 builds that do not pass an explicit -DCMAKE_CUDA_ARCHITECTURES.
if(CUVSLAM_CUDA_ARCH_ALL)
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-arch=all>)
endif()

add_subdirectory(libs)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(doc)

# generate setup_env scripts
set(SET_ENV_BAT ${CMAKE_BINARY_DIR}/cuvslam_vars.sh)

file(WRITE ${SET_ENV_BAT}   "#!/bin/bash\n"
                            "echo Welcome to cuVSLAM Tools Environment!\n"
                            "[ -f ${CUVSLAM_TOOLS_PYENV}/bin/activate ] && source ${CUVSLAM_TOOLS_PYENV}/bin/activate\n"
                            "export CUVSLAM_BUILD_DIR=${CMAKE_BINARY_DIR}/\n"
                            "export CUVSLAM_DATASETS=${CUVSLAM_DATASETS}/\n"
                            "export CUVSLAM_OUTPUT=${CUVSLAM_OUTPUT}/\n")
