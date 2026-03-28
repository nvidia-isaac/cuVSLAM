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
option(USE_CUNLS "Build with cunls support." OFF)
option(USE_CERES "Build with ceres support." OFF)
option(USE_NVTX "Build with nvtx tracing support." OFF)
option(USE_SLAM_OUTPUT "Build with option to enable debug output of SLAM" OFF)
option(USE_LMDB "Use lmdb library for the database of landmarks (map)" ON)
option(USE_CUDA "Use CUDA Toolkit" ON)
option(ENFORCE_GPU "Disable the usage of pure CPU version of cuVSLAM" ON)
option(CUVSLAM_LOG_ENABLE "Enable root cuvslam::Log." OFF)
option(SOF_USE_SMALLER_NCC "Use 3-pixel area for NCC. Possible it is useful for fisheye. By default, NCC uses 5 pixels." OFF)
option(DECREASE_RANSAC_AREA "Do not use 10% tracks for RANSAC near the image border. Possible useful for fisheye. The Default is to use all tracks." OFF)
option(CUVSLAM_BUILD_SHARED_LIB "Build shared library version of cuVSLAM" TRUE)
option(USE_RERUN "Use Rerun for visualization" OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "all" CACHE STRING "CUDA architectures to compile for (e.g. 87, 86, all)")
endif()

include(cmake/cuVSLAMUtils.cmake)
setup_cuvslam_settings()

include(cmake/FindExt.cmake)

# Add testing functionalities so that we can take advantage of GTest and CTest
enable_testing()

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
