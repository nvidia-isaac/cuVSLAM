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

# This file allows to find the third-party packages necessary for the build.
# FetchContent is preferred to download and build the dependencies.

include(GNUInstallDirs)

set(BUILD_SHARED_LIBS OFF)

# suppress warnings from FetchContent deps
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW) # option() honors normal variables
endif()
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Build third-party dependencies
include(${CMAKE_CURRENT_LIST_DIR}/ext/eigen.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/libjpeg.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/googletest.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/gflags.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/jsoncpp.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/spdlog.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/zlib.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/libpng.cmake) # depends on zlib
include(${CMAKE_CURRENT_LIST_DIR}/ext/cnpy.cmake) # depends on zlib
include(${CMAKE_CURRENT_LIST_DIR}/ext/circularbuffer.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/dense_hash_map.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/ext/yaml-cpp.cmake)
if(USE_LMDB)
    include(${CMAKE_CURRENT_LIST_DIR}/ext/lmdb.cmake)
endif()
if(USE_NVTX)
    include(${CMAKE_CURRENT_LIST_DIR}/ext/nvtx.cmake)
endif()
if(USE_DBOW2)
    include(${CMAKE_CURRENT_LIST_DIR}/ext/dbow2.cmake)
endif()
if(USE_ONNXRUNTIME)
    include(${CMAKE_CURRENT_LIST_DIR}/ext/onnxruntime.cmake)
endif()
if(USE_RERUN)
    include(${CMAKE_CURRENT_LIST_DIR}/ext/rerun.cmake) # downloads and builds Arrow automatically
endif()
if(USE_CERES)
    message(FATAL_ERROR "Ceres is not supported yet")
    # include(cmake/ext/ceres.cmake)
endif()

if(USE_CUNLS AND NOT USE_CUDA)
    message(FATAL_ERROR "USE_CUNLS requires USE_CUDA")
endif()

if (MSVC)
    # TODO: move to cuVSLAMUtils.cmake?
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE INTERNAL "hardcoded. do not change it. hide from user" FORCE)
    set(CMAKE_INSTALL_PREFIX "" CACHE INTERNAL "not used. hide from user" FORCE)

    find_library(SHLWAPI Shlwapi.lib)
endif()

if(USE_CUDA)
    # if(("${CUDAToolkit_ROOT}" STREQUAL ""))
    #     message(FATAL_ERROR "Please set CUDAToolkit_ROOT as cmake variable")
    # endif()
    find_package(CUDAToolkit REQUIRED)

    if(USE_CUNLS)
        include(${CMAKE_CURRENT_LIST_DIR}/ext/cunls.cmake)
    endif()

    # Print CUDA information
    message(STATUS "Found CUDA version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit target architecture directory: ${CUDAToolkit_TARGET_DIR}")
endif()
