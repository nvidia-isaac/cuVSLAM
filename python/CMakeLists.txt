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

# Separate pycuvslam CMakeLists.txt to use with pip install -e .

# 3.26 for FindPython's Development.SABIModule; scikit-build-core fetches a newer CMake when the system one is older
cmake_minimum_required(VERSION 3.26)
project(pycuvslam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Without SKBUILD_SABI_COMPONENT (Development.SABIModule), nanobind silently drops STABLE_ABI while scikit-build-core
# still tags the wheel abi3.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)

set(CUVSLAM_BUILD_DIR_HELP "Use absolute path or shell expansion:\n"
"  CUVSLAM_BUILD_DIR=/absolute/path/to/build pip install -e .\n"
"  CUVSLAM_BUILD_DIR=$PWD/build pip install -e .")

if(NOT DEFINED ENV{CUVSLAM_BUILD_DIR})
    message(FATAL_ERROR
        "CUVSLAM_BUILD_DIR environment variable is not set.\n"
        ${CUVSLAM_BUILD_DIR_HELP})
endif()

set(CUVSLAM_BUILD_DIR_PATH "$ENV{CUVSLAM_BUILD_DIR}")

# Check if path is absolute (starts with /)
if(NOT IS_ABSOLUTE "${CUVSLAM_BUILD_DIR_PATH}")
    message(FATAL_ERROR
        "CUVSLAM_BUILD_DIR must be an absolute path.\n"
        ${CUVSLAM_BUILD_DIR_HELP})
endif()

message(STATUS "pycuvslam: Using local build from ${CUVSLAM_BUILD_DIR_PATH}")

set(CUVSLAM_LIBRARY "${CUVSLAM_BUILD_DIR_PATH}/bin/libcuvslam.so")
set(CUVSLAM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs)

# Verify the library exists
if(NOT EXISTS "${CUVSLAM_LIBRARY}")
    message(FATAL_ERROR
        "libcuvslam.so not found at: ${CUVSLAM_LIBRARY}."
        ${CUVSLAM_BUILD_DIR_HELP})
endif()

nanobind_add_module(_cuvslam STABLE_ABI NB_STATIC src/cuvslam2.cpp)
target_link_libraries(_cuvslam PRIVATE ${CUVSLAM_LIBRARY})
target_include_directories(_cuvslam PRIVATE ${CUVSLAM_INCLUDE_DIR})

# Set RPATH so _cuvslam.so can find libcuvslam.so in the same directory
set_target_properties(_cuvslam PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS _cuvslam LIBRARY DESTINATION cuvslam)
install(FILES ${CUVSLAM_LIBRARY} DESTINATION cuvslam)

# The AnyLoc engine builder and model for build_anyloc_engine(), when the build made them
set(ANYLOC_DIR "${CUVSLAM_BUILD_DIR_PATH}/anyloc")
if(EXISTS "${CUVSLAM_BUILD_DIR_PATH}/bin/anyloc_engine_builder")
    install(PROGRAMS "${CUVSLAM_BUILD_DIR_PATH}/bin/anyloc_engine_builder" DESTINATION cuvslam/anyloc)
    # The build's RPATH would outrank the LD_LIBRARY_PATH that build_anyloc_engine() sets
    install(CODE "file(RPATH_REMOVE FILE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/cuvslam/anyloc/anyloc_engine_builder\")")
    install(FILES "${ANYLOC_DIR}/anyloc_vits14_l9_value_322.onnx"
                  "${ANYLOC_DIR}/anyloc_vits14_l9_value_322_reference_input.npy"
                  "${ANYLOC_DIR}/anyloc_vits14_l9_value_322_reference_output.npy"
                  "${ANYLOC_DIR}/LICENSE.dinov2"
            DESTINATION cuvslam/anyloc)
endif()

# The C++ build's notices plus nanobind and its bundled tsl::robin_map, both compiled into _cuvslam (NB_STATIC)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ThirdPartyNotices.cmake)
set(CUVSLAM_NOTICES "${CUVSLAM_BUILD_DIR_PATH}/THIRD_PARTY_NOTICES.txt")
if(NOT EXISTS "${CUVSLAM_NOTICES}")
    message(FATAL_ERROR "${CUVSLAM_NOTICES} not found; configure the cuVSLAM build in ${CUVSLAM_BUILD_DIR_PATH} again.")
endif()
# A pip installed nanobind keeps its license in the dist-info, a source checkout at the top
file(GLOB NANOBIND_LICENSE
    "${NB_DIR}/LICENSE"
    "${NB_DIR}/../nanobind-${nanobind_VERSION}.dist-info/licenses/LICENSE"
    "${NB_DIR}/../nanobind-${nanobind_VERSION}.dist-info/LICENSE")
if(NOT NANOBIND_LICENSE)
    message(FATAL_ERROR "Cannot find the license of nanobind ${nanobind_VERSION} in or next to ${NB_DIR}")
endif()
list(GET NANOBIND_LICENSE 0 NANOBIND_LICENSE)
cuvslam_third_party_notice(NAME nanobind VERSION ${nanobind_VERSION}
    URL https://github.com/wjakob/nanobind LICENSE BSD-3-Clause
    FILES ${NANOBIND_LICENSE})
# The nanobind wheel ships robin_map's headers but not its LICENSE file; the license is the first comment of each
file(READ "${NB_DIR}/ext/robin_map/include/tsl/robin_map.h" ROBIN_MAP_HEADER)
string(FIND "${ROBIN_MAP_HEADER}" "*/" ROBIN_MAP_LICENSE_END)
string(SUBSTRING "${ROBIN_MAP_HEADER}" 0 ${ROBIN_MAP_LICENSE_END} ROBIN_MAP_LICENSE)
string(REGEX REPLACE "^/\\*\\*\n" "" ROBIN_MAP_LICENSE "${ROBIN_MAP_LICENSE}")
string(REGEX REPLACE "(^|\n) \\* ?" "\\1" ROBIN_MAP_LICENSE "${ROBIN_MAP_LICENSE}")
if(NOT ROBIN_MAP_LICENSE MATCHES "^MIT License")
    message(FATAL_ERROR "Unexpected license comment in ${NB_DIR}/ext/robin_map/include/tsl/robin_map.h")
endif()
cuvslam_third_party_notice(NAME tsl::robin_map VERSION "bundled with nanobind ${nanobind_VERSION}"
    URL https://github.com/Tessil/robin-map LICENSE MIT
    TEXT "${ROBIN_MAP_LICENSE}")
cuvslam_write_third_party_notices(${CMAKE_CURRENT_BINARY_DIR}/THIRD_PARTY_NOTICES.txt BASE ${CUVSLAM_NOTICES})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/THIRD_PARTY_NOTICES.txt DESTINATION cuvslam)

nanobind_add_stub(_cuvslam_stub
    MODULE _cuvslam
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_cuvslam.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_cuvslam>
    DEPENDS _cuvslam
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_cuvslam.pyi DESTINATION cuvslam)
