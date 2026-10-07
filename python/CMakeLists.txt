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

cmake_minimum_required(VERSION 3.19)
project(pycuvslam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
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

nanobind_add_module(pycuvslam STABLE_ABI NB_STATIC cuvslam2.cpp)
target_link_libraries(pycuvslam PRIVATE ${CUVSLAM_LIBRARY})
target_include_directories(pycuvslam PRIVATE ${CUVSLAM_INCLUDE_DIR})

# Set RPATH so pycuvslam.so can find libcuvslam.so in the same directory
set_target_properties(pycuvslam PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS pycuvslam LIBRARY DESTINATION .)
install(FILES ${CUVSLAM_LIBRARY} DESTINATION .)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
    DESTINATION .
)

nanobind_add_stub(pycuvslam_stub
    MODULE pycuvslam
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pycuvslam.pyi
    MARKER_FILE ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    PYTHON_PATH $<TARGET_FILE_DIR:pycuvslam>
    DEPENDS pycuvslam
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/pycuvslam.pyi
    ${CMAKE_CURRENT_BINARY_DIR}/py.typed
    DESTINATION .
)

add_subdirectory(test)
