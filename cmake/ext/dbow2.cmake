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

include(FetchContent)

# Upstream has no release tag newer than v1.1 (2015); master HEAD is the maintained revision.
set(DBOW2_COMMIT "3924753db6145f12618e7de09b7e6b258db93c6e")

# DBoW2 stores descriptors as cv::Mat, so OpenCV is unavoidable. The library itself needs only the
# core module; the ORB extraction in libs/slam/vpr needs features2d and imgproc as well.
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d)
set(VPR_DBOW2_OPENCV_LIBS opencv_imgproc opencv_features2d CACHE INTERNAL "OpenCV modules the VPR DBoW2 backend needs")

# Download DBoW2 sources and ignore their CMakeLists.txt: it builds a SHARED library even with
# BUILD_SHARED_LIBS=OFF, copies demo/images into CMAKE_BINARY_DIR, inherits install() rules into the
# consuming project, and declares cmake_minimum_required(VERSION 3.0), which CMake 4 rejects.
FetchContent_Declare(
    dbow2
    GIT_REPOSITORY https://github.com/dorian3d/DBoW2.git
    GIT_TAG ${DBOW2_COMMIT}
)

FetchContent_GetProperties(dbow2)
if(NOT dbow2_POPULATED)
    FetchContent_Populate(dbow2)
endif()

# Build our own static library from the upstream sources. This is upstream's own source list;
# FSurf64.cpp is excluded there too, and it is not reachable from DBoW2.h.
add_library(dbow2-static STATIC
    ${dbow2_SOURCE_DIR}/src/BowVector.cpp
    ${dbow2_SOURCE_DIR}/src/FBrief.cpp
    ${dbow2_SOURCE_DIR}/src/FORB.cpp
    ${dbow2_SOURCE_DIR}/src/FeatureVector.cpp
    ${dbow2_SOURCE_DIR}/src/QueryResults.cpp
    ${dbow2_SOURCE_DIR}/src/ScoringObject.cpp
)

# SYSTEM so third-party headers do not trip cuVSLAM's warning flags. Only the parent include
# directory is exposed, so consumers write #include <DBoW2/DBoW2.h>.
target_include_directories(dbow2-static SYSTEM PUBLIC
    $<BUILD_INTERFACE:${dbow2_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# DBoW2's own .cpp files include their headers unprefixed (#include "BowVector.h") while the headers
# live in include/DBoW2/, so that directory has to be on the include path to compile them. PRIVATE,
# so consumers are not polluted with the unprefixed names.
target_include_directories(dbow2-static PRIVATE
    ${dbow2_SOURCE_DIR}/include/DBoW2
)

# PUBLIC: the DBoW2 headers expose cv::Mat in their templated API.
target_link_libraries(dbow2-static PUBLIC opencv_core)

set_target_properties(dbow2-static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

if(NOT TARGET dbow2::dbow2)
    add_library(dbow2::dbow2 ALIAS dbow2-static)
endif()
