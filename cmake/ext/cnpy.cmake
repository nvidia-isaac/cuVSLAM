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

set(CNPY_COMMIT "4e8810b1a8637695171ed346ce68f6984e585ef4")

# cnpy depends on zlib - ensure it's available
if(NOT TARGET ZLIB::ZLIB)
    include(cmake/ext/zlib.cmake)
endif()

# Download cnpy source (ignore their CMakeLists.txt)
FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG ${CNPY_COMMIT}
)

FetchContent_GetProperties(cnpy)
if(NOT cnpy_POPULATED)
    FetchContent_Populate(cnpy)
endif()
cuvslam_third_party_notice(NAME cnpy VERSION ${CNPY_COMMIT}
    URL https://github.com/rogersce/cnpy LICENSE MIT
    FILES ${cnpy_SOURCE_DIR}/LICENSE)

# Create our own static library target directly from sources
# cnpy is just cnpy.h and cnpy.cpp in the root directory
add_library(cnpy-static STATIC
    ${cnpy_SOURCE_DIR}/cnpy.cpp
)

target_include_directories(cnpy-static PUBLIC
    $<BUILD_INTERFACE:${cnpy_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cnpy-static PUBLIC ZLIB::ZLIB)

# Create modern namespaced target for consistency
if(NOT TARGET cnpy::cnpy)
    add_library(cnpy::cnpy ALIAS cnpy-static)
endif()
