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

set(CIRCULARBUFFER_COMMIT "cef66805cb5424e27300a966becc7c2678117c27")

FetchContent_Declare(
    circularbuffer
    GIT_REPOSITORY https://github.com/vinitjames/circularbuffer.git
    GIT_TAG ${CIRCULARBUFFER_COMMIT}
)

# Fetch without building (header-only library)
FetchContent_GetProperties(circularbuffer)
if(NOT circularbuffer_POPULATED)
    FetchContent_Populate(circularbuffer)
    # Don't call add_subdirectory - we don't want their CMakeLists.txt
    # which tries to build tests with ExternalProject_Add
endif()
cuvslam_third_party_notice(NAME circularbuffer VERSION ${CIRCULARBUFFER_COMMIT}
    URL https://github.com/vinitjames/circularbuffer LICENSE MIT
    FILES ${circularbuffer_SOURCE_DIR}/LICENSE)

# Create interface target for the header-only library
if(NOT TARGET circularbuffer::circularbuffer)
    add_library(circularbuffer::circularbuffer INTERFACE IMPORTED GLOBAL)
    set_target_properties(circularbuffer::circularbuffer PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${circularbuffer_SOURCE_DIR}"
    )
endif()
