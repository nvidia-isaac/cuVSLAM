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

set(ZLIB_VERSION "1.3.1")

# Configure zlib options before fetching
set(ZLIB_BUILD_EXAMPLES OFF)
set(SKIP_INSTALL_ALL ON)

FetchContent_Declare(
    zlib
    URL https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VERSION}.tar.gz
    URL_HASH SHA256=17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c
)

# Make available (download and build)
FetchContent_MakeAvailable(zlib)
cuvslam_third_party_notice(NAME zlib VERSION ${ZLIB_VERSION}
    URL https://github.com/madler/zlib LICENSE Zlib
    FILES ${zlib_SOURCE_DIR}/LICENSE)

# zlib provides these targets:
# - zlib (shared library)
# - zlibstatic (static library)

# Create modern namespaced target for consistency
if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
endif()
