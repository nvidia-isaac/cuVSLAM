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

set(GFLAGS_VERSION "2.3.0")

# gflags needs to know it's a subproject to avoid install()
set(GFLAGS_IS_SUBPROJECT TRUE)

# Configure gflags options before fetching
set(INSTALL_HEADERS OFF)
set(INSTALL_SHARED_LIBS OFF)
set(INSTALL_STATIC_LIBS OFF)
set(BUILD_gflags_LIB ON)
set(BUILD_gflags_nothreads_LIB OFF)
set(BUILD_TESTING OFF)

FetchContent_Declare(
    gflags
    URL https://github.com/gflags/gflags/archive/v${GFLAGS_VERSION}.tar.gz
    URL_HASH SHA256=f619a51371f41c0ad6837b2a98af9d4643b3371015d873887f7e8d3237320b2f
)

FetchContent_MakeAvailable(gflags)
cuvslam_third_party_notice(NAME gflags VERSION ${GFLAGS_VERSION}
    URL https://github.com/gflags/gflags LICENSE BSD-3-Clause
    FILES ${gflags_SOURCE_DIR}/COPYING.txt)

# gflags provides these targets automatically:
# - gflags::gflags
