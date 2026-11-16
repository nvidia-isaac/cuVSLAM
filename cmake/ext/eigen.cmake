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

set(EIGEN_VERSION "3.4.1")

# Configure Eigen options before fetching
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_MPL2_ONLY ON)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz
    URL_HASH SHA256=b93c667d1b69265cdb4d9f30ec21f8facbbe8b307cf34c0b9942834c6d4fdbe2
)

FetchContent_MakeAvailable(eigen)
# MPL-2.0 section 3.2: binary recipients are told where the source is
cuvslam_third_party_notice(NAME Eigen VERSION ${EIGEN_VERSION}
    URL https://gitlab.com/libeigen/eigen LICENSE "MPL-2.0 AND Apache-2.0 AND BSD-3-Clause"
    NOTE "Used unmodified as header-only C++ templates, built with EIGEN_MPL2_ONLY. The source code of\n\
this version is available at\nhttps://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz"
    FILES ${eigen_SOURCE_DIR}/COPYING.MPL2 ${eigen_SOURCE_DIR}/COPYING.APACHE ${eigen_SOURCE_DIR}/COPYING.BSD)

# Eigen automatically provides these targets:
# - Eigen3::Eigen (modern CMake target, alias to 'eigen')

# Mark Eigen headers as SYSTEM to suppress warnings from third-party code
# Note: Eigen3::Eigen is an ALIAS, so we need to set properties on the actual target 'eigen'
get_target_property(EIGEN_INCLUDE_DIRS eigen INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(eigen PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIRS}"
)
