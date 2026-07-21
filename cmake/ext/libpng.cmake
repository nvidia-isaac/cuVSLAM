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

set(LIBPNG_VERSION "1.6.55")

# libpng depends on zlib - ensure it's available
if(NOT TARGET ZLIB::ZLIB)
    include(cmake/ext/zlib.cmake)
endif()

# Configure libpng options before fetching
set(PNG_SHARED OFF)
set(PNG_STATIC ON)
set(PNG_TESTS OFF)
set(PNG_BUILD_ZLIB OFF)
set(SKIP_INSTALL_ALL ON)

# # Tell libpng where to find zlib
set(ZLIB_LIBRARY ZLIB::ZLIB)
set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})

FetchContent_Declare(
    libpng
    URL https://github.com/pnggroup/libpng/archive/refs/tags/v${LIBPNG_VERSION}.tar.gz
    URL_HASH SHA256=71a2c5b1218f60c4c6d2f1954c7eb20132156cae90bdb90b566c24db002782a6
)

FetchContent_MakeAvailable(libpng)

# libpng provides these targets:
# - png_static

# Ensure png_static properly links against ZLIB::ZLIB
target_link_libraries(png_static PUBLIC ZLIB::ZLIB)

# Create modern namespaced target for consistency
if(NOT TARGET PNG::PNG)
    add_library(PNG::PNG ALIAS png_static)
endif()
