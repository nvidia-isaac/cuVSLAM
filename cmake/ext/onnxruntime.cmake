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

# ONNX Runtime ships prebuilt binaries per release; there is no source build here.
set(ONNXRUNTIME_VERSION "1.30.0")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(ONNXRUNTIME_ARCH "aarch64")
    set(ONNXRUNTIME_SHA256 "e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da")
else()
    set(ONNXRUNTIME_ARCH "x64")
    set(ONNXRUNTIME_SHA256 "a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd")
endif()

FetchContent_Declare(
    onnxruntime_prebuilt
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz
    URL_HASH SHA256=${ONNXRUNTIME_SHA256}
)

FetchContent_GetProperties(onnxruntime_prebuilt)
if(NOT onnxruntime_prebuilt_POPULATED)
    FetchContent_Populate(onnxruntime_prebuilt)
endif()

set(ONNXRUNTIME_LIBRARY
    "${onnxruntime_prebuilt_SOURCE_DIR}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}"
    CACHE INTERNAL "Full path of the ONNX Runtime shared library that has to ship next to libcuvslam.so")

# The lib/cmake/onnxruntime package config shipped inside the tarball is broken: it points at
# lib64/ and include/onnxruntime/, and the tarball has neither, so find_package(onnxruntime CONFIG)
# fails with "The installation package was faulty". Declare the imported target by hand.
if(NOT TARGET onnxruntime)
    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        IMPORTED_SONAME "libonnxruntime.so.1"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_prebuilt_SOURCE_DIR}/include")
endif()

if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime ALIAS onnxruntime)
endif()
