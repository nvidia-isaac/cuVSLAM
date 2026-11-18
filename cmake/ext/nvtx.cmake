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

set(NVTX_VERSION "3.4.0")

FetchContent_Declare(
    nvtx
    URL https://github.com/NVIDIA/NVTX/archive/refs/tags/v${NVTX_VERSION}.tar.gz
    URL_HASH SHA256=99a3e97d7fe90d5195e87256492bf9cd42476d72cbc79ba477011a2384b88f92
)

FetchContent_MakeAvailable(nvtx)
cuvslam_third_party_notice(NAME NVTX VERSION ${NVTX_VERSION}
    URL https://github.com/NVIDIA/NVTX LICENSE "Apache-2.0 WITH LLVM-exception"
    FILES ${nvtx_SOURCE_DIR}/LICENSE.txt)

# Create interface library (header-only)
if(NOT TARGET NVTX::nvtx3)
    add_library(NVTX::nvtx3 INTERFACE IMPORTED GLOBAL)
    target_include_directories(NVTX::nvtx3 INTERFACE
        ${nvtx_SOURCE_DIR}/c/include
    )
endif()
