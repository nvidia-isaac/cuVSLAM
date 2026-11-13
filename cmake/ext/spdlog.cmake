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

set(SPDLOG_VERSION "1.17.0")

# Configure spdlog options before fetching
set(SPDLOG_BUILD_SHARED OFF)
set(SPDLOG_BUILD_EXAMPLE OFF)
set(SPDLOG_BUILD_TESTS OFF)
set(SPDLOG_BUILD_BENCH OFF)
set(SPDLOG_FMT_EXTERNAL OFF)
set(SPDLOG_INSTALL OFF)

FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/v${SPDLOG_VERSION}.tar.gz
    URL_HASH SHA256=d8862955c6d74e5846b3f580b1605d2428b11d97a410d86e2fb13e857cd3a744
)

FetchContent_MakeAvailable(spdlog)
cuvslam_third_party_notice(NAME spdlog VERSION ${SPDLOG_VERSION}
    URL https://github.com/gabime/spdlog LICENSE MIT
    NOTE "Built with its bundled copy of {fmt} (SPDLOG_FMT_EXTERNAL=OFF), whose license follows spdlog's."
    FILES ${spdlog_SOURCE_DIR}/LICENSE ${spdlog_SOURCE_DIR}/include/spdlog/fmt/bundled/fmt.license.rst)

# spdlog provides these targets automatically:
# - spdlog::spdlog (modern CMake target)
# - spdlog::spdlog_header_only (if header-only mode)
