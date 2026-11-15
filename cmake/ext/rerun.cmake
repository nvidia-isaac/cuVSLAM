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

set(RERUN_VERSION "0.33.1")

# Configure Rerun options before fetching
set(RERUN_DOWNLOAD_AND_BUILD_ARROW ON)
set(RERUN_ARROW_LINK_SHARED OFF)

FetchContent_Declare(
    rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/${RERUN_VERSION}/rerun_cpp_sdk.zip
    URL_HASH SHA256=f7caf9ac607ca75e9e2e543e30b4752480efc1933b05972026acc0e7032a1071
)

# Rerun automatically downloads and builds Arrow via download_and_build_arrow.cmake
FetchContent_MakeAvailable(rerun_sdk)
cuvslam_third_party_not_distributed(rerun_sdk
    "USE_RERUN development builds only; a package with it would also need Rerun's and Apache Arrow's notices")

# rerun_sdk automatically provides:
# - rerun_sdk
# - rerun_c (imported static library)
# - rerun_arrow_target (Arrow, downloaded and built automatically)

# Create modern namespaced target for consistency
if(NOT TARGET rerun::sdk)
    add_library(rerun::sdk ALIAS rerun_sdk)
endif()
