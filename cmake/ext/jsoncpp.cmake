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

set(JSONCPP_VERSION "1.9.6")

# Configure jsoncpp options before fetching
set(JSONCPP_WITH_TESTS OFF)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF)
set(JSONCPP_WITH_PKGCONFIG_SUPPORT OFF)
set(BUILD_OBJECT_LIBS OFF)

FetchContent_Declare(
    jsoncpp
    URL https://github.com/open-source-parsers/jsoncpp/archive/${JSONCPP_VERSION}.tar.gz
    URL_HASH SHA256=f93b6dd7ce796b13d02c108bc9f79812245a82e577581c4c9aabe57075c90ea2
)

# Make available (download and build)
FetchContent_MakeAvailable(jsoncpp)
cuvslam_third_party_notice(NAME JsonCpp VERSION ${JSONCPP_VERSION}
    URL https://github.com/open-source-parsers/jsoncpp LICENSE MIT
    FILES ${jsoncpp_SOURCE_DIR}/LICENSE)

# jsoncpp provides these targets automatically:
# - jsoncpp_static
# - jsoncpp_lib (shared library)

# Create modern namespaced target for consistency
if(NOT TARGET jsoncpp::jsoncpp)
    add_library(jsoncpp::jsoncpp ALIAS jsoncpp_static)
endif()
