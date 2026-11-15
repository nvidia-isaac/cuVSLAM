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

set(LIBJPEG_VERSION "3.1.3")
set(LIBJPEG_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/_install/libjpeg-turbo-${LIBJPEG_VERSION})
set(LIBJPEG_LIBRARY ${LIBJPEG_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}jpeg${CMAKE_STATIC_LIBRARY_SUFFIX})

FetchContent_Declare(
    libjpeg
    URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_VERSION}/libjpeg-turbo-${LIBJPEG_VERSION}.tar.gz
    URL_HASH SHA256=075920b826834ac4ddf97661cc73491047855859affd671d52079c6867c1c6c0
)

FetchContent_GetProperties(libjpeg)
if(NOT libjpeg_POPULATED)
    FetchContent_Populate(libjpeg)
endif()
cuvslam_third_party_notice(NAME libjpeg-turbo VERSION ${LIBJPEG_VERSION}
    URL https://github.com/libjpeg-turbo/libjpeg-turbo LICENSE "IJG AND BSD-3-Clause AND Zlib"
    NOTE "This software is based in part on the work of the Independent JPEG Group."
    FILES ${libjpeg_SOURCE_DIR}/LICENSE.md ${libjpeg_SOURCE_DIR}/README.ijg)

# Pre-create directories so CMake does not reject imported target include paths at configure time.
file(MAKE_DIRECTORY ${LIBJPEG_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${LIBJPEG_INSTALL_DIR}/lib)

# libjpeg-turbo rejects add_subdirectory(), so FetchContent only fetches the source. Build it with a standalone
# CMake invocation and expose the installed archive through an imported target.
if(NOT TARGET libjpeg_external)
    add_custom_command(
        OUTPUT ${LIBJPEG_LIBRARY}
        COMMAND
            ${CMAKE_COMMAND}
            -S ${libjpeg_SOURCE_DIR}
            -B ${libjpeg_BINARY_DIR}
            -DCMAKE_INSTALL_PREFIX=${LIBJPEG_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DENABLE_SHARED=OFF
            -DENABLE_STATIC=ON
            -DWITH_SIMD=OFF
            -DWITH_TOOLS=OFF
            -DWITH_TESTS=OFF
            -DWITH_TURBOJPEG=OFF
        COMMAND
            ${CMAKE_COMMAND}
            --build ${libjpeg_BINARY_DIR}
            --target install
        COMMENT "Building libjpeg-turbo ${LIBJPEG_VERSION}"
        VERBATIM
    )

    add_custom_target(libjpeg_external DEPENDS ${LIBJPEG_LIBRARY})
endif()

# Create modern namespaced target for consistency.
if(NOT TARGET jpeg::jpeg)
    add_library(jpeg::jpeg STATIC IMPORTED GLOBAL)
    set_target_properties(jpeg::jpeg PROPERTIES
        IMPORTED_LOCATION ${LIBJPEG_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${LIBJPEG_INSTALL_DIR}/include
    )
    add_dependencies(jpeg::jpeg libjpeg_external)
endif()
