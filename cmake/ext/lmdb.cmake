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

set(LMDB_VERSION "0.9.31")

FetchContent_Declare(
    lmdb
    URL https://github.com/LMDB/lmdb/archive/refs/tags/LMDB_${LMDB_VERSION}.tar.gz
    URL_HASH SHA256=dd70a8c67807b3b8532b3e987b0a4e998962ecc28643e1af5ec77696b081c9b0
)

# Fetch without building - LMDB doesn't have CMake, we'll build manually
FetchContent_GetProperties(lmdb)
if(NOT lmdb_POPULATED)
    FetchContent_Populate(lmdb)

    # LMDB has no CMake, need to build the library manually
    set(LMDB_SOURCE_DIR ${lmdb_SOURCE_DIR}/libraries/liblmdb)

    add_library(lmdb_build STATIC
        ${LMDB_SOURCE_DIR}/mdb.c
        ${LMDB_SOURCE_DIR}/midl.c
    )

    target_include_directories(lmdb_build PUBLIC
        ${LMDB_SOURCE_DIR}
    )

    find_package(Threads REQUIRED)
    target_link_libraries(lmdb_build PRIVATE Threads::Threads)
endif()
cuvslam_third_party_notice(NAME LMDB VERSION ${LMDB_VERSION}
    URL https://github.com/LMDB/lmdb LICENSE OLDAP-2.8
    FILES ${lmdb_SOURCE_DIR}/libraries/liblmdb/COPYRIGHT ${lmdb_SOURCE_DIR}/libraries/liblmdb/LICENSE)

# Create modern namespaced target
if(NOT TARGET lmdb::lmdb)
    add_library(lmdb::lmdb ALIAS lmdb_build)
endif()
