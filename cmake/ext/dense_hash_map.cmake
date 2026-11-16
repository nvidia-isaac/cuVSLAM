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

set(DENSE_HASH_MAP_COMMIT "74277fc4813028ae4a9e8d9176788eb8001177a6")

FetchContent_Declare(
    dense_hash_map
    GIT_REPOSITORY https://github.com/Jiwan/dense_hash_map.git
    GIT_TAG ${DENSE_HASH_MAP_COMMIT}
    GIT_SHALLOW TRUE
)

# Fetch without building (header-only library)
FetchContent_GetProperties(dense_hash_map)
if(NOT dense_hash_map_POPULATED)
    FetchContent_Populate(dense_hash_map)
    # Don't call add_subdirectory - we don't want their CMakeLists.txt
    # which tries to build tests and benchmarks
endif()
cuvslam_third_party_notice(NAME dense_hash_map VERSION ${DENSE_HASH_MAP_COMMIT}
    URL https://github.com/Jiwan/dense_hash_map LICENSE MIT
    FILES ${dense_hash_map_SOURCE_DIR}/LICENSE.md)

# Create interface target for the header-only library
if(NOT TARGET dense_hash_map::dense_hash_map)
    add_library(dense_hash_map::dense_hash_map INTERFACE IMPORTED GLOBAL)
    set_target_properties(dense_hash_map::dense_hash_map PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${dense_hash_map_SOURCE_DIR}/include"
    )
endif()
