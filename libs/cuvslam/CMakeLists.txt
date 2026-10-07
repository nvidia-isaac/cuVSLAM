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

set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)

add_custom_target(
    version_info ALL
    BYPRODUCTS ${GENERATED_DIR}/version.h
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DOUTPUT_FILE=${GENERATED_DIR}/version.h
        -P ${CMAKE_SOURCE_DIR}/cmake/UpdateVersion.cmake
    COMMENT "Updating version header"
)

if(CUVSLAM_BUILD_SHARED_LIB)
    set(LIB_TYPE SHARED)
else()
    set(LIB_TYPE STATIC)
endif()

set(INCLUDE_DIRS
    ${GENERATED_DIR}
    )

set(LIBS
    slam
    camera
    sof
    sba
    common
    pnp
    epipolar
    cuvslam_math
    log
    profiler
    odometry
    pipelines
    map
    imu
    cnpy::cnpy
    )

if(USE_CUDA)
    set(LIBS ${LIBS} cuda_modules)
endif()

if(USE_CUNLS)
    set(LIBS ${LIBS} cunls::cunls)
endif()

if(USE_RERUN)
    set(LIBS ${LIBS} visualizer)
endif()

set(SOURCES
    cuvslam2.cpp
    cuvslam2.h
    cuvslam_gpu.cpp
    cuvslam_gpu.h
    internal.h
    debug_dump.cpp
    debug_dump.h
    ground_constraint2.cpp
    ground_constraint2.h
    tracker.cpp
    version.cpp
    )

add_library(cuvslam ${LIB_TYPE} ${SOURCES})
target_include_directories(cuvslam PUBLIC ${INCLUDE_DIRS})
target_link_libraries(cuvslam ${LIBS})
add_dependencies(cuvslam version_info)
set_target_properties(cuvslam PROPERTIES
    OUTPUT_NAME cuvslam
    LIBRARY_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
)

if(USE_CUDA)
    set_target_properties(cuvslam PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# interface target for projects that link cuvslam libraries but not cuvslam top-level target
add_library(cuvslam_libs INTERFACE)
target_link_libraries(cuvslam_libs INTERFACE ${LIBS})

add_subdirectory(test)
