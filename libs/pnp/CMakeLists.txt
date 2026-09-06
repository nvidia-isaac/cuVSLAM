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

set(MODULE_NAME "pnp")

set(HEADERS
    mono_pnp.h
    multicam_pnp.h
    )

set(SOURCES
    mono_pnp.cpp
    multicam_pnp.cpp
    )


set(LIBS
    camera
    common
    cuvslam_math
    epipolar
    params
    profiler
)

if(USE_CUDA)
    set(HEADERS ${HEADERS} visual_icp.h)
    set(SOURCES ${SOURCES} visual_icp.cpp lift_tracks.cpp)
    set(LIBS ${LIBS} cuda_modules)
endif()

if(USE_CUNLS)
    enable_language(CUDA)
    set(HEADERS ${HEADERS} multisensor_pose_estimator.h)
    # Host-only source (no device code); built by the host compiler. Not LANGUAGE CUDA — see the
    # note in libs/map/CMakeLists.txt (nvcc cannot parse Eigen's ARM NEON headers on aarch64).
    set(SOURCES ${SOURCES} multisensor_pose_estimator.cpp)
    # PlaneMap / DepthPointMap / DepthCameraInfo / Plane now live in libs/map/ (under USE_CUNLS),
    # but multisensor_pose_estimator.h still references map::Plane in its public API and the
    # solver pulls in map/plane_map.h, so the pnp lib gains a map dep on this branch.
    set(LIBS ${LIBS} cunls::cunls cuda_modules imu map)
endif()

if(USE_RERUN)
    set(HEADERS ${HEADERS} visualizer.h)
    set(SOURCES ${SOURCES} visualizer.cpp)
    set(LIBS ${LIBS} visualizer)
endif()

setup_staticlib_project(
        MODULE_NAME ${MODULE_NAME}
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES ${LIBS}
)

add_subdirectory(test)
