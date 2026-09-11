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

set(MODULE_NAME "pipelines")

set(HEADERS
    feature_predictor.h
    internal/mono/sba_mono_wrapper.h
    internal/mono/workthread.h
    service_sba.h
    track.h
    track_online_mono.h
    track_online_multi.h
    tracker_state_machine.h
    triangulator.h
    winsorizer.h
    )

set(SOURCES
    feature_predictor.cpp
    sba_mono_wrapper.cpp
    service_sba.cpp
    track_online_mono.cpp
    track_online_multi.cpp
    tracker_state_machine.cpp
    triangulator.cpp
    winsorizer.cpp
    )


set(LIBS
    camera
    common
    epipolar
    imu
    map
    params
    pnp
    profiler
    sba
    sof
    )

if(USE_CUDA)
    set(HEADERS ${HEADERS} service_sba_gpu.h track_online_rgbd.h)
    set(SOURCES ${SOURCES} track_online_rgbd.cpp)
endif()

# Multisensor pipeline depends on cuNLS-based MultisensorPoseEstimator / PlaneMap / DepthPointMap
# in the pnp library, all of which are gated behind USE_CUNLS. USE_CUNLS implies USE_CUDA
# (enforced in cmake/FindExt.cmake), so no separate USE_CUDA guard is needed here. TrackLifter is
# also gated here because it includes pnp/multisensor_pose_estimator.h for the RGBDInfo struct.
if(USE_CUNLS)
    enable_language(CUDA)
    set(HEADERS ${HEADERS} track_online_multisensor.h depth_maps_context.h imu_fusion_context.h track_lifter.h)
    # All host-only sources (no device code); built by the host compiler. track_lifter.cpp is not
    # marked LANGUAGE CUDA — see the note in libs/map/CMakeLists.txt (nvcc cannot parse Eigen's ARM
    # NEON headers on aarch64).
    set(SOURCES ${SOURCES} track_online_multisensor.cpp depth_maps_context.cpp imu_fusion_context.cpp track_lifter.cpp)
endif()

if(USE_RERUN)
    set(HEADERS ${HEADERS} visualizer.h)
    set(SOURCES ${SOURCES} visualizer.cpp)
    set(LIBS ${LIBS} visualizer)
endif()

setup_staticlib_project(
        MODULE_NAME sfm_online_lite
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES ${LIBS}
)

setup_staticlib_project(
        MODULE_NAME pipelines
        HEADERS ${HEADERS} inertial_pnp.h track_online_inertial.h
        SOURCES ${SOURCES} inertial_pnp.cpp track_online_inertial.cpp
        LIBRARIES ${LIBS}
)

add_subdirectory(test)
