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

set(MODULE_NAME "odometry")

set(HEADERS
    ground_integrator.h
    increment_pose.h
    ipredictor.h
    ivisual_odometry.h
    mono_visual_odometry.h
    pose_prediction.h
    multi_visual_odometry.h
    multi_visual_odometry_base.h
    svo_config.h
    svo_config_gflags.h
    )

set(SOURCES
    ground_integrator.cpp
    mono_visual_odometry.cpp
    pose_prediction.cpp
    stereo_inertial_odometry.cpp
    multi_visual_odometry.cpp
    multi_visual_odometry_base.cpp
    )


set(LIBS
    camera
    common
    cuvslam_math
    imu
    map
    params
    pipelines
    profiler
    sof
)

if(USE_CUDA)
    set(SOURCES ${SOURCES} rgbd_odometry.cpp)
    set(HEADERS ${HEADERS} rgbd_odometry.h)
endif()

# multisensor_odometry pulls in the cuNLS-based SolverSfMMultisensor pipeline. Gate it on
# USE_CUNLS (which implies USE_CUDA) so the file isn't compiled when cuNLS is unavailable.
if(USE_CUNLS)
    set(SOURCES ${SOURCES} multisensor_odometry.cpp)
    set(HEADERS ${HEADERS} multisensor_odometry.h)
endif()

setup_staticlib_project(
        MODULE_NAME ${MODULE_NAME}
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES ${LIBS}
)

add_subdirectory(test)
