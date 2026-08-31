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

# libs/imu CMake file

set(MODULE_NAME "imu")

set(HEADERS
    imu_preintegration.h
    imu_sba.h
    imu_sba_problem.h
    inertial_optimization.h
    linear_filter.h
    linear_solver.h
    positive_matrix.h
    soft_inertial_pnp.h
    )

set(SOURCES
    imu_preintegration.cpp
    imu_sba_fixed_vel.cpp
    imu_sba_problem.cpp
    inertial_optimization.cpp
    linear_filter.cpp
    linear_solver.cpp
    positive_matrix.cpp
    soft_inertial_pnp.cpp
    )


set(LIBS
    camera
    common
    cuvslam_math
    epipolar
    profiler
)

if(USE_CUDA)
    set(HEADERS ${HEADERS} imu_sba_gpu.h)
    set(SOURCES ${SOURCES} imu_sba_gpu_fixed_vel.cpp)
    set(LIBS ${LIBS} cuda_modules)
endif()

set(HEADERS ${HEADERS} gyro_bias_nec.h)
set(SOURCES ${SOURCES} gyro_bias_nec.cpp)

setup_staticlib_project(
        MODULE_NAME ${MODULE_NAME}
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES ${LIBS}
)

add_subdirectory(test)
