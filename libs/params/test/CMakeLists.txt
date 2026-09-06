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

# libs/params/test CMake file

# config_params_test.cpp covers the descriptors for Odometry::Config and Slam::Config. Those live
# in libs/cuvslam/cuvslam_params.h rather than next to the structs, because cuvslam2.h is the public
# API boundary; the test lives here so every "what does the parameter surface cover" check stays in
# one place.
set(SOURCES
    config_params_test.cpp
    params_test.cpp
    sof_params_test.cpp
    solver_params_test.cpp
    )

set(LIBS
    cuvslam
    odometry
    params
    sof
    )

setup_test_project(
    MODULE_NAME ${MODULE_NAME}
    SOURCES ${SOURCES}
    LIBRARIES ${LIBS}
    )
