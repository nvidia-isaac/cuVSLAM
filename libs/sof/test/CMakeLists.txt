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

# libs/sof/tst CMake file

set(HEADERS
    ../image_pyramid_float.h
)

# Source files
set(SOURCES
    basic_convolutor_test.cpp
    interpolation_test.cpp
    image_test.cpp
    kf_selector_test.cpp
    lk_tracker_test.cpp
    sof_l2r_test.cpp
    st_tracker_test.cpp
)

# Linking all the required libraries
set(LIBS
    testing
    sof
    epipolar
    utils
    edex
    cuvslam_math
    camera
    camera_rig_edex
    pnp
    log
    common
    profiler
)

if(USE_CUDA)
    set(LIBS ${LIBS} cuda_modules)
endif()

if(USE_RERUN)
    set(LIBS ${LIBS} visualizer rerun::sdk)
endif()

setup_test_project(
    MODULE_NAME ${MODULE_NAME}
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    LIBRARIES ${LIBS}
)
