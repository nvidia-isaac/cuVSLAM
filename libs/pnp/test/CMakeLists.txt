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

set(SOURCES)

set(LIBS
    testing
    pnp
    common
    camera
    cuvslam_math
    log
)

if(USE_CUDA)
    list(APPEND SOURCES visual_icp_test.cpp)
    list(APPEND LIBS cuda_modules)
endif()

if(USE_CUNLS)
    enable_language(CUDA)
    # Host-only tests; built by the host compiler (not LANGUAGE CUDA) — see libs/map/CMakeLists.txt.
    list(APPEND SOURCES multisensor_pose_estimator_test.cpp point_to_point_icp_test.cpp)
    list(APPEND LIBS cunls::cunls)
endif()

if(SOURCES)
    setup_test_project(
        MODULE_NAME ${MODULE_NAME}
        SOURCES ${SOURCES}
        LIBRARIES ${LIBS}
    )
endif()
