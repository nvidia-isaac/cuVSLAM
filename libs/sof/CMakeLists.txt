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

set(MODULE_NAME "sof")

set(HEADERS
    basic_convolutor.h
    basic_image_downscaler.h
    box_blur.h
    convolutor.h
    epipolar_curves.h
    feature_prediction_interface.h
    feature_tracker.h
    gaussian_coefficients.h
    gftt.h
    gradient_pyramid.h
    image_context.h
    image_manager.h
    image_pyramid_float.h
    image_pyramid_u8.h
    internal/sof_mono_base.h
    internal/sof_mono_cpu.h
    internal/sof_multicamera_base.h
    internal/sof_multicamera_cpu.h
    kernel_operator.h
    kf_selector.h
    klt_tracker.h
    lk_tracker.h
    selection.h
    selector_interface.h
    selector_mono.h
    selector_stereo.h
    sof.h
    sof_config.h
    sof_config_gflags.h
    sof_create.h
    sof_mono_interface.h
    sof_multicamera_interface.h
    st_tracker.h
    )

set(SOURCES
    basic_convolutor.cpp
    basic_image_downscaler.cpp
    box_blur.cpp
    convolutor.cpp
    epipolar_curves.cpp
    gftt.cpp
    gradient_pyramid.cpp
    image_context.cpp
    image_manager.cpp
    image_pyramid_float.cpp
    image_pyramid_u8.cpp
    kf_selector.cpp
    klt_tracker.cpp
    lk_tracker.cpp
    selection.cpp
    selector_mono.cpp
    selector_stereo.cpp
    sof.cpp
    sof_config.cpp
    sof_create.cpp
    sof_mono_base.cpp
    sof_mono_cpu.cpp
    sof_multicamera_base.cpp
    sof_multicamera_cpu.cpp
    st_tracker.cpp
    )


set(LIBS
    camera
    common
    epipolar
    profiler
)

if(USE_CUDA)
    set(HEADERS ${HEADERS}
        internal/sof_mono_gpu.h
        internal/sof_multicamera_gpu.h
        )
    set(SOURCES ${SOURCES}
        sof_mono_gpu.cpp
        sof_multicamera_gpu.cpp
        )
    set(LIBS ${LIBS} cuda_modules)
endif()

setup_staticlib_project(
        MODULE_NAME "${MODULE_NAME}"
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES ${LIBS}
)

add_subdirectory(test)
