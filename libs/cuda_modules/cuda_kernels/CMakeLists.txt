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

project(cuda_kernels LANGUAGES CXX CUDA)

set(SOURCES
    convolutions.cu
    gftt.cu
    image_pyramid.cu
    lk_tracker.cu
    selection_v2.cu
    image_cast.cu
    sort.cu
    st_tracker.cu
    sba_v1.cu
    sba_imu_v1.cu
    window_matcher.cu
    )

add_library(cuda_kernels STATIC ${SOURCES})

set_target_properties(cuda_kernels PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC,-fvisibility=hidden>
    # Enable device debug info for compute-sanitizer support
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo>
)

target_link_libraries(cuda_kernels PUBLIC
    cuvslam_settings
    CUDA::cudart_static
    CUDA::cublas
    CUDA::cusolver
)
