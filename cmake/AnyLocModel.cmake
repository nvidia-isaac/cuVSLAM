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

# Where the two AnyLoc model build steps of tools/anyloc_model put their artifacts, and the GPU the engine is built
# for. Included from the root CMakeLists.txt after FindExt.cmake and before libs/, so that the tests in libs/ can
# name the engine tools/anyloc_model builds.
#
# Defines, with CUVSLAM_BUILD_ANYLOC_ENGINE
#   CUVSLAM_ANYLOC_ONNX              step 1 output, the ONNX export of DINOv2's AnyLoc read-out
#   CUVSLAM_ANYLOC_REFERENCE_INPUT   network input both steps validate on, .npy
#   CUVSLAM_ANYLOC_REFERENCE_OUTPUT  AnyLoc's fp32 descriptors of that input, .npy
#   CUVSLAM_ANYLOC_ENGINE            step 2 output, the TensorRT engine; empty when there is no GPU to build it for
#   CUVSLAM_ANYLOC_ENGINE_SM         compute capability that engine is built for, e.g. 89

set(CUVSLAM_ANYLOC_LAYER 9 CACHE STRING "DINOv2 block whose value facet AnyLoc reads, 0 based")
set(CUVSLAM_ANYLOC_INPUT_SIZE 322 CACHE STRING "Square input size of the AnyLoc network in pixels, a multiple of 14")
set(CUVSLAM_ANYLOC_GPU_SM "" CACHE STRING
    "Compute capability the AnyLoc engine is built for, e.g. 89; empty detects the GPU of this machine")

set(CUVSLAM_ANYLOC_ENGINE "")
set(CUVSLAM_ANYLOC_ENGINE_SM "")
if(NOT CUVSLAM_BUILD_ANYLOC_ENGINE)
    return()
endif()

set(_anyloc_name "dinov2_vits14_l${CUVSLAM_ANYLOC_LAYER}_${CUVSLAM_ANYLOC_INPUT_SIZE}")
set(CUVSLAM_ANYLOC_ONNX "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}.onnx")
set(CUVSLAM_ANYLOC_REFERENCE_INPUT "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}_reference_input.npy")
set(CUVSLAM_ANYLOC_REFERENCE_OUTPUT "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}_reference_output.npy")

# An engine is compiled for one GPU architecture and one TensorRT version, so its name records both: building for
# another GPU, or with another TensorRT, then makes another file instead of reusing one that no longer loads.
set(_anyloc_sm "${CUVSLAM_ANYLOC_GPU_SM}")
if(NOT _anyloc_sm AND CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
    message(WARNING "Cross compiling, so there is no GPU to build the AnyLoc TensorRT engine for and only its ONNX "
                    "model is built. Build the engine on the target with anyloc_engine_builder.")
elseif(NOT _anyloc_sm)
    try_run(_anyloc_sm_run _anyloc_sm_compiled
        ${CMAKE_BINARY_DIR}/anyloc/query_gpu_sm
        ${CMAKE_CURRENT_LIST_DIR}/anyloc/query_gpu_sm.cpp
        LINK_LIBRARIES CUDA::cudart_static
        RUN_OUTPUT_VARIABLE _anyloc_sm
    )
    if(NOT _anyloc_sm_compiled OR NOT _anyloc_sm_run EQUAL 0 OR NOT _anyloc_sm MATCHES "^[0-9]+$")
        message(WARNING "No CUDA GPU to build the AnyLoc TensorRT engine for, so only its ONNX model is built. Set "
                        "CUVSLAM_ANYLOC_GPU_SM to build the engine anyway (the build machine still needs that GPU), "
                        "or build it on the target with anyloc_engine_builder.")
        set(_anyloc_sm "")
    endif()
endif()

if(_anyloc_sm)
    set(CUVSLAM_ANYLOC_ENGINE_SM "${_anyloc_sm}")
    set(CUVSLAM_ANYLOC_ENGINE
        "${EXECUTABLE_OUTPUT_PATH}/${_anyloc_name}_fp16_trt${TensorRT_VERSION}_sm${_anyloc_sm}.engine")
    message(STATUS "AnyLoc TensorRT engine: ${CUVSLAM_ANYLOC_ENGINE}")
endif()
