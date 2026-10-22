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

# AnyLoc model paths and GPU target; included before libs/, whose tests use them.
# Defines, with CUVSLAM_BUILD_ANYLOC_ENGINE
#   CUVSLAM_ANYLOC_ONNX              step 1 output, the ONNX model
#   CUVSLAM_ANYLOC_REFERENCE_INPUT   validation input of both steps, .npy
#   CUVSLAM_ANYLOC_REFERENCE_OUTPUT  its fp32 AnyLoc descriptors, .npy
#   CUVSLAM_ANYLOC_ENGINE            step 2 output, the TensorRT engine; empty when cross compiling
#   CUVSLAM_ANYLOC_ENGINE_SM         compute capability of CUVSLAM_ANYLOC_TARGET

set(CUVSLAM_ANYLOC_ENGINE "")
set(CUVSLAM_ANYLOC_ENGINE_SM "")
if(NOT CUVSLAM_BUILD_ANYLOC_ENGINE)
    return()
endif()

set(CUVSLAM_ANYLOC_LAYER 9 CACHE STRING "DINOv2 block whose value facet AnyLoc reads, 0 based")
set(CUVSLAM_ANYLOC_INPUT_SIZE 322 CACHE STRING "Square input size of the AnyLoc network in pixels, a multiple of 14")

# TensorRT builds only for the GPU it runs on, so each target is built on its own device.
set(_anyloc_targets rtx4090 orin_nano)
set(_anyloc_sm_rtx4090 89)
set(_anyloc_sm_orin_nano 87)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_anyloc_default_target orin_nano)
else()
    set(_anyloc_default_target rtx4090)
endif()
set(CUVSLAM_ANYLOC_TARGET "${_anyloc_default_target}" CACHE STRING
    "GPU the AnyLoc engine is built for: rtx4090 (sm_89) or orin_nano (sm_87); the default follows the processor")
set_property(CACHE CUVSLAM_ANYLOC_TARGET PROPERTY STRINGS ${_anyloc_targets})
if(NOT CUVSLAM_ANYLOC_TARGET IN_LIST _anyloc_targets)
    list(JOIN _anyloc_targets ", " _anyloc_target_names)
    message(FATAL_ERROR
        "CUVSLAM_ANYLOC_TARGET is \"${CUVSLAM_ANYLOC_TARGET}\", which is none of: ${_anyloc_target_names}")
endif()

set(_anyloc_name "anyloc_vits14_l${CUVSLAM_ANYLOC_LAYER}_value_${CUVSLAM_ANYLOC_INPUT_SIZE}")
set(CUVSLAM_ANYLOC_ONNX "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}.onnx")
set(CUVSLAM_ANYLOC_REFERENCE_INPUT "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}_reference_input.npy")
set(CUVSLAM_ANYLOC_REFERENCE_OUTPUT "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}_reference_output.npy")

# An engine only loads with the TensorRT version and SM that built it, so its name records both.
if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
    message(WARNING "Cross compiling, so the AnyLoc TensorRT engine cannot be built here and only its ONNX model "
                    "is. Build the engine on the ${CUVSLAM_ANYLOC_TARGET} with anyloc_engine_builder.")
else()
    set(CUVSLAM_ANYLOC_ENGINE_SM "${_anyloc_sm_${CUVSLAM_ANYLOC_TARGET}}")
    set(CUVSLAM_ANYLOC_ENGINE
        "${EXECUTABLE_OUTPUT_PATH}/${_anyloc_name}_fp16_trt${TensorRT_VERSION}_sm${CUVSLAM_ANYLOC_ENGINE_SM}.engine")
    message(STATUS "AnyLoc TensorRT engine for ${CUVSLAM_ANYLOC_TARGET}: ${CUVSLAM_ANYLOC_ENGINE}")
endif()
