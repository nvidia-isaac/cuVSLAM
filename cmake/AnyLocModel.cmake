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

# AnyLoc model paths; included before libs/, whose tests use them.
# Defines, with CUVSLAM_BUILD_ANYLOC_ENGINE
#   CUVSLAM_ANYLOC_ONNX              step 1 output, the ONNX model
#   CUVSLAM_ANYLOC_REFERENCE_INPUT   validation input of both steps, .npy
#   CUVSLAM_ANYLOC_REFERENCE_OUTPUT  its fp32 AnyLoc descriptors, .npy
#   CUVSLAM_ANYLOC_ENGINE            step 2 output, the TensorRT engine; empty when cross compiling

set(CUVSLAM_ANYLOC_ENGINE "")
if(NOT CUVSLAM_BUILD_ANYLOC_ENGINE)
    return()
endif()

set(CUVSLAM_ANYLOC_LAYER 9 CACHE STRING "DINOv2 block whose value facet AnyLoc reads, 0 based")
set(CUVSLAM_ANYLOC_INPUT_SIZE 322 CACHE STRING "Square input size of the AnyLoc network in pixels, a multiple of 14")

set(_anyloc_name "anyloc_vits14_l${CUVSLAM_ANYLOC_LAYER}_value_${CUVSLAM_ANYLOC_INPUT_SIZE}")
set(CUVSLAM_ANYLOC_ONNX "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}.onnx")
set(CUVSLAM_ANYLOC_REFERENCE_INPUT "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}_reference_input.npy")
set(CUVSLAM_ANYLOC_REFERENCE_OUTPUT "${CMAKE_BINARY_DIR}/anyloc/${_anyloc_name}_reference_output.npy")

# An engine only loads in the TensorRT version that built it, so its name records it.
if(CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
    message(WARNING "Cross compiling, so the AnyLoc TensorRT engine cannot be built here and only its ONNX model "
                    "is. Build the engine on the target with anyloc_engine_builder.")
else()
    set(CUVSLAM_ANYLOC_ENGINE "${EXECUTABLE_OUTPUT_PATH}/${_anyloc_name}_fp16_trt${TensorRT_VERSION}.engine")
    message(STATUS "AnyLoc TensorRT engine: ${CUVSLAM_ANYLOC_ENGINE}")
endif()
