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

# The AnyLoc model, built in two steps from the inputs cmake/ext/dinov2.cmake pins (see README.md):
#   anyloc_onnx    exports the part of DINOv2 that AnyLoc reads to ONNX, in a Python venv made from requirements.txt
#   anyloc_engine  builds a TensorRT engine of that export with anyloc_engine_builder, for this machine's GPU
# Both check what they make against AnyLoc's own fp32 descriptors of a reference image and fail the build rather than
# leave a model behind that does not reproduce them. The paths come from cmake/AnyLocModel.cmake; the engine lands
# next to libcuvslam.so and is the file Slam::Config::vpr_model_path names.

# Step 2's builder needs TensorRT and nothing else, so it is there with USE_TENSORRT alone, also where the model is not
# built: copied to a machine with another GPU, the ONNX model becomes an engine for that GPU with it.
get_filename_component(_tensorrt_library_dir "${TensorRT_nvinfer_LIBRARY}" DIRECTORY)
setup_app(MODULE_NAME anyloc_engine_builder
          SOURCES build_engine.cpp
          LIBRARIES TensorRT::nvonnxparser TensorRT::nvinfer CUDA::cudart_static cnpy::cnpy gflags::gflags)
# libnvonnxparser loads libnvinfer_plugin, and with it cuBLAS and cuDNN, which need not be on the loader's path. The
# linker writes the rpath as DT_RUNPATH by default, which only applies to the builder's own dependencies; DT_RPATH also
# applies to the libraries those load.
set_target_properties(anyloc_engine_builder PROPERTIES BUILD_RPATH "${_tensorrt_library_dir};${CUDAToolkit_LIBRARY_DIR}")
target_link_options(anyloc_engine_builder PRIVATE LINKER:--disable-new-dtags)

if(NOT CUVSLAM_BUILD_ANYLOC_ENGINE)
    return()
endif()

set(CUVSLAM_ANYLOC_PYTHON "" CACHE FILEPATH
    "Python with tools/anyloc_model/requirements.txt installed to run the AnyLoc ONNX export; empty makes a venv")

set(_python_tools ${CMAKE_SOURCE_DIR}/tools/python_tools)
set(_exporter_sources
    ${_python_tools}/cuvslam_tools/vpr/export_dinov2_onnx.py
    ${_python_tools}/cuvslam_tools/vpr/dinov2_value_facet.py
)
# The pin itself, so that moving it re-exports, and the model code it pins, for a local checkout that is edited.
file(GLOB _dinov2_sources
    ${dinov2_SOURCE_DIR}/dinov2/hub/backbones.py
    ${dinov2_SOURCE_DIR}/dinov2/models/*.py
    ${dinov2_SOURCE_DIR}/dinov2/layers/*.py
)
set(_dinov2_inputs ${CMAKE_SOURCE_DIR}/cmake/ext/dinov2.cmake ${_dinov2_sources} ${DINOV2_CHECKPOINT})
set(_reference_image ${CMAKE_SOURCE_DIR}/test_data/sof/left.png)

if(CUVSLAM_ANYLOC_PYTHON)
    set(_python ${CUVSLAM_ANYLOC_PYTHON})
    set(_python_stamp "")
else()
    # The pinned numpy and onnx have wheels up to Python 3.12 only.
    find_package(Python3 3.10...<3.13 COMPONENTS Interpreter)
    if(NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "The AnyLoc ONNX export needs Python 3.10 to 3.12 for its venv. Install one, point "
                            "Python3_EXECUTABLE at it, or set CUVSLAM_ANYLOC_PYTHON to an interpreter that already "
                            "has tools/anyloc_model/requirements.txt (or set CUVSLAM_BUILD_ANYLOC_ENGINE=OFF).")
    endif()
    set(_venv ${CMAKE_BINARY_DIR}/anyloc/venv)
    set(_python ${_venv}/bin/python)
    set(_python_stamp ${_venv}/requirements.stamp)
    add_custom_command(
        OUTPUT ${_python_stamp}
        COMMAND ${CMAKE_COMMAND}
                -DPYTHON=${Python3_EXECUTABLE}
                -DVENV=${_venv}
                -DREQUIREMENTS=${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
                -DSTAMP=${_python_stamp}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/ensure_venv.cmake
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt ${CMAKE_CURRENT_SOURCE_DIR}/ensure_venv.cmake
        COMMENT "AnyLoc: creating the Python venv the ONNX export runs in"
        VERBATIM
    )
endif()
add_custom_target(anyloc_venv DEPENDS ${_python_stamp})

# Step 1. The exporter is run from the source tree, so the venv only needs the export's own requirements.
add_custom_command(
    OUTPUT ${CUVSLAM_ANYLOC_ONNX} ${CUVSLAM_ANYLOC_REFERENCE_INPUT} ${CUVSLAM_ANYLOC_REFERENCE_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${_python_tools} PYTHONDONTWRITEBYTECODE=1
            ${_python} -m cuvslam_tools.vpr.export_dinov2_onnx
            --dinov2_src ${dinov2_SOURCE_DIR}
            --checkpoint ${DINOV2_CHECKPOINT}
            --layer ${CUVSLAM_ANYLOC_LAYER}
            --size ${CUVSLAM_ANYLOC_INPUT_SIZE}
            --output ${CUVSLAM_ANYLOC_ONNX}
            --reference_image ${_reference_image}
            --reference_input ${CUVSLAM_ANYLOC_REFERENCE_INPUT}
            --reference_output ${CUVSLAM_ANYLOC_REFERENCE_OUTPUT}
    DEPENDS ${_python_stamp} ${_exporter_sources} ${_dinov2_inputs} ${_reference_image}
    COMMENT "AnyLoc step 1/2: exporting DINOv2 ViT-S/14 up to block ${CUVSLAM_ANYLOC_LAYER} to ONNX"
    VERBATIM
)
add_custom_target(anyloc_onnx ALL DEPENDS ${CUVSLAM_ANYLOC_ONNX})
add_dependencies(anyloc_onnx anyloc_venv)

# Step 2.
if(CUVSLAM_ANYLOC_ENGINE)
    add_custom_command(
        OUTPUT ${CUVSLAM_ANYLOC_ENGINE}
        COMMAND anyloc_engine_builder
                --onnx=${CUVSLAM_ANYLOC_ONNX}
                --engine=${CUVSLAM_ANYLOC_ENGINE}
                --reference_input=${CUVSLAM_ANYLOC_REFERENCE_INPUT}
                --reference_output=${CUVSLAM_ANYLOC_REFERENCE_OUTPUT}
                --expect_sm=${CUVSLAM_ANYLOC_ENGINE_SM}
        DEPENDS anyloc_engine_builder ${CUVSLAM_ANYLOC_ONNX} ${CUVSLAM_ANYLOC_REFERENCE_INPUT}
                ${CUVSLAM_ANYLOC_REFERENCE_OUTPUT}
        COMMENT "AnyLoc step 2/2: building the FP16 TensorRT engine ${CUVSLAM_ANYLOC_ENGINE}"
        VERBATIM
    )
    add_custom_target(anyloc_engine ALL DEPENDS ${CUVSLAM_ANYLOC_ENGINE})
    add_dependencies(anyloc_engine anyloc_onnx)
endif()

# The exporter's unit tests need torch, onnx and the DINOv2 source, which only the export's environment has; the
# tools test suite skips them elsewhere, here they must run.
add_test(NAME anyloc_export_python_test
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${_python_tools}:${dinov2_SOURCE_DIR} PYTHONDONTWRITEBYTECODE=1
            XFORMERS_DISABLED=1 CUVSLAM_ANYLOC_EXPORT_TESTS=required
            ${_python} -m unittest -v cuvslam_tools.tests.test_export_dinov2
)
