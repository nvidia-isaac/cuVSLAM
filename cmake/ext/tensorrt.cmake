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

# Found, not fetched: an engine only loads in the TensorRT version that built it.
# Defines
#   TensorRT::headers       headers only; libcuvslam loads the runtime with dlopen()
#   TensorRT::nvinfer       runtime; needs no cuBLAS or cuDNN
#   TensorRT::nvonnxparser  ONNX parser for the engine builder, with CUVSLAM_BUILD_ANYLOC_ENGINE; needs cuBLAS, cuDNN
#   TensorRT_VERSION        e.g. 8.6.1

set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH
    "TensorRT install prefix; empty searches next to CUDA and in the system paths")

set(_tensorrt_hints ${TENSORRT_ROOT} ${CUDAToolkit_TARGET_DIR} ${CUDAToolkit_LIBRARY_ROOT})
find_path(TensorRT_INCLUDE_DIR NvInferVersion.h
    HINTS ${_tensorrt_hints}
    PATH_SUFFIXES include include/${CMAKE_LIBRARY_ARCHITECTURE}
)
find_library(TensorRT_nvinfer_LIBRARY nvinfer
    HINTS ${_tensorrt_hints} ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib/${CMAKE_LIBRARY_ARCHITECTURE}
)
if(CUVSLAM_BUILD_ANYLOC_ENGINE)
    find_library(TensorRT_nvonnxparser_LIBRARY nvonnxparser
        HINTS ${_tensorrt_hints} ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib lib64 lib/${CMAKE_LIBRARY_ARCHITECTURE}
    )
endif()
if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_nvinfer_LIBRARY
   OR (CUVSLAM_BUILD_ANYLOC_ENGINE AND NOT TensorRT_nvonnxparser_LIBRARY))
    message(FATAL_ERROR "USE_TENSORRT is ON but TensorRT was not found (NvInferVersion.h: ${TensorRT_INCLUDE_DIR}, "
                        "libnvinfer: ${TensorRT_nvinfer_LIBRARY}, libnvonnxparser: ${TensorRT_nvonnxparser_LIBRARY}). "
                        "Install it or set TENSORRT_ROOT to its install prefix.")
endif()

# TensorRT 10 sets the version numbers in TRT_*_ENTERPRISE, 8.x in NV_TENSORRT_*.
file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _tensorrt_version_lines
     REGEX "#define (NV_TENSORRT|TRT)_(MAJOR|MINOR|PATCH)(_ENTERPRISE)? +[0-9]+")
foreach(_part MAJOR MINOR PATCH)
    string(REGEX MATCH "(NV_TENSORRT|TRT)_${_part}(_ENTERPRISE)? +([0-9]+)" _match "${_tensorrt_version_lines}")
    set(TensorRT_VERSION_${_part} "${CMAKE_MATCH_3}")
endforeach()
set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
if(TensorRT_VERSION VERSION_LESS 8.6)
    # 8.6 is the first with the name based I/O API (enqueueV3) on all supported platforms.
    message(FATAL_ERROR
        "TensorRT ${TensorRT_VERSION} in ${TensorRT_INCLUDE_DIR} is too old, cuVSLAM needs 8.6 or newer")
endif()
message(STATUS "Found TensorRT ${TensorRT_VERSION}: ${TensorRT_nvinfer_LIBRARY}")

get_filename_component(_tensorrt_library_dir "${TensorRT_nvinfer_LIBRARY}" DIRECTORY)

if(NOT TARGET TensorRT::headers)
    add_library(TensorRT::headers INTERFACE IMPORTED GLOBAL)
    set_target_properties(TensorRT::headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR};${CUDAToolkit_INCLUDE_DIRS}"
    )
endif()

if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer SHARED IMPORTED GLOBAL)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR};${CUDAToolkit_INCLUDE_DIRS}"
    )
endif()

if(CUVSLAM_BUILD_ANYLOC_ENGINE AND NOT TARGET TensorRT::nvonnxparser)
    add_library(TensorRT::nvonnxparser SHARED IMPORTED GLOBAL)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
        INTERFACE_LINK_LIBRARIES TensorRT::nvinfer
        # Lets --no-allow-shlib-undefined resolve libnvinfer_plugin's cuBLAS and cuDNN.
        INTERFACE_LINK_OPTIONS
            "LINKER:-rpath-link,${_tensorrt_library_dir};LINKER:-rpath-link,${CUDAToolkit_LIBRARY_DIR}"
    )
endif()
