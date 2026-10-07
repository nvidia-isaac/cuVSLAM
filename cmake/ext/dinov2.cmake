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

# DINOv2's model code and ViT-S/14 weights, the inputs of the AnyLoc ONNX export (tools/anyloc_model).
#
# Both are pinned: the source at a commit, the weights by their SHA256. torch.hub, which the export used before,
# fetches whatever the repository's main branch holds and downloads the weights on its own.
#
# Defines
#   dinov2_SOURCE_DIR   the source tree; only its Python model code is used, nothing is built from it
#   DINOV2_CHECKPOINT   dinov2_vits14_pretrain.pth

include(FetchContent)

set(DINOV2_COMMIT "7764ea0f912e53c92e82eb78a2a1631e92725fc8")

FetchContent_Declare(
    dinov2
    URL https://github.com/facebookresearch/dinov2/archive/${DINOV2_COMMIT}.tar.gz
    URL_HASH SHA256=c27dcdaf50e9fb5bbdf2bb529da357716372e19c6afab17d5350f3f0094aed4b
)

FetchContent_GetProperties(dinov2)
if(NOT dinov2_POPULATED)
    FetchContent_Populate(dinov2)
endif()

set(DINOV2_CHECKPOINT_URL "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth")
set(DINOV2_CHECKPOINT_SHA256 "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9")
set(CUVSLAM_DINOV2_CHECKPOINT "" CACHE FILEPATH
    "Local copy of dinov2_vits14_pretrain.pth, checked against its SHA256; empty downloads it into the build tree")

if(CUVSLAM_DINOV2_CHECKPOINT)
    set(DINOV2_CHECKPOINT "${CUVSLAM_DINOV2_CHECKPOINT}")
else()
    set(DINOV2_CHECKPOINT "${FETCHCONTENT_BASE_DIR}/dinov2-weights/dinov2_vits14_pretrain.pth")
endif()

set(_dinov2_checkpoint_hash "")
if(EXISTS "${DINOV2_CHECKPOINT}")
    file(SHA256 "${DINOV2_CHECKPOINT}" _dinov2_checkpoint_hash)
endif()
if(NOT _dinov2_checkpoint_hash STREQUAL DINOV2_CHECKPOINT_SHA256)
    if(CUVSLAM_DINOV2_CHECKPOINT)
        message(FATAL_ERROR "CUVSLAM_DINOV2_CHECKPOINT=${CUVSLAM_DINOV2_CHECKPOINT} is not the DINOv2 ViT-S/14 "
                            "checkpoint the AnyLoc export is pinned to (SHA256 ${DINOV2_CHECKPOINT_SHA256}); "
                            "download it from ${DINOV2_CHECKPOINT_URL}")
    endif()
    message(STATUS "Downloading the DINOv2 ViT-S/14 checkpoint (88 MB) for the AnyLoc model")
    file(DOWNLOAD "${DINOV2_CHECKPOINT_URL}" "${DINOV2_CHECKPOINT}"
         EXPECTED_HASH SHA256=${DINOV2_CHECKPOINT_SHA256}
         STATUS _dinov2_download_status)
    list(GET _dinov2_download_status 0 _dinov2_download_code)
    if(NOT _dinov2_download_code EQUAL 0)
        file(REMOVE "${DINOV2_CHECKPOINT}")
        message(FATAL_ERROR "Could not download ${DINOV2_CHECKPOINT_URL}: ${_dinov2_download_status}. Without network "
                            "access, set CUVSLAM_DINOV2_CHECKPOINT to a copy of it.")
    endif()
endif()
