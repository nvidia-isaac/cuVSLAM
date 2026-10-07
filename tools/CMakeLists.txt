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

if(USE_LMDB)
    add_subdirectory(check_externals)
endif()
add_subdirectory(cuvslam_api_launcher)
add_subdirectory(cuvslam_api_test)
add_subdirectory(gravity_calc)
add_subdirectory(map_extractor)
add_subdirectory(tracker)

if(USE_TENSORRT)
    add_subdirectory(anyloc_model)
endif()

find_package(OpenCV QUIET)

if(OpenCV_FOUND)
    if(USE_CERES)
        add_subdirectory(edex_refinement)
    endif()
    add_subdirectory(selection_visualizer)
    add_subdirectory(sof_visualizer)
    add_subdirectory(undistort)
endif()
