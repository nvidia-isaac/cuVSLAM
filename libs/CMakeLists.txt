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

# libs CMake file
# Please maintain in alphabetical order - it helps for merges

add_subdirectory(camera)
add_subdirectory(camera_rig_edex)
add_subdirectory(common)
if(USE_CUDA)
    add_subdirectory(cuda_modules)
endif()
add_subdirectory(edex)
add_subdirectory(epipolar)
add_subdirectory(launcher)
add_subdirectory(log)
add_subdirectory(math)
add_subdirectory(map)
add_subdirectory(params)
add_subdirectory(pnp)
add_subdirectory(profiler)
add_subdirectory(cuvslam)
if(USE_CERES)
    add_subdirectory(refinement)
endif()
add_subdirectory(sba)
add_subdirectory(imu)
add_subdirectory(pipelines)
add_subdirectory(slam)
add_subdirectory(sof)
add_subdirectory(utils)
add_subdirectory(testing)
add_subdirectory(odometry)
if(USE_RERUN)
    add_subdirectory(visualizer)
endif()
