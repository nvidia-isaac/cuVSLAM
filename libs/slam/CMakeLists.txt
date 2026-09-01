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

set(MODULE_NAME "slam")

set(HEADERS
    localizer/localizer.h
    async_slam/async_slam.h
    async_slam/tail.h
    common/blob.h
    common/blob_eigen.h
    common/slam_common.h
    common/slam_input_image.h
    common/slam_log_types.h
    map/map.h
    map/database/slam_database.h
    map/descriptor/feature_descriptor.h
    map/descriptor/st_descriptor_gpu_ops.h
    map/descriptor/st_descriptor_merge_details.h
    map/descriptor/st_descriptor_ops.h
    map/pose_graph/pose_graph.h
    map/pose_graph/pose_graph_hypothesis.h
    map/pose_graph/slam_posegraph_keyframe_info.h
    map/spatial_index/landmark.h
    map/spatial_index/lsi_grid.h
    map/spatial_index/lsi_grid_cell_id.h
    slam/slam.h
    slam/slam_check_hypothesis.h
    slam/loop_closure_solver/iloop_closure_solver.h
    slam/loop_closure_solver/lcs_dummy.h
    slam/loop_closure_solver/lcs_simple.h
    slam/loop_closure_solver/lcs_simple_single_kf.h
    slam/loop_closure_solver/lcs_two_steps_easy.h
    slam/loop_closure_solver/pnp_ransac_hypothesis.h
    view/map_to_view.h
    view/view_landmarks.h
    view/view_manager.h
    view/view_pose_graph.h
)

set(SOURCES
    localizer/localizer.cpp
    async_slam/async_slam.cpp
    async_slam/async_slam_localize.cpp
    async_slam/async_slam_worker.cpp
    async_slam/tail.cpp
    common/slam_common.cpp
    map/map.cpp
    map/descriptor/st_descriptor_gpu_ops.cpp
    map/descriptor/st_descriptor_merge_details.cpp
    map/descriptor/st_descriptor_ops.cpp
    map/pose_graph/pose_graph.cpp
    map/pose_graph/pose_graph_hypothesis.cpp
    map/pose_graph/pose_graph_optimize.cpp
    map/pose_graph/pose_graph_reduce.cpp
    map/spatial_index/lsi_grid.cpp
    slam/slam.cpp
    slam/slam_map_building.cpp
    slam/loop_closure_solver/iloop_closure_solver.cpp
    slam/loop_closure_solver/lcs_dummy.cpp
    slam/loop_closure_solver/lcs_simple.cpp
    slam/loop_closure_solver/lcs_simple_single_kf.cpp
    slam/loop_closure_solver/lcs_two_steps_easy.cpp
    slam/slam_check_hypothesis.cpp
    view/map_to_view.cpp
)


set(LIBS
    camera
    common
    cuvslam_math
    epipolar
    log
    pnp
    profiler
)

if (USE_CUDA)
  set(LIBS ${LIBS} cuda_modules)
endif()

if (USE_LMDB)
  set(HEADERS ${HEADERS} map/database/lmdb_slam_database.h)
  set(SOURCES ${SOURCES} map/database/lmdb_slam_database.cpp)
  set(LIBS ${LIBS} lmdb::lmdb)
endif (USE_LMDB)

if (USE_RERUN)
  set(HEADERS ${HEADERS} map/spatial_index/visualizer.h)
  set(SOURCES ${SOURCES} map/spatial_index/visualizer.cpp)
  # async_slam.cpp calls pipelines::logTrajectory() in its USE_RERUN path, so slam must link pipelines.
  set(LIBS ${LIBS} visualizer pipelines)
endif (USE_RERUN)

setup_staticlib_project(
    MODULE_NAME ${MODULE_NAME}
    HEADERS ${HEADERS}
    SOURCES ${SOURCES}
    LIBRARIES ${LIBS}
)

add_subdirectory(test)
