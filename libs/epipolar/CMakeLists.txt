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

set(MODULE_NAME "epipolar")

set(HEADERS
    camera_projection.h
    camera_selection.h
    fundamental_matrix_utils.h
    fundamental_ransac.h
    homography.h
    homography_ransac.h
    near_plane.h
    optscalar_triangle_ransac.h
    point_reconstruction.h
    points_normalization.h
    resectioning_utils.h
    )

set(SOURCES
    camera_projection.cpp
    fundamental_matrix_utils.cpp
    homography.cpp
    points_normalization.cpp
    resectioning_utils.cpp
    )

setup_staticlib_project(
        MODULE_NAME "${MODULE_NAME}"
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES common cuvslam_math
)

add_subdirectory(test)
