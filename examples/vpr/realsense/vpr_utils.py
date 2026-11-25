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

"""Files, place recognition settings and the RealSense frame source shared by the tutorial's steps."""

import numpy as np
import pyrealsense2 as rs

import cuvslam as vslam

ENGINE_PATH = 'anyloc.engine'
VOCABULARY_PATH = 'vpr.vocabulary'
MAP_PATH = 'map'
IMAGES_DIR = 'images'

REALSENSE_RESOLUTION = (640, 360)
REALSENSE_FPS = 30
REALSENSE_WARMUP_FRAMES = 60


def slam_config() -> vslam.Slam.Config:
    """Returns the SLAM configuration of mapping and localizing: place recognition with the vocabulary of step 1."""
    vpr_settings = vslam.Slam.VprSettings(model_path=ENGINE_PATH, vocabulary_path=VOCABULARY_PATH)
    # max_map_size=0 keeps every keyframe: merging one away would also drop its place
    return vslam.Slam.Config(use_vpr=True, vpr_settings=vpr_settings, max_map_size=0)


def realsense_frames():
    """Returns the rig of the RealSense camera's infrared pair and a generator of (timestamp_ns, images)."""
    config = rs.config()
    for index in (1, 2):
        config.enable_stream(rs.stream.infrared, index, *REALSENSE_RESOLUTION, rs.format.y8, REALSENSE_FPS)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)  # its dots would be tracked as features
    left, right = (profile.get_stream(rs.stream.infrared, index).as_video_stream_profile() for index in (1, 2))
    intrinsics = left.get_intrinsics()
    # The infrared pair is rectified: the right camera only moves along its baseline
    rig = vslam.Rig([
        vslam.Camera(size=(intrinsics.width, intrinsics.height), principal=(intrinsics.ppx, intrinsics.ppy),
                     focal=(intrinsics.fx, intrinsics.fy),
                     rig_from_camera=vslam.Pose(rotation=[0, 0, 0, 1], translation=translation))
        for translation in ([0, 0, 0], right.get_extrinsics_to(left).translation)
    ])

    def frames():
        try:
            for _ in range(REALSENSE_WARMUP_FRAMES):  # auto exposure settles
                pipeline.wait_for_frames()
            start = None
            while True:
                frameset = pipeline.wait_for_frames()
                left_frame, right_frame = frameset.get_infrared_frame(1), frameset.get_infrared_frame(2)
                images = [np.asanyarray(frame.get_data()) for frame in (left_frame, right_frame)]
                timestamp = int(left_frame.timestamp * 1e6)  # milliseconds to nanoseconds
                start = timestamp if start is None else start
                yield timestamp - start, images  # from 0 in every run, as in a recording
        finally:
            pipeline.stop()

    return rig, frames()
