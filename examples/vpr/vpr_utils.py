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

"""Files, place recognition settings and stereo frame sources shared by the tutorial's steps."""

import glob
import json
import os
import time

import cv2
import numpy as np

import cuvslam as vslam

ENGINE_PATH = 'anyloc.engine'
VOCABULARY_PATH = 'vpr.vocabulary'
MAP_PATH = 'map'
IMAGES_DIR = 'images'

# Mapping and localizing must recognize places with the same vocabulary
VPR_SETTINGS = vslam.Slam.VprSettings(model_path=ENGINE_PATH, vocabulary_path=VOCABULARY_PATH)

REALSENSE_RESOLUTION = (640, 360)
REALSENSE_FPS = 30
REALSENSE_WARMUP_FRAMES = 60


def stereo_rig(size, principal, focal, right_translation) -> vslam.Rig:
    """Returns a rig of two rectified pinhole cameras, the right one at right_translation from the left one."""
    return vslam.Rig([
        vslam.Camera(size=size, principal=principal, focal=focal,
                     rig_from_camera=vslam.Pose(rotation=[0, 0, 0, 1], translation=translation))
        for translation in ([0, 0, 0], right_translation)
    ])


def realsense_frames():
    """Returns the rig of a live RealSense camera's infrared pair and a generator of (timestamp_ns, images)."""
    import pyrealsense2 as rs  # only the camera needs librealsense

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
    rig = stereo_rig((intrinsics.width, intrinsics.height), (intrinsics.ppx, intrinsics.ppy),
                     (intrinsics.fx, intrinsics.fy), right.get_extrinsics_to(left).translation)

    def frames():
        try:
            for _ in range(REALSENSE_WARMUP_FRAMES):  # auto exposure settles
                pipeline.wait_for_frames()
            while True:
                frameset = pipeline.wait_for_frames()
                left_frame, right_frame = frameset.get_infrared_frame(1), frameset.get_infrared_frame(2)
                images = [np.asanyarray(frame.get_data()) for frame in (left_frame, right_frame)]
                yield int(left_frame.timestamp * 1e6), images  # milliseconds to nanoseconds
        finally:
            pipeline.stop()

    return rig, frames()


def coda_frames(sequence):
    """Returns the rig of a converted CODa sequence and a generator of (timestamp_ns, images) at its frame rate."""
    with open(os.path.join(sequence, 'stereo.edex')) as file:
        header, body = json.load(file)
    left, right = header['cameras']
    intrinsics = left['intrinsics']
    # The images are rectified: the right camera only moves along x
    rig = stereo_rig(intrinsics['size'], intrinsics['principal'], intrinsics['focal'],
                     [right['transform'][0][3], 0, 0])
    pairs = list(zip(*(sorted(glob.glob(os.path.join(sequence, folder, '*.png'))) for folder in ('00', '01'))))
    period_ns = 10**9 // body['fps']

    def frames():
        start = time.monotonic()
        for index, pair in enumerate(pairs):
            timestamp = index * period_ns
            time.sleep(max(0.0, start + timestamp / 1e9 - time.monotonic()))  # as fast as a camera delivers them
            yield timestamp, [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in pair]

    return rig, frames()


def stereo_frames(coda_sequence):
    """Returns coda_frames(coda_sequence) when it is given, realsense_frames() otherwise."""
    return coda_frames(coda_sequence) if coda_sequence else realsense_frames()
