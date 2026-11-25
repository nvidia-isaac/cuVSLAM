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

"""Files, place recognition settings and CODa reading shared by the tutorial's steps."""

import glob
import json
import os

import cv2

import cuvslam as vslam

CODA_DIR = os.path.expanduser('~/datasets/coda')  # the converted sequences 0 and 5
ENGINE_PATH = 'anyloc.engine'
VOCABULARY_PATH = 'vpr.vocabulary'
MAP_PATH = 'map'


def slam_config(**kwargs) -> vslam.Slam.Config:
    """Returns the SLAM configuration of mapping and localizing: place recognition with the vocabulary of step 1."""
    vpr_settings = vslam.Slam.VprSettings(model_path=ENGINE_PATH, vocabulary_path=VOCABULARY_PATH)
    # max_map_size=0 keeps every keyframe: merging one away would also drop its place
    return vslam.Slam.Config(use_vpr=True, vpr_settings=vpr_settings, max_map_size=0, **kwargs)


def image_paths(sequence: int, camera: str) -> list:
    """Returns the image paths of one camera of a CODa sequence, 00 (left) or 01 (right), in recording order."""
    folder = os.path.join(CODA_DIR, str(sequence), camera)
    paths = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not paths:
        raise SystemExit(f'No images in {folder}: convert CODa first, or set CODA_DIR')
    return paths


def coda_frames(sequence: int, step: int = 1):
    """Returns the rig of a CODa sequence and a generator of (timestamp_ns, images) over every step-th frame."""
    with open(os.path.join(CODA_DIR, str(sequence), 'stereo.edex')) as file:
        header, body = json.load(file)
    left, right = header['cameras']
    intrinsics = left['intrinsics']
    # The images are rectified: the right camera only moves along x
    rig = vslam.Rig([
        vslam.Camera(size=intrinsics['size'], principal=intrinsics['principal'], focal=intrinsics['focal'],
                     rig_from_camera=vslam.Pose(rotation=[0, 0, 0, 1], translation=translation))
        for translation in ([0, 0, 0], [right['transform'][0][3], 0, 0])
    ])
    pairs = list(zip(image_paths(sequence, '00'), image_paths(sequence, '01')))
    period_ns = 10**9 // body['fps']
    frames = ((index * period_ns, [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in pairs[index]])
              for index in range(0, len(pairs), step))
    return rig, frames
