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

"""Step 2: maps the area with place recognition while the camera moves through it, then saves the map."""

import cuvslam as vslam
from vpr_utils import MAP_PATH, realsense_frames, slam_config

rig, frames = realsense_frames()
tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryWithSlamRealtime,
                        vslam.Odometry.Config(rectified_stereo_camera=True), slam_config())
print('Mapping: walk through the area, then press Ctrl+C')
try:
    for timestamp, images in frames:
        tracker.track(timestamp, images)
except KeyboardInterrupt:
    pass
if not tracker.slam.save_map(MAP_PATH).result():
    raise RuntimeError(f'Could not save the map to {MAP_PATH}')
print(f'Saved the map to {MAP_PATH}')
