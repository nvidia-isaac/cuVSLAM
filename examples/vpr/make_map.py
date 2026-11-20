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

"""Step 2: maps the area with place recognition and saves the map."""

import argparse

import cuvslam as vslam
from vpr_utils import MAP_PATH, slam_config, stereo_frames


def main() -> None:
    """Tracks until Ctrl+C or the end of the recording, then saves the map to MAP_PATH."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coda_sequence', nargs='?', help='converted CODa sequence to map instead of the camera')
    args = parser.parse_args()

    rig, frames = stereo_frames(args.coda_sequence)
    tracker = vslam.Tracker(rig, vslam.Tracker.Mode.OdometryWithSlamRealtime,
                            vslam.Odometry.Config(rectified_stereo_camera=True), slam_config())
    print('Mapping until Ctrl+C or the end of the recording')
    try:
        for timestamp, images in frames:
            tracker.track(timestamp, images)
    except KeyboardInterrupt:
        pass
    if not tracker.slam.save_map(MAP_PATH).result():
        raise RuntimeError(f'Could not save the map to {MAP_PATH}')
    print(f'Saved the map to {MAP_PATH}')


if __name__ == '__main__':
    main()
