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

"""Step 3: every 5 seconds, localizes the current frame in the saved map without a guess pose."""

import argparse

import cuvslam as vslam
from vpr_utils import MAP_PATH, VPR_SETTINGS, stereo_frames

PERIOD_NS = 5 * 10**9


def main() -> None:
    """Prints where each try finds the rig in the map, or why it failed, until Ctrl+C or the end of the recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('coda_sequence', nargs='?', help='converted CODa sequence to localize instead of the camera')
    args = parser.parse_args()

    rig, frames = stereo_frames(args.coda_sequence)
    config = vslam.Slam.Config(use_vpr=True, vpr_settings=VPR_SETTINGS, sync_mode=True)
    # Without a guess pose the radii are unused; the steps refine the pose around each recognized place
    settings = vslam.Slam.LocalizationSettings(horizontal_search_radius=8., vertical_search_radius=2.,
                                               horizontal_step=0.5, vertical_step=0.2, angular_step_rads=0.03)
    start = next_try = None
    try:
        for timestamp, images in frames:
            if start is None:
                start = next_try = timestamp
            if timestamp < next_try:
                continue
            next_try = timestamp + PERIOD_NS
            # A new session for each try: nothing tracked, no prior pose, nothing kept from the previous try
            slam = vslam.Slam(rig, [0], config)
            request = slam.localize_in_map(MAP_PATH, timestamp, None, images, settings)
            pose = request.result()
            seconds = (timestamp - start) / 1e9
            if pose is None:
                print(f'{seconds:7.1f} s: not localized: {request.error_message}')
            else:
                print(f'{seconds:7.1f} s: at ' + ', '.join(f'{value:.2f}' for value in pose.translation))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
