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

"""Step 3: every 5 seconds, recognizes the place of the current frame and localizes it in the saved map, both without
a guess pose."""

import cuvslam as vslam
from vpr_utils import MAP_PATH, realsense_frames, slam_config

PERIOD_NS = 5 * 10**9

rig, frames = realsense_frames()
config = slam_config()
# Without a guess pose the radii are unused; the steps refine the pose around each recognized place
settings = vslam.Slam.LocalizationSettings(horizontal_search_radius=8., vertical_search_radius=2.,
                                           horizontal_step=0.5, vertical_step=0.2, angular_step_rads=0.03)
recognizer = None
next_try = 0
tries = localized = recognition_tries = recognized = 0
try:
    for timestamp, images in frames:
        if timestamp < next_try:
            continue
        next_try = timestamp + PERIOD_NS
        if recognizer is None:
            found, place = None, 'no map loaded yet'
        else:
            request = recognizer.recognize_place_by_frame(images, timestamp)
            found = request.result()
            place = (f'recognized the place mapped at {found.timestamp_ns / 1e9:.1f} s (score {found.score:.2f})'
                     if found is not None else f'not recognized: {request.error_message}')
        # A new session for each try: nothing tracked, no prior pose, nothing kept from the previous try
        slam = vslam.Slam(rig, [0], config)
        request = slam.localize_in_map(MAP_PATH, timestamp, None, images, settings)
        pose = request.result()
        tries += 1
        recognition_tries += recognizer is not None
        recognized += found is not None
        if pose is None:
            print(f'{timestamp / 1e9:7.1f} s: {place}; not localized: {request.error_message}')
            continue
        localized += 1
        print(f'{timestamp / 1e9:7.1f} s: {place}; localized at ' +
              ', '.join(f'{value:.2f}' for value in pose.translation))
        if recognizer is None:
            recognizer = slam  # place recognition searches its map: the saved one, loaded by this localization
except KeyboardInterrupt:
    pass
# Successes only: without ground truth the tutorial can't tell a right place from a wrong one
print(f'Summary: recognized {recognized} of {recognition_tries} tries '
      f'({100 * recognized / max(recognition_tries, 1):.1f} %), localized {localized} of {tries} '
      f'({100 * localized / max(tries, 1):.1f} %)')
