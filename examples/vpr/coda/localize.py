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

"""Step 3: every 5 seconds of CODa sequence 5, recognizes the place of its frame and localizes it in the saved map,
both without a guess pose, and checks both against ground truth."""

import numpy as np
from scipy.spatial.transform import Rotation

import cuvslam as vslam
from vpr_utils import FRAME_NS, MAP_PATH, coda_frames, ground_truth, slam_config

RIGHT_PLACE_M = 10  # a place mapped this close to the frame is the right one, as cuvslam_vpr_reporter counts it

mapped, queried = ground_truth(0), ground_truth(5)
rig, frames = coda_frames(5, step=50)  # every 5 seconds at 10 fps
config = slam_config(sync_mode=True)
# Without a guess pose the radii are unused; the steps refine the pose around each recognized place
settings = vslam.Slam.LocalizationSettings(horizontal_search_radius=8., vertical_search_radius=2.,
                                           horizontal_step=0.5, vertical_step=0.2, angular_step_rads=0.03)
recognizer = None
tries = localized = recognition_tries = recognized = right = 0
errors = []
for timestamp, images in frames:
    query = queried[timestamp // FRAME_NS]
    reference = None  # where the frame is in the map, by ground truth, when its place was recognized right
    if recognizer is None:
        found, place = None, 'no map loaded yet'
    else:
        request = recognizer.recognize_place_by_frame(images, timestamp)
        found = request.result()
        if found is None:
            place = f'not recognized: {request.error_message}'
        else:
            node = mapped[found.timestamp_ns // FRAME_NS]
            distance = np.linalg.norm(query[:, 3] - node[:, 3])
            place = (f'recognized the place mapped at {found.timestamp_ns / 1e9:.1f} s, {distance:.1f} m away '
                     f'(score {found.score:.2f})')
            if distance <= RIGHT_PLACE_M:
                right += 1
                # Ground truth gives the frame's offset from that place, the map gives the place's pose: no map drift
                offset = node[:, :3].T @ (query[:, 3] - node[:, 3])
                reference = Rotation.from_quat(found.pose.rotation).apply(offset) + found.pose.translation
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
    check = ''
    if reference is not None:
        errors.append(np.linalg.norm(np.array(pose.translation) - reference))
        check = f' ({errors[-1]:.2f} m off)'
    print(f'{timestamp / 1e9:7.1f} s: {place}; localized at ' +
          ', '.join(f'{value:.2f}' for value in pose.translation) + check)
    if recognizer is None:
        recognizer = slam  # place recognition searches its map: the saved one, loaded by this localization
print(f'Summary: recognized {recognized} of {recognition_tries} tries '
      f'({100 * recognized / max(recognition_tries, 1):.1f} %), localized {localized} of {tries} '
      f'({100 * localized / tries:.1f} %)')
checked = f'median {np.median(errors):.2f} m, max {np.max(errors):.2f} m off' if errors else 'none'
print(f'Ground truth: {right} of {recognized} recognized places within {RIGHT_PLACE_M} m '
      f'({100 * right / max(recognized, 1):.1f} %); {len(errors)} localizations checked: {checked}')
