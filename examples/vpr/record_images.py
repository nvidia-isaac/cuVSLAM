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

"""Step 1 with a RealSense camera: records left infrared images of the area into images/ until Ctrl+C."""

import os

import cv2

from vpr_utils import IMAGES_DIR, realsense_frames

SAVE_EVERY = 10  # frames, so 3 images per second at 30 fps


def main() -> None:
    """Saves every SAVE_EVERY-th left image, named by its timestamp, so later recordings add to earlier ones."""
    _, frames = realsense_frames()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    print(f'Recording into {IMAGES_DIR}/: walk through the area, then press Ctrl+C')
    saved = 0
    try:
        for index, (timestamp, images) in enumerate(frames):
            if index % SAVE_EVERY == 0:
                cv2.imwrite(os.path.join(IMAGES_DIR, f'{timestamp}.png'), images[0])
                saved += 1
    except KeyboardInterrupt:
        pass
    print(f'Saved {saved} images')


if __name__ == '__main__':
    main()
