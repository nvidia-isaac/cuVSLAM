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

"""Step 1: fits the place recognition vocabulary to every left image of CODa sequence 0."""

import cv2

import cuvslam as vslam
from vpr_utils import ENGINE_PATH, VOCABULARY_PATH, image_paths

paths = image_paths(0, '00')
# Every image counts; sessions that load the vocabulary keep the default limits
builder = vslam.VprVocabularyBuilder(vslam.Slam.VprSettings(model_path=ENGINE_PATH,
                                                            vocabulary_max_keyframes=len(paths)))
print(f'Fitting the vocabulary to {len(paths)} images')
for path in paths:
    builder.add_image(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
builder.save(VOCABULARY_PATH)
print(f'Saved the vocabulary to {VOCABULARY_PATH}')
