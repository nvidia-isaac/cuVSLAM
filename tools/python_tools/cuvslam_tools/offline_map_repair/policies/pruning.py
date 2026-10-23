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


"""Shared landmark-pruning predicates used by more than one repair policy.

These are *policies*, not correctness rules: cuVSLAM itself is happy to keep the
landmarks selected here. They exist to test whether dropping weakly constrained
points helps or hurts relocalization, so keep the thresholds explicit and named.
"""

import numpy as np

# A point seen from a single keyframe has no triangulation baseline, and the further
# away it is the worse its depth estimate. Past ~15 m a lone observation contributes
# almost no geometric constraint while still costing a descriptor match.
SINGLE_OBSERVATION_RANGE_M = 15.0


def distant_single_observation_landmarks(map_data):
    """Return landmarks with exactly one observation taken from over 15 m away."""
    return [
        identifier
        for identifier, landmark in map_data.landmarks.items()
        if len(landmark["observations"]) == 1
        and np.linalg.norm(landmark["observations"][0]["xyz"] - landmark["observations"][0]["eye"])
        > SINGLE_OBSERVATION_RANGE_M
    ]
