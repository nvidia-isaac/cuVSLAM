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

"""Selection and loading of the ground truth a tracker run is scored against.

A run has exactly one reference: a KITTI-format pose file, the forward pass of a shuttle replay, or
nothing at all. Which one it is comes from the config, never from what happens to be on disk.
"""

import math
import os
from typing import List, Optional

import numpy as np


def resolve_gt_file(dataset_path: str,
                    gt_path: Optional[str],
                    gt_from_shuttle: bool,
                    repeat_type: str,
                    num_loops: int) -> Optional[str]:
    """Validate the requested ground-truth source and return the pose file to read, if any.

    Returns None when the run has no pose file to read, which covers both the shuttle reference and
    a run with no ground truth. Raises when the request cannot be honoured, so a missing or
    contradictory ground truth stops the run instead of quietly producing metrics against something
    the caller did not ask for.
    """
    if gt_from_shuttle:
        if gt_path:
            raise ValueError(
                f"Conflicting ground truth: gt_from_shuttle is set alongside the pose file {gt_path!r}. "
                "Pick one source."
            )
        if repeat_type != "shuttle" or num_loops <= 0:
            raise ValueError(
                "gt_from_shuttle scores the backward shuttle pass against the forward one, so it needs "
                f"repeat_type='shuttle' with num_loops > 0; got repeat_type={repeat_type!r}, "
                f"num_loops={num_loops}."
            )
        return None

    if not gt_path:
        return None

    # A video input names the file itself, so its poses sit beside it, the way stereo.edex does.
    base = os.path.dirname(dataset_path) if os.path.isfile(dataset_path) else dataset_path
    resolved = gt_path if os.path.isabs(gt_path) else os.path.join(base, gt_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Ground-truth file not found: {resolved}. Provide the file, drop the ground-truth path to "
            "run without accuracy metrics, or set gt_from_shuttle to score a shuttle run against its "
            "own forward pass."
        )
    return resolved


def load_gt_transforms(gt_file: str) -> List[np.ndarray]:
    """Read a KITTI-format pose file into a list of 4x4 transforms."""
    bottom_row = np.array([0.0, 0.0, 0.0, 1.0])
    transforms = []
    with open(gt_file, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            values = [float(value) for value in line.split()]
            if len(values) != 12:
                raise ValueError(
                    f"{gt_file}:{line_number} holds {len(values)} values; KITTI poses need 12 per line."
                )
            # float() takes "nan" and "inf", and either one only surfaces later as an SVD that
            # will not converge inside the metrics.
            for value in values:
                if not math.isfinite(value):
                    raise ValueError(f"{gt_file}:{line_number} holds a non-finite value ({value}).")
            transforms.append(np.vstack((np.array(values).reshape(3, 4), bottom_row)))
    return transforms
