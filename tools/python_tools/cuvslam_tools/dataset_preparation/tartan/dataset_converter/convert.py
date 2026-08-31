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

"""Callable entry point for the classic TartanAir to EDEX conversion."""

import os
from typing import List

from .pipeline import TartanAirPipeline

TARTAN_DIRS = {"image_left", "image_right"}
TARTAN_FILES = {"pose_left.txt", "pose_right.txt"}


def find_sequences(seq_path: str) -> List[str]:
    """Return every classic TartanAir sequence folder under seq_path."""
    sequences = []
    for path, dirs, files in os.walk(seq_path):
        if TARTAN_DIRS.issubset(set(dirs)) and TARTAN_FILES.issubset(set(files)):
            sequences.append(path)
    return sequences


def convert_sequences(seq_path: str, save_gt_folder: str, save_edex_folder: str) -> List[str]:
    """Convert every classic TartanAir sequence under seq_path in place.

    Each sequence is rewritten to the EDEX layout (``00``/``01`` image folders,
    ``gt.txt``, ``cfg.edex``). Returns the sequence folders that were converted.
    """
    sequences = find_sequences(seq_path)
    for sequence in sequences:
        TartanAirPipeline(sequence, save_gt_folder, save_edex_folder)()
    return sequences
