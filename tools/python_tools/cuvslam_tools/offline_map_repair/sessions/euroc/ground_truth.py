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


"""EVALUATION ONLY: read EuRoC ground truth. No repair policy may import this."""

import csv
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def ground_truth(dataset):
    """EVALUATION ONLY. Load EuRoC ground truth expressed in the cam0 frame.

    The CSV gives T_world_body; cuVSLAM poses are in the cam0 rig frame, so compose
    with cam0's ``T_BS`` to make the two comparable. Never used to compute an edit.
    """
    root = Path(dataset) / "mav0"
    with (root / "state_groundtruth_estimate0/data.csv").open() as stream:
        rows = [r for r in csv.reader(stream) if r and not r[0].startswith("#")]
    ts = np.array([int(r[0]) for r in rows], dtype=np.int64)
    values = np.array([[float(v) for v in r[1:8]] for r in rows])
    transforms = np.tile(np.eye(4), (len(rows), 1, 1))
    transforms[:, :3, 3] = values[:, :3]
    transforms[:, :3, :3] = Rotation.from_quat(values[:, [4, 5, 6, 3]]).as_matrix()
    calibration = yaml.safe_load((root / "cam0/sensor.yaml").read_text())
    body_from_camera = np.array(calibration["T_BS"]["data"]).reshape(4, 4)
    return ts, transforms @ body_from_camera
