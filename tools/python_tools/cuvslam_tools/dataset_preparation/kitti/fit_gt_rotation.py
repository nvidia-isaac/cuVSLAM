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

"""Reproduce gt_rotations.json: per-date camera-to-GT rotations fitted on odometry runs, like SOFT2's KITTI_TO_CAM1.

Each date's rotation starts from the analytic frame change and minimizes the pooled KITTI t_err of its sequences
with ground truth over all runs given.
"""

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from . import recalibration as cal

POSES_ZIP = "data_odometry_poses.zip"
# Reporter trajectories flip y and z relative to KITTI's ground truth.
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
SEGMENT_LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)


def homogeneous(rows: np.ndarray) -> np.ndarray:
    """KITTI-format poses (N x 12) as N x 4 x 4."""
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3] = rows.reshape(-1, 3, 4)
    return poses


class Segments:
    """KITTI devkit segments (every 10th frame, 100-800 m) of one estimated trajectory against ground truth."""

    def __init__(self, gt: np.ndarray, est: np.ndarray):
        steps = np.linalg.norm((np.linalg.inv(gt[:-1]) @ gt[1:])[:, :3, 3], axis=1)
        dist = np.r_[0.0, np.cumsum(steps)]
        pairs = []
        for first in range(0, len(gt), 10):
            for length in SEGMENT_LENGTHS:
                last = np.searchsorted(dist, dist[first] + length, side="right")
                if last < len(gt):
                    pairs.append((first, last, length))
        first, last, self.length = np.array(pairs).T
        self.gt = np.linalg.inv(gt[first]) @ gt[last]
        self.est = np.linalg.inv(est[first]) @ est[last]

    def t_err(self, A: np.ndarray) -> np.ndarray:
        """Translation error per segment with estimates rotated into the GT frame by A."""
        error = np.linalg.inv(A @ self.est @ A.T) @ self.gt
        return np.linalg.norm(error[:, :3, 3], axis=1) / self.length


def pooled_t_err(segments: List[Segments], A: np.ndarray) -> float:
    """Mean t_err (%) over all segments."""
    return float(np.concatenate([s.t_err(A) for s in segments]).mean() * 100)


def rotation(theta_deg, A0: np.ndarray) -> np.ndarray:
    """4 x 4 rotation: the rotation vector theta (deg) applied after A0."""
    A = np.eye(4)
    A[:3, :3] = Rotation.from_rotvec(np.radians(theta_deg)).as_matrix() @ A0
    return A


def fit(segments: List[Segments], A0: np.ndarray) -> np.ndarray:
    """The rotation near A0 (3 x 3) with the lowest pooled t_err."""
    result = minimize(lambda theta: pooled_t_err(segments, rotation(theta, A0)), np.zeros(3), method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 800})
    return rotation(result.x, A0)[:3, :3]


def write_rotations(path: Path, rotations: dict) -> None:
    """JSON with one matrix row per line."""
    blocks = [f'    "{date}": [\n' + ",\n".join(f"        {json.dumps(row)}" for row in rotations[date].tolist())
              + "\n    ]" for date in sorted(rotations)]
    path.write_text("{\n" + ",\n".join(blocks) + "\n}\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and write the fitted rotations."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="KITTI archives directory (prepare_kitti --raw-dir ... --recalibrated).")
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="cuvslam_reporter output directories of kitti-vio_gt.cfg on the recalibrated dataset.")
    parser.add_argument("--out", type=Path, required=True, help="Rotations JSON to write.")
    args = parser.parse_args(argv)

    by_date = defaultdict(list)
    with zipfile.ZipFile(args.raw_dir / POSES_ZIP) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".txt")):
            seq = Path(name).stem
            gt = homogeneous(np.loadtxt(archive.read(name).decode().splitlines()))
            for run in args.runs:
                est = homogeneous(np.loadtxt(run / f"KITTI-{seq}-ODOM.txt"))
                by_date[cal.SEQUENCE_DATES[seq]].append(Segments(gt, FLIP @ est @ FLIP))
    rotations = {}
    for date, segments in sorted(by_date.items()):
        kitti, _ = cal.read_kitti_calibration(args.raw_dir / f"{date}_calib.zip", date)
        A0 = cal.frame_rotation(kitti, cal.new_calibration(args.raw_dir, date)).T
        rotations[date] = fit(segments, A0)
        print(f"{date}: rotation {np.round(Rotation.from_matrix(rotations[date]).as_rotvec(degrees=True), 3)} deg, "
              f"t_err {pooled_t_err(segments, rotation(np.zeros(3), A0)):.3f} -> "
              f"{pooled_t_err(segments, rotation(np.zeros(3), rotations[date])):.3f} %")
    write_rotations(args.out, rotations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
