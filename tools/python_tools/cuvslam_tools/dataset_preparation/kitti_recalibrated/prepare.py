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

"""Re-rectify the converted KITTI odometry dataset with the Cvišić et al. (ECMR 2021) stereo calibration."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from cuvslam_tools.dataset_preparation.common import (
    PreparationError,
    dataset_file,
    run_download_script,
    run_preparation,
)

from . import calibration as cal

DATASET_NAME = "kitti_recalibrated"
DOWNLOAD_SCRIPT = "download_kitti_recalibrated.sh"
# The authors' calibrations (downloaded); they published none for 2011_09_26, which uses ours (bundled).
AUTHORS_CALIBRATIONS = {"2011_10_03": "lamor_00_02.yaml", "2011_09_30": "lamor_04_12.yaml"}
OWN_CALIBRATION = "calib_2011_09_26.json"
GT_ROTATIONS = "gt_rotations.json"

_MAPS = {}


def new_calibration(raw_dir: Path, date: str) -> cal.Stereo:
    """The recalibration of one recording date."""
    name = AUTHORS_CALIBRATIONS.get(date)
    return cal.read_stereo_calibration(raw_dir / name if name else dataset_file(__file__, OWN_CALIBRATION))


def _set_maps(maps):
    _MAPS["maps"] = maps


def _remap(job):
    camera, src, dst = job
    image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PreparationError(f"cannot read {src}")
    map_x, map_y = _MAPS["maps"][camera]
    if not cv2.imwrite(dst, cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)):
        raise PreparationError(f"cannot write {dst}")


def rerectify(kitti_dir: Path, output_dir: Path, sequences: List[str], kitti: cal.Stereo, new: cal.Stereo,
              gt_rotation: np.ndarray) -> None:
    """Write sequences of one recording date: images, stereo.edex with KITTI's baseline, gt.txt as A^T G A."""
    maps = cal.rerectification_maps(kitti, new)
    top, bottom, left, right = cal.common_crop(maps, kitti.size)
    maps = [tuple(m[top:bottom + 1, left:right + 1].astype(np.float32) for m in xy) for xy in maps]
    size = [right - left + 1, bottom - top + 1]
    A = np.eye(4)
    A[:3, :3] = gt_rotation
    with ProcessPoolExecutor(initializer=_set_maps, initargs=(maps,)) as pool:
        for seq in sequences:
            src, dst = kitti_dir / seq, output_dir / seq
            edex = json.loads((src / "stereo.edex").read_text())
            cameras = edex[0]["cameras"]
            focal = cameras[0]["intrinsics"]["focal"][0]
            if abs(focal - kitti.cameras[0].P[0, 0]) > 1e-3:
                raise PreparationError(f"{src}: focal length {focal} is not KITTI's rectification")
            for camera, new_camera in zip(cameras, new.cameras):
                camera["intrinsics"]["focal"] = [float(new_camera.P[0, 0]), float(new_camera.P[1, 1])]
                camera["intrinsics"]["principal"] = [float(new_camera.P[0, 2] - left), float(new_camera.P[1, 2] - top)]
                camera["intrinsics"]["size"] = size
            cameras[1]["transform"][0][3] = float(-kitti.cameras[1].P[0, 3] / kitti.cameras[1].P[0, 0])
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "stereo.edex").write_text(json.dumps(edex, indent=4))

            if (src / "gt.txt").is_file():
                lines = []
                for g in np.loadtxt(src / "gt.txt").reshape(-1, 3, 4):
                    pose = A.T @ np.vstack([g, [0, 0, 0, 1]]) @ A
                    lines.append(" ".join(f"{v:.9e}" for v in pose[:3].ravel()))
                (dst / "gt.txt").write_text("\n".join(lines) + "\n")

            jobs = []
            for camera, sub in enumerate(("00", "01")):
                (dst / sub).mkdir(exist_ok=True)
                jobs += [(camera, str(p), str(dst / sub / p.name)) for p in sorted((src / sub).glob("*.png"))]
            list(pool.map(_remap, jobs, chunksize=64))
            print(f"{seq}: {len(jobs)} images, {size[0]}x{size[1]}", flush=True)


def write_configs(output_dir: Path, sequences: List[str]) -> None:
    """Reporter configs for odometry and SLAM on the sequences with ground truth."""
    for name, label, use_slam in (("vio_gt", "ODOM", False), ("slam_gt", "SLAM", True)):
        config = {
            "dataset_folder": f"{DATASET_NAME}/",
            "segment_lengths": [100, 200, 300, 400, 500, 600, 700, 800],
            "sequence_cfgs": [{"sequence_folder": s, "sequence_title": f"KITTI-{s}-{label}", "gt_file_path": "gt.txt",
                               "use_slam": use_slam} for s in sequences],
        }
        (output_dir / f"{DATASET_NAME}-{name}.cfg").write_text(json.dumps(config, indent=4) + "\n")


def prepare(kitti_dir: Path, raw_dir: Path, output_dir: Path) -> Path:
    """Download the calibrations into raw_dir, re-rectify kitti_dir into output_dir, and return output_dir."""
    kitti_dir, raw_dir, output_dir = Path(kitti_dir), Path(raw_dir), Path(output_dir)
    run_download_script(dataset_file(__file__, DOWNLOAD_SCRIPT), [str(raw_dir)])
    by_date = defaultdict(list)
    for edex in sorted(kitti_dir.glob("*/stereo.edex")):
        if edex.parent.name not in cal.SEQUENCE_DATES:
            raise PreparationError(f"{edex.parent} is not a KITTI odometry sequence (00-21)")
        by_date[cal.SEQUENCE_DATES[edex.parent.name]].append(edex.parent.name)
    if not by_date:
        raise PreparationError(f"no KITTI sequences (<seq>/stereo.edex) in {kitti_dir}")

    rotations = json.loads(dataset_file(__file__, GT_ROTATIONS).read_text())
    for date, sequences in sorted(by_date.items()):
        kitti, _ = cal.read_kitti_calibration(raw_dir / f"{date}_calib.zip", date)
        rerectify(kitti_dir, output_dir, sequences, kitti, new_calibration(raw_dir, date), np.array(rotations[date]))
    with_gt = [s for seqs in by_date.values() for s in seqs if (output_dir / s / "gt.txt").is_file()]
    write_configs(output_dir, sorted(with_gt))
    return output_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the recalibrated KITTI dataset."""
    parser = argparse.ArgumentParser(prog="prepare_kitti_recalibrated", description=__doc__, allow_abbrev=False)
    parser.add_argument("--kitti-dir", type=Path, required=True, help="KITTI odometry dataset from prepare_kitti.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory for the downloaded calibrations.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the recalibrated dataset.")
    args = parser.parse_args(argv)
    return run_preparation(lambda: prepare(args.kitti_dir, args.raw_dir, args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
