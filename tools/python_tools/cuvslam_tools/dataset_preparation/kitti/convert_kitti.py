#!/usr/bin/env python3
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

"""Convert KITTI odometry dataset from raw zip archives to cuVSLAM converted format.

Input (raw/):
  data_odometry_gray.zip  — images for all 22 sequences (00-21)
  data_odometry_poses.zip — ground-truth poses for sequences 00-10

Output (converted/):
  <seq>/00/<seq>.0.XXXX.png   — left camera images, 1-indexed
  <seq>/01/<seq>.1.XXXX.png   — right camera images, 1-indexed
  <seq>/stereo.edex           — calibration in cuVSLAM edex format
  <seq>/gt.txt                — ground-truth poses (sequences 00-10 only)
  kitti-slam_gt.cfg           — reporter config: GT sequences, SLAM mode
  kitti-vio_gt.cfg            — reporter config: GT sequences, VIO mode
  kitti-vio_slam.cfg          — reporter config: all sequences, both modes
  kitti-vio_slam_gt.cfg       — reporter config: GT sequences, both modes

Usage:
  python3 convert_kitti.py [RAW_DIR] [OUT_DIR]

  RAW_DIR  defaults to the directory containing this script
  OUT_DIR  defaults to <parent_of_RAW_DIR>/converted
"""

import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

# ---------------------------------------------------------------------------
# Sequences with ground-truth poses
# ---------------------------------------------------------------------------
GT_SEQS = {f"{i:02d}" for i in range(11)}  # '00'..'10'


class ConversionError(ValueError):
    """Raised when the raw KITTI odometry archives cannot be converted."""

# ---------------------------------------------------------------------------
# Calibration: exact string representations matched to the reference converted
# folder.  Keyed by rounded focal length (integer) to identify the recording
# date group:
#   2011_10_03 → fx ≈ 719  (seqs 00-02, 13-21)
#   2011_09_26 → fx ≈ 722  (seq 03)
#   2011_09_30 → fx ≈ 707  (seqs 04-12)
# The baseline is always taken from the 2011_10_03 calibration
# (0.5371657188644179), matching the existing converted folder.
# ---------------------------------------------------------------------------
_FOCAL_TO_CALIB = {
    719: {"focal": "718.855999999", "cx": "607.192799999", "cy": "185.2157"},
    722: {"focal": "721.537700002", "cx": "609.5593",      "cy": "172.854"},
    707: {"focal": "707.091199998", "cx": "601.8873",      "cy": "183.1104"},
}
_BASELINE_STR = "0.5371657188644179"


def _calib_strs(fx: float, cx: float, cy: float) -> dict:
    """Return exact string representations of calibration values."""
    key = round(fx)
    if key in _FOCAL_TO_CALIB:
        return _FOCAL_TO_CALIB[key]
    # Fallback for unknown calibrations — use Python's repr
    return {"focal": repr(fx), "cx": repr(cx), "cy": repr(cy)}


# ---------------------------------------------------------------------------
# PNG utilities
# ---------------------------------------------------------------------------

def read_png_dimensions(data: bytes) -> tuple:
    """Return (width, height) from the first 24 bytes of a PNG file."""
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError("Not a PNG file")
    w = struct.unpack('>I', data[16:20])[0]
    h = struct.unpack('>I', data[20:24])[0]
    return w, h


def _safe_extractall(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a ZIP archive after validating all member paths stay in destination."""
    destination = destination.resolve()
    for member in archive.infolist():
        posix_path = PurePosixPath(member.filename)
        windows_path = PureWindowsPath(member.filename)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        try:
            (destination / member.filename).resolve().relative_to(destination)
        except ValueError as exc:
            raise ValueError(f"Unsafe ZIP member path: {member.filename}") from exc
    archive.extractall(destination)


# ---------------------------------------------------------------------------
# Calibration parsing
# ---------------------------------------------------------------------------

def parse_calib(calib_text: str) -> dict:
    """Parse KITTI calib.txt.  Returns {fx, cx, cy, baseline}."""
    matrices = {}
    for line in calib_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        key, *vals = line.split()
        matrices[key.rstrip(":")] = list(map(float, vals))

    P0 = matrices["P0"]   # row-major 3x4
    P1 = matrices["P1"]

    return {
        "fx":       P0[0],           # P0[0,0]
        "cx":       P0[2],           # P0[0,2]
        "cy":       P0[6],           # P0[1,2]
        "baseline": -P1[3] / P0[0],  # -P1[0,3] / fx
    }


# ---------------------------------------------------------------------------
# Edex generation
# ---------------------------------------------------------------------------

_EDEX_TEMPLATE = """\
[
    {{
        "cameras": [
            {{
                "intrinsics": {{
                    "distortion_model": "pinhole",
                    "distortion_params": [],
                    "focal": [{focal}, {focal}],
                    "principal": [{cx}, {cy}],
                    "size": [{width}, {height}]
                }},
                "transform": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ]
            }},
            {{
                "intrinsics": {{
                    "distortion_model": "pinhole",
                    "distortion_params": [],
                    "focal": [{focal}, {focal}],
                    "principal": [{cx}, {cy}],
                    "size": [{width}, {height}]
                }},
                "transform": [
                    [1.0, 0.0, 0.0, {baseline}],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]
                ]
            }}
        ],
        "frame_end": {frame_end},
        "frame_start": 1,
        "version": "0.9"
    }},
    {{
        "fps": 10,
        "points2d": {{}},
        "points3d": {{}},
        "rig_positions": {{}},
        "sequence": [["00/{seq}.0.0001.png"], ["01/{seq}.1.0001.png"]]
    }}
]"""


def make_edex(seq: str, fx: float, cx: float, cy: float,
              w: int, h: int, num_frames: int) -> str:
    """Return stereo.edex content (no trailing newline)."""
    strs = _calib_strs(fx, cx, cy)
    return _EDEX_TEMPLATE.format(
        focal=strs["focal"],
        cx=strs["cx"],
        cy=strs["cy"],
        width=w,
        height=h,
        baseline=_BASELINE_STR,
        frame_end=num_frames,
        seq=seq,
    )


# ---------------------------------------------------------------------------
# Config file generation
# ---------------------------------------------------------------------------

def _json_val(v) -> str:
    """Format a scalar as JSON for generated reporter config files."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return f'"{v}"'


def _seq_cfg_entry(seq: str, mode: str) -> list:
    """Return the ordered dict items for one sequence entry in a cfg file.

    mode: 'odom' or 'slam'
    """
    is_slam = mode == "slam"
    label = "SLAM" if is_slam else "ODOM"

    fields = []
    fields.append(("enable", True))
    fields.append(("sequence_folder", seq))
    fields.append(("edex_file", "stereo.edex"))
    fields.append(("precompute_2d_tracks", False))
    fields.append(("precompute_key_frames", False))
    fields.append(("use_gt_scale", False))
    fields.append(("sequence_title", f"KITTI-{int(seq):02d}-{label}"))
    if seq in GT_SEQS:
        fields.append(("gt_file_path", "gt.txt"))
    if is_slam:
        fields.append(("use_slam", True))
    return fields


def _format_cfg(sequences: list, slam: bool, odom: bool) -> str:
    """Build the exact cfg file format matching the reference converted folder."""
    lines = ["{"]
    lines.append('    "version": "0.1",')
    lines.append('    "write_cache": false,')
    lines.append('    "use_cuda": false,')
    lines.append('    "dataset_folder": "kitti/",')
    lines.append('    "use_icp_scaling": false,')
    lines.append('    "segment_lengths": [100, 200, 300, 400, 500, 600, 700, 800],')
    lines.append('    "sequence_cfgs": [')

    entries = []
    for seq in sequences:
        if odom:
            entries.append(_seq_cfg_entry(seq, "odom"))
        if slam:
            entries.append(_seq_cfg_entry(seq, "slam"))

    for ei, fields in enumerate(entries):
        lines.append("        {")
        for fi, (k, v) in enumerate(fields):
            comma = "," if fi < len(fields) - 1 else ""
            lines.append(f'            "{k}": {_json_val(v)}{comma}')
        comma = "," if ei < len(entries) - 1 else ""
        lines.append(f"        }}{comma}")

    lines.append("  ]")
    lines.append("}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def write_configs(out_dir: Path, all_seqs: list):
    """Write the standard KITTI reporter config files to the output directory."""
    gt_seqs = [s for s in all_seqs if s in GT_SEQS]

    configs = {
        "kitti-slam_gt.cfg":     _format_cfg(gt_seqs,  slam=True,  odom=False),
        "kitti-vio_gt.cfg":      _format_cfg(gt_seqs,  slam=False, odom=True),
        "kitti-vio_slam.cfg":    _format_cfg(all_seqs, slam=True,  odom=True),
        "kitti-vio_slam_gt.cfg": _format_cfg(gt_seqs,  slam=True,  odom=True),
    }

    for name, text in configs.items():
        (out_dir / name).write_text(text)
        print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(raw_dir: Path, out_dir: Path):
    """Convert KITTI odometry archives into cuVSLAM sequence folders."""
    gray_zip  = raw_dir / "data_odometry_gray.zip"
    poses_zip = raw_dir / "data_odometry_poses.zip"
    calib_zip = raw_dir / "data_odometry_calib.zip"

    for archive in (gray_zip, poses_zip, calib_zip):
        if not archive.exists():
            raise ConversionError(f"{archive} not found")

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kitti_convert_") as tmp:
        tmp_path  = Path(tmp)
        gray_tmp  = tmp_path / "gray"
        poses_tmp = tmp_path / "poses"
        calib_tmp = tmp_path / "calib"

        print("Extracting data_odometry_gray.zip …")
        with zipfile.ZipFile(gray_zip) as archive:
            _safe_extractall(archive, gray_tmp)

        print("Extracting data_odometry_poses.zip …")
        with zipfile.ZipFile(poses_zip) as archive:
            _safe_extractall(archive, poses_tmp)

        print("Extracting data_odometry_calib.zip …")
        with zipfile.ZipFile(calib_zip) as archive:
            _safe_extractall(archive, calib_tmp)

        sequences_root = gray_tmp  / "dataset" / "sequences"
        poses_root     = poses_tmp / "dataset" / "poses"
        calib_root     = calib_tmp / "dataset" / "sequences"

        sequences = sorted(p.name for p in sequences_root.iterdir() if p.is_dir())
        print(f"Found sequences: {sequences}\n")

        for seq in sequences:
            seq_src  = sequences_root / seq
            seq_dst  = out_dir / seq
            cam0_dst = seq_dst / "00"
            cam1_dst = seq_dst / "01"
            cam0_dst.mkdir(parents=True, exist_ok=True)
            cam1_dst.mkdir(parents=True, exist_ok=True)

            # Parse calibration from data_odometry_calib.zip
            calib = parse_calib((calib_root / seq / "calib.txt").read_text())
            fx, cx, cy = calib["fx"], calib["cx"], calib["cy"]

            # Sorted image lists (raw names: 000000.png … NNNNNN.png)
            cam0_imgs = sorted((seq_src / "image_0").glob("*.png"))
            cam1_imgs = sorted((seq_src / "image_1").glob("*.png"))

            num_frames = min(len(cam0_imgs), len(cam1_imgs))
            if len(cam0_imgs) != len(cam1_imgs):
                print(f"  WARNING: seq {seq}: {len(cam0_imgs)} left vs "
                      f"{len(cam1_imgs)} right images — using {num_frames}")

            # Image dimensions from first left image (first 24 bytes suffice)
            with open(cam0_imgs[0], "rb") as f:
                w, h = read_png_dimensions(f.read(24))

            print(f"Sequence {seq}: {num_frames} frames, {w}x{h}, "
                  f"fx={fx:.4f}, cx={cx:.4f}, cy={cy:.4f}")

            # Rename and copy images: raw 000000.png → out seq.0.0001.png (1-indexed)
            for i, src in enumerate(cam0_imgs[:num_frames], start=1):
                shutil.copy2(src, cam0_dst / f"{seq}.0.{i:04d}.png")

            for i, src in enumerate(cam1_imgs[:num_frames], start=1):
                shutil.copy2(src, cam1_dst / f"{seq}.1.{i:04d}.png")

            # stereo.edex (no trailing newline, matching reference)
            edex_text = make_edex(seq, fx, cx, cy, w, h, num_frames)
            (seq_dst / "stereo.edex").write_text(edex_text)

            # Ground-truth poses (sequences 00-10)
            poses_file = poses_root / f"{int(seq):02d}.txt"
            if poses_file.exists():
                (seq_dst / "gt.txt").write_bytes(poses_file.read_bytes())
                print("  copied gt.txt")

        print("\nGenerating config files …")
        write_configs(out_dir, sequences)

    print(f"\nDone.  Output written to: {out_dir}")


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _repo_root  = _script_dir.parents[4]

    args = sys.argv[1:]
    raw = Path(args[0]) if len(args) >= 1 else _repo_root / "datasets" / "kitti" / "raw"
    out = Path(args[1]) if len(args) >= 2 else _repo_root / "datasets" / "converted"

    print(f"RAW dir : {raw}")
    print(f"OUT dir : {out}\n")
    try:
        convert(raw, out)
    except ConversionError as exc:
        sys.exit(f"ERROR: {exc}")
