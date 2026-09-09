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
"""
Patch a cuVSLAM debug-dump EDEX so the tracker CLI can consume it.

The debug dump written by isaac_ros_visual_slam has several format issues
that prevent the tracker from reading it. This script fixes them and writes
patched files to an output directory.

Usage:
    python3 patch_debug_dump_edex.py <dump_dir> [output_dir]

    dump_dir   -- directory containing stereo.edex, frame_metadata.jsonl, images/
    output_dir -- where to write patched files (default: <dump_dir>_patched)

The output directory contains:
    stereo.edex          -- patched EDEX (same name so tracker uses it by default)
    frame_metadata.jsonl -- patched metadata
    images/              -- symlink to original images (not copied, too large)
    IMU.jsonl            -- copied from dump_dir if present
"""

import json
import shutil
import sys
from pathlib import Path


# RealSense D435i defaults -- update for other sensors
IMU_DEFAULTS = {
    "gyro_noise_density": 1.0e-3,
    "accel_noise_density": 1.0e-2,
    "frequency": 200.0,
    "g": 9.81,
}


def patch_debug_dump_edex(dump_dir: Path, out_dir: Path) -> None:
    # --- Validate inputs before doing any work ---
    if not dump_dir.exists() or not dump_dir.is_dir():
        raise FileNotFoundError(f"dump_dir does not exist or is not a directory: {dump_dir}")

    edex_src = dump_dir / "stereo.edex"
    meta_src = dump_dir / "frame_metadata.jsonl"
    images_dir = dump_dir / "images"

    if not edex_src.is_file():
        raise FileNotFoundError(f"Required file not found: edex_src={edex_src}")
    if not meta_src.is_file():
        raise FileNotFoundError(f"Required file not found: meta_src={meta_src}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Required directory not found: images_dir={images_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Count actual frames from images directory ---
    cam0_frames = sorted(images_dir.glob("cam0.*.png"))
    frame_count = len(cam0_frames)
    if frame_count == 0:
        print(f"WARNING: no cam0.*.png found in {images_dir} — frame_end fix will be skipped")
    else:
        print(f"Found {frame_count} frames in {images_dir}")

    # --- Patch stereo.edex ---
    with open(edex_src) as f:
        edex = json.load(f)

    changed = False
    for entry in edex:
        if not isinstance(entry, dict):
            continue

        # Fix 1: frame_end should be last frame index (frame_count - 1)
        if "frame_end" in entry and frame_count > 0:
            expected = frame_count - 1
            if entry["frame_end"] != expected:
                print(f"  [fix 1] frame_end: {entry['frame_end']} -> {expected}")
                entry["frame_end"] = expected
                changed = True

        # Fix 2: distortion_params missing from camera intrinsics
        for cam in entry.get("cameras", []):
            intr = cam.get("intrinsics", {})
            if "distortion_params" not in intr:
                print(f"  [fix 2] added distortion_params: [] to camera {cam.get('name', '?')}")
                intr["distortion_params"] = []
                changed = True

        # Fix 3: sequence entries are bare strings -- must be lists of strings
        if "sequence" in entry:
            new_seq = []
            any_fixed = False
            for item in entry["sequence"]:
                if isinstance(item, str):
                    new_seq.append([item])
                    any_fixed = True
                else:
                    new_seq.append(item)
            if any_fixed:
                print(f"  [fix 3] wrapped {len(new_seq)} sequence entries in lists")
                entry["sequence"] = new_seq
                changed = True

        # Fix 4: IMU section missing noise params
        imu = entry.get("imu")
        if imu is not None:
            for key, val in IMU_DEFAULTS.items():
                if key not in imu:
                    print(
                        f"  [fix 4] added imu.{key} = {val}"
                        f"  (D435i default -- update for other sensors)"
                    )
                    imu[key] = val
                    changed = True

    edex_out = out_dir / "stereo.edex"
    with open(edex_out, "w") as f:
        json.dump(edex, f, indent=2)
    status = "patched" if changed else "no fixes needed, copied"
    print(f"  -> stereo.edex {status}: {edex_out}")

    # --- Patch frame_metadata.jsonl ---
    raw = meta_src.read_text()
    # Fix 5: "depths":[} -> "depths":[]}  (missing ] before closing })
    fixed = raw.replace('"depths":[}', '"depths":[]}')
    n_fixed = raw.count('"depths":[}')
    meta_out = out_dir / "frame_metadata.jsonl"
    meta_out.write_text(fixed)
    if n_fixed:
        print(f"  [fix 5] frame_metadata.jsonl: fixed {n_fixed} occurrences of [}} -> []}} in {meta_out}")
    else:
        print(f"  frame_metadata.jsonl: no fixes needed, copied to {meta_out}")

    # --- Symlink images/ (not copied -- too large) ---
    images_link = out_dir / "images"
    if not images_link.exists():
        images_link.symlink_to(images_dir.resolve())
        print(f"  -> symlinked images/ -> {images_dir.resolve()}")

    # --- Copy IMU.jsonl if present ---
    imu_src = dump_dir / "IMU.jsonl"
    if imu_src.exists():
        shutil.copy2(imu_src, out_dir / "IMU.jsonl")
        print("  -> copied IMU.jsonl")

    print(f"\nPatched EDEX ready at: {out_dir}")
    print("\nRun tracker with:")
    print(f"  CUVSLAM_DATASETS={out_dir.parent}/ \\")
    print("  CUVSLAM_OUTPUT=/tmp/cuvslam_output/ \\")
    print("  ./build/bin/tracker \\")
    print(f"  -edex {out_dir.name} \\")
    print("  -edex_filename stereo.edex \\")
    print(f"  -output_edex {out_dir}/result.edex \\")
    print("  -mode multicamera")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    dump = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else dump.parent / (dump.name + "_patched")
    patch_debug_dump_edex(dump, out)
