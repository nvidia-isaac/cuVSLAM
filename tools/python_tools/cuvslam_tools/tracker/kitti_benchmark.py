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

"""Export tracker poses in KITTI benchmark trajectory format."""

import os
from typing import Any, Mapping, Optional

import numpy as np

from cuvslam_tools.tracker.pose_utils import pose_to_transform


_KITTI_BASIS = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _pose_to_kitti_benchmark_transform(pose: Any) -> np.ndarray:
    """Convert a cuVSLAM pose to KITTI benchmark coordinate basis."""
    transform = pose_to_transform(pose)
    return _KITTI_BASIS @ transform @ np.linalg.inv(_KITTI_BASIS)


def save_poses_to_kitti_benchmark(poses: Mapping[int, Optional[Any]], output_path: str) -> int:
    """Write valid poses to a KITTI benchmark text file and return the count."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    n_skipped = sum(1 for pose in poses.values() if pose is None)
    if n_skipped:
        print(f"Warning: skipping {n_skipped} invalid poses while writing {output_path}")

    n_written = 0
    with open(output_path, "w") as f:
        for frame_id in sorted(poses):
            pose = poses[frame_id]
            if pose is None:
                continue
            transform = _pose_to_kitti_benchmark_transform(pose)
            values = [transform[row, col] for row in range(3) for col in range(4)]
            f.write(" ".join(f"{float(value):.12g}" for value in values))
            f.write("\n")
            n_written += 1
    return n_written


def export_kitti_benchmark_artifacts(
    world_from_rig: Mapping[int, Optional[Any]],
    loop_closures: Mapping[int, Any],
    output_dir: str,
    sequence_title: str,
    *,
    use_slam: bool,
    suffix: str = "",
) -> None:
    """Export odometry and optional loop-closure KITTI benchmark files."""
    if not output_dir or not sequence_title:
        return

    output_path = os.path.join(output_dir, f"{sequence_title}{suffix}.txt")
    n_written = save_poses_to_kitti_benchmark(world_from_rig, output_path)
    print(f"Saved KITTI benchmark poses ({n_written}) to {output_path}")

    if use_slam and loop_closures:
        lc_output_path = os.path.join(output_dir, f"{sequence_title}{suffix}_LC.txt")
        n_lc_written = save_poses_to_kitti_benchmark(loop_closures, lc_output_path)
        print(f"Saved KITTI benchmark loop-closure poses ({n_lc_written}) to {lc_output_path}")
