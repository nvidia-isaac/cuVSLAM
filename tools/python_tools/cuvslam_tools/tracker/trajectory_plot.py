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

"""Static trajectory plotting shared by Python and C++ tracking backends."""

from typing import Any, Mapping, Optional, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from cuvslam_tools.tracker.pose_utils import pose_to_transform


def plot_trajectory(
    poses: Mapping[int, Any],
    loop_closures: Mapping[int, Any],
    gt_poses: Optional[Sequence] = None,
    visualize_plot: bool = False,
    save_path: Optional[str] = None,
    gt_from_shuttle: bool = False,
    title: str = "Trajectory (X vs Z)",
) -> None:
    """Plot estimated poses and optional ground truth without binding-specific types."""
    pose_transforms = [pose_to_transform(poses[key]) for key in sorted(poses) if poses[key] is not None]
    if not pose_transforms:
        return
    lc_transforms = [pose_to_transform(loop_closures[key]) for key in sorted(loop_closures)]

    trajectory = [transform[:3, 3] for transform in pose_transforms]
    rotations = [Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True) for transform in pose_transforms]
    trans_zipped = list(zip(*trajectory))
    euler_zipped = list(zip(*rotations))
    lc_zipped = list(zip(*[transform[:3, 3] for transform in lc_transforms]))

    if gt_poses:
        gt_transforms = [pose_to_transform(pose) for pose in gt_poses]
        gt_trans_zipped = list(zip(*[pose[:3, 3] for pose in gt_transforms]))
        gt_euler_zipped = list(
            zip(*[Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True) for pose in gt_transforms])
        )
    else:
        gt_trans_zipped = []
        gt_euler_zipped = []

    grid = gridspec.GridSpec(2, 2, width_ratios=[1, 1])
    figure = plt.figure(figsize=(20, 10))
    translation_axes = figure.add_subplot(grid[0, 0])
    translation_axes.set_title("Translation (X, Y, Z) vs. Time")
    for index, label in enumerate(("x", "y", "z")):
        translation_axes.plot(trans_zipped[index], label=label)
        if gt_trans_zipped:
            translation_axes.plot(gt_trans_zipped[index], label=f"gt_{label}")
    translation_axes.set_xlabel("Time")
    translation_axes.set_ylabel("meters")
    translation_axes.legend()

    rotation_axes = figure.add_subplot(grid[1, 0])
    rotation_axes.set_title("Rotation (X, Y, Z) vs. Time")
    for index, label in enumerate(("roll", "pitch", "yaw")):
        rotation_axes.plot(euler_zipped[index], label=label)
        if gt_euler_zipped:
            rotation_axes.plot(gt_euler_zipped[index], label=f"gt_{label}")
    rotation_axes.set_xlabel("Time")
    rotation_axes.set_ylabel("degrees")
    rotation_axes.legend()

    bird_axes = figure.add_subplot(grid[:, 1])
    bird_axes.set_title(title)
    if gt_trans_zipped:
        bird_axes.plot(trans_zipped[0], trans_zipped[2], label="Backward pass" if gt_from_shuttle else "VO")
        bird_axes.plot(
            gt_trans_zipped[0],
            gt_trans_zipped[2],
            label="Forward pass" if gt_from_shuttle else "GT",
            linewidth=2,
        )
    else:
        bird_axes.plot(trans_zipped[0], trans_zipped[2], label="VO")
    if lc_zipped:
        bird_axes.scatter(lc_zipped[0], lc_zipped[2], label="LC", c="green", s=20)

    x_min, x_max = min(trans_zipped[0]), max(trans_zipped[0])
    z_min, z_max = min(trans_zipped[2]), max(trans_zipped[2])
    max_range = max(x_max - x_min, z_max - z_min) * 1.1
    if max_range == 0:
        max_range = 1.0
    bird_axes.set_xlim((x_min + x_max - max_range) / 2, (x_min + x_max + max_range) / 2)
    bird_axes.set_ylim((z_min + z_max - max_range) / 2, (z_min + z_max + max_range) / 2)
    bird_axes.set_xlabel("x")
    bird_axes.set_ylabel("z")
    bird_axes.legend()
    bird_axes.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if visualize_plot:
        plt.show(block=True)
    plt.close()
