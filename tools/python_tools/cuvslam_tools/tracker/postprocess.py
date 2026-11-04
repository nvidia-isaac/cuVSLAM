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

"""Backend-neutral metrics, plot, and benchmark export finalization."""

import os
from typing import Any, Mapping, Optional, Sequence

from cuvslam_tools.tracker.kitti_benchmark import export_kitti_benchmark_artifacts
from cuvslam_tools.tracker.metrics import calculate_sequence_errors
from cuvslam_tools.tracker.results import Stat
from cuvslam_tools.tracker.trajectory_plot import plot_trajectory


def finalize_trajectory(
    poses: Mapping[int, Any],
    loop_closures: Mapping[int, Any],
    gt_transforms: Sequence,
    stat: Stat,
    *,
    frame_metadata: Mapping[int, dict],
    use_segments: bool,
    segment_lengths: list[int],
    num_loops: int,
    repeat_type: str,
    output_dir: str,
    sequence_title: str,
    use_slam: bool,
    visualize_plot: bool,
    gt_from_shuttle: bool,
    frame_mapping: Optional[Mapping[int, int]] = None,
    suffix: str = "",
) -> None:
    """Populate metrics and write artifacts for a completed backend run."""
    if gt_transforms:
        calculate_sequence_errors(
            poses,
            list(gt_transforms),
            stat,
            dict(frame_metadata),
            use_segments,
            segment_lengths,
            num_loops,
            repeat_type,
            frame_mapping=dict(frame_mapping) if frame_mapping is not None else None,
        )

    plot_path = None
    if output_dir:
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        plot_path = os.path.join(output_dir, "plots", f"{sequence_title}{suffix}.png")
        stat.bird_view_with_errors_path = os.path.abspath(plot_path)

    plot_trajectory(
        poses,
        loop_closures,
        list(gt_transforms),
        visualize_plot,
        plot_path,
        gt_from_shuttle,
    )
    export_kitti_benchmark_artifacts(
        poses,
        loop_closures,
        output_dir,
        sequence_title,
        use_slam=use_slam,
        suffix=suffix,
    )
