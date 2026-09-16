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

"""Binding-free result models shared by tracking backends."""

from dataclasses import dataclass, field


@dataclass
class Stat:
    """Summary statistics and report paths from one tracking run."""

    sequence_title: str = ""
    n_frames: int = 0
    tracking_time: float = 0.0
    average_fps: float = -1.0
    bird_view_with_errors_path: str = ""
    gt_av_translation_error: float = 0.0
    gt_av_rotation_error: float = 0.0
    gt_n_error_segments: int = 0
    gt_simple_error: float = 0.0
    num_tracking_losts: int = 0
    odometry_mode: str = ""
    seg_err_points: list[dict[str, float]] = field(default_factory=list)
