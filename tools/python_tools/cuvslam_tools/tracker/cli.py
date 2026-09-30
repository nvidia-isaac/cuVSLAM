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

"""Console entry point and shared argument definitions for cuVSLAM tracking."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Union


def _parse_int_list(value: Union[str, Iterable[int]]) -> list[int]:
    """Parse a comma-separated string or iterable into a list of integers."""
    if isinstance(value, str):
        if not value:
            return []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _str2bool(value: Union[str, bool]) -> bool:
    """Parse common command-line spellings for boolean values."""
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false, 1/0, yes/no, y/n).")


def _normalized_path(path: str) -> Path:
    """Resolve a command-line path without requiring that it exists."""
    return Path(path).expanduser().resolve(strict=False)


def _normalize_tracker_binding_args(args: argparse.Namespace) -> None:
    """Convert pure-Python parser values into cuVSLAM binding enum values."""
    from cuvslam_tools.tracker import conversions as conv

    if isinstance(args.multicam_mode, str):
        args.multicam_mode = conv.str2multicam_mode(args.multicam_mode)
    if isinstance(args.odometry_mode, str):
        args.odometry_mode = conv.str2odometry_mode(args.odometry_mode)
    if isinstance(getattr(args, 'vpr_mode', None), str):
        args.vpr_mode = conv.str2vpr_mode(args.vpr_mode)
    if isinstance(getattr(args, 'loop_closure_mode', None), str):
        args.loop_closure_mode = conv.str2loop_closure_mode(args.loop_closure_mode)


def add_tracker_arguments(parser: argparse.ArgumentParser) -> None:
    """Add tracker workflow arguments to a parser."""
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help=(
            "Path to an EDEX sequence directory, or to a supported video file. "
            "For ROS bag input, first convert the bag to EDEX with rosbag_extract_edex."
        ),
    )
    parser.add_argument("--edex", type=str, default="", help="Alias for --dataset when running an EDEX sequence.")
    parser.add_argument(
        "--edex_filename",
        type=str,
        default="stereo.edex",
        help="EDEX config filename used with --edex when --config_path is not provided.",
    )
    parser.add_argument("--sequence_title", type=str, default="sequence", help="Title of the sequence.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for plots, stats, and optional tracker data.",
    )
    parser.add_argument("--save_output_tracker_data", action="store_true", help="Save output tracker data.")
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print the complete odometry and SLAM configurations.",
    )
    parser.add_argument(
        "--visualize_rerun",
        action="store_true",
        help="Enable real-time visualization of tracking results using Rerun viewer.",
    )
    parser.add_argument("--visualize_plot", action="store_true", help="Show trajectory plots.")
    parser.add_argument("--num_loops", type=int, default=0, help="Number of repeat or shuttle loops.")
    parser.add_argument(
        "--repeat_type",
        choices=["none", "repeat", "shuttle"],
        default="none",
        help="Replay mode.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="gt.txt",
        help=(
            "KITTI-format ground-truth poses, absolute or relative to --dataset. "
            "Tracking stops with an error when the file is missing."
        ),
    )
    parser.add_argument(
        "--gt_from_shuttle",
        type=_str2bool,
        default=False,
        help=(
            "Score the backward shuttle pass against the forward one instead of a ground-truth file. "
            "Requires --repeat_type shuttle with --num_loops > 0, and rules out --gt_path."
        ),
    )
    parser.add_argument("--blackout_period", type=int, default=0, help="Blackout series period in frames.")
    parser.add_argument("--blackout_duration", type=int, default=10, help="Consecutive blacked-out frames per series.")
    parser.add_argument("--config_path", type=str, default="", help="Path to the stereo.edex file.")
    parser.add_argument("--use_segments", action="store_true", default=False, help="Use segments for error calculation.")
    parser.add_argument(
        "--segment_lengths",
        type=_parse_int_list,
        default=[],
        help="Comma-separated segment lengths for error calculation.",
    )

    parser.add_argument(
        "--multicam_mode",
        type=str,
        choices=["performance", "precision", "moderate"],
        default="performance",
        help="Multicamera mode: performance, precision, or moderate.",
    )
    parser.add_argument(
        "--odometry_mode",
        type=str,
        choices=["mono", "multicamera", "inertial", "rgbd"],
        default="multicamera",
        help="Odometry mode: mono, multicamera, inertial, rgbd.",
    )
    parser.add_argument(
        "--use_gpu",
        type=_str2bool,
        default=True,
        help="Enable GPU acceleration.",
    )
    parser.add_argument(
        "--async_sba",
        type=_str2bool,
        default=False,
        help="Enable asynchronous Sparse Bundle Adjustment. Selects the realtime tracker mode, "
             "in which case --sync_slam must be false.",
    )
    parser.add_argument(
        "--use_motion_model",
        type=_str2bool,
        default=True,
        help="Enable motion model for prediction.",
    )
    parser.add_argument(
        "--use_denoising",
        type=_str2bool,
        default=False,
        help="Enable denoising of input images.",
    )
    parser.add_argument(
        "--rectified_stereo_camera",
        type=_str2bool,
        default=False,
        help="Enable rectified stereo camera tracking mode.",
    )
    parser.add_argument(
        "--enable_observations_export",
        type=_str2bool,
        default=False,
        help="Enable exporting landmark observations during tracking.",
    )
    parser.add_argument(
        "--enable_landmarks_export",
        type=_str2bool,
        default=False,
        help="Enable exporting landmarks during tracking.",
    )
    parser.add_argument(
        "--enable_final_landmarks_export",
        type=_str2bool,
        default=False,
        help="Enable exporting final landmarks.",
    )
    parser.add_argument(
        "--max_frame_delta_s",
        type=float,
        default=0.1,
        help="Maximum time difference between frames in seconds.",
    )
    parser.add_argument(
        "--debug_dump_directory",
        type=str,
        default="",
        help="Directory for debug data dumps.",
    )
    parser.add_argument(
        "--debug_imu_mode",
        type=_str2bool,
        default=False,
        help="Enable IMU debug mode.",
    )
    parser.add_argument("--use_slam", type=_str2bool, default=False, help="Enable SLAM mode.")
    parser.add_argument("--sync_slam", type=_str2bool, default=True,
                        help="Run SLAM in the calling thread. Must be the opposite of --async_sba.")

    parser.add_argument(
        "--depth_scale_factor",
        type=float,
        default=1.0,
        help="Depth scale factor for RGBD mode.",
    )
    parser.add_argument(
        "--enable_depth_stereo_tracking",
        type=_str2bool,
        default=False,
        help="Enable depth-stereo tracking in RGBD mode.",
    )
    parser.add_argument(
        "--cache_uncompressed",
        "--save_tga_images",
        dest="cache_uncompressed",
        action="store_true",
        help="Cache PNG images as .png.tga sidecar files while loading EDEX datasets.",
    )


def add_loop_closure_arguments(parser: argparse.ArgumentParser) -> None:
    """Add SLAM loop closure and place recognition arguments to a parser.

    Separate from add_tracker_arguments() because cuvslam_vpr_reporter calls that too but owns its own place
    recognition flags, and configures SLAM for each backend it evaluates itself.
    """
    parser.add_argument(
        "--loop_closure_mode",
        type=str.lower,
        choices=["default", "vpr"],
        default="default",
        help="Loop closure backend: default (spatial search) or vpr (visual place recognition candidates). "
             "Case insensitive.",
    )
    parser.add_argument(
        "--vpr_mode",
        type=str.lower,
        choices=["off", "simple", "bow", "dbow2", "anyloc"],
        default="off",
        help="Visual place recognition backend. Required (non-off) when --loop_closure_mode=vpr. Case insensitive.",
    )
    parser.add_argument(
        "--vpr_model_path",
        type=str,
        default="",
        help="Model file the VPR backend needs; --vpr_mode=anyloc reads a DINOv2 ONNX model from here.",
    )
    parser.add_argument(
        "--max_map_size",
        type=int,
        default=None,
        help="Maximum number of pose graph nodes SLAM keeps, 0 for no limit; by default the SLAM default. A node "
             "merged away also leaves the place recognition map, so compare --loop_closure_mode vpr with 0.",
    )


def validate_loop_closure_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace,
                                    require_use_slam: bool = True) -> None:
    """Exit through parser.error() on loop closure flags that cannot work, before any tracking starts.

    Args:
        parser: Parser that defined the arguments with add_loop_closure_arguments().
        args: The parsed arguments.
        require_use_slam: Also reject a non-default --loop_closure_mode or --vpr_mode without --use_slam true.
            cuvslam_reporter passes False: it takes use_slam from each sequence of its config.
    """
    if args.loop_closure_mode == "vpr" and args.vpr_mode == "off":
        parser.error("--loop_closure_mode vpr needs a place recognition backend; "
                     "set --vpr_mode to simple, bow, dbow2 or anyloc")
    if require_use_slam and not args.use_slam and (args.loop_closure_mode != "default" or args.vpr_mode != "off"):
        parser.error("--loop_closure_mode and --vpr_mode only affect SLAM; add --use_slam true")
    if args.vpr_mode == "anyloc" and not os.path.isfile(args.vpr_model_path):
        parser.error("--vpr_mode anyloc needs a DINOv2 ONNX model; set --vpr_model_path to one exported by "
                     "cuvslam_export_dinov2")
    if args.max_map_size is not None and args.max_map_size < 0:
        parser.error("--max_map_size must be 0 (no limit) or a positive number of pose graph nodes")


def stat_to_dict(stat: Any) -> dict:
    """Convert a tracker Stat object to a JSON-serializable dictionary."""
    return {
        "sequence_title": stat.sequence_title,
        "n_frames": stat.n_frames,
        "tracking_time": stat.tracking_time,
        "average_fps": stat.average_fps,
        "bird_view_with_errors_path": stat.bird_view_with_errors_path,
        "gt_av_translation_error": stat.gt_av_translation_error,
        "gt_av_rotation_error": stat.gt_av_rotation_error,
        "gt_n_error_segments": stat.gt_n_error_segments,
        "gt_simple_error": stat.gt_simple_error,
        "num_tracking_losts": stat.num_tracking_losts,
        "odometry_mode": stat.odometry_mode,
        "num_loop_closures": stat.num_loop_closures,
        "loop_closure_mode": stat.loop_closure_mode,
        "seg_err_points": getattr(stat, "seg_err_points", []),
    }


def save_tracker_stats(stat: Any, output_dir: str) -> None:
    """Write tracker statistics JSON under the output directory."""
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "tracker_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stat_to_dict(stat), f, indent=2)
    print(f"Saved tracker stats to {stats_path}")


def main(argv: Optional[list[str]] = None) -> int:
    """Parse tracker CLI arguments, run tracking, and print or save stats."""
    parser = argparse.ArgumentParser(prog="cuvslam_tracker")
    add_tracker_arguments(parser)
    add_loop_closure_arguments(parser)
    args = parser.parse_args(argv)
    validate_loop_closure_arguments(parser, args)

    if args.edex:
        if args.dataset and _normalized_path(args.dataset) != _normalized_path(args.edex):
            parser.error("--dataset and --edex point to different inputs")
        args.dataset = args.edex
        if not args.config_path:
            args.config_path = os.path.join(args.edex, args.edex_filename)

    if not args.dataset:
        parser.error("--dataset is required")

    try:
        from cuvslam_tools.tracker.runner import track
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        if getattr(exc, "name", None) == "cuvslam" or "cuvslam" in str(exc):
            parser.error("cuvslam_tracker requires the cuVSLAM Python binding to be installed")
        raise

    try:
        _normalize_tracker_binding_args(args)
        tracker_results = track(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output_dir:
        save_tracker_stats(tracker_results.stat, args.output_dir)
    else:
        print(json.dumps(stat_to_dict(tracker_results.stat), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
