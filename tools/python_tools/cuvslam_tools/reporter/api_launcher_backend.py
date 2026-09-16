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

"""Subprocess adapter from cuvslam_reporter to cuvslam_api_launcher."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cuvslam_tools.tracker.ground_truth import load_gt_transforms, resolve_gt_file
from cuvslam_tools.tracker.pose_utils import json_pose_to_transform
from cuvslam_tools.tracker.postprocess import finalize_trajectory
from cuvslam_tools.tracker.results import Stat


_ODOMETRY_MODES = {"multicamera": 0, "inertial": 1, "rgbd": 2, "mono": 3}
_MULTICAM_MODES = {"performance": 0, "precision": 1, "moderate": 2}
_RESERVED_FLAGS = {
    "edex",
    "cameras",
    "repeat",
    "shuttle",
    "ignore_tracking_errors",
    "blackout_period",
    "blackout_duration",
    "report_output",
    "print_odom_poses",
    "print_slam_poses",
    "print_format",
    "print_stats",
    "debug_dump",
    "depth_scale_factor",
    "cache_uncompressed",
}


@dataclass
class FrameRecord:
    """One attempted launcher frame."""

    frame_id: int
    source_frame_number: int
    timestamp_ns: int
    tracking_valid: bool
    odom_pose: Optional[np.ndarray]
    slam_pose: Optional[np.ndarray]


@dataclass
class LauncherRecords:
    """Parsed machine-readable launcher output."""

    frames: list[FrameRecord]
    final_slam_poses: dict[int, np.ndarray]
    loop_closures: dict[int, np.ndarray]
    summary: dict


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _flag_name(argument: str) -> str:
    return argument.lstrip("-").split("=", 1)[0]


def validate_launcher_args(arguments: list[str]) -> None:
    """Reject passthrough flags that would break the reporter-owned contract."""
    previous_may_take_value = False
    for argument in arguments:
        if not argument.startswith("-"):
            if previous_may_take_value:
                previous_may_take_value = False
                continue
            raise ValueError(f"Launcher passthrough arguments must be gflags, got {argument!r}")
        name = _flag_name(argument)
        if name in _RESERVED_FLAGS or name.startswith("cfg_"):
            raise ValueError(f"Launcher flag --{name} is managed by cuvslam_reporter")
        previous_may_take_value = "=" not in argument


def resolve_api_launcher(explicit_path: str = "") -> str:
    """Find an executable API launcher using the documented lookup order."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    else:
        build_dir = os.environ.get("CUVSLAM_BUILD_DIR")
        if build_dir:
            candidates.append(Path(build_dir) / "bin" / "cuvslam_api_launcher")
        path_match = shutil.which("cuvslam_api_launcher")
        if path_match:
            candidates.append(Path(path_match))

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    if explicit_path:
        raise FileNotFoundError(f"cuvslam_api_launcher is not executable: {explicit_path}")
    raise FileNotFoundError(
        "cuvslam_api_launcher not found; pass --api_launcher_path, set CUVSLAM_BUILD_DIR, or add it to PATH"
    )


def _rgbd_edex_settings(edex_path: str) -> tuple[int, float]:
    """Read the depth camera and integer-PNG divisor used by the Python EDEX path."""
    with open(edex_path, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, list) or not config or not isinstance(config[0], dict):
        raise ValueError(f"Invalid RGB-D EDEX configuration: {edex_path}")
    for camera in config[0].get("cameras", []):
        if "depth_id" in camera:
            return int(camera["depth_id"]), float(camera.get("depth_scale_factor", 1.0))
    raise ValueError(f"RGB-D EDEX has no camera depth_id: {edex_path}")


def build_launcher_command(args, launcher: str, report_path: str) -> list[str]:
    """Translate reporter tracker options to launcher-owned gflags."""
    odometry_mode = str(args.odometry_mode).lower()
    multicam_mode = str(args.multicam_mode).lower()
    if odometry_mode not in _ODOMETRY_MODES:
        raise ValueError(f"Unsupported C++ odometry mode: {args.odometry_mode}")
    if multicam_mode not in _MULTICAM_MODES:
        raise ValueError(f"Unsupported C++ multicamera mode: {args.multicam_mode}")

    repeat_type = (args.repeat_type or "none").lower()
    repeat = args.num_loops if args.num_loops > 0 else 1
    if repeat_type == "none" and args.num_loops > 0:
        repeat_type = "repeat"

    command = [
        launcher,
        f"--edex={args.config_path}",
        f"--report_output={report_path}",
        "--ignore_tracking_errors=true",
        "--print_stats=false",
        f"--repeat={repeat}",
        f"--shuttle={_bool(repeat_type == 'shuttle')}",
        f"--blackout_period={args.blackout_period}",
        f"--blackout_duration={args.blackout_duration}",
        f"--cfg_odom_mode={_ODOMETRY_MODES[odometry_mode]}",
        f"--cfg_multicam_mode={_MULTICAM_MODES[multicam_mode]}",
        f"--cfg_use_gpu={_bool(args.use_gpu)}",
        f"--cfg_async_sba={_bool(args.async_sba)}",
        f"--cfg_use_motion_model={_bool(args.use_motion_model)}",
        f"--cfg_denoising={_bool(args.use_denoising)}",
        f"--cfg_debug_imu_mode={_bool(args.debug_imu_mode)}",
        f"--cfg_horizontal={_bool(args.rectified_stereo_camera)}",
        f"--cfg_max_frame_delta_s={args.max_frame_delta_s}",
        f"--cfg_enable_slam={_bool(args.use_slam)}",
        f"--cfg_sync_slam={_bool(args.sync_slam)}",
        "--cfg_enable_export="
        + _bool(args.use_slam or args.enable_observations_export or args.enable_landmarks_export),
        f"--cfg_enable_final_landmarks_export={_bool(args.enable_final_landmarks_export)}",
        f"--cache_uncompressed={_bool(args.cache_uncompressed)}",
    ]
    if args.camera_ids:
        command.append("--cameras=" + ",".join(str(camera) for camera in args.camera_ids))
    if args.debug_dump_directory:
        command.append(f"--debug_dump={args.debug_dump_directory}")
    if odometry_mode == "rgbd":
        depth_camera, edex_scale = _rgbd_edex_settings(args.config_path)
        input_scale = args.depth_scale_factor if args.depth_scale_factor != 1.0 else edex_scale
        command.extend(
            [
                f"--cfg_depth_camera={depth_camera}",
                "--cfg_depth_scale_factor=1.0",
                f"--cfg_enable_depth_stereo_tracking={_bool(args.enable_depth_stereo_tracking)}",
                f"--depth_scale_factor={input_scale}",
            ]
        )

    passthrough = list(getattr(args, "api_launcher_args", []))
    validate_launcher_args(passthrough)
    return command + passthrough


def parse_launcher_output(path: str) -> LauncherRecords:
    """Parse and validate the launcher's JSONL protocol."""
    frames: list[FrameRecord] = []
    final_slam_poses: dict[int, np.ndarray] = {}
    loop_closures: dict[int, np.ndarray] = {}
    summary = None
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid launcher JSON: {error}") from error
            record_type = record.get("type")
            try:
                if record_type == "frame":
                    frames.append(
                        FrameRecord(
                            frame_id=int(record["frame_id"]),
                            source_frame_number=int(record["source_frame_number"]),
                            timestamp_ns=int(record["timestamp_ns"]),
                            tracking_valid=bool(record["tracking_valid"]),
                            odom_pose=(
                                json_pose_to_transform(record["odom_pose"])
                                if record.get("odom_pose") is not None
                                else None
                            ),
                            slam_pose=(
                                json_pose_to_transform(record["slam_pose"])
                                if record.get("slam_pose") is not None
                                else None
                            ),
                        )
                    )
                elif record_type == "final_slam_pose":
                    final_slam_poses[int(record["timestamp_ns"])] = json_pose_to_transform(record["pose"])
                elif record_type == "loop_closure":
                    loop_closures[int(record["timestamp_ns"])] = json_pose_to_transform(record["pose"])
                elif record_type == "summary":
                    if summary is not None:
                        raise ValueError("duplicate summary")
                    summary = record
                else:
                    raise ValueError(f"unknown record type {record_type!r}")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: malformed {record_type!r} record: {error}") from error

    if summary is None:
        raise ValueError(f"{path}: launcher output has no summary")
    if int(summary.get("attempted_frames", -1)) != len(frames):
        raise ValueError(f"{path}: frame count does not match summary")
    frame_ids = [frame.frame_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError(f"{path}: launcher output has duplicate frame ids")
    return LauncherRecords(sorted(frames, key=lambda frame: frame.frame_id), final_slam_poses, loop_closures, summary)


def _frame_poses(records: LauncherRecords, use_slam: bool) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Select final poses and reproduce Python's previous-pose fallback."""
    raw: dict[int, np.ndarray] = {}
    previous = None
    by_timestamp = {frame.timestamp_ns: frame for frame in records.frames}
    selected_by_timestamp = {
        timestamp: pose for timestamp, pose in records.final_slam_poses.items() if timestamp in by_timestamp
    }
    for frame in records.frames:
        pose = selected_by_timestamp.get(frame.timestamp_ns)
        if pose is None:
            pose = frame.slam_pose if use_slam and frame.slam_pose is not None else frame.odom_pose
        if pose is None:
            pose = previous
        if pose is not None:
            raw[frame.frame_id] = pose
            previous = pose
    return raw, records.loop_closures


def _ground_truth(
    args, records: LauncherRecords, poses: dict[int, np.ndarray]
) -> tuple[list[np.ndarray], dict[int, int]]:
    """Load file GT or synthesize shuttle GT and map the final pass by source frame."""
    repeat_type = (args.repeat_type or "none").lower()
    gt_file = resolve_gt_file(args.dataset, args.gt_path, args.gt_from_shuttle, repeat_type, args.num_loops)

    source_order: dict[int, int] = {}
    first_frame: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    for frame in records.frames:
        source = frame.source_frame_number
        if source not in source_order:
            source_order[source] = len(source_order)
            first_frame[source] = frame.frame_id
        last_frame[source] = frame.frame_id

    if args.gt_from_shuttle:
        gt_transforms = [poses[first_frame[source]] for source in source_order if first_frame[source] in poses]
        valid_sources = [source for source in source_order if first_frame[source] in poses]
        mapping = {
            last_frame[source]: index
            for index, source in enumerate(valid_sources)
            if last_frame[source] in poses
        }
        return gt_transforms, mapping

    if gt_file is None:
        return [], {}
    gt_transforms = load_gt_transforms(gt_file)
    mapping = {}
    for source, frame_id in last_frame.items():
        gt_index = source if 0 <= source < len(gt_transforms) else source_order[source]
        if gt_index < len(gt_transforms) and frame_id in poses:
            mapping[frame_id] = gt_index
    return gt_transforms, mapping


def run_api_launcher(args) -> Stat:
    """Run one reporter sequence through the C++ API launcher."""
    if str(args.config_path).lower().endswith(".mp4"):
        raise ValueError("api_launcher backend supports EDEX input only, not MP4")
    if args.visualize_rerun:
        raise ValueError("--visualize_rerun is available only with --tracker_backend=python")
    if args.save_output_tracker_data:
        raise ValueError("--save_output_tracker_data is available only with --tracker_backend=python")

    launcher = resolve_api_launcher(getattr(args, "api_launcher_path", ""))
    safe_title = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.sequence_title)
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / f"{safe_title}.api_launcher.jsonl"
    log_path = log_dir / f"{safe_title}.api_launcher.log"
    command = build_launcher_command(args, launcher, str(report_path))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(
        f"$ {shlex.join(command)}\n\nstdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"api_launcher failed for {args.sequence_title} with exit code {completed.returncode}; "
            f"see {log_path}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    records = parse_launcher_output(str(report_path))
    if not records.summary.get("success", False):
        raise RuntimeError(f"api_launcher reported an unsuccessful run for {args.sequence_title}; see {log_path}")
    poses, loop_closures = _frame_poses(records, args.use_slam)
    gt_transforms, frame_mapping = _ground_truth(args, records, poses)

    stat = Stat(
        sequence_title=args.sequence_title,
        n_frames=int(records.summary["attempted_frames"]),
        tracking_time=float(records.summary["total_track_seconds"]),
        num_tracking_losts=int(records.summary["lost_frames"]),
        odometry_mode=str(args.odometry_mode),
    )
    stat.average_fps = stat.n_frames / stat.tracking_time if stat.tracking_time > 0 else -1
    finalize_trajectory(
        poses,
        loop_closures,
        gt_transforms,
        stat,
        frame_metadata={},
        use_segments=args.use_segments,
        segment_lengths=args.segment_lengths,
        num_loops=args.num_loops,
        repeat_type=args.repeat_type,
        output_dir=args.output_dir,
        sequence_title=args.sequence_title,
        use_slam=args.use_slam,
        visualize_plot=args.visualize_plot,
        gt_from_shuttle=args.gt_from_shuttle,
        frame_mapping=frame_mapping,
    )
    return stat
