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

"""Detect and replay EuRoC, KITTI, EDEX, or ROS bag data through cuVSLAM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repository_root() -> Path:
    """Find the enclosing cuVSLAM checkout without depending on skill path depth."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "VERSION").is_file() and (candidate / "tools" / "python_tools").is_dir():
            return candidate
    return Path.cwd().resolve()


REPO_DEFAULT = _find_repository_root()
sys.path.insert(0, str(SCRIPT_DIR))

from validate_tum import validate_tum  # noqa: E402


class ReplayError(RuntimeError):
    """Raised for an actionable replay failure."""


@dataclass(frozen=True)
class DatasetDescription:
    """Resolved input type and paths used by the replay."""

    kind: str
    original: Path
    replay_path: Path
    config_path: Path | None = None

    @property
    def output_directory(self) -> Path:
        return self.original if self.original.is_dir() else self.original.parent


def _is_rosbag_directory(path: Path) -> bool:
    return (path / "metadata.yaml").is_file() and any(
        candidate.suffix in {".db3", ".mcap"} for candidate in path.iterdir() if candidate.is_file()
    )


def _euroc_root(path: Path) -> Path | None:
    candidates = [path / "mav0", path]
    for candidate in candidates:
        if all((candidate / camera / "data.csv").is_file() for camera in ("cam0", "cam1")):
            return candidate
    return None


def _kitti_sequence(path: Path, sequence: str | None) -> Path | None:
    if all((path / name).exists() for name in ("image_0", "image_1", "calib.txt", "times.txt")):
        return path
    sequences = path / "sequences"
    if not sequences.is_dir():
        return None
    candidates = sorted(
        candidate
        for candidate in sequences.iterdir()
        if candidate.is_dir()
        and all((candidate / name).exists() for name in ("image_0", "image_1", "calib.txt", "times.txt"))
    )
    if sequence:
        selected = sequences / sequence
        if selected not in candidates:
            raise ReplayError(f"KITTI sequence {sequence!r} was not found under {sequences}")
        return selected
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(candidate.name for candidate in candidates)
        raise ReplayError(f"KITTI root contains multiple sequences ({names}); pass --sequence")
    return None


def _find_edex(path: Path, explicit_config: Path | None = None) -> tuple[Path, Path] | None:
    if explicit_config:
        config = explicit_config.expanduser().resolve()
        if not config.is_file():
            raise ReplayError(f"EDEX config does not exist: {config}")
        directory = path if path.is_dir() else config.parent
        return directory, config
    if path.is_file() and (path.suffix == ".edex" or path.name == "edex"):
        return path.parent, path
    if not path.is_dir():
        return None
    preferred = path / "stereo.edex"
    if preferred.is_file():
        return path, preferred
    candidates = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and (candidate.suffix == ".edex" or candidate.name == "edex")
    )
    if len(candidates) == 1:
        return path, candidates[0]
    if len(candidates) > 1:
        raise ReplayError("multiple EDEX configs found; select one with --edex-config")
    return None


def detect_dataset(
    dataset: Path,
    *,
    forced_kind: str = "auto",
    sequence: str | None = None,
    edex_config: Path | None = None,
) -> DatasetDescription:
    """Detect a supported dataset and normalize its replay root."""
    original = dataset.expanduser().resolve()
    if not original.exists():
        raise ReplayError(f"dataset does not exist: {original}")

    if forced_kind in {"auto", "rosbag"}:
        is_rosbag = (
            _is_rosbag_directory(original)
            if original.is_dir()
            else original.suffix.lower() in {".bag", ".db3", ".mcap"}
        )
        if is_rosbag:
            replay_path = original
            if original.is_file() and original.suffix.lower() in {".db3", ".mcap"}:
                if (original.parent / "metadata.yaml").is_file():
                    replay_path = original.parent
                else:
                    raise ReplayError(
                        f"ROS 2 storage file has no sibling metadata.yaml: {original}"
                    )
            return DatasetDescription("rosbag", original, replay_path)
        if forced_kind == "rosbag":
            raise ReplayError(f"input does not look like a ROS bag: {original}")

    if forced_kind in {"auto", "euroc"}:
        root = _euroc_root(original) if original.is_dir() else None
        if root:
            return DatasetDescription("euroc", original, root)
        if forced_kind == "euroc":
            raise ReplayError(f"input does not contain a EuRoC mav0 layout: {original}")

    if forced_kind in {"auto", "kitti"}:
        root = _kitti_sequence(original, sequence) if original.is_dir() else None
        if root:
            return DatasetDescription("kitti", original, root)
        if forced_kind == "kitti":
            raise ReplayError(f"input does not contain a KITTI odometry sequence: {original}")

    if forced_kind in {"auto", "edex"}:
        result = _find_edex(original, edex_config)
        if result:
            root, config = result
            return DatasetDescription("edex", original, root, config)
        if forced_kind == "edex":
            raise ReplayError(f"input does not contain an EDEX config: {original}")

    raise ReplayError(
        f"could not identify {original} as EuRoC, KITTI, EDEX, or a ROS bag; "
        "pass --dataset-format and the relevant config explicitly"
    )


def _output_path(description: DatasetDescription, output: str | None) -> Path:
    if not output:
        return description.output_directory / "trajectory.txt"
    requested = Path(output).expanduser()
    if not requested.is_absolute():
        requested = description.output_directory / requested
    if requested.is_dir():
        requested = requested / "trajectory.txt"
    return requested.resolve()


def _read_edex_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rig, metadata = config[0], config[1]
    except (OSError, json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise ReplayError(f"could not read EDEX config {config_path}: {exc}") from exc
    if not isinstance(rig, dict) or not isinstance(metadata, dict):
        raise ReplayError(f"EDEX config must contain rig and metadata objects: {config_path}")
    return rig, metadata


def _edex_mode(config_path: Path) -> str:
    rig, metadata = _read_edex_config(config_path)
    cameras = rig.get("cameras", [])
    if not isinstance(cameras, list):
        raise ReplayError(f"EDEX cameras must be a list: {config_path}")
    if "depth_sequence" in metadata or any(
        isinstance(camera, dict) and "depth_id" in camera for camera in cameras
    ):
        return "rgbd"
    if "imu" in rig and len(cameras) >= 2:
        return "inertial"
    if len(cameras) >= 2:
        return "multicamera"
    if len(cameras) == 1:
        return "mono"
    raise ReplayError(f"EDEX config has no cameras: {config_path}")


def _default_mode(description: DatasetDescription) -> str:
    if description.kind == "euroc":
        has_imu = (description.replay_path / "imu0" / "data.csv").is_file()
        return "inertial" if has_imu else "multicamera"
    if description.kind == "kitti":
        return "multicamera"
    if description.kind == "edex" and description.config_path:
        return _edex_mode(description.config_path)
    return "auto"


def _validate_mode(description: DatasetDescription, mode: str) -> None:
    if description.kind == "euroc":
        if mode == "rgbd":
            raise ReplayError("raw EuRoC data has no aligned depth stream")
        if mode == "inertial" and not (description.replay_path / "imu0" / "data.csv").is_file():
            raise ReplayError("EuRoC inertial mode requires imu0/data.csv")
        return
    if description.kind == "kitti":
        if mode not in {"mono", "multicamera"}:
            raise ReplayError("raw KITTI odometry supports mono or multicamera mode")
        return
    if description.kind != "edex" or description.config_path is None:
        return

    rig, metadata = _read_edex_config(description.config_path)
    cameras = rig.get("cameras", [])
    if not isinstance(cameras, list) or not cameras:
        raise ReplayError(f"EDEX config has no cameras: {description.config_path}")
    has_imu = "imu" in rig
    has_depth = "depth_sequence" in metadata or any(
        isinstance(camera, dict) and "depth_id" in camera for camera in cameras
    )
    if mode == "multicamera" and len(cameras) < 2:
        raise ReplayError("EDEX multicamera mode requires at least two cameras")
    if mode == "inertial" and (len(cameras) < 2 or not has_imu):
        raise ReplayError("EDEX inertial mode requires at least two cameras and one IMU")
    if mode == "rgbd" and not has_depth:
        raise ReplayError("EDEX RGB-D mode requires a depth stream and depth camera mapping")


def _load_runtime(repo: Path) -> tuple[Any, Any, Any]:
    tools_path = repo / "tools" / "python_tools"
    if not tools_path.is_dir():
        raise ReplayError(f"cuVSLAM tools package was not found under {repo}")
    sys.path.insert(0, str(tools_path))
    try:
        import cuvslam
        import numpy as np
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise ReplayError(
            f"replay runtime is not ready ({exc}); run bootstrap_runtime.py --repo {repo}"
        ) from exc

    required_api = (
        "Odometry",
        "Tracker",
        "Camera",
        "Pose",
        "Rig",
        "ImuCalibration",
        "ImuMeasurement",
    )
    if any(not hasattr(cuvslam, name) for name in required_api):
        owners = [
            owner
            for owner_name in ("core", "pycuvslam")
            if (owner := getattr(cuvslam, owner_name, None)) is not None
        ]
        for name in required_api:
            if hasattr(cuvslam, name):
                continue
            for owner in owners:
                if hasattr(owner, name):
                    setattr(cuvslam, name, getattr(owner, name))
                    break
    missing = [name for name in required_api if not hasattr(cuvslam, name)]
    if missing:
        raise ReplayError(
            "installed cuVSLAM binding lacks required APIs: " + ", ".join(missing)
        )
    return cuvslam, np, Image


def _enum_value(vslam: Any, mode: str) -> Any:
    odometry = vslam.Odometry
    tracker = getattr(vslam, "Tracker", None)
    enum = getattr(odometry, "OdometryMode", None) or getattr(
        tracker, "OdometryMode", None
    )
    if enum is None:
        raise ReplayError("installed cuVSLAM binding exposes no OdometryMode enum")
    members = {
        "mono": "Mono",
        "multicamera": "Multicamera",
        "inertial": "Inertial",
        "rgbd": "RGBD",
    }
    return getattr(enum, members[mode])


def _make_tracker(
    vslam: Any,
    rig: Any,
    mode: str,
    *,
    rectified: bool,
    rgbd_settings: Any = None,
) -> Any:
    config_type = getattr(vslam.Odometry, "Config", None) or getattr(
        vslam.Tracker, "OdometryConfig", None
    )
    if config_type is None:
        raise ReplayError("installed cuVSLAM binding exposes no odometry config type")
    config = config_type()
    settings = {
        "odometry_mode": _enum_value(vslam, mode),
        "async_sba": False,
        "use_gpu": True,
        "rectified_stereo_camera": rectified,
        "enable_observations_export": False,
        "enable_landmarks_export": False,
        "enable_final_landmarks_export": False,
    }
    for name, value in settings.items():
        if hasattr(config, name):
            setattr(config, name, value)
    if rgbd_settings is not None and hasattr(config, "rgbd_settings"):
        config.rgbd_settings = rgbd_settings

    warm_up = getattr(vslam, "warm_up_gpu", None)
    if callable(warm_up):
        warm_up()

    failures = []
    tracker_mode = getattr(vslam.Tracker, "Mode", None)
    if tracker_mode is not None:
        try:
            return vslam.Tracker(rig, tracker_mode.OdometryOnlyOffline, config)
        except TypeError as exc:
            failures.append(str(exc))
    try:
        return vslam.Tracker(rig, config)
    except TypeError as exc:
        failures.append(str(exc))
    raise ReplayError("no supported Tracker constructor accepted the replay config: " + " | ".join(failures))


class ReplayProcessor:
    """Small adapter that records timestamped poses from an EDEX or raw replay."""

    def __init__(self, vslam: Any, tracker: Any, mode: str):
        self.vslam = vslam
        self.tracker = tracker
        self.mode = mode
        self.poses: dict[int, tuple[int, Any]] = {}
        self.lost = 0
        self._track_variant: int | None = None

    def _track(self, timestamp: int, images: Sequence[Any], masks: Sequence[Any], depths: Any) -> Any:
        calls = (
            lambda: self.tracker.track(timestamp, images, masks, depths),
            lambda: self.tracker.track(timestamp, images, masks),
            lambda: self.tracker.track(timestamp, images),
        )
        if self._track_variant is not None:
            return calls[self._track_variant]()
        errors = []
        for index, call in enumerate(calls):
            try:
                result = call()
            except TypeError as exc:
                errors.append(str(exc))
                continue
            self._track_variant = index
            return result
        raise ReplayError("no supported Tracker.track signature accepted the frame: " + " | ".join(errors))

    def process_images(
        self,
        frame_id: int,
        timestamps: Sequence[int],
        images: Sequence[Any],
        masks: Sequence[Any],
        depths: Any = None,
    ) -> None:
        timestamp = max(int(value) for value in timestamps)
        result = self._track(timestamp, images, masks, depths)
        odometry_result = result[0] if isinstance(result, (tuple, list)) else result
        world_from_rig = getattr(odometry_result, "world_from_rig", None)
        pose = getattr(world_from_rig, "pose", world_from_rig)
        if pose is None:
            self.lost += 1
        else:
            self.poses[frame_id] = (timestamp, pose)
        processed = len(self.poses) + self.lost
        if processed % 100 == 0:
            print(f"Processed {processed} frames; poses={len(self.poses)} lost={self.lost}", flush=True)

    def process_imu(
        self,
        timestamp: int,
        linear_accelerations: Sequence[float],
        angular_velocities: Sequence[float],
    ) -> None:
        if self.mode != "inertial":
            return
        measurement = self.vslam.ImuMeasurement()
        measurement.timestamp_ns = int(timestamp)
        measurement.linear_accelerations = linear_accelerations
        measurement.angular_velocities = angular_velocities
        self.tracker.register_imu_measurement(0, measurement)

    def get_camera_pose(self, frame_id: int) -> Any:
        value = self.poses.get(frame_id)
        return value[1] if value else None

    def set_frame_metadata(self, frame_id: int, metadata: dict[str, Any]) -> None:
        del frame_id, metadata


def _load_image(path: Path, np: Any, Image: Any) -> Any:
    with Image.open(path) as image:
        pixels = np.asarray(image)
        if image.mode == "RGB":
            pixels = np.ascontiguousarray(pixels[:, :, ::-1])
        else:
            pixels = np.ascontiguousarray(pixels)
    return pixels


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReplayError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_euroc_rig(root: Path, helpers: Any, vslam: Any, mode: str) -> Any:
    """Load only a complete calibration family, without changing dataset files."""
    sensor_names = ["cam0"] if mode == "mono" else ["cam0", "cam1"]
    if mode == "inertial":
        sensor_names.append("imu0")
    recalibrated = {
        name: root / name / "sensor_recalibrated.yaml" for name in sensor_names
    }
    defaults = {name: root / name / "sensor.yaml" for name in sensor_names}
    if all(path.is_file() for path in recalibrated.values()):
        paths = recalibrated
        is_default = False
    else:
        partial = [name for name, path in recalibrated.items() if path.is_file()]
        if partial:
            print(
                "Ignoring partial recalibration set "
                f"({', '.join(partial)}); using the complete default calibration",
                flush=True,
            )
        missing = [str(path) for path in defaults.values() if not path.is_file()]
        if missing:
            raise ReplayError("missing EuRoC calibration file(s): " + ", ".join(missing))
        paths = defaults
        is_default = True

    configs = {
        name: helpers._load_yaml_config(str(path)) for name, path in paths.items()
    }
    cam0_transform = helpers.get_transform_from_config(configs["cam0"])
    cam0 = helpers._create_camera_from_config(configs["cam0"], is_default)
    cam0.rig_from_camera = vslam.Pose(
        rotation=[0.0, 0.0, 0.0, 1.0],
        translation=[0.0, 0.0, 0.0],
    )
    cameras = [cam0]
    if mode != "mono":
        cam1 = helpers._create_camera_from_config(configs["cam1"], is_default)
        cam1_transform = helpers.get_transform_from_config(configs["cam1"])
        if is_default:
            cam1_transform = helpers.transform_to_cam0_reference(cam0_transform, cam1_transform)
        cam1.rig_from_camera = helpers.transform_to_pose(cam1_transform.flatten().tolist())
        cameras.append(cam1)

    imus = []
    if mode == "inertial":
        imu_config = configs["imu0"]
        imu_transform = helpers.get_transform_from_config(imu_config)
        if is_default:
            imu_transform = helpers.transform_to_cam0_reference(cam0_transform, imu_transform)
        imu = vslam.ImuCalibration()
        imu.rig_from_imu = helpers.transform_to_pose(imu_transform.flatten().tolist())
        imu.gyroscope_noise_density = imu_config["gyroscope_noise_density"]
        imu.gyroscope_random_walk = imu_config["gyroscope_random_walk"]
        imu.accelerometer_noise_density = imu_config["accelerometer_noise_density"]
        imu.accelerometer_random_walk = imu_config["accelerometer_random_walk"]
        imu.frequency = imu_config["rate_hz"]
        imus.append(imu)

    rig = vslam.Rig()
    rig.cameras = cameras
    rig.imus = imus
    return rig


def _run_euroc(
    description: DatasetDescription,
    repo: Path,
    vslam: Any,
    np: Any,
    Image: Any,
    mode: str,
) -> tuple[ReplayProcessor, int]:
    helpers = _load_module(repo / "examples" / "euroc" / "dataset_utils.py", "cuvslam_euroc_helpers")
    root = description.replay_path
    rig = _get_euroc_rig(root, helpers, vslam, mode)

    left = helpers.read_csv_data(str(root / "cam0" / "data.csv"), "camera")
    right = helpers.read_csv_data(str(root / "cam1" / "data.csv"), "camera")
    left_by_timestamp = {row["timestamp"]: row["filename"] for row in left}
    right_by_timestamp = {row["timestamp"]: row["filename"] for row in right}
    timestamps = sorted(set(left_by_timestamp) & set(right_by_timestamp))
    if not timestamps:
        raise ReplayError("EuRoC camera streams have no common timestamps")
    dropped_left = len(left) - len(timestamps)
    dropped_right = len(right) - len(timestamps)
    if dropped_left or dropped_right:
        print(
            f"Pairing EuRoC cameras by timestamp; unmatched cam0={dropped_left} cam1={dropped_right}",
            flush=True,
        )

    imu_rows = []
    if mode == "inertial":
        imu_rows = helpers.read_csv_data(str(root / "imu0" / "data.csv"), "imu")
    tracker = _make_tracker(vslam, rig, mode, rectified=False)
    processor = ReplayProcessor(vslam, tracker, mode)
    empty_masks = [np.array([])] * (1 if mode == "mono" else 2)
    imu_index = 0
    for frame_id, timestamp in enumerate(timestamps):
        while imu_index < len(imu_rows) and imu_rows[imu_index]["timestamp"] <= timestamp:
            row = imu_rows[imu_index]
            processor.process_imu(row["timestamp"], row["accel"], row["gyro"])
            imu_index += 1
        paths = [root / "cam0" / "data" / left_by_timestamp[timestamp]]
        if mode != "mono":
            paths.append(root / "cam1" / "data" / right_by_timestamp[timestamp])
        images = [_load_image(path, np, Image) for path in paths]
        processor.process_images(frame_id, [timestamp] * len(images), images, empty_masks)
    return processor, len(timestamps)


def _parse_kitti_calibration(path: Path) -> dict[str, list[float]]:
    matrices = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, values = line.split(":", 1)
        matrices[key] = [float(value) for value in values.split()]
    if "P0" not in matrices or "P1" not in matrices:
        raise ReplayError(f"KITTI calibration must contain P0 and P1: {path}")
    for key in ("P0", "P1"):
        if len(matrices[key]) != 12:
            raise ReplayError(
                f"KITTI calibration {key} must contain 12 values: {path}"
            )
    if matrices["P0"][0] <= 0 or matrices["P0"][5] <= 0 or matrices["P1"][0] <= 0:
        raise ReplayError(f"KITTI calibration has non-positive focal length: {path}")
    return matrices


def _run_kitti(
    description: DatasetDescription,
    vslam: Any,
    np: Any,
    Image: Any,
    mode: str,
) -> tuple[ReplayProcessor, int]:
    if mode not in {"mono", "multicamera"}:
        raise ReplayError("raw KITTI odometry supports mono or multicamera mode")
    root = description.replay_path
    calibration = _parse_kitti_calibration(root / "calib.txt")
    p0, p1 = calibration["P0"], calibration["P1"]
    left_files = {path.stem: path for path in (root / "image_0").glob("*.png")}
    right_files = {path.stem: path for path in (root / "image_1").glob("*.png")}
    if mode == "mono":
        frame_names = sorted(left_files)
    else:
        frame_names = sorted(set(left_files) & set(right_files))
        dropped_left = len(left_files) - len(frame_names)
        dropped_right = len(right_files) - len(frame_names)
        if dropped_left or dropped_right:
            print(
                f"Pairing KITTI cameras by frame id; unmatched image_0={dropped_left} "
                f"image_1={dropped_right}",
                flush=True,
            )
    times = [
        float(line)
        for line in (root / "times.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not frame_names:
        raise ReplayError("KITTI input contains no replayable image frames")
    if any(not name.isdigit() for name in frame_names):
        raise ReplayError("KITTI image filenames must have numeric frame ids")
    if max(int(name) for name in frame_names) >= len(times):
        raise ReplayError(
            f"KITTI image frame id exceeds available timestamps ({len(times)} rows)"
        )
    with Image.open(left_files[frame_names[0]]) as image:
        size = list(image.size)

    cameras = []
    for index, projection in enumerate((p0, p1) if mode == "multicamera" else (p0,)):
        camera = vslam.Camera()
        camera.size = size
        camera.focal = [projection[0], projection[5]]
        camera.principal = [projection[2], projection[6]]
        translation = [0.0, 0.0, 0.0]
        if index == 1:
            translation[0] = -p1[3] / p1[0]
        camera.rig_from_camera = vslam.Pose(rotation=[0.0, 0.0, 0.0, 1.0], translation=translation)
        cameras.append(camera)
    rig = vslam.Rig()
    rig.cameras = cameras
    rig.imus = []
    tracker = _make_tracker(vslam, rig, mode, rectified=True)
    processor = ReplayProcessor(vslam, tracker, mode)
    empty_masks = [np.array([])] * len(cameras)
    for frame_id, frame_name in enumerate(frame_names):
        paths = [left_files[frame_name]]
        if mode == "multicamera":
            paths.append(right_files[frame_name])
        images = [_load_image(path, np, Image) for path in paths]
        timestamp = int(round(times[int(frame_name)] * 1e9))
        processor.process_images(frame_id, [timestamp] * len(images), images, empty_masks)
    return processor, len(frame_names)


def _run_edex(
    description: DatasetDescription,
    vslam: Any,
    mode: str,
) -> tuple[ReplayProcessor, int]:
    from cuvslam_tools.tracker.edex_reader import EdexReader

    if description.config_path is None:
        raise ReplayError("EDEX replay requires a config path")
    reader_config = description.config_path
    temporary_config: Path | None = None
    rig, metadata = _read_edex_config(description.config_path)
    sequence = metadata.get("sequence")
    flat_sequence = (
        isinstance(sequence, list)
        and sequence
        and all(isinstance(path, str) for path in sequence)
    )
    if flat_sequence:
        cameras = rig.get("cameras", [])
        if len(sequence) != len(cameras):
            raise ReplayError(
                "legacy flat EDEX sequence is ambiguous because its entry count does not "
                "match the camera count"
            )
        config = json.loads(description.config_path.read_text(encoding="utf-8"))
        config[1]["sequence"] = [[path] for path in sequence]
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="cuvslam-edex-",
            suffix=".edex",
            delete=False,
        ) as stream:
            json.dump(config, stream)
            temporary_config = Path(stream.name)
        reader_config = temporary_config
        print("Normalizing legacy flat EDEX sequence entries through a temporary config", flush=True)
    try:
        reader = EdexReader(
            str(description.replay_path),
            stereo_edex=str(reader_config),
            rgbd_mode=mode == "rgbd",
            camera_ids=[0] if mode == "mono" else None,
        )
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)
    if not reader.validate_rig() or reader.rig is None:
        raise ReplayError(f"invalid EDEX rig: {description.config_path}")
    if mode != "inertial":
        reader.rig.imus = []
    tracker = _make_tracker(
        vslam,
        reader.rig,
        mode,
        rectified=False,
        rgbd_settings=reader.rgbd_settings,
    )
    processor = ReplayProcessor(vslam, tracker, mode)
    reader.replay(processor)
    return processor, reader.total_frames


def _find_converted_edex(path: Path) -> tuple[Path, Path]:
    preferred = path / "stereo.edex"
    if preferred.is_file():
        return path, preferred
    candidates = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and (candidate.suffix == ".edex" or candidate.name == "edex")
    )
    if len(candidates) != 1:
        raise ReplayError(
            f"ROS bag conversion produced {len(candidates)} EDEX configs under {path}; "
            "pass --edex-output pointing to a single converted sequence"
        )
    return candidates[0].parent, candidates[0]


def _convert_rosbag(
    description: DatasetDescription,
    repo: Path,
    rosbag_config: Path | None,
    edex_output: Path | None,
) -> DatasetDescription:
    if rosbag_config is None:
        raise ReplayError(
            "ROS bag replay requires --rosbag-config because topics, rig frame, and calibration "
            "cannot be inferred safely"
        )
    config = rosbag_config.expanduser().resolve()
    if not config.is_file():
        raise ReplayError(f"ROS bag config does not exist: {config}")
    output = (
        edex_output.expanduser().resolve()
        if edex_output
        else description.output_directory / ".cuvslam-edex"
    )
    if output.exists():
        try:
            root, edex = _find_converted_edex(output)
            print(f"Reusing converted EDEX input: {edex}")
            return DatasetDescription("edex", description.original, root, edex)
        except ReplayError as exc:
            raise ReplayError(
                f"refusing to overwrite existing incomplete EDEX output {output}: {exc}"
            ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.in-progress-", dir=output.parent)
    )
    command = [
        sys.executable,
        "-m",
        "cuvslam_tools.bag2edex.scripts.extract_edex",
        "--config_path",
        str(config),
        "--rosbag_path",
        str(description.replay_path),
        "--output_path",
        str(staging),
    ]
    print("Converting ROS bag to EDEX:", " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=repo, check=True)
    except subprocess.CalledProcessError as exc:
        raise ReplayError(
            f"ROS bag conversion failed with exit code {exc.returncode}; "
            f"partial output is preserved at {staging}"
        ) from exc
    root, edex = _find_converted_edex(staging)
    root_relative = root.relative_to(staging)
    edex_relative = edex.relative_to(staging)
    os.replace(staging, output)
    return DatasetDescription(
        "edex",
        description.original,
        output / root_relative,
        output / edex_relative,
    )


def _write_tum(
    processor: ReplayProcessor,
    output: Path,
    *,
    expected_frames: int,
    min_coverage: float,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    ordered = sorted(processor.poses.values(), key=lambda value: value[0])
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for timestamp, pose in ordered:
                translation = list(pose.translation)
                rotation = list(pose.rotation)
                values = [timestamp / 1e9, *translation, *rotation]
                stream.write(" ".join(f"{float(value):.12g}" for value in values) + "\n")
        stats = validate_tum(
            temporary,
            expected_rows=expected_frames,
            min_coverage=min_coverage,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    stats["path"] = str(output)
    return stats


def _write_kitti(
    processor: ReplayProcessor,
    output: Path,
    *,
    expected_frames: int,
    min_coverage: float,
) -> dict[str, Any]:
    from cuvslam_tools.tracker.kitti_benchmark import save_poses_to_kitti_benchmark

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    poses = {frame_id: value[1] for frame_id, value in processor.poses.items()}
    try:
        save_poses_to_kitti_benchmark(poses, str(temporary))
        rows = [
            line.split()
            for line in temporary.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows or any(len(row) != 12 for row in rows):
            raise ReplayError(f"KITTI output validation failed: {output}")
        if not all(math.isfinite(float(value)) for row in rows for value in row):
            raise ReplayError(f"KITTI output contains non-finite values: {output}")
        coverage = len(rows) / expected_frames
        if coverage < min_coverage:
            raise ReplayError(
                f"trajectory coverage {coverage:.3%} is below {min_coverage:.3%}"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "rows": len(rows),
        "expected_rows": expected_frames,
        "coverage": coverage,
        "valid": True,
    }


def _plan(description: DatasetDescription, output: Path, mode: str, output_format: str) -> dict[str, Any]:
    return {
        "dataset_format": description.kind,
        "dataset_input": str(description.original),
        "replay_path": str(description.replay_path),
        "config_path": str(description.config_path) if description.config_path else None,
        "mode": mode,
        "output_format": output_format,
        "output_path": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "euroc", "kitti", "edex", "rosbag"],
        default="auto",
    )
    parser.add_argument("--sequence", help="KITTI sequence name when a dataset root contains several")
    parser.add_argument("--edex-config", type=Path, help="explicit EDEX config file")
    parser.add_argument("--rosbag-config", type=Path, help="topic/calibration YAML for ROS bag conversion")
    parser.add_argument("--edex-output", type=Path, help="persistent ROS-bag-to-EDEX output directory")
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT, help="cuVSLAM repository root")
    parser.add_argument(
        "--mode",
        choices=["auto", "mono", "multicamera", "inertial", "rgbd"],
        default="auto",
    )
    parser.add_argument("--output", help="output path; relative paths are resolved beside the dataset")
    parser.add_argument("--output-format", choices=["tum", "kitti"], default="tum")
    parser.add_argument("--min-coverage", type=float, default=0.9)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="detect and print the replay plan without running",
    )
    args = parser.parse_args()

    try:
        description = detect_dataset(
            args.dataset,
            forced_kind=args.dataset_format,
            sequence=args.sequence,
            edex_config=args.edex_config,
        )
        output = _output_path(description, args.output)
        mode = args.mode if args.mode != "auto" else _default_mode(description)
        _validate_mode(description, mode)
        plan = _plan(description, output, mode, args.output_format)
        if description.kind == "rosbag":
            plan["rosbag_config"] = str(args.rosbag_config.resolve()) if args.rosbag_config else None
            plan["edex_output"] = str(
                (args.edex_output or description.output_directory / ".cuvslam-edex").resolve()
            )
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        if args.inspect:
            return 0

        repo = args.repo.expanduser().resolve()
        vslam, np, Image = _load_runtime(repo)
        print(f"cuVSLAM version: {vslam.get_version()}", flush=True)
        if description.kind == "rosbag":
            description = _convert_rosbag(
                description,
                repo,
                args.rosbag_config,
                args.edex_output,
            )
            if args.mode == "auto":
                mode = _default_mode(description)
            _validate_mode(description, mode)

        if description.kind == "euroc":
            processor, expected_frames = _run_euroc(description, repo, vslam, np, Image, mode)
        elif description.kind == "kitti":
            processor, expected_frames = _run_kitti(description, vslam, np, Image, mode)
        elif description.kind == "edex":
            processor, expected_frames = _run_edex(description, vslam, mode)
        else:
            raise ReplayError(f"unsupported replay route: {description.kind}")

        if not processor.poses:
            raise ReplayError("cuVSLAM produced no poses")
        if args.output_format == "tum":
            stats = _write_tum(
                processor,
                output,
                expected_frames=expected_frames,
                min_coverage=args.min_coverage,
            )
        else:
            stats = _write_kitti(
                processor,
                output,
                expected_frames=expected_frames,
                min_coverage=args.min_coverage,
            )
        print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
        print(
            f"Completed replay: frames={expected_frames} poses={len(processor.poses)} "
            f"lost={processor.lost} output={output}",
            flush=True,
        )
        return 0
    except (IndexError, KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Replay failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
