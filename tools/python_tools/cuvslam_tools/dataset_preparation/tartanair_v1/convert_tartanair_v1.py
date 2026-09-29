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

"""Convert TartanAir V1 Hard into the stereo evaluation the OSMO pipeline ran, from public zips.

Reads the downloaded zips directly, with no extracted copy, and writes per sequence::

    <env>/<traj>/00/000000.png          left colour, copied byte for byte
    <env>/<traj>/01/000000.png          right colour, copied byte for byte
    <env>/<traj>/depth_00/000000.png    left depth, uint16 millimetres, 0 = no return
    <env>/<traj>/gt.txt                 3x4 pose of the left camera, relative to frame 0
    <env>/<traj>/pose_left.txt          source poses, kept for provenance
    <env>/<traj>/pose_right.txt
    <env>/<traj>/stereo.edex            rig, intrinsics, depth scale, frame rate

plus three reporter configs at the root: ``tartan-vo_slam.cfg`` (KPI prefix
``TARTAN``) with every trajectory, and the OSMO split into
``tartan_stable-vo_slam.cfg`` (``TARTAN_STABLE``, OSMO's ``TARTAN``) and
``tartan_flaky-vo_slam.cfg`` (``TARTAN_FLAKY``). The sequence folders and titles
match that pipeline's, so a per-sequence comparison lines up. Ground truth is
written by the same code the OSMO dataset was built with, which reproduces its
``gt.txt`` on every sequence.

The selection is every Hard trajectory of 16 environments. The OSMO dataset also
omitted ``abandonedfactory_night`` and ``gascola``; of what it kept, two
trajectories were in no config and are placed by how they track:
``neighborhood/P003`` in the stable set, ``westerndesert/P001`` in the flaky one.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from cuvslam_tools.dataset_preparation import rgbd
from cuvslam_tools.dataset_preparation.tartan.dataset_converter.utils import Normalizer, TartanAirReader, Writer

DATASET_ID = "tartanair_v1"
SPLIT = "Hard"

SOURCE_NAME = "TartanAir V1"
SOURCE_URL = "https://theairlab.org/tartanair-dataset/"

ZIP_KINDS = ("image_left", "image_right", "depth_left")
MANIFEST = Path(__file__).with_name("zipfiles.txt")

# Published intrinsics and baseline; the right camera sits along +x of the left.
IMAGE_SIZE = (640, 480)
FOCAL = (320.0, 320.0)
PRINCIPAL = (320.0, 240.0)
BASELINE_M = 0.25

# V1 publishes no frame rate or timestamps. The OSMO dataset declared 30 fps, and
# the reader derives every frame time from this field, so it is kept.
FPS = 30

# uint16 millimetres reach 65.535 m. Sky (about 10000 m in V1) and anything
# beyond is written as 0: the track lifter accepts any depth above 1 mm with no
# upper bound, so a clipped value would place landmarks on a wall, while 0 is
# rejected everywhere depth is read.
DEPTH_SCALE_FACTOR = 1000.0
_MAX_DEPTH_UNITS = np.iinfo(np.uint16).max

SEGMENT_LENGTHS: Tuple[float, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)

STABLE: Dict[str, Tuple[str, ...]] = {
    "abandonedfactory": (
        "P000", "P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009", "P010", "P011",
    ),
    "amusement": ("P000", "P001", "P002", "P003", "P004", "P005", "P006", "P007",),
    "carwelding": ("P000", "P001", "P002", "P003",),
    "endofworld": ("P000", "P001", "P002", "P005", "P006",),
    "hospital": (
        "P037", "P038", "P039", "P040", "P041", "P042", "P043", "P044", "P045", "P046", "P047", "P048",
        "P049",
    ),
    "japanesealley": ("P000", "P001", "P002", "P003", "P004", "P005",),
    "neighborhood": (
        "P000", "P001", "P002", "P003", "P004", "P006", "P007", "P008", "P009", "P010", "P011", "P012",
        "P013", "P014", "P015", "P016", "P017",
    ),
    "ocean": ("P000", "P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009",),
    "office": ("P000", "P001", "P003", "P004", "P005", "P006",),
    "office2": ("P000", "P001", "P002", "P005", "P006", "P007", "P008", "P010",),
    "oldtown": ("P000", "P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008",),
    "seasidetown": ("P000", "P001", "P002", "P004",),
    "seasonsforest": ("P001", "P002", "P004", "P005", "P006",),
    "seasonsforest_winter": ("P010", "P011", "P012", "P013", "P014", "P015", "P016", "P017", "P018",),
    "soulcity": ("P000", "P001", "P002", "P003", "P004", "P005", "P008", "P009",),
    "westerndesert": ("P000", "P002", "P003", "P004", "P006", "P007",),
}

FLAKY: Dict[str, Tuple[str, ...]] = {
    "neighborhood": ("P005",),
    "office": ("P002", "P007",),
    "office2": ("P003", "P004", "P009",),
    "seasidetown": ("P003",),
    "westerndesert": ("P001", "P005",),
}

ENVIRONMENTS: Tuple[str, ...] = tuple(sorted(set(STABLE) | set(FLAKY)))

ALL: Dict[str, Tuple[str, ...]] = {
    env: tuple(sorted(STABLE.get(env, ()) + FLAKY.get(env, ()))) for env in ENVIRONMENTS
}

# Reporter config -> the trajectories it evaluates. The KPI prefix is the name
# before the first hyphen, and an underscore keeps TARTAN_STABLE and
# TARTAN_FLAKY from colliding with TARTAN. The configs ship inside the dataset
# tarball, so all three are written whichever ones eval runs: switching between
# the full set and the split is then a registry change, not a reprovisioning.
CONFIGS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "tartan-vo_slam.cfg": ALL,
    "tartan_stable-vo_slam.cfg": STABLE,
    "tartan_flaky-vo_slam.cfg": FLAKY,
}

EDEX_FILE = rgbd.EDEX_FILE
GROUND_TRUTH_FILE = rgbd.GROUND_TRUTH_FILE

_SCHEMA_VERSION = 1
_CONVERTER_VERSION = 1

# The right camera's pose recomputed per frame must agree with the fixed
# baseline to within these.
_BASELINE_TOLERANCE_M = 1e-3
_ROTATION_TOLERANCE = 1e-3

_TRAJECTORY = re.compile(r"^[^/]+/Hard/(P\d{3})/pose_left\.txt$")


class ConversionError(RuntimeError):
    """Raised when the downloaded data does not match what this converter pins."""


def zip_filename(env: str, kind: str) -> str:
    """``<env>/Hard/<kind>.zip`` as the download script saves it."""
    return f"{env}_{SPLIT}_{kind}.zip"


def selected(envs: Optional[Sequence[str]] = None) -> Dict[str, Tuple[str, ...]]:
    """Every pinned trajectory, per environment, optionally limited to some environments."""
    if envs:
        unknown = sorted(set(envs) - set(ENVIRONMENTS))
        if unknown:
            raise ConversionError(f"unknown environments {unknown}; known: {', '.join(ENVIRONMENTS)}")
    chosen = envs or ENVIRONMENTS
    return {env: ALL[env] for env in ENVIRONMENTS if env in chosen}


def config_subsets(chosen: Dict[str, Tuple[str, ...]]) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """Each reporter config's trajectories within a selection, omitting configs left empty.

    Only a partial run can empty one, when none of its environments has a flaky trajectory.
    """
    subsets = {}
    for name, sequences in CONFIGS.items():
        subset = {env: trajs for env, trajs in sequences.items() if env in chosen}
        if subset:
            subsets[name] = subset
    return subsets


def _poses(text: str, source: str) -> np.ndarray:
    values = np.loadtxt(io.StringIO(text), ndmin=2)
    if values.shape[1] != 7:
        raise ConversionError(f"{source}: expected 7 values per pose, got {values.shape[1]}")
    transforms = np.tile(np.eye(4), (len(values), 1, 1))
    transforms[:, :3, :3] = Rotation.from_quat(values[:, 3:]).as_matrix()
    transforms[:, :3, 3] = values[:, :3]
    return transforms


def _check_rig(left: np.ndarray, right: np.ndarray, source: str) -> None:
    """The right camera must sit at the published baseline, unrotated, on every frame.

    Poses are NED, so the baseline runs along +y of the left camera's frame.
    """
    if len(left) != len(right):
        raise ConversionError(f"{source}: {len(left)} left poses, {len(right)} right")
    relative = np.linalg.inv(left) @ right
    offset = np.abs(relative[:, :3, 3] - [0.0, BASELINE_M, 0.0]).max()
    rotation = np.abs(relative[:, :3, :3] - np.eye(3)).max()
    if offset > _BASELINE_TOLERANCE_M or rotation > _ROTATION_TOLERANCE:
        raise ConversionError(f"{source}: right camera departs from the {BASELINE_M} m baseline "
                              f"(offset {offset:.2e} m, rotation {rotation:.2e})")


def convert_depth(data: bytes, source: str) -> Tuple[np.ndarray, float]:
    """V1 float32 metres to uint16 millimetres, and the fraction written as no return."""
    metres = np.load(io.BytesIO(data))
    if metres.shape != IMAGE_SIZE[::-1]:
        raise ConversionError(f"{source}: expected {IMAGE_SIZE[::-1]} depth, got {metres.shape}")
    units = np.round(metres.astype(np.float64) * DEPTH_SCALE_FACTOR)
    invalid = ~np.isfinite(units) | (units <= 0) | (units > _MAX_DEPTH_UNITS)
    units[invalid] = 0
    return units.astype(np.uint16), float(invalid.mean())


def edex_document(frame_count: int) -> List[Dict[str, object]]:
    """The OSMO dataset's stereo rig, with depth declared on the left camera."""
    intrinsics = {
        "distortion_model": "pinhole",
        "distortion_params": [],
        "focal": list(FOCAL),
        "principal": list(PRINCIPAL),
        "size": list(IMAGE_SIZE),
    }
    return [
        {
            "version": "0.9",
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "cameras": [
                {
                    "transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    "intrinsics": intrinsics,
                    "depth_id": 0,
                    "depth_scale_factor": DEPTH_SCALE_FACTOR,
                },
                {
                    "transform": [[1.0, 0.0, 0.0, BASELINE_M], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                    "intrinsics": intrinsics,
                },
            ],
        },
        {
            "fps": FPS,
            "points2d": {},
            "points3d": {},
            "rig_positions": {},
            "sequence": [["00/000000.png"], ["01/000000.png"]],
            "depth_sequence": [["depth_00/000000.png"]],
        },
    ]


def convert_trajectory(raw_dir: str, env: str, trajectory: str, output_dir: str) -> Dict[str, object]:
    """Convert one trajectory from its three zips into ``output_dir/<env>/<trajectory>``."""
    source = f"{env}/{SPLIT}/{trajectory}"
    destination = Path(output_dir) / env / trajectory
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    for directory in ("00", "01", "depth_00"):
        (staging / directory).mkdir(parents=True)

    archives = {kind: zipfile.ZipFile(Path(raw_dir) / zip_filename(env, kind)) for kind in ZIP_KINDS}
    try:
        pose_left = archives["image_left"].read(f"{source}/pose_left.txt").decode()
        pose_right = archives["image_right"].read(f"{source}/pose_right.txt").decode()
        left, right = _poses(pose_left, f"{source}/pose_left.txt"), _poses(pose_right, f"{source}/pose_right.txt")
        _check_rig(left, right, source)
        frame_count = len(left)

        invalid_fractions = []
        for frame in range(frame_count):
            name = f"{frame:06d}"
            members = (
                ("image_left", f"{source}/image_left/{name}_left.png", staging / "00" / f"{name}.png"),
                ("image_right", f"{source}/image_right/{name}_right.png", staging / "01" / f"{name}.png"),
            )
            for kind, member, target in members:
                try:
                    data = archives[kind].read(member)
                except KeyError:
                    raise ConversionError(f"{zip_filename(env, kind)} is missing {member}") from None
                if frame == 0:
                    try:
                        size = Image.open(io.BytesIO(data)).size
                    except OSError:
                        size = None
                    if size != IMAGE_SIZE:
                        raise ConversionError(f"{member}: expected a {IMAGE_SIZE} image, got {size}")
                target.write_bytes(data)

            member = f"{source}/depth_left/{name}_left_depth.npy"
            try:
                depth, invalid = convert_depth(archives["depth_left"].read(member), member)
            except KeyError:
                raise ConversionError(f"{zip_filename(env, 'depth_left')} is missing {member}") from None
            Image.fromarray(depth).save(staging / "depth_00" / f"{name}.png", format="PNG")
            invalid_fractions.append(invalid)
    finally:
        for archive in archives.values():
            archive.close()

    (staging / "pose_left.txt").write_text(pose_left, encoding="utf-8")
    (staging / "pose_right.txt").write_text(pose_right, encoding="utf-8")
    Writer(str(staging))(Normalizer()(TartanAirReader(str(staging / "pose_left.txt"))()))
    (staging / EDEX_FILE).write_text(json.dumps(edex_document(frame_count), indent=4) + "\n", encoding="utf-8")

    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)

    positions = left[:, :3, 3]
    return {
        "sequence": f"{env}/{trajectory}",
        "frames": frame_count,
        "path_length_m": round(float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()), 3),
        "invalid_depth_fraction": round(float(np.mean(invalid_fractions)), 4),
    }


def _zip_trajectories(raw_dir: Path, env: str) -> Tuple[str, ...]:
    path = raw_dir / zip_filename(env, "image_left")
    if not path.is_file():
        raise ConversionError(f"missing {path}; run the download step first")
    with zipfile.ZipFile(path) as archive:
        return tuple(sorted(m.group(1) for m in map(_TRAJECTORY.match, archive.namelist()) if m))


def _config_entries(sequences: Dict[str, Tuple[str, ...]]) -> List[List[Tuple[str, object]]]:
    entries = []
    for env, trajectories in sequences.items():
        for trajectory in trajectories:
            for use_slam in (False, True):
                entries.append(rgbd.reporter_sequence_entry(
                    sequence_folder=f"{env}/{trajectory}",
                    sequence_title=f"{env}-{SPLIT}-{trajectory}-{'SLAM' if use_slam else 'ODOM'}",
                    use_slam=use_slam,
                ))
    return entries


def _manifest() -> Dict[str, Tuple[str, int]]:
    rows = [line.split() for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {key: (md5, int(size)) for key, md5, size in rows}


def convert(raw_dir: Path, output_dir: Path, envs: Optional[Sequence[str]] = None,
            workers: Optional[int] = None) -> Path:
    """Convert the downloaded zips into ``output_dir`` and write its configs.

    ``envs`` limits a partial local run; the configs then list only those
    environments' sequences. Returns ``output_dir``, the dataset root.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    chosen = selected(envs)
    for env, pinned in chosen.items():
        found = _zip_trajectories(raw_dir, env)
        if found != pinned:
            raise ConversionError(
                f"{zip_filename(env, 'image_left')} holds {list(found)}, the selection pins {list(pinned)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(raw_dir), env, trajectory, str(output_dir)) for env, trajs in chosen.items() for trajectory in trajs]
    with ProcessPoolExecutor(max_workers=workers or min(16, os.cpu_count() or 1)) as pool:
        sequences = list(pool.map(convert_trajectory, *zip(*jobs)))

    subsets = config_subsets(chosen)
    for name, subset in subsets.items():
        (output_dir / name).write_text(
            rgbd.format_reporter_config(_config_entries(subset), f"{DATASET_ID}/", SEGMENT_LENGTHS),
            encoding="utf-8",
        )

    manifest = _manifest()
    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "converter_version": _CONVERTER_VERSION,
        "source": {
            "name": SOURCE_NAME,
            "url": SOURCE_URL,
            "split": SPLIT,
            "zips": [{"key": key, "md5": md5, "size": size}
                     for key, (md5, size) in manifest.items() if key.split("/")[0] in chosen],
        },
        "rig": {
            "cameras": ["left", "right"],
            "baseline_m": BASELINE_M,
            "size": list(IMAGE_SIZE),
            "focal": list(FOCAL),
            "principal": list(PRINCIPAL),
            "fps": FPS,
            "depth": "left camera only; uint16 png, millimetres; 0 marks sky and anything beyond 65.535 m",
        },
        "configs": {name: sum(map(len, subset.values())) for name, subset in subsets.items()},
        "sequences": sequences,
    }
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir
