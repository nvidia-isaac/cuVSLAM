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

"""Convert CODa sequence archives to the cuVSLAM reporter layout.

Produces, per sequence::

    <sequence>/00/<sequence>.0.00001.png   left rectified image, 1-indexed
    <sequence>/01/<sequence>.1.00001.png   right rectified image, 1-indexed
    <sequence>/gt.txt                      3x4 pose per frame, relative to frame 0
    <sequence>/stereo.edex                 rig, intrinsics, and baseline

plus a ``dataset_metadata.json`` and the reporter configs at the dataset root.

Members are read straight out of the zip rather than extracted first: one CODa
sequence unpacks to tens of gigabytes, and only the rectified stereo pair, the
calibration, and the poses are needed.

CODa is not redistributable, so the archives are supplied by the user rather
than downloaded; see ``download_coda.sh``.
"""

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_SCHEMA_VERSION = 1
_CONVERTER_VERSION = 1

DATASET_ID = "coda"

SOURCE_NAME = "UT Campus Object Dataset (CODa)"
SOURCE_URL = "https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV"
SOURCE_CITATION = (
    "A. Zhang et al., Towards Robust Robot 3D Perception in Urban Environments: "
    "The UT Campus Object Dataset, Texas Data Repository, doi:10.18738/T8/BBOQMV"
)

# CODa ships 23 sequences, one zip each, named by their index.
ALL_SEQS: Tuple[str, ...] = tuple(str(index) for index in range(23))

# Globally optimized poses where they exist, per-scan poses otherwise. Sequences
# 8, 14, and 15 only ship poses/dense/.
_GT_PREFERRED = "poses/dense_global"
_GT_FALLBACK = "poses/dense"

# The longest sequence has roughly 21k frames at 10 Hz, so four digits (what
# KITTI uses) would not be enough to keep the output names sorted.
_FRAME_DIGITS = 5

_FPS = 10
SEGMENT_LENGTHS: Tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800)

_IDENTITY_POSE_LINE = (
    "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 "
    "0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 "
    "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
)


class ConversionError(RuntimeError):
    """Raised when the source CODa archives cannot be converted."""


def archive_name(sequence: str) -> str:
    """Return the source archive filename for one sequence."""
    return f"{sequence}.zip"


def _validated_sequences(sequences: Sequence[str]) -> List[str]:
    selected = [str(sequence) for sequence in sequences]
    if not selected:
        raise ConversionError("no sequences selected")
    unknown = [sequence for sequence in selected if sequence not in ALL_SEQS]
    if unknown:
        raise ConversionError(
            f"unknown sequence(s): {', '.join(sorted(unknown))}; known: {', '.join(ALL_SEQS)}"
        )
    duplicated = sorted({sequence for sequence in selected if selected.count(sequence) > 1})
    if duplicated:
        raise ConversionError(f"duplicate sequence(s): {', '.join(duplicated)}")
    return [sequence for sequence in ALL_SEQS if sequence in set(selected)]


def _discovered_sequences(raw_dir: Path) -> List[str]:
    """Return the sequences whose archive is present, in numeric order.

    An omitted selection converts what the user actually downloaded: CODa is
    fetched by hand one sequence at a time, so having all 23 is the exception.
    """
    present = [sequence for sequence in ALL_SEQS if (raw_dir / archive_name(sequence)).is_file()]
    if not present:
        raise ConversionError(
            f"no CODa sequence archives found in {raw_dir}; expected one or more of "
            f"{ALL_SEQS[0]}.zip … {ALL_SEQS[-1]}.zip"
        )
    return present


def required_archives(sequences: Optional[Sequence[str]] = None) -> List[str]:
    """Return the archives an explicit selection needs.

    An omitted selection converts whatever is already on disk, so nothing is
    required up front and the list is empty.
    """
    if sequences is None:
        return []
    return [archive_name(sequence) for sequence in _validated_sequences(sequences)]


# ---------------------------------------------------------------------------
# Calibration parsing
#
# CODa calibration files are OpenCV-style YAML with a fixed shape, so they are
# read with targeted patterns instead of pulling in a YAML parser.
# ---------------------------------------------------------------------------

def _read_matrix(text: str, key: str, sequence: str) -> Optional[List[float]]:
    """Return the float list under ``<key>: ... data: [...]``, spanning lines.

    An absent key returns None, which the callers turn into their own message. A
    key that is present but malformed is reported here instead, so a truncated or
    corrupted calibration file fails as a ConversionError rather than escaping as
    a bare ValueError.
    """
    lines = text.splitlines()
    index = 0
    in_section = False
    while index < len(lines):
        line = lines[index]
        if re.match(r"^" + re.escape(key) + r":\s*$", line):
            in_section = True
            index += 1
            continue
        if in_section and line and not line[0].isspace():
            in_section = False
        if in_section:
            match = re.match(r"^\s+data:\s*\[(.*)$", line)
            if match:
                buffered = match.group(1)
                while "]" not in buffered and index + 1 < len(lines):
                    index += 1
                    buffered += " " + lines[index].strip()
                if "]" not in buffered:
                    raise ConversionError(f"sequence {sequence}: {key} data list is never closed")
                inner = buffered[: buffered.index("]")]
                try:
                    values = [float(value.strip()) for value in inner.split(",") if value.strip()]
                except ValueError as exc:
                    raise ConversionError(f"sequence {sequence}: {key} holds a non-numeric entry ({exc})") from exc
                # float() takes "nan" and "inf" without complaint, and those would reach
                # the EDEX as a rig nothing can use.
                for value in values:
                    if not math.isfinite(value):
                        raise ConversionError(f"sequence {sequence}: {key} holds a non-finite entry ({value})")
                return values
        index += 1
    return None


def _read_int(text: str, key: str) -> Optional[int]:
    match = re.search(r"^" + re.escape(key) + r":\s*(\d+)\s*$", text, re.M)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Rigid transform helpers
#
# Poses are carried as a (rotation, translation) pair rather than a 4x4 matrix,
# matching the SE(3) helpers the other converters use.
# ---------------------------------------------------------------------------

def quaternion_to_matrix(quaternion: Sequence[float]) -> List[List[float]]:
    """Convert a wxyz quaternion to a row-major 3x3 rotation matrix."""
    w, x, y, z = quaternion
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise ConversionError("ground-truth pose carries a zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def invert_transform(
    rotation: Sequence[Sequence[float]], translation: Sequence[float]
) -> Tuple[List[List[float]], List[float]]:
    """Invert a rigid transform given as a rotation matrix and translation."""
    transposed = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(transposed[row][column] * translation[column] for column in range(3)) for row in range(3)
    ]
    return transposed, inverse_translation


def compose_transforms(
    first_rotation: Sequence[Sequence[float]],
    first_translation: Sequence[float],
    second_rotation: Sequence[Sequence[float]],
    second_translation: Sequence[float],
) -> Tuple[List[List[float]], List[float]]:
    """Return the composition of two rigid transforms, first applied last."""
    rotation = [
        [sum(first_rotation[row][inner] * second_rotation[inner][column] for inner in range(3)) for column in range(3)]
        for row in range(3)
    ]
    translation = [
        sum(first_rotation[row][inner] * second_translation[inner] for inner in range(3)) + first_translation[row]
        for row in range(3)
    ]
    return rotation, translation


def _pose_line(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> str:
    values = (
        list(rotation[0])
        + [translation[0]]
        + list(rotation[1])
        + [translation[1]]
        + list(rotation[2])
        + [translation[2]]
    )
    return " ".join(f"{value:.6e}" for value in values)


# ---------------------------------------------------------------------------
# Output documents
# ---------------------------------------------------------------------------

def _camera(focal: Sequence[float], principal: Sequence[float], size: Sequence[int], offset: float) -> Dict:
    return {
        "intrinsics": {
            "distortion_model": "pinhole",
            "distortion_params": [],
            "focal": list(focal),
            "principal": list(principal),
            "size": list(size),
        },
        "transform": [
            [1.0, 0.0, 0.0, offset],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    }


def edex_document(calibration: Dict, frames: int, left_name: str, right_name: str) -> List[Dict]:
    """Return the ``stereo.edex`` document for one converted sequence.

    Both cameras carry the same intrinsics: CODa ships rectified images, so the
    right camera differs from the left only by the baseline along x.
    """
    focal = calibration["focal"]
    principal = calibration["principal"]
    size = calibration["size"]
    return [
        {
            "cameras": [
                _camera(focal, principal, size, 0.0),
                _camera(focal, principal, size, calibration["baseline"]),
            ],
            "frame_end": frames,
            "frame_start": 1,
            "version": "0.9",
        },
        {
            "fps": _FPS,
            "points2d": {},
            "points3d": {},
            "rig_positions": {},
            "sequence": [[f"00/{left_name}"], [f"01/{right_name}"]],
        },
    ]


def reporter_sequence_entry(sequence: str, use_slam: bool, has_ground_truth: bool) -> List[Tuple[str, object]]:
    """Build one reporter ``sequence_cfgs`` entry as ordered key/value pairs."""
    fields: List[Tuple[str, object]] = [
        ("enable", True),
        ("sequence_folder", sequence),
        ("edex_file", "stereo.edex"),
        ("precompute_2d_tracks", False),
        ("precompute_key_frames", False),
        ("use_gt_scale", False),
        ("sequence_title", f"CODA-{int(sequence):02d}-{'SLAM' if use_slam else 'ODOM'}"),
    ]
    # Naming a missing gt.txt makes the reporter fail the sequence, so the key is
    # only written when the archive actually carried poses.
    if has_ground_truth:
        fields.append(("gt_file_path", "gt.txt"))
    if use_slam:
        fields.append(("use_slam", True))
    return fields


def format_reporter_config(
    entries: Sequence[Sequence[Tuple[str, object]]],
    dataset_folder: str,
    segment_lengths: Sequence[float],
) -> str:
    """Render a reporter config file.

    Emitted as text rather than through ``json.dumps`` so the layout matches the
    KITTI and EuRoC converters, which the reporter fixtures were written against.
    """
    if not entries:
        raise ConversionError("cannot write a reporter config with no sequences")
    lines = [
        "{",
        '    "version": "0.1",',
        '    "write_cache": false,',
        '    "use_cuda": false,',
        f'    "dataset_folder": {json.dumps(dataset_folder)},',
        '    "use_icp_scaling": false,',
        f'    "segment_lengths": {json.dumps(list(segment_lengths))},',
        '    "sequence_cfgs": [',
    ]
    for entry_index, fields in enumerate(entries):
        lines.append("        {")
        for field_index, (key, value) in enumerate(fields):
            comma = "," if field_index + 1 < len(fields) else ""
            lines.append(f'            "{key}": {json.dumps(value)}{comma}')
        comma = "," if entry_index + 1 < len(entries) else ""
        lines.append(f"        }}{comma}")
    lines.extend(["  ]", "}", ""])
    return "\n".join(lines)


def _config_entries(
    sequences: Sequence[str], ground_truth: Sequence[str], slam: bool, odom: bool
) -> List[List[Tuple[str, object]]]:
    with_ground_truth = set(ground_truth)
    entries = []
    for sequence in sequences:
        has_ground_truth = sequence in with_ground_truth
        if odom:
            entries.append(reporter_sequence_entry(sequence, use_slam=False, has_ground_truth=has_ground_truth))
        if slam:
            entries.append(reporter_sequence_entry(sequence, use_slam=True, has_ground_truth=has_ground_truth))
    return entries


def _write_configs(output_dir: Path, sequences: Sequence[str], ground_truth: Sequence[str]) -> List[str]:
    """Write the reporter configs, returning their names.

    The name determines the KPI prefix: the collector takes everything before the
    first hyphen, so every ``coda-*.cfg`` here yields ``CODA``. The ``_gt``
    configs cover only the sequences that carried poses, so they are skipped
    entirely when none did.
    """
    with_ground_truth = [sequence for sequence in sequences if sequence in set(ground_truth)]
    configs = {
        f"{DATASET_ID}-vio_slam.cfg": _config_entries(sequences, ground_truth, slam=True, odom=True),
    }
    if with_ground_truth:
        configs.update(
            {
                f"{DATASET_ID}-slam_gt.cfg": _config_entries(
                    with_ground_truth, ground_truth, slam=True, odom=False
                ),
                f"{DATASET_ID}-vio_gt.cfg": _config_entries(
                    with_ground_truth, ground_truth, slam=False, odom=True
                ),
                f"{DATASET_ID}-vio_slam_gt.cfg": _config_entries(
                    with_ground_truth, ground_truth, slam=True, odom=True
                ),
            }
        )
    for name, entries in configs.items():
        (output_dir / name).write_text(
            format_reporter_config(entries, f"{DATASET_ID}/", SEGMENT_LENGTHS),
            encoding="utf-8",
        )
    return sorted(configs)


def _dataset_metadata(
    archives: List[Dict[str, object]],
    config_names: List[str],
    sequence_metadata: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "converter_version": _CONVERTER_VERSION,
        "source": {
            "name": SOURCE_NAME,
            "url": SOURCE_URL,
            "citation": SOURCE_CITATION,
            # Size rather than a checksum: the archives run to tens of gigabytes
            # each and hashing them would double the read for the whole dataset.
            "archives": archives,
        },
        "reporter_configs": config_names,
        "sequences": sequence_metadata,
    }


# ---------------------------------------------------------------------------
# Per-sequence conversion
# ---------------------------------------------------------------------------

def _read_member(archive: zipfile.ZipFile, name: str, sequence: str) -> str:
    try:
        return archive.read(name).decode()
    except KeyError as exc:
        raise ConversionError(f"sequence {sequence}: {name} not found in {Path(archive.filename).name}") from exc


def _read_calibration(archive: zipfile.ZipFile, sequence: str) -> Dict:
    """Return the rectified stereo calibration for one sequence.

    Only cam0's file is read. CODa stores rectified images under
    ``2d_rect/cam{0,1}``, so the right camera's intrinsics equal the left's by
    construction. The baseline comes from cam0's ``disparity_matrix`` entry
    ``Q[14] = -1/Tx`` rather than cam1's ``projection_matrix`` ``P[0,3]``,
    because CODa encodes the latter off by roughly 1000x against the physical
    cam0-to-cam1 extrinsic while Q agrees with it.
    """
    text = _read_member(archive, f"calibrations/{sequence}/calib_cam0_intrinsics.yaml", sequence)

    projection = _read_matrix(text, "projection_matrix", sequence)
    if projection is None or len(projection) < 8:
        raise ConversionError(f"sequence {sequence}: cannot parse cam0 projection_matrix")
    width = _read_int(text, "image_width")
    height = _read_int(text, "image_height")
    if width is None or height is None:
        raise ConversionError(f"sequence {sequence}: cannot parse cam0 image dimensions")

    disparity = _read_matrix(text, "disparity_matrix", sequence)
    if disparity is None or len(disparity) < 15:
        raise ConversionError(f"sequence {sequence}: cannot parse cam0 disparity_matrix")
    if abs(disparity[14]) < 1e-9:
        raise ConversionError(
            f"sequence {sequence}: cam0 disparity_matrix has zero/invalid Q[14] "
            f"(={disparity[14]}); cannot derive baseline"
        )

    return {
        "focal": [projection[0], projection[5]],
        "principal": [projection[2], projection[6]],
        "size": [width, height],
        "baseline": abs(1.0 / disparity[14]),
    }


def _read_lidar_to_camera(archive: zipfile.ZipFile, sequence: str) -> Tuple[List[List[float]], List[float]]:
    """Return T_os1_from_cam0, the transform ground truth is re-grounded through."""
    text = _read_member(archive, f"calibrations/{sequence}/calib_os1_to_cam0.yaml", sequence)
    extrinsic = _read_matrix(text, "extrinsic_matrix", sequence)
    if extrinsic is None or len(extrinsic) != 16:
        raise ConversionError(f"sequence {sequence}: cannot parse os1_to_cam0 extrinsic_matrix")
    rotation = [extrinsic[row * 4: row * 4 + 3] for row in range(3)]
    translation = [extrinsic[row * 4 + 3] for row in range(3)]
    return invert_transform(rotation, translation)


def _index_images(archive: zipfile.ZipFile, sequence: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    patterns = [
        re.compile(
            rf"^2d_rect/cam{camera}/{re.escape(sequence)}/"
            rf"2d_rect_cam{camera}_{re.escape(sequence)}_(\d+)\.png$"
        )
        for camera in (0, 1)
    ]
    indexed: List[Dict[int, str]] = [{}, {}]
    for info in archive.infolist():
        for camera, pattern in enumerate(patterns):
            match = pattern.match(info.filename)
            if match:
                indexed[camera][int(match.group(1))] = info.filename
                break
    return indexed[0], indexed[1]


def _ground_truth_member(archive: zipfile.ZipFile, sequence: str) -> Optional[str]:
    names = set(archive.namelist())
    for prefix in (_GT_PREFERRED, _GT_FALLBACK):
        if f"{prefix}/{sequence}.txt" in names:
            return f"{prefix}/{sequence}.txt"
    return None


def _ground_truth_lines(
    poses: Sequence[str],
    frames: Sequence[int],
    lidar_to_camera: Tuple[List[List[float]], List[float]],
    sequence: str,
) -> List[str]:
    """Render ``gt.txt``: one row-major 3x4 pose per frame, relative to frame 0.

    CODa stores ``T_world_from_os1`` in the LiDAR frame, while cuVSLAM tracks in
    the cam0 frame with the first pose pinned to identity, so each pose is moved
    onto cam0 and then expressed relative to the first one.
    """
    camera_rotation, camera_translation = lidar_to_camera
    on_camera = []
    for frame in frames:
        row = poses[frame].split()
        if len(row) < 8:
            raise ConversionError(f"sequence {sequence}: pose row {frame} has {len(row)} fields, expected 8")
        translation = [float(value) for value in row[1:4]]
        rotation = quaternion_to_matrix([float(value) for value in row[4:8]])
        on_camera.append(compose_transforms(rotation, translation, camera_rotation, camera_translation))

    inverse = invert_transform(*on_camera[0])
    lines = [_IDENTITY_POSE_LINE]
    for rotation, translation in on_camera[1:]:
        lines.append(_pose_line(*compose_transforms(*inverse, rotation, translation)))
    return lines


def convert_sequence(archive_path: Path, sequence: str, output_dir: Path) -> Dict[str, object]:
    """Convert one sequence archive into ``output_dir/<sequence>``."""
    sequence_dir = output_dir / sequence
    left_dir = sequence_dir / "00"
    right_dir = sequence_dir / "01"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        calibration = _read_calibration(archive, sequence)
        lidar_to_camera = _read_lidar_to_camera(archive, sequence)

        left_images, right_images = _index_images(archive, sequence)
        frames = sorted(set(left_images) & set(right_images))
        if len(frames) != len(left_images) or len(frames) != len(right_images):
            print(
                f"  WARNING: sequence {sequence}: {len(left_images)} left / {len(right_images)} right / "
                f"{len(frames)} paired — using paired only"
            )
        if not frames:
            raise ConversionError(f"sequence {sequence}: no paired stereo frames in {archive_path.name}")

        # Ground truth is read before any image is written so a frame whose pose
        # row is missing can be dropped from both, keeping gt.txt one row per
        # image pair, which is what the reporter assumes.
        member = _ground_truth_member(archive, sequence)
        poses: Optional[List[str]] = None
        dropped = 0
        if member is None:
            print(f"  WARNING: sequence {sequence}: no pose file in the archive, skipping gt.txt")
        else:
            poses = archive.read(member).decode().strip().splitlines()
            print(f"  reading ground truth from {member} ({len(poses)} rows)")
            kept = [frame for frame in frames if frame < len(poses)]
            dropped = len(frames) - len(kept)
            if dropped:
                print(
                    f"  WARNING: sequence {sequence}: {dropped} frame(s) past the end of the pose "
                    "file — dropping them from both the images and gt.txt"
                )
            frames = kept
            if not frames:
                raise ConversionError(f"sequence {sequence}: every frame is past the end of {member}")

        print(
            f"Sequence {sequence}: {len(frames)} frames, {calibration['size'][0]}x{calibration['size'][1]}, "
            f"focal={calibration['focal'][0]:.4f},{calibration['focal'][1]:.4f}, "
            f"principal={calibration['principal'][0]:.4f},{calibration['principal'][1]:.4f}, "
            f"baseline={calibration['baseline']:.6f} m"
        )

        left_names, right_names = [], []
        for index, frame in enumerate(frames, start=1):
            left_names.append(f"{sequence}.0.{index:0{_FRAME_DIGITS}d}.png")
            right_names.append(f"{sequence}.1.{index:0{_FRAME_DIGITS}d}.png")
            (left_dir / left_names[-1]).write_bytes(archive.read(left_images[frame]))
            (right_dir / right_names[-1]).write_bytes(archive.read(right_images[frame]))

        if poses is not None:
            (sequence_dir / "gt.txt").write_text(
                "\n".join(_ground_truth_lines(poses, frames, lidar_to_camera, sequence)) + "\n",
                encoding="utf-8",
            )

    (sequence_dir / "stereo.edex").write_text(
        json.dumps(edex_document(calibration, len(frames), left_names[0], right_names[0]), indent=4) + "\n",
        encoding="utf-8",
    )

    return {
        "sequence": sequence,
        "source_archive": archive_path.name,
        "ground_truth_source": member,
        "camera": {
            "focal": calibration["focal"],
            "principal": calibration["principal"],
            "size": calibration["size"],
            "baseline": calibration["baseline"],
        },
        "converted_counts": {
            "frames": len(frames),
            "ground_truth_poses": len(frames) if poses is not None else 0,
        },
        "dropped_outside_ground_truth": dropped,
    }


def convert(
    raw_dir: Path,
    output_dir: Path,
    sequences: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Convert the selected sequences from ``raw_dir`` into ``output_dir``.

    ``sequences`` defaults to every sequence whose archive is present.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    selected = _validated_sequences(sequences) if sequences is not None else _discovered_sequences(raw_dir)

    missing = [
        archive_name(sequence)
        for sequence in selected
        if not (raw_dir / archive_name(sequence)).is_file()
    ]
    if missing:
        raise ConversionError(f"missing archive(s) in {raw_dir}: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_metadata: List[Dict[str, object]] = []
    sequence_metadata: List[Dict[str, object]] = []
    for sequence in selected:
        archive = raw_dir / archive_name(sequence)
        archive_metadata.append({"name": archive.name, "size_bytes": archive.stat().st_size})
        sequence_metadata.append(convert_sequence(archive, sequence, output_dir))

    with_ground_truth = [
        str(entry["sequence"]) for entry in sequence_metadata if entry["ground_truth_source"] is not None
    ]
    config_names = _write_configs(output_dir, selected, with_ground_truth)
    metadata = _dataset_metadata(archive_metadata, config_names, sequence_metadata)
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
