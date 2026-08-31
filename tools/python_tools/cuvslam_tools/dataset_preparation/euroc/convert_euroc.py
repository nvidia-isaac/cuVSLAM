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

"""Convert the official EuRoC MAV nested archives to portable cuVSLAM EDEX data."""

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Sequence, Tuple


_OUTER_ZIPS = [
    (
        "machine_hall.zip",
        ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult"],
    ),
    ("vicon_room1.zip", ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]),
    ("vicon_room2.zip", ["V2_01_easy", "V2_02_medium", "V2_03_difficult"]),
]
ALL_SEQS = [sequence for _, sequences in _OUTER_ZIPS for sequence in sequences]

_SOURCE_DOI = "10.3929/ethz-b-000690084"
_SOURCE_URL = "https://doi.org/10.3929/ethz-b-000690084"
_SCHEMA_VERSION = "1.0"
_CONVERTER_VERSION = "1.0"


class ConversionError(ValueError):
    """Raised when EuRoC input cannot be converted safely and unambiguously."""


# Recalibrated parameters from examples/euroc, matching the benchmark corpus.
_EDEX_TEMPLATE = """\
[
    {
        "cameras": [
            {
                "intrinsics": {
                    "distortion_model": "fisheye",
                    "distortion_params": [
                        -0.0062748193357009315,
                        0.029005519692414498,
                        -0.03438856012105873,
                        0.014830434499283266
                    ],
                    "focal": [
                        460.9855047976205,
                        459.67892586299877
                    ],
                    "principal": [
                        366.09923470990486,
                        249.22157605207943
                    ],
                    "size": [
                        752,
                        480
                    ]
                },
                "transform": [
                    [
                        1,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        1,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        1,
                        0
                    ]
                ]
            },
            {
                "intrinsics": {
                    "distortion_model": "fisheye",
                    "distortion_params": [
                        0.0030523152970989243,
                        0.0022729295767180894,
                        -0.0023088086978921007,
                        0.002031411542915807
                    ],
                    "focal": [
                        459.56983030590357,
                        458.20957848757143
                    ],
                    "principal": [
                        379.5888918566419,
                        255.9525258537914
                    ],
                    "size": [
                        752,
                        480
                    ]
                },
                "transform": [
                    [
                        0.9999967,
                        0.0021889,
                        -0.0013548,
                        0.1099839
                    ],
                    [
                        -0.0022078,
                        0.9998979,
                        -0.0141205,
                        0.0005322
                    ],
                    [
                        0.0013237,
                        0.0141234,
                        0.9998994,
                        -0.0004407
                    ]
                ]
            }
        ],
        "frame_end": ___FRAME_END___,
        "frame_start": 0,
        "imu": {
            "g": [
                0.33835679,
                -9.43382516,
                -2.54067297
            ],
            "measurements": "IMU.jsonl",
            "transform": [
                [ 0.0149006,  0.9996865, -0.0201192,  0.0683705],
                [ 0.9998883, -0.014921 , -0.0008608,  0.0158797],
                [-0.0011607, -0.0201041, -0.9997972,  0.0035799]
            ],
            "gyroscope_noise_density": 0.00016968,
            "gyroscope_random_walk": 0.000019393,
            "accelerometer_noise_density": 0.002,
            "accelerometer_random_walk": 0.003,
            "frequency": 200
        },
        "version": "0.9"
    },
    {
        "frame_metadata": "frame_metadata.jsonl",
        "points2d": {},
        "points3d": {},
        "rig_positions": {},
        "sequence": [
            [
                "00/l.000000.png"
            ],
            [
                "01/r.000000.png"
            ]
        ]
    }
]
"""


def _make_edex(frame_end: int) -> str:
    return _EDEX_TEMPLATE.replace("___FRAME_END___", str(frame_end))


# Official EuRoC cam0 sensor-to-body transform used for GT alignment.
_T_BS_DATA = [
    0.0148655429818,
    -0.999880929698,
    0.00414029679422,
    -0.0216401454975,
    0.999557249008,
    0.0149672133247,
    0.025715529948,
    -0.064676986768,
    -0.0257744366974,
    0.00375618835797,
    0.999660727178,
    0.00981073058949,
]
_R_BC0 = [[_T_BS_DATA[row * 4 + column] for column in range(3)] for row in range(3)]
_T_BC0 = [_T_BS_DATA[row * 4 + 3] for row in range(3)]


def _matrix_multiply(left: List[List[float]], right: List[List[float]]) -> List[List[float]]:
    return [
        [sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _matrix_vector(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return [sum(matrix[row][index] * vector[index] for index in range(3)) for row in range(3)]


def _invert_transform(
    rotation: List[List[float]], translation: List[float]
) -> Tuple[List[List[float]], List[float]]:
    inverse_rotation = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_translation = _matrix_vector(inverse_rotation, [-value for value in translation])
    return inverse_rotation, inverse_translation


def _compose_transforms(
    left_rotation: List[List[float]],
    left_translation: List[float],
    right_rotation: List[List[float]],
    right_translation: List[float],
) -> Tuple[List[List[float]], List[float]]:
    rotated_translation = _matrix_vector(left_rotation, right_translation)
    return (
        _matrix_multiply(left_rotation, right_rotation),
        [rotated_translation[index] + left_translation[index] for index in range(3)],
    )


def _normalize_quaternion(quaternion: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        raise ConversionError("malformed ground-truth CSV: zero-length quaternion")
    return [value / norm for value in quaternion]


def _quaternion_to_rotation(quaternion: List[float]) -> List[List[float]]:
    w, x, y, z = _normalize_quaternion(quaternion)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def _slerp(left: List[float], right: List[float], alpha: float) -> List[float]:
    left = _normalize_quaternion(left)
    right = _normalize_quaternion(right)
    dot = sum(left[index] * right[index] for index in range(4))
    if dot < 0.0:
        right = [-value for value in right]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(
            [left[index] + alpha * (right[index] - left[index]) for index in range(4)]
        )
    angle = math.acos(dot)
    sine = math.sin(angle)
    left_weight = math.sin((1.0 - alpha) * angle) / sine
    right_weight = math.sin(alpha * angle) / sine
    return [left_weight * left[index] + right_weight * right[index] for index in range(4)]


GroundTruthRow = Tuple[int, List[float], List[float]]


def _interpolate_body_pose(
    ground_truth: List[GroundTruthRow], ground_truth_times: List[int], timestamp: int
) -> Tuple[List[float], List[float]]:
    index = bisect.bisect_left(ground_truth_times, timestamp)
    if index < len(ground_truth_times) and ground_truth_times[index] == timestamp:
        return ground_truth[index][1], ground_truth[index][2]
    if index == 0 or index == len(ground_truth):
        raise ConversionError(f"camera timestamp {timestamp} lies outside the ground-truth range")
    left_time, left_position, left_quaternion = ground_truth[index - 1]
    right_time, right_position, right_quaternion = ground_truth[index]
    alpha = (timestamp - left_time) / (right_time - left_time)
    position = [
        left_position[axis] + alpha * (right_position[axis] - left_position[axis]) for axis in range(3)
    ]
    return position, _slerp(left_quaternion, right_quaternion, alpha)


def _cam0_world_pose(
    ground_truth: List[GroundTruthRow], ground_truth_times: List[int], timestamp: int
) -> Tuple[List[List[float]], List[float]]:
    body_position, body_quaternion = _interpolate_body_pose(ground_truth, ground_truth_times, timestamp)
    return _compose_transforms(
        _quaternion_to_rotation(body_quaternion),
        body_position,
        _R_BC0,
        _T_BC0,
    )


def _validate_safe_archive_path(path: str, description: str) -> None:
    """Reject path forms that could escape an archive-relative namespace."""
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        not path
        or "\x00" in path
        or path.startswith(("/", "\\"))
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ConversionError(f"unsafe {description}: {path!r}")


def _validate_zip_members(archive: zipfile.ZipFile, description: str) -> None:
    for member in archive.infolist():
        _validate_safe_archive_path(member.filename, f"{description} member path")


def _required_member(archive: zipfile.ZipFile, member_name: str, sequence: str) -> zipfile.ZipInfo:
    try:
        member = archive.getinfo(member_name)
    except KeyError as exc:
        raise ConversionError(f"{sequence}: required archive member missing: {member_name}") from exc
    if member.is_dir():
        raise ConversionError(f"{sequence}: required archive member is a directory: {member_name}")
    return member


def _read_member_bytes(archive: zipfile.ZipFile, member_name: str, sequence: str) -> bytes:
    member = _required_member(archive, member_name, sequence)
    try:
        return archive.read(member)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ConversionError(f"{sequence}: failed to read archive member {member_name}: {exc}") from exc


def _read_member_text(archive: zipfile.ZipFile, member_name: str, sequence: str) -> str:
    try:
        return _read_member_bytes(archive, member_name, sequence).decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConversionError(f"{sequence}: malformed CSV {member_name}: not UTF-8") from exc


def _csv_rows(text: str, member_name: str, sequence: str) -> List[Tuple[int, List[str]]]:
    rows = []
    try:
        reader = csv.reader(io.StringIO(text))
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            rows.append((line_number, row))
    except csv.Error as exc:
        raise ConversionError(f"{sequence}: malformed CSV {member_name}: {exc}") from exc
    if not rows:
        raise ConversionError(f"{sequence}: malformed CSV {member_name}: no data rows")
    return rows


def _parse_camera_csv(text: str, member_name: str, sequence: str) -> List[Tuple[int, str]]:
    entries = []
    timestamps = set()
    for line_number, row in _csv_rows(text, member_name, sequence):
        if len(row) != 2:
            raise ConversionError(
                f"{sequence}: malformed CSV {member_name} line {line_number}: expected 2 columns"
            )
        try:
            timestamp = int(row[0].strip())
        except ValueError as exc:
            raise ConversionError(
                f"{sequence}: malformed CSV {member_name} line {line_number}: invalid timestamp"
            ) from exc
        filename = row[1].strip()
        _validate_safe_archive_path(filename, f"image filename in {member_name} line {line_number}")
        if timestamp in timestamps:
            raise ConversionError(
                f"{sequence}: malformed CSV {member_name} line {line_number}: duplicate timestamp {timestamp}"
            )
        timestamps.add(timestamp)
        entries.append((timestamp, filename))
    return sorted(entries)


def _parse_imu_csv(text: str, member_name: str, sequence: str) -> List[Tuple[int, List[float]]]:
    entries = []
    for line_number, row in _csv_rows(text, member_name, sequence):
        if len(row) != 7:
            raise ConversionError(
                f"{sequence}: malformed CSV {member_name} line {line_number}: expected 7 columns"
            )
        try:
            entries.append((int(row[0].strip()), [float(value.strip()) for value in row[1:]]))
        except ValueError as exc:
            raise ConversionError(
                f"{sequence}: malformed CSV {member_name} line {line_number}: invalid numeric value"
            ) from exc
    entries.sort(key=lambda entry: entry[0])
    return entries


def _parse_ground_truth_csv(text: str, member_name: str, sequence: str) -> List[GroundTruthRow]:
    entries = []
    previous_timestamp = None
    for line_number, row in _csv_rows(text, member_name, sequence):
        if len(row) < 8:
            raise ConversionError(
                f"{sequence}: malformed CSV {member_name} line {line_number}: expected at least 8 columns"
            )
        try:
            timestamp = int(row[0].strip())
            position = [float(value.strip()) for value in row[1:4]]
            quaternion = [float(value.strip()) for value in row[4:8]]
        except ValueError as exc:
            raise ConversionError(
                f"{sequence}: malformed CSV {member_name} line {line_number}: invalid numeric value"
            ) from exc
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ConversionError(
                f"{sequence}: malformed CSV {member_name} line {line_number}: "
                "timestamps must be strictly increasing"
            )
        _normalize_quaternion(quaternion)
        entries.append((timestamp, position, quaternion))
        previous_timestamp = timestamp
    return entries


def _json_scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _sequence_config_entry(sequence: str, mode: str) -> List[Tuple[str, object]]:
    is_slam = mode == "slam"
    label = "SLAM" if is_slam else "ODOM"
    fields = [
        ("enable", True),
        ("sequence_folder", sequence),
        ("edex_file", "stereo.edex"),
        ("precompute_2d_tracks", False),
        ("precompute_key_frames", False),
        ("use_gt_scale", False),
        ("sequence_title", "EUROC-" + sequence.replace("_", "-") + "-" + label),
        ("gt_file_path", "gt.txt"),
    ]
    if is_slam:
        fields.append(("use_slam", True))
    return fields


def _format_config(
    entries: List[List[Tuple[str, object]]], dataset_folder: str
) -> str:
    lines = [
        "{",
        '    "version": "0.1",',
        '    "write_cache": false,',
        '    "use_cuda": false,',
        f'    "dataset_folder": {_json_scalar(dataset_folder)},',
        '    "use_icp_scaling": false,',
        '    "segment_lengths": [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 35, 45],',
        '    "sequence_cfgs": [',
    ]
    for entry_index, fields in enumerate(entries):
        lines.append("        {")
        for field_index, (key, value) in enumerate(fields):
            comma = "," if field_index + 1 < len(fields) else ""
            lines.append(f'            "{key}": {_json_scalar(value)}{comma}')
        comma = "," if entry_index + 1 < len(entries) else ""
        lines.append(f"        }}{comma}")
    lines.extend(["  ]", "}", ""])
    return "\n".join(lines)


def _write_configs(output_dir: Path, sequences: Sequence[str]) -> List[str]:
    if not sequences:
        raise ConversionError("no sequences selected")

    dataset_folder = output_dir.resolve().name + "/"
    odometry_entries = [_sequence_config_entry(sequence, "odom") for sequence in sequences]
    slam_entries = [_sequence_config_entry(sequence, "slam") for sequence in sequences]
    combined_entries = [
        entry
        for sequence in sequences
        for entry in (_sequence_config_entry(sequence, "odom"), _sequence_config_entry(sequence, "slam"))
    ]

    configs = {
        "euroc-vio.cfg": _format_config(odometry_entries, dataset_folder),
        "euroc-slam.cfg": _format_config(slam_entries, dataset_folder),
        "euroc-vio_slam.cfg": _format_config(combined_entries, dataset_folder),
    }
    for config_name, contents in configs.items():
        (output_dir / config_name).write_text(contents, encoding="utf-8")
    return sorted(configs)


def _ground_truth_lines(
    ground_truth: List[GroundTruthRow], pairs: List[Tuple[int, str, str]]
) -> List[str]:
    ground_truth_times = [entry[0] for entry in ground_truth]
    first_rotation, first_translation = _cam0_world_pose(ground_truth, ground_truth_times, pairs[0][0])
    inverse_rotation, inverse_translation = _invert_transform(first_rotation, first_translation)
    identity = (
        "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 "
        "0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 "
        "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
    )
    lines = [identity]
    for timestamp, _, _ in pairs[1:]:
        rotation, translation = _cam0_world_pose(ground_truth, ground_truth_times, timestamp)
        relative_rotation, relative_translation = _compose_transforms(
            inverse_rotation,
            inverse_translation,
            rotation,
            translation,
        )
        values = (
            relative_rotation[0]
            + [relative_translation[0]]
            + relative_rotation[1]
            + [relative_translation[1]]
            + relative_rotation[2]
            + [relative_translation[2]]
        )
        lines.append(" ".join(f"{value:.6e}" for value in values))
    return lines


def _convert_sequence(
    sequence: str, inner_archive_path: Path, output_dir: Path, source_archive: str
) -> Dict[str, object]:
    camera_members = ["mav0/cam0/data.csv", "mav0/cam1/data.csv"]
    imu_member = "mav0/imu0/data.csv"
    ground_truth_member = "mav0/state_groundtruth_estimate0/data.csv"

    try:
        with zipfile.ZipFile(inner_archive_path) as archive:
            _validate_zip_members(archive, f"{sequence} inner archive")
            cam0_entries = _parse_camera_csv(
                _read_member_text(archive, camera_members[0], sequence),
                camera_members[0],
                sequence,
            )
            cam1_entries = _parse_camera_csv(
                _read_member_text(archive, camera_members[1], sequence),
                camera_members[1],
                sequence,
            )
            imu_entries = _parse_imu_csv(
                _read_member_text(archive, imu_member, sequence),
                imu_member,
                sequence,
            )
            ground_truth = _parse_ground_truth_csv(
                _read_member_text(archive, ground_truth_member, sequence),
                ground_truth_member,
                sequence,
            )

            ground_truth_start = ground_truth[0][0]
            ground_truth_end = ground_truth[-1][0]
            cam0_by_timestamp = dict(cam0_entries)
            cam1_by_timestamp = dict(cam1_entries)
            associated_timestamps = sorted(
                timestamp
                for timestamp in cam0_by_timestamp.keys() & cam1_by_timestamp.keys()
                if ground_truth_start <= timestamp <= ground_truth_end
            )
            pairs = [
                (timestamp, cam0_by_timestamp[timestamp], cam1_by_timestamp[timestamp])
                for timestamp in associated_timestamps
            ]
            if not pairs:
                raise ConversionError(
                    f"{sequence}: no associated cam0/cam1 frames within the ground-truth range"
                )

            # Preserve benchmark data selection: camera frames include the GT
            # boundaries, while IMU samples must lie strictly inside them.
            filtered_imu_entries = [
                entry
                for entry in imu_entries
                if ground_truth_start < entry[0] < ground_truth_end
            ]
            if not filtered_imu_entries:
                raise ConversionError(
                    f"{sequence}: no IMU samples inside the ground-truth range"
                )

            image_members = []
            for _, cam0_filename, cam1_filename in pairs:
                image_members.append(f"mav0/cam0/data/{cam0_filename}")
                image_members.append(f"mav0/cam1/data/{cam1_filename}")
            for image_member in image_members:
                _required_member(archive, image_member, sequence)

            sequence_dir = output_dir / sequence
            if sequence_dir.is_symlink():
                raise ConversionError(f"{sequence}: refusing to replace symlinked output directory")
            if sequence_dir.exists():
                shutil.rmtree(sequence_dir)
            cam0_output = sequence_dir / "00"
            cam1_output = sequence_dir / "01"
            cam0_output.mkdir(parents=True)
            cam1_output.mkdir()

            frame_metadata = []
            for frame_index, (timestamp, cam0_filename, cam1_filename) in enumerate(pairs):
                left_name = f"l.{frame_index:06d}.png"
                right_name = f"r.{frame_index:06d}.png"
                (cam0_output / left_name).write_bytes(
                    _read_member_bytes(archive, f"mav0/cam0/data/{cam0_filename}", sequence)
                )
                (cam1_output / right_name).write_bytes(
                    _read_member_bytes(archive, f"mav0/cam1/data/{cam1_filename}", sequence)
                )
                frame_metadata.append(
                    json.dumps(
                        {
                            "frame_id": frame_index,
                            "cams": [
                                {
                                    "id": 0,
                                    "filename": f"00/{left_name}",
                                    "timestamp": timestamp,
                                },
                                {
                                    "id": 1,
                                    "filename": f"01/{right_name}",
                                    "timestamp": timestamp,
                                },
                            ],
                        },
                        separators=(",", ":"),
                    )
                )
            (sequence_dir / "frame_metadata.jsonl").write_text(
                "\n".join(frame_metadata) + "\n", encoding="utf-8"
            )

            output_imu = []
            for timestamp, values in filtered_imu_entries:
                wx, wy, wz, ax, ay, az = values
                output_imu.append(
                    json.dumps(
                        {
                            "AngularVelocityX": wx,
                            "AngularVelocityY": wy,
                            "AngularVelocityZ": wz,
                            "LinearAccelerationX": ax,
                            "LinearAccelerationY": ay,
                            "LinearAccelerationZ": az,
                            "timestamp": timestamp,
                            "type": "imu_data",
                        }
                    )
                )
            (sequence_dir / "IMU.jsonl").write_text("\n".join(output_imu), encoding="utf-8")
            (sequence_dir / "stereo.edex").write_text(
                _make_edex(len(pairs) - 1), encoding="utf-8"
            )
            (sequence_dir / "gt.txt").write_text(
                "\n".join(_ground_truth_lines(ground_truth, pairs)) + "\n",
                encoding="utf-8",
            )
    except zipfile.BadZipFile as exc:
        raise ConversionError(f"{sequence}: malformed inner ZIP archive: {exc}") from exc

    return {
        "name": sequence,
        "source_archive": source_archive,
        "source_counts": {
            "cam0_frames": len(cam0_entries),
            "cam1_frames": len(cam1_entries),
            "imu_samples": len(imu_entries),
            "ground_truth_poses": len(ground_truth),
        },
        "converted_counts": {
            "frames": len(pairs),
            "imu_samples": len(output_imu),
            "ground_truth_poses": len(pairs),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _selected_sequences(sequences: Optional[Sequence[str]]) -> List[str]:
    if sequences is None:
        return list(ALL_SEQS)
    unknown = sorted(set(sequences) - set(ALL_SEQS))
    if unknown:
        raise ConversionError("unknown sequence(s): " + ", ".join(unknown))
    selected = set(sequences)
    ordered = [sequence for sequence in ALL_SEQS if sequence in selected]
    if not ordered:
        raise ConversionError("no sequences selected")
    return ordered


def required_archives(sequences: Optional[Sequence[str]] = None) -> List[str]:
    """Return the source bundle archives needed to convert the selected sequences."""
    selected = set(_selected_sequences(sequences))
    return [
        archive_name
        for archive_name, archive_sequences in _OUTER_ZIPS
        if selected.intersection(archive_sequences)
    ]


def _dataset_metadata(
    archives: List[Dict[str, str]],
    config_names: List[str],
    sequence_metadata: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "converter_version": _CONVERTER_VERSION,
        "source": {
            "name": "EuRoC MAV Dataset",
            "doi": _SOURCE_DOI,
            "url": _SOURCE_URL,
            "archives": archives,
        },
        "media_layout": {
            "storage": "loose_files",
            "cam0": "{sequence}/00/l.{frame:06d}.png",
            "cam1": "{sequence}/01/r.{frame:06d}.png",
            "frame_metadata": "{sequence}/frame_metadata.jsonl",
        },
        "generated_configs": config_names,
        "sequences": sequence_metadata,
    }


def convert(
    raw_dir: Path, output_dir: Path, sequences: Optional[Sequence[str]] = None
) -> Dict[str, object]:
    """Convert selected EuRoC sequences and return the deterministic metadata document."""
    selected = _selected_sequences(sequences)
    selected_set = set(selected)
    required_archives = [
        (archive_name, archive_sequences)
        for archive_name, archive_sequences in _OUTER_ZIPS
        if selected_set.intersection(archive_sequences)
    ]
    for archive_name, archive_sequences in required_archives:
        archive_path = raw_dir / archive_name
        if not archive_path.is_file():
            needed = [sequence for sequence in archive_sequences if sequence in selected_set]
            raise ConversionError(
                f"missing archive {archive_path} (required for {', '.join(needed)})"
            )

    archive_metadata = [
        {"name": archive_name, "sha256": _sha256(raw_dir / archive_name)}
        for archive_name, _ in required_archives
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_metadata = []

    # Keep the nested sequence archive on the same conversion volume. A EuRoC
    # sequence can be too large for hosts where /tmp is backed by a small tmpfs.
    with tempfile.TemporaryDirectory(
        prefix="euroc_convert_",
        dir=str(output_dir.parent),
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)
        for archive_name, archive_sequences in required_archives:
            archive_path = raw_dir / archive_name
            try:
                with zipfile.ZipFile(archive_path) as outer_archive:
                    _validate_zip_members(outer_archive, f"{archive_name} outer archive")
                    group = archive_name[:-4]
                    for sequence in archive_sequences:
                        if sequence not in selected_set:
                            continue
                        nested_member = f"{group}/{sequence}/{sequence}.zip"
                        _validate_safe_archive_path(nested_member, "nested archive member path")
                        nested_info = _required_member(outer_archive, nested_member, sequence)
                        nested_path = temporary_path / f"{sequence}.zip"
                        try:
                            with outer_archive.open(nested_info) as source, nested_path.open("wb") as destination:
                                shutil.copyfileobj(source, destination)
                        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                            raise ConversionError(
                                f"{sequence}: failed to read nested archive {nested_member}: {exc}"
                            ) from exc
                        sequence_metadata.append(
                            _convert_sequence(sequence, nested_path, output_dir, archive_name)
                        )
                        nested_path.unlink()
            except zipfile.BadZipFile as exc:
                raise ConversionError(f"malformed outer ZIP archive {archive_path}: {exc}") from exc

    config_names = _write_configs(output_dir, selected)
    metadata = _dataset_metadata(archive_metadata, config_names, sequence_metadata)
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_dir",
        nargs="?",
        type=Path,
        default=repo_root / "datasets" / "euroc" / "raw",
        help="directory containing the three official EuRoC bundle ZIPs",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=repo_root / "datasets" / "converted" / "euroc",
        help="directory for converted EuRoC data",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        metavar="SEQUENCE",
        help="explicit sequence subset (default: all 11 official sequences)",
    )
    args = parser.parse_args(argv)
    try:
        convert(args.raw_dir, args.output_dir, args.sequences)
    except (ConversionError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
