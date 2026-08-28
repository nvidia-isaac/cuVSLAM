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

"""Shared primitives for converting RGB-D datasets to the cuVSLAM reporter layout.

TUM RGB-D and ICL-NUIM both ship independently timestamped colour and depth
streams plus a sparse ground-truth trajectory, and both need the same four
steps: associate colour to depth, interpolate ground truth onto the associated
frame times, emit ``frame_metadata.jsonl`` and ``gt.txt``, and emit a
``stereo.edex`` rig description.

Timestamps are parsed with :class:`decimal.Decimal` and scaled to integer
nanoseconds, so a text timestamp maps to exactly one integer regardless of
binary floating-point rounding.

The SE(3) and quaternion helpers duplicate private equivalents in
``euroc/convert_euroc.py``. Folding EuRoC onto these is a separate cleanup;
changing the EuRoC converter here would put an unrelated dataset's output at
risk.
"""

import bisect
import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Sequence, Tuple

NANOSECONDS_PER_SECOND = 1_000_000_000

# Reporter reads "dataset_folder" relative to CUVSLAM_DATASETS and every other
# path relative to the sequence directory.
EDEX_FILE = "stereo.edex"
GROUND_TRUTH_FILE = "gt.txt"
FRAME_METADATA_FILE = "frame_metadata.jsonl"

# Colour is camera 0. Depth shares that camera's id because it is registered to
# the colour image; the folder names are labels only.
COLOR_DIR = "00"
DEPTH_DIR = "01"


class RgbdConversionError(RuntimeError):
    """Raised when a source RGB-D sequence cannot be converted."""


# (timestamp_ns, relative source path)
IndexEntry = Tuple[int, str]
# (timestamp_ns, translation xyz, quaternion xyzw)
TrajectoryRow = Tuple[int, List[float], List[float]]
# (color_timestamp_ns, color_path, depth_timestamp_ns, depth_path)
FramePair = Tuple[int, str, int, str]


@dataclass(frozen=True)
class PinholeCamera:
    """Pinhole intrinsics for a colour camera with registered depth."""

    focal: Tuple[float, float]
    principal: Tuple[float, float]
    size: Tuple[int, int]
    # Denominator converting raw depth samples to metres. Native TUM 16-bit PNG
    # depth uses 5000; float32 depth already in metres uses 1.
    depth_scale_factor: float


def _parse_timestamp_ns(text: str, source: str, line_number: int) -> int:
    try:
        seconds = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise RgbdConversionError(
            f"{source} line {line_number}: invalid timestamp {text!r}"
        ) from exc
    # Decimal accepts "inf" and "nan"; converting either to an integer raises
    # something outside this module's error contract.
    if not seconds.is_finite():
        raise RgbdConversionError(
            f"{source} line {line_number}: non-finite timestamp {text!r}"
        )
    return int(seconds * NANOSECONDS_PER_SECOND)


def _data_lines(text: str) -> List[Tuple[int, List[str]]]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append((line_number, stripped.split()))
    return rows


def read_timestamp_index(text: str, source: str) -> List[IndexEntry]:
    """Parse a ``<timestamp> <relative path>`` index such as TUM's ``rgb.txt``."""
    entries: List[IndexEntry] = []
    previous = None
    for line_number, columns in _data_lines(text):
        if len(columns) != 2:
            raise RgbdConversionError(
                f"{source} line {line_number}: expected 2 columns, got {len(columns)}"
            )
        timestamp = _parse_timestamp_ns(columns[0], source, line_number)
        if previous is not None and timestamp <= previous:
            raise RgbdConversionError(
                f"{source} line {line_number}: timestamps must be strictly increasing"
            )
        entries.append((timestamp, columns[1]))
        previous = timestamp
    if not entries:
        raise RgbdConversionError(f"{source}: no entries")
    return entries


def _normalize_quaternion(quaternion: List[float], source: str, line_number: int) -> List[float]:
    # hypot rather than sum-of-squares: squaring overflows to infinity for large
    # components, which normalizes to an all-zero quaternion and then to an
    # identity rotation, and underflows to zero for tiny ones, which reads as a
    # zero-length quaternion. Both are silent corruption of the ground truth.
    norm = math.hypot(*quaternion)
    if norm <= 0.0 or not math.isfinite(norm):
        raise RgbdConversionError(f"{source} line {line_number}: zero-length quaternion")
    return [component / norm for component in quaternion]


def read_tum_trajectory(text: str, source: str) -> List[TrajectoryRow]:
    """Parse a ``<timestamp> tx ty tz qx qy qz qw`` trajectory file.

    This is the TUM RGB-D ground-truth format: the pose of the colour camera's
    optical frame in the world frame, with the same axis convention cuVSLAM uses
    (x right, y down, z forward), so no axis permutation is applied.
    """
    rows: List[TrajectoryRow] = []
    previous = None
    for line_number, columns in _data_lines(text):
        if len(columns) != 8:
            raise RgbdConversionError(
                f"{source} line {line_number}: expected 8 columns, got {len(columns)}"
            )
        timestamp = _parse_timestamp_ns(columns[0], source, line_number)
        try:
            values = [float(column) for column in columns[1:]]
        except ValueError as exc:
            raise RgbdConversionError(
                f"{source} line {line_number}: invalid numeric value"
            ) from exc
        # float() accepts "inf" and "nan". Left alone they propagate into every
        # interpolated pose as silently wrong ground truth rather than an error.
        if not all(math.isfinite(value) for value in values):
            raise RgbdConversionError(
                f"{source} line {line_number}: non-finite pose value"
            )
        if previous is not None and timestamp <= previous:
            raise RgbdConversionError(
                f"{source} line {line_number}: timestamps must be strictly increasing"
            )
        rows.append(
            (timestamp, values[0:3], _normalize_quaternion(values[3:7], source, line_number))
        )
        previous = timestamp
    if not rows:
        raise RgbdConversionError(f"{source}: no ground-truth poses")
    return rows


def associate(
    color: Sequence[IndexEntry],
    depth: Sequence[IndexEntry],
    max_difference_ns: int,
) -> List[FramePair]:
    """Match colour to depth frames, closest pair first, each frame used once.

    This reproduces the selection in TUM's published ``associate.py``: every
    candidate pair within the tolerance is considered, the smallest time
    difference wins, and both frames are then consumed. Ties break on timestamp,
    so the result does not depend on iteration order.

    Both indices must be sorted by timestamp, which
    :func:`read_timestamp_index` guarantees. Unsorted input does not raise; the
    sliding window below simply stops early and silently drops pairs.
    """
    if max_difference_ns <= 0:
        raise RgbdConversionError("association tolerance must be positive")

    # Both indices are sorted, so the window of depth frames within tolerance of
    # a colour frame only ever moves forward. Enumerating the full cross product
    # instead would be several million comparisons on a single TUM sequence.
    candidates = []
    window_start = 0
    for color_ts, _ in color:
        while window_start < len(depth) and depth[window_start][0] <= color_ts - max_difference_ns:
            window_start += 1
        index = window_start
        while index < len(depth) and depth[index][0] < color_ts + max_difference_ns:
            candidates.append((abs(color_ts - depth[index][0]), color_ts, depth[index][0]))
            index += 1
    candidates.sort()
    color_paths = dict(color)
    depth_paths = dict(depth)
    used_color = set()
    used_depth = set()
    matched: List[FramePair] = []
    for _, color_ts, depth_ts in candidates:
        if color_ts in used_color or depth_ts in used_depth:
            continue
        used_color.add(color_ts)
        used_depth.add(depth_ts)
        matched.append((color_ts, color_paths[color_ts], depth_ts, depth_paths[depth_ts]))
    matched.sort()
    if not matched:
        raise RgbdConversionError("no colour frame is within the association tolerance of a depth frame")
    return matched


def restrict_to_trajectory(pairs: Sequence[FramePair], trajectory: Sequence[TrajectoryRow]) -> List[FramePair]:
    """Drop frames outside the ground-truth time span so every frame has a pose."""
    first = trajectory[0][0]
    last = trajectory[-1][0]
    inside = [pair for pair in pairs if first <= pair[0] <= last]
    if not inside:
        raise RgbdConversionError("no associated frame falls inside the ground-truth time span")
    return inside


def quaternion_to_matrix(quaternion: Sequence[float]) -> List[List[float]]:
    """Convert an xyzw quaternion to a row-major 3x3 rotation matrix."""
    x, y, z, w = quaternion
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _slerp(first: Sequence[float], second: Sequence[float], weight: float) -> List[float]:
    dot = sum(a * b for a, b in zip(first, second))
    target = list(second)
    if dot < 0.0:
        # A quaternion and its negation are the same rotation; take the short arc.
        target = [-component for component in second]
        dot = -dot
    if dot > 0.9995:
        # Nearly parallel: slerp is numerically unstable here and lerp is within
        # float noise of it.
        blended = [a + weight * (b - a) for a, b in zip(first, target)]
    else:
        sine = (1.0 - dot * dot) ** 0.5
        angle = math.atan2(sine, dot)
        first_weight = math.sin((1.0 - weight) * angle) / sine
        second_weight = math.sin(weight * angle) / sine
        blended = [a * first_weight + b * second_weight for a, b in zip(first, target)]
    norm = math.hypot(*blended)
    return [component / norm for component in blended]


def interpolate_pose(
    trajectory: Sequence[TrajectoryRow], timestamps: Sequence[int], timestamp: int
) -> Tuple[List[List[float]], List[float]]:
    """Interpolate the trajectory at one frame time.

    Translation is interpolated linearly and rotation with slerp. ``timestamps``
    is the precomputed list of trajectory times, kept as an argument so callers
    can build it once per sequence.
    """
    if timestamp <= timestamps[0]:
        row = trajectory[0]
        return quaternion_to_matrix(row[2]), list(row[1])
    if timestamp >= timestamps[-1]:
        row = trajectory[-1]
        return quaternion_to_matrix(row[2]), list(row[1])

    index = bisect.bisect_left(timestamps, timestamp)
    if timestamps[index] == timestamp:
        row = trajectory[index]
        return quaternion_to_matrix(row[2]), list(row[1])

    before = trajectory[index - 1]
    after = trajectory[index]
    span = after[0] - before[0]
    weight = (timestamp - before[0]) / span
    translation = [a + weight * (b - a) for a, b in zip(before[1], after[1])]
    rotation = _slerp(before[2], after[2], weight)
    return quaternion_to_matrix(rotation), translation


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


def relative_ground_truth_lines(
    trajectory: Sequence[TrajectoryRow], pairs: Sequence[FramePair]
) -> List[str]:
    """Render ``gt.txt``: one row-major 3x4 pose per frame, relative to frame 0.

    The reporter compares odometry against a trajectory that starts at the
    origin, so the first row is exactly the identity and later rows are
    ``pose(frame 0)^-1 * pose(frame i)``.
    """
    timestamps = [row[0] for row in trajectory]
    first_rotation, first_translation = interpolate_pose(trajectory, timestamps, pairs[0][0])
    inverse_rotation, inverse_translation = invert_transform(first_rotation, first_translation)
    lines = [
        "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 "
        "0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 "
        "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
    ]
    for pair in pairs[1:]:
        rotation, translation = interpolate_pose(trajectory, timestamps, pair[0])
        relative_rotation, relative_translation = compose_transforms(
            inverse_rotation, inverse_translation, rotation, translation
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


def frame_metadata_lines(
    pairs: Sequence[FramePair], color_names: Sequence[str], depth_names: Sequence[str]
) -> List[str]:
    """Render ``frame_metadata.jsonl``, one JSON object per associated frame."""
    if not (len(pairs) == len(color_names) == len(depth_names)):
        raise RgbdConversionError("frame metadata inputs have mismatched lengths")
    lines = []
    for frame_index, (pair, color_name, depth_name) in enumerate(zip(pairs, color_names, depth_names)):
        color_timestamp, _, depth_timestamp, _ = pair
        lines.append(
            json.dumps(
                {
                    "frame_id": frame_index,
                    "cams": [
                        {
                            "id": 0,
                            "filename": f"{COLOR_DIR}/{color_name}",
                            "timestamp": color_timestamp,
                        }
                    ],
                    "depth": [
                        {
                            "id": 0,
                            "filename": f"{DEPTH_DIR}/{depth_name}",
                            "timestamp": depth_timestamp,
                        }
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return lines


def edex_document(
    camera: PinholeCamera,
    frame_count: int,
    first_color_name: str,
    first_depth_name: str,
) -> List[Dict[str, object]]:
    """Build the two-section EDEX document for a single RGB-D camera.

    ``frame_metadata`` drives replay, so ``sequence`` and ``depth_sequence`` are
    templates the reader only consults when frame metadata is absent. They are
    still written with real filenames because other tooling reads them, and the
    depth entry's extension is what tells the reader whether depth samples are
    millimetres in a PNG or metres in a ``.npy``.
    """
    if frame_count <= 0:
        raise RgbdConversionError("cannot describe an empty sequence")
    return [
        {
            "version": "0.9",
            "frame_start": 0,
            "frame_end": frame_count - 1,
            "cameras": [
                {
                    "transform": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                    "intrinsics": {
                        "distortion_model": "pinhole",
                        "distortion_params": [],
                        "focal": [camera.focal[0], camera.focal[1]],
                        "principal": [camera.principal[0], camera.principal[1]],
                        "size": [camera.size[0], camera.size[1]],
                    },
                    "depth_id": 0,
                    "depth_scale_factor": camera.depth_scale_factor,
                }
            ],
        },
        {
            "frame_metadata": FRAME_METADATA_FILE,
            "points2d": {},
            "points3d": {},
            "rig_positions": {},
            "sequence": [[f"{COLOR_DIR}/{first_color_name}"]],
            "depth_sequence": [[f"{DEPTH_DIR}/{first_depth_name}"]],
        },
    ]


def reporter_sequence_entry(
    sequence_folder: str,
    sequence_title: str,
    use_slam: bool,
    gt_file: Optional[str] = GROUND_TRUTH_FILE,
) -> List[Tuple[str, object]]:
    """Build one reporter ``sequence_cfgs`` entry as ordered key/value pairs.

    ``edex_file`` and ``gt_file_path`` are always written even though the
    reporter defaults the former to ``stereo.edex``: omitting ``gt_file_path``
    makes the reporter run the sequence with no ground truth at all, which is how
    the legacy TUM config silently produced ungated numbers.
    """
    fields: List[Tuple[str, object]] = [
        ("enable", True),
        ("sequence_folder", sequence_folder),
        ("edex_file", EDEX_FILE),
        ("precompute_2d_tracks", False),
        ("precompute_key_frames", False),
        ("use_gt_scale", False),
        ("sequence_title", sequence_title),
    ]
    if gt_file:
        fields.append(("gt_file_path", gt_file))
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
    ``dataset_folder`` must equal ``<dataset id>/``; the registry validates that.
    """
    if not entries:
        raise RgbdConversionError("cannot write a reporter config with no sequences")
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
