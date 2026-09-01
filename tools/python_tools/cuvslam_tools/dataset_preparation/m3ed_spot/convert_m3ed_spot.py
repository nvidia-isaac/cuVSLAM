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

"""Convert the 19 M3ED SPOT stereo sequences to the cuVSLAM reporter layout.

Produces, per sequence::

    <sequence>/00/000000.png          OVC left, mono8
    <sequence>/01/000000.png          OVC right, mono8
    <sequence>/frame_metadata.jsonl   per-frame timestamps for both cameras
    <sequence>/gt.txt                 3x4 pose per frame, relative to frame 0
    <sequence>/stereo.edex            rig, per-camera intrinsics, and baseline

plus a ``dataset_metadata.json`` and the reporter config at the dataset root.

The images come from ``/ovc/{left,right}/data`` in the published ``_data.h5``,
which is already time synchronized: both cameras share ``/ovc/ts``, so there is
no association step. Ground truth comes from the small ``_pose_gt.h5``, whose
FasterLIO poses are expressed in the left *event* camera frame, so the OVC left
extrinsic is applied before the poses are made relative.

Measured against the retired output on ``skatepark_2``, the images this produces
are byte-identical to it, and the trajectory matches to 490 um rms per frame.
Four things are deliberately different:

- Calibration is read per sequence from the source. The retired pipeline wrote
  one hardcoded intrinsic and baseline into all 16 sequences, which cannot be
  right for a dataset that ships a calibration session per recording group.
- ``stereo.edex`` references ``frame_metadata.jsonl``, so replay uses the real
  25 Hz frame times. The retired EDEX omitted that reference and declared
  ``fps: 30``, so replay synthesized timestamps 33.3 ms apart for frames that
  are actually 40 ms apart.
- The ground-truth pose of each frame is the pose at that frame's own timestamp.
  In the retired output the ground truth belongs to the frame five positions
  after the image it is paired with, about 200 mm of offset at this cadence:
  its images align with ours at an offset of six frames while its poses align at
  one.
- Images are written as mono8, which is what the sensor produces. The retired
  output replicated each pixel across three RGB channels, which is why it is
  2.3 times larger for the same content.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from cuvslam_tools.dataset_preparation import rgbd
from cuvslam_tools.dataset_preparation.rgbd import RgbdConversionError as ConversionError

_SCHEMA_VERSION = 1
_CONVERTER_VERSION = 1

SOURCE_NAME = "M3ED: Multi-Robot, Multi-Sensor, Multi-Environment Event Dataset"
SOURCE_URL = "https://m3ed.io"
SOURCE_CITATION = (
    "K. Chaney, F. Cladera, Z. Wang, A. Bisulco, M. A. Hsieh, C. Korpela, V. Kumar, "
    "C. J. Taylor and K. Daniilidis, M3ED: Multi-Robot, Multi-Sensor, "
    "Multi-Environment Event Dataset, CVPRW 2023"
)

# The 19 SPOT sequences the retired pipeline converted, paired with their
# published names. The retired output used these short names as directory names
# and they stay the sequence folders here, so its per-sequence results remain
# comparable.
#
# Its reporter config enabled only 16 of them: hard, srt_green_loop and
# stairwell were converted but never evaluated. They are converted here too, so
# the corpus matches what the retired pipeline held, and enabling them is a
# separate decision made in the registry.
SEQUENCES: Tuple[Tuple[str, str], ...] = (
    ("art_plaza_loop", "spot_outdoor_day_art_plaza_loop"),
    ("building_loop", "spot_indoor_building_loop"),
    ("easy_1", "spot_forest_easy_1"),
    ("easy_2", "spot_forest_easy_2"),
    ("hard", "spot_forest_hard"),
    ("obstacles", "spot_indoor_obstacles"),
    ("penno_plaza_lights", "spot_outdoor_night_penno_plaza_lights"),
    ("penno_short_loop", "spot_outdoor_day_penno_short_loop"),
    ("penno_short_loop_night", "spot_outdoor_night_penno_short_loop"),
    ("road_1", "spot_forest_road_1"),
    ("road_3", "spot_forest_road_3"),
    ("rocky_steps", "spot_outdoor_day_rocky_steps"),
    ("skatepark_1", "spot_outdoor_day_skatepark_1"),
    ("skatepark_2", "spot_outdoor_day_skatepark_2"),
    ("srt_green_loop", "spot_outdoor_day_srt_green_loop"),
    ("srt_under_bridge_1", "spot_outdoor_day_srt_under_bridge_1"),
    ("srt_under_bridge_2", "spot_outdoor_day_srt_under_bridge_2"),
    ("stairs", "spot_indoor_stairs"),
    ("stairwell", "spot_indoor_stairwell"),
)

# The three the retired reporter config left out.
UNEVALUATED_SEQS: Tuple[str, ...] = ("hard", "srt_green_loop", "stairwell")

ALL_SEQS: Tuple[str, ...] = tuple(short for short, _ in SEQUENCES)
_SOURCE_BY_SHORT: Dict[str, str] = dict(SEQUENCES)

DATASET_ID = "m3ed_spot"
LEFT_DIR = "00"
RIGHT_DIR = "01"

# Outdoor trajectories, so the retired suite measured drift over much longer
# segments than the indoor RGB-D corpora.
SEGMENT_LENGTHS: Tuple[float, ...] = (5, 10, 15, 20, 25, 50, 100)

# Kalibr writes radtan coefficients in OpenCV order, and cuVSLAM's Polynomial
# model takes the first eight OpenCV coefficients, so the four published values
# are k1, k2, p1, p2 and the remaining rational terms are zero.
_POLYNOMIAL_PARAMETER_COUNT = 8

# PNG level 1. The images are 1280x800 mono8 and the suite decodes every one of
# them on every run, so this trades archive size for replay throughput the same
# way the retired output did.
_PNG_COMPRESS_LEVEL = 1

# Frames are stored one per HDF5 chunk, and reading them one at a time over the
# network wastes most of each range request. Pulling a run of frames per camera
# turns the read pattern into two long sequential scans.
_FRAME_BATCH = 16

_OVC_TIMESTAMP_UNIT_NS = 1_000  # /ovc/ts and pose ts are microseconds


@dataclass(frozen=True)
class CameraCalibration:
    """One OVC camera: intrinsics, distortion, and its event-camera extrinsic."""

    focal: Tuple[float, float]
    principal: Tuple[float, float]
    size: Tuple[int, int]
    distortion: Tuple[float, ...]
    # Maps a point in this camera's frame into the left event camera frame.
    prophesee_left_from_camera: Tuple[Tuple[Tuple[float, ...], ...], Tuple[float, ...]]


def source_name(sequence: str) -> str:
    """Return the published M3ED sequence name for a short sequence name."""
    try:
        return _SOURCE_BY_SHORT[sequence]
    except KeyError:
        raise ConversionError(
            f"unknown sequence '{sequence}'; known: {', '.join(ALL_SEQS)}"
        ) from None


def _selected_sequences(sequences: Optional[Sequence[str]]) -> List[str]:
    if sequences is None:
        return list(ALL_SEQS)
    selected = list(sequences)
    if not selected:
        raise ConversionError("no sequences selected")
    unknown = [sequence for sequence in selected if sequence not in _SOURCE_BY_SHORT]
    if unknown:
        raise ConversionError(
            f"unknown sequence(s): {', '.join(sorted(unknown))}; known: {', '.join(ALL_SEQS)}"
        )
    duplicated = sorted({sequence for sequence in selected if selected.count(sequence) > 1})
    if duplicated:
        raise ConversionError(f"duplicate sequence(s): {', '.join(duplicated)}")
    # Keep the retired report order regardless of the order requested.
    return [sequence for sequence in ALL_SEQS if sequence in set(selected)]


# A published rotation should be orthonormal to far better than this; the bound
# only has to reject corruption, not measure numerical quality.
_ROTATION_TOLERANCE = 1e-6


def _split_matrix(matrix) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[float, ...]]:
    """Split a 4x4 homogeneous transform into a 3x3 rotation and a translation.

    The rotation block is checked for orthonormality, because a corrupt pose
    would otherwise pass silently into interpolation and produce ground truth
    that looks plausible.
    """
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ConversionError(f"expected a 4x4 transform, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ConversionError("transform contains non-finite values")
    block = array[:3, :3]
    if not np.allclose(block @ block.T, np.eye(3), atol=_ROTATION_TOLERANCE):
        raise ConversionError("transform rotation block is not orthonormal")
    if abs(float(np.linalg.det(block)) - 1.0) > _ROTATION_TOLERANCE:
        raise ConversionError("transform rotation block is not a right-handed rotation")
    rotation = tuple(tuple(float(value) for value in row[:3]) for row in array[:3])
    translation = tuple(float(row[3]) for row in array[:3])
    return rotation, translation


def read_camera_calibration(handle, side: str) -> CameraCalibration:
    """Read one OVC camera's calibration from an open ``_data.h5``."""
    group_name = f"ovc/{side}/calib"
    try:
        group = handle[group_name]
    except KeyError:
        raise ConversionError(f"missing {group_name} in the source file") from None

    model = group["distortion_model"][()]
    model_name = model.decode() if isinstance(model, bytes) else str(model)
    if model_name != "radtan":
        raise ConversionError(
            f"{group_name}: unsupported distortion model '{model_name}'; expected 'radtan'"
        )

    intrinsics = np.asarray(group["intrinsics"][()], dtype=float)
    if intrinsics.shape != (4,):
        raise ConversionError(f"{group_name}: expected 4 intrinsics, got {intrinsics.shape}")
    coefficients = np.asarray(group["distortion_coeffs"][()], dtype=float)
    if coefficients.shape != (4,):
        raise ConversionError(
            f"{group_name}: expected 4 radtan coefficients, got {coefficients.shape}"
        )
    resolution = np.asarray(group["resolution"][()], dtype=int)
    if resolution.shape != (2,):
        raise ConversionError(f"{group_name}: expected a 2-element resolution")
    if not (np.isfinite(intrinsics).all() and np.isfinite(coefficients).all()):
        raise ConversionError(f"{group_name}: non-finite calibration value")

    padding = (0.0,) * (_POLYNOMIAL_PARAMETER_COUNT - coefficients.size)
    return CameraCalibration(
        focal=(float(intrinsics[0]), float(intrinsics[1])),
        principal=(float(intrinsics[2]), float(intrinsics[3])),
        size=(int(resolution[0]), int(resolution[1])),
        distortion=tuple(float(value) for value in coefficients) + padding,
        prophesee_left_from_camera=_split_matrix(group["T_to_prophesee_left"][()]),
    )


def read_frame_timestamps(handle) -> List[int]:
    """Read ``/ovc/ts`` as nanoseconds and check it is usable as a frame clock."""
    try:
        stamps = np.asarray(handle["ovc/ts"][()], dtype=np.int64)
    except KeyError:
        raise ConversionError("missing ovc/ts in the source file") from None
    if stamps.ndim != 1 or stamps.size == 0:
        raise ConversionError("ovc/ts must be a non-empty one-dimensional dataset")
    if np.any(np.diff(stamps) <= 0):
        raise ConversionError("ovc/ts must be strictly increasing")
    return [int(value) * _OVC_TIMESTAMP_UNIT_NS for value in stamps]


def read_trajectory(handle) -> List[rgbd.TrajectoryRow]:
    """Read ``_pose_gt.h5`` into trajectory rows in the left event camera frame.

    ``Cn_T_C0`` follows the M3ED convention of naming the transform that takes a
    point from ``C0`` into ``Cn``, so the pose of the camera in the first camera's
    frame is its inverse.
    """
    for name in ("Cn_T_C0", "ts"):
        if name not in handle:
            raise ConversionError(f"missing /{name} in the pose ground-truth file")
    poses = np.asarray(handle["Cn_T_C0"][()], dtype=float)
    stamps = np.asarray(handle["ts"][()], dtype=np.int64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ConversionError(f"Cn_T_C0 must have shape (n, 4, 4), got {poses.shape}")
    if stamps.shape != (poses.shape[0],):
        raise ConversionError(
            f"ts has {stamps.shape} entries but Cn_T_C0 has {poses.shape[0]} poses"
        )
    if poses.shape[0] < 2:
        raise ConversionError("pose ground truth needs at least two samples to interpolate")
    if np.any(np.diff(stamps) <= 0):
        raise ConversionError("pose ground-truth timestamps must be strictly increasing")

    rows: List[rgbd.TrajectoryRow] = []
    for stamp, matrix in zip(stamps, poses):
        rotation, translation = rgbd.invert_transform(*_split_matrix(matrix))
        rows.append(
            (
                int(stamp) * _OVC_TIMESTAMP_UNIT_NS,
                list(translation),
                rgbd.matrix_to_quaternion(rotation),
            )
        )
    return rows


def left_from_right(left: CameraCalibration, right: CameraCalibration):
    """Return the right camera's pose in the left camera frame."""
    left_from_prophesee = rgbd.invert_transform(*left.prophesee_left_from_camera)
    return rgbd.compose_transforms(
        left_from_prophesee[0],
        left_from_prophesee[1],
        right.prophesee_left_from_camera[0],
        right.prophesee_left_from_camera[1],
    )


def edex_camera_transform(
    rotation: Sequence[Sequence[float]], translation: Sequence[float]
) -> List[List[float]]:
    """Render a camera pose into the EDEX ``transform`` matrix.

    The EDEX matrix is not a homogeneous transform. The reader consumes its
    rotation block as camera-from-rig while taking the translation as the camera
    centre in rig coordinates, so it means ``p_camera = R (p_rig - c)``. Passing
    a homogeneous rig-from-camera pose instead leaves the rotation transposed,
    which on a 120 mm baseline at 1050 px focal biases every disparity: measured
    on skatepark_2, that scores 134% ATE where this convention scores 3.5%.

    ``rotation`` and ``translation`` are the camera's pose in the rig frame.
    """
    transposed = [[rotation[column][row] for column in range(3)] for row in range(3)]
    return [list(transposed[row]) + [float(translation[row])] for row in range(3)]


def _intrinsics_document(camera: CameraCalibration) -> Dict[str, object]:
    return {
        "distortion_model": "polynomial",
        "distortion_params": list(camera.distortion),
        "focal": [camera.focal[0], camera.focal[1]],
        "principal": [camera.principal[0], camera.principal[1]],
        "size": [camera.size[0], camera.size[1]],
    }


def edex_document(
    left: CameraCalibration, right: CameraCalibration, frame_count: int
) -> List[Dict[str, object]]:
    """Build the two-section EDEX document for the OVC stereo pair."""
    if frame_count <= 0:
        raise ConversionError("cannot describe an empty sequence")
    rotation, translation = left_from_right(left, right)
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
                    "intrinsics": _intrinsics_document(left),
                },
                {
                    "transform": edex_camera_transform(rotation, translation),
                    "intrinsics": _intrinsics_document(right),
                },
            ],
        },
        {
            "frame_metadata": rgbd.FRAME_METADATA_FILE,
            "points2d": {},
            "points3d": {},
            "rig_positions": {},
            "sequence": [[f"{LEFT_DIR}/000000.png"], [f"{RIGHT_DIR}/000000.png"]],
        },
    ]


def frame_metadata_lines(timestamps: Sequence[int], names: Sequence[str]) -> List[str]:
    """Render ``frame_metadata.jsonl`` for a synchronized stereo pair."""
    if len(timestamps) != len(names):
        raise ConversionError("frame metadata inputs have mismatched lengths")
    lines = []
    for frame_index, (timestamp, name) in enumerate(zip(timestamps, names)):
        lines.append(
            json.dumps(
                {
                    "frame_id": frame_index,
                    "cams": [
                        {"id": 0, "filename": f"{LEFT_DIR}/{name}", "timestamp": timestamp},
                        {"id": 1, "filename": f"{RIGHT_DIR}/{name}", "timestamp": timestamp},
                    ],
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return lines


def _write_mono_png(frame: np.ndarray, destination: Path, expected_size: Tuple[int, int]) -> None:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ConversionError(f"expected a mono image, got shape {array.shape}")
    height, width = array.shape
    if (width, height) != expected_size:
        raise ConversionError(
            f"image is {width}x{height} but calibration declares {expected_size[0]}x{expected_size[1]}"
        )
    if array.dtype != np.uint8:
        raise ConversionError(f"expected uint8 image data, got {array.dtype}")
    Image.fromarray(array, mode="L").save(
        destination, format="PNG", compress_level=_PNG_COMPRESS_LEVEL
    )


def convert_sequence(
    data_handle,
    pose_handle,
    sequence: str,
    output_dir: Path,
    frame_limit: Optional[int] = None,
) -> Dict[str, object]:
    """Convert one open M3ED sequence and return its metadata.

    ``frame_limit`` truncates the output, which keeps local validation on a
    prefix of a sequence affordable; production conversion leaves it unset.
    """
    left_calibration = read_camera_calibration(data_handle, "left")
    right_calibration = read_camera_calibration(data_handle, "right")
    if left_calibration.size != right_calibration.size:
        raise ConversionError(
            f"{sequence}: stereo cameras disagree on resolution: "
            f"{left_calibration.size} and {right_calibration.size}"
        )

    timestamps = read_frame_timestamps(data_handle)
    trajectory = read_trajectory(pose_handle)

    left_images = data_handle["ovc/left/data"]
    right_images = data_handle["ovc/right/data"]
    if left_images.shape[0] != right_images.shape[0]:
        raise ConversionError(
            f"{sequence}: left has {left_images.shape[0]} frames, right has {right_images.shape[0]}"
        )
    if left_images.shape[0] != len(timestamps):
        raise ConversionError(
            f"{sequence}: ovc/ts has {len(timestamps)} entries but the images have "
            f"{left_images.shape[0]} frames"
        )

    # Every frame needs a pose, and the trajectory starts and ends inside the
    # recording, so frames outside its span are dropped rather than clamped to
    # the first or last pose.
    first_pose_ns, last_pose_ns = rgbd.trajectory_span(trajectory)
    selected = [
        index
        for index, timestamp in enumerate(timestamps)
        if first_pose_ns <= timestamp <= last_pose_ns
    ]
    if not selected:
        raise ConversionError(f"{sequence}: no frame falls inside the ground-truth time span")
    dropped_outside_ground_truth = len(timestamps) - len(selected)
    if frame_limit is not None:
        selected = selected[:frame_limit]

    sequence_dir = output_dir / sequence
    if sequence_dir.is_symlink():
        raise ConversionError(f"{sequence}: refusing to replace symlinked output directory")
    if sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    left_output = sequence_dir / LEFT_DIR
    right_output = sequence_dir / RIGHT_DIR
    left_output.mkdir(parents=True)
    right_output.mkdir()

    names: List[str] = []
    frame_timestamps: List[int] = []
    for batch_start in range(0, len(selected), _FRAME_BATCH):
        batch = selected[batch_start : batch_start + _FRAME_BATCH]
        left_batch = left_images[batch[0] : batch[-1] + 1]
        right_batch = right_images[batch[0] : batch[-1] + 1]
        for offset, source_index in enumerate(batch):
            frame_index = batch_start + offset
            name = f"{frame_index:06d}.png"
            within = source_index - batch[0]
            _write_mono_png(left_batch[within], left_output / name, left_calibration.size)
            _write_mono_png(right_batch[within], right_output / name, right_calibration.size)
            names.append(name)
            frame_timestamps.append(timestamps[source_index])

    (sequence_dir / rgbd.FRAME_METADATA_FILE).write_text(
        "\n".join(frame_metadata_lines(frame_timestamps, names)) + "\n", encoding="utf-8"
    )
    (sequence_dir / rgbd.GROUND_TRUTH_FILE).write_text(
        "\n".join(
            rgbd.relative_ground_truth_lines(
                trajectory,
                frame_timestamps,
                body_from_camera=left_calibration.prophesee_left_from_camera,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (sequence_dir / rgbd.EDEX_FILE).write_text(
        json.dumps(edex_document(left_calibration, right_calibration, len(names)), indent=4) + "\n",
        encoding="utf-8",
    )

    rotation, translation = left_from_right(left_calibration, right_calibration)
    return {
        "sequence": sequence,
        "source_sequence": source_name(sequence),
        "source_counts": {
            "frames": int(left_images.shape[0]),
            "ground_truth_poses": len(trajectory),
        },
        "converted_counts": {"frames": len(names), "ground_truth_poses": len(names)},
        "dropped_outside_ground_truth": dropped_outside_ground_truth,
        "baseline_m": abs(translation[0]),
        "left_intrinsics": {
            "focal": list(left_calibration.focal),
            "principal": list(left_calibration.principal),
        },
        "right_intrinsics": {
            "focal": list(right_calibration.focal),
            "principal": list(right_calibration.principal),
        },
        "frame_limit": frame_limit,
    }


def _config_entries(sequences: Sequence[str], modes: Sequence[bool]) -> List[List[Tuple[str, object]]]:
    entries = []
    for sequence in sequences:
        title = f"M3ED-{sequence.replace('_', '-')}"
        for use_slam in modes:
            entries.append(
                rgbd.reporter_sequence_entry(
                    sequence_folder=sequence,
                    sequence_title=f"{title}-{'SLAM' if use_slam else 'ODOM'}",
                    use_slam=use_slam,
                )
            )
    return entries


def _write_configs(output_dir: Path, sequences: Sequence[str]) -> List[str]:
    """Write the reporter configs, returning their names.

    Three, as for KITTI and EuRoC: odometry only, SLAM only, and both. All three
    start with the dataset ID, which is what sets the KPI prefix, since the
    collector takes everything before the first hyphen and yields ``M3ED_SPOT``.
    """
    configs = {
        f"{DATASET_ID}-vo.cfg": _config_entries(sequences, (False,)),
        f"{DATASET_ID}-slam.cfg": _config_entries(sequences, (True,)),
        f"{DATASET_ID}-vo_slam.cfg": _config_entries(sequences, (False, True)),
    }
    for name, entries in configs.items():
        (output_dir / name).write_text(
            rgbd.format_reporter_config(entries, f"{DATASET_ID}/", SEGMENT_LENGTHS),
            encoding="utf-8",
        )
    return sorted(configs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_metadata(
    config_names: List[str], sequence_metadata: List[Dict[str, object]]
) -> Dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "converter_version": _CONVERTER_VERSION,
        "source": {
            "name": SOURCE_NAME,
            "url": SOURCE_URL,
            "citation": SOURCE_CITATION,
            "product": "processed/<sequence>/<sequence>_{data,pose_gt}.h5",
        },
        "rig": {
            "cameras": ["ovc/left", "ovc/right"],
            "encoding": "mono8 png",
            "distortion_model": "polynomial",
            "calibration": "per sequence, from the source file",
        },
        "ground_truth": {
            "source": "FasterLIO poses from _pose_gt.h5 (Cn_T_C0)",
            "frame": "left event camera",
            "camera_extrinsic_applied": True,
        },
        "reporter_configs": config_names,
        "sequences": sequence_metadata,
    }


def convert(
    output_dir: Path,
    sequences: Optional[Sequence[str]] = None,
    *,
    open_sequence,
    frame_limit: Optional[int] = None,
) -> Dict[str, object]:
    """Convert the selected sequences into ``output_dir``.

    ``open_sequence(published_name, kind)`` yields an open ``h5py.File`` for
    ``kind`` in ``{"data", "pose_gt"}`` together with a provenance dictionary.
    Injecting it keeps the transport, whether an HTTP range reader or a local
    file, out of the conversion logic.
    """
    selected = _selected_sequences(sequences)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_metadata = []
    for sequence in selected:
        published = source_name(sequence)
        print(f"Converting {sequence} from {published} …")
        with open_sequence(published, "data") as (data_handle, data_provenance):
            with open_sequence(published, "pose_gt") as (pose_handle, pose_provenance):
                metadata = convert_sequence(
                    data_handle, pose_handle, sequence, output_dir, frame_limit=frame_limit
                )
        metadata["source_files"] = {"data": data_provenance, "pose_gt": pose_provenance}
        sequence_metadata.append(metadata)

    config_names = _write_configs(output_dir, selected)
    metadata = _dataset_metadata(config_names, sequence_metadata)
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
