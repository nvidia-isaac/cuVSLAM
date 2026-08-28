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

"""Convert the TUM RGB-D freiburg3 sequences to the cuVSLAM reporter layout.

Produces, per sequence::

    <sequence>/00/000000.png          colour, copied byte for byte
    <sequence>/01/000000.png          depth, copied byte for byte (16-bit, scale 5000)
    <sequence>/frame_metadata.jsonl   associated colour/depth frames with timestamps
    <sequence>/gt.txt                 3x4 pose per frame, relative to frame 0
    <sequence>/stereo.edex            rig, intrinsics, and depth scale

plus a ``dataset_metadata.json`` and the reporter configs at the dataset root.

Depth is copied unchanged rather than expanded to float32 ``.npy`` as the
retired pipeline did. The reader accepts a 16-bit PNG directly and applies
``depth_scale_factor``, so the extra step only cost precision (it quantised to
whole millimetres) and roughly five times the disk.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from cuvslam_tools.dataset_preparation import rgbd
from cuvslam_tools.dataset_preparation.rgbd import RgbdConversionError as ConversionError

_SCHEMA_VERSION = 1
_CONVERTER_VERSION = 1

SOURCE_NAME = "TUM RGB-D Benchmark"
SOURCE_URL = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"
SOURCE_CITATION = (
    "J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers, "
    "A Benchmark for the Evaluation of RGB-D SLAM Systems, IROS 2012"
)

# The 15 freiburg3 sequences the retired nightly evaluated, in report order.
ALL_SEQS: Tuple[str, ...] = (
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_nostructure_texture_far",
    "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop",
    "rgbd_dataset_freiburg3_teddy",
    "rgbd_dataset_freiburg3_structure_texture_far",
    "rgbd_dataset_freiburg3_walking_halfsphere",
    "rgbd_dataset_freiburg3_structure_notexture_far",
    "rgbd_dataset_freiburg3_sitting_xyz",
    "rgbd_dataset_freiburg3_cabinet",
    "rgbd_dataset_freiburg3_nostructure_texture_near_withloop",
    "rgbd_dataset_freiburg3_nostructure_notexture_far",
    "rgbd_dataset_freiburg3_sitting_xyz_validation",
    "rgbd_dataset_freiburg3_structure_texture_near",
    "rgbd_dataset_freiburg3_long_office_household",
    "rgbd_dataset_freiburg3_large_cabinet_validation",
)

# TUM is a full-suite corpus only: ICL-NUIM covers RGB-D in the smoke suite at
# less than half the staging cost, so no subset config is generated here.

# Published ROS defaults for the freiburg3 camera series. All 15 selected
# sequences come from that series, so one intrinsics set covers them, and the
# series is rectified: its distortion coefficients are all zero.
FREIBURG3_CAMERA = rgbd.PinholeCamera(
    focal=(535.4, 539.2),
    principal=(320.1, 247.6),
    size=(640, 480),
    depth_scale_factor=5000.0,
)

# Colour and depth are two views of the same Kinect capture, so their timestamps
# normally differ by tens of microseconds. 1 ms keeps pairs that came from one
# capture and rejects pairs stitched across neighbouring captures, which would
# break the pixel-to-pixel alignment RGB-D tracking assumes: relaxing this to
# half a frame interval (20 ms) adds 43 pairs on long_office_household whose
# colour and depth are up to 8.2 ms apart. It also matches the retired pipeline,
# whose largest shipped colour-to-depth offset is 990 us.
ASSOCIATION_TOLERANCE_NS = 1_000_000

SEGMENT_LENGTHS: Tuple[float, ...] = (1, 2, 3, 5, 7.5, 10, 15, 20, 25, 35, 45)

DATASET_ID = "tum"
_INDEX_FILES = ("rgb.txt", "depth.txt", "groundtruth.txt")


def archive_name(sequence: str) -> str:
    """Return the source archive filename for one sequence."""
    return f"{sequence}.tgz"


def _selected_sequences(sequences: Optional[Sequence[str]]) -> List[str]:
    if sequences is None:
        return list(ALL_SEQS)
    selected = list(sequences)
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
    # Keep the retired report order regardless of the order requested.
    return [sequence for sequence in ALL_SEQS if sequence in set(selected)]


def required_archives(sequences: Optional[Sequence[str]] = None) -> List[str]:
    """Return the source archives needed to convert the selected sequences."""
    return [archive_name(sequence) for sequence in _selected_sequences(sequences)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_index_text(sequence_root: Path, name: str, sequence: str) -> str:
    path = sequence_root / name
    if not path.is_file():
        raise ConversionError(f"{sequence}: missing {name} in {sequence_root}")
    return path.read_text(encoding="utf-8")


def _copy_media(
    sequence_root: Path,
    sequence: str,
    pairs: Sequence[rgbd.FramePair],
    color_output: Path,
    depth_output: Path,
) -> Tuple[List[str], List[str]]:
    color_names: List[str] = []
    depth_names: List[str] = []
    for frame_index, (_, color_source, _, depth_source) in enumerate(pairs):
        name = f"{frame_index:06d}.png"
        for source_relative, destination in (
            (color_source, color_output / name),
            (depth_source, depth_output / name),
        ):
            source = sequence_root / source_relative
            if not source.is_file():
                raise ConversionError(f"{sequence}: missing image {source_relative}")
            shutil.copyfile(source, destination)
        color_names.append(name)
        depth_names.append(name)
    return color_names, depth_names


def convert_sequence(sequence_root: Path, sequence: str, output_dir: Path) -> Dict[str, object]:
    """Convert one extracted TUM sequence and return its metadata."""
    texts = {name: _read_index_text(sequence_root, name, sequence) for name in _INDEX_FILES}
    color_index = rgbd.read_timestamp_index(texts["rgb.txt"], f"{sequence}/rgb.txt")
    depth_index = rgbd.read_timestamp_index(texts["depth.txt"], f"{sequence}/depth.txt")
    trajectory = rgbd.read_tum_trajectory(texts["groundtruth.txt"], f"{sequence}/groundtruth.txt")

    associated = rgbd.associate(color_index, depth_index, ASSOCIATION_TOLERANCE_NS)
    pairs = rgbd.restrict_to_trajectory(associated, trajectory)

    sequence_dir = output_dir / sequence
    if sequence_dir.is_symlink():
        raise ConversionError(f"{sequence}: refusing to replace symlinked output directory")
    if sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    color_output = sequence_dir / rgbd.COLOR_DIR
    depth_output = sequence_dir / rgbd.DEPTH_DIR
    color_output.mkdir(parents=True)
    depth_output.mkdir()

    color_names, depth_names = _copy_media(sequence_root, sequence, pairs, color_output, depth_output)

    (sequence_dir / rgbd.FRAME_METADATA_FILE).write_text(
        "\n".join(rgbd.frame_metadata_lines(pairs, color_names, depth_names)) + "\n",
        encoding="utf-8",
    )
    (sequence_dir / rgbd.GROUND_TRUTH_FILE).write_text(
        "\n".join(rgbd.relative_ground_truth_lines(trajectory, pairs)) + "\n",
        encoding="utf-8",
    )
    (sequence_dir / rgbd.EDEX_FILE).write_text(
        json.dumps(
            rgbd.edex_document(FREIBURG3_CAMERA, len(pairs), color_names[0], depth_names[0]),
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "sequence": sequence,
        "source_archive": archive_name(sequence),
        "source_counts": {
            "color_frames": len(color_index),
            "depth_frames": len(depth_index),
            "ground_truth_poses": len(trajectory),
        },
        "converted_counts": {
            "frames": len(pairs),
            "ground_truth_poses": len(pairs),
        },
        "association_tolerance_ns": ASSOCIATION_TOLERANCE_NS,
        "dropped_outside_ground_truth": len(associated) - len(pairs),
    }


def _config_entries(sequences: Sequence[str]) -> List[List[Tuple[str, object]]]:
    entries = []
    for sequence in sequences:
        # freiburg3-sitting-xyz-ODOM, matching the retired report titles.
        title = sequence.replace("rgbd_dataset_", "").replace("_", "-")
        for use_slam in (False, True):
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

    One config, covering every converted sequence in both ODOM and SLAM modes.
    Its name determines the KPI prefix: the collector takes everything before the
    first hyphen, so ``tum-rgbd_slam.cfg`` yields ``TUM``.
    """
    configs = {f"{DATASET_ID}-rgbd_slam.cfg": _config_entries(sequences)}
    dataset_folder = f"{DATASET_ID}/"
    for name, entries in configs.items():
        (output_dir / name).write_text(
            rgbd.format_reporter_config(entries, dataset_folder, SEGMENT_LENGTHS),
            encoding="utf-8",
        )
    return sorted(configs)


def _dataset_metadata(
    archives: List[Dict[str, str]],
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
            "archives": archives,
        },
        "camera": {
            "series": "freiburg3",
            "distortion_model": "pinhole",
            "focal": list(FREIBURG3_CAMERA.focal),
            "principal": list(FREIBURG3_CAMERA.principal),
            "size": list(FREIBURG3_CAMERA.size),
            "depth_scale_factor": FREIBURG3_CAMERA.depth_scale_factor,
            "depth_encoding": "uint16 png",
        },
        "reporter_configs": config_names,
        "sequences": sequence_metadata,
    }


def convert(
    raw_dir: Path,
    output_dir: Path,
    sequences: Optional[Sequence[str]] = None,
    *,
    extract_sequence: Callable[[Path, Path], Path],
) -> Dict[str, object]:
    """Convert the selected sequences from ``raw_dir`` into ``output_dir``.

    ``extract_sequence(archive, destination) -> Path`` unpacks one source archive
    and returns the extracted sequence directory. It is injected so archive
    safety stays in ``prepare.py`` and tests can convert from a plain directory.
    """
    selected = _selected_sequences(sequences)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    missing = [
        archive_name(sequence)
        for sequence in selected
        if not (raw_dir / archive_name(sequence)).is_file()
    ]
    if missing:
        raise ConversionError(f"missing archive(s) in {raw_dir}: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_metadata = []
    sequence_metadata = []
    for sequence in selected:
        archive = raw_dir / archive_name(sequence)
        archive_metadata.append({"name": archive.name, "sha256": _sha256(archive)})
        print(f"Converting {sequence} …")
        # Extract beside the output so a small /tmp cannot fail the conversion,
        # and delete each extraction before starting the next sequence.
        staging = output_dir.parent / f".tum_extract_{sequence}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            sequence_root = extract_sequence(archive, staging)
            sequence_metadata.append(convert_sequence(sequence_root, sequence, output_dir))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    config_names = _write_configs(output_dir, selected)
    metadata = _dataset_metadata(archive_metadata, config_names, sequence_metadata)
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
