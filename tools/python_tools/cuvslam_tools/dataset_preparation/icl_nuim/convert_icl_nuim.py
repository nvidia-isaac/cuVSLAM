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

"""Convert the ICL-NUIM TUM-compatible sequences to the cuVSLAM reporter layout.

Produces, per sequence, the same layout as the TUM RGB-D converter::

    <sequence>/00/000000.png          colour, copied byte for byte
    <sequence>/01/000000.png          depth, copied byte for byte (16-bit, scale 5000)
    <sequence>/frame_metadata.jsonl   one frame per ground-truth pose
    <sequence>/gt.txt                 3x4 pose per frame, relative to frame 0
    <sequence>/stereo.edex            rig, intrinsics, and depth scale

ICL-NUIM is rendered, not recorded, and that changes three things against TUM:

* There are no timestamps anywhere. Frames are numbered, the archive's
  ``associations.txt`` pairs colour to depth by that number, and the pose file's
  first column is the same number. Nothing is matched on time, and timestamps
  are synthesized at the published 30 Hz so the output is reproducible.
* The pose file does not cover every rendered frame. ``traj*`` poses start at
  index 2 while its images start at 1, so the leading frame has no pose and is
  dropped.
* Poses are exact per frame, so nothing is interpolated.

Each archive bundles its own ``associations.txt`` and ``.gt.freiburg`` pose
file, which is byte-identical to the copy published separately on the dataset
page, so the archive is the only download needed.
"""

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from cuvslam_tools.dataset_preparation import rgbd
from cuvslam_tools.dataset_preparation.rgbd import RgbdConversionError as ConversionError

_SCHEMA_VERSION = 1
_CONVERTER_VERSION = 1

SOURCE_NAME = "ICL-NUIM RGB-D Benchmark Dataset"
SOURCE_URL = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html"
SOURCE_LICENSE = "CC BY 3.0"
SOURCE_CITATION = (
    "A. Handa, T. Whelan, J.B. McDonald and A.J. Davison, "
    "A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM, ICRA 2014"
)

DATASET_ID = "icl_nuim"

# Rendered at 30 Hz per the dataset page. Frame index is the only ordering the
# source provides, so timestamps are derived from it rather than invented at
# conversion time; that keeps the output byte-identical across runs.
FRAME_RATE_HZ = 30
NANOSECONDS_PER_FRAME = rgbd.NANOSECONDS_PER_SECOND // FRAME_RATE_HZ


@dataclass(frozen=True)
class SequenceSpec:
    """One ICL-NUIM trajectory: its archive, its pose file, and its report name."""

    # Directory name inside the converted dataset, matching the archive stem.
    name: str
    # Pose file bundled inside the archive. Named after the scene rather than the
    # archive, so it cannot be derived from `name`.
    ground_truth: str
    # Reporter title stem. "traj0_frei_png" is the office scene, so the archive
    # stem alone would not say which room a report row came from.
    title: str


ASSOCIATIONS_FILE = "associations.txt"


ALL_SEQS: Tuple[SequenceSpec, ...] = (
    SequenceSpec("living_room_traj0_frei_png", "livingRoom0.gt.freiburg", "living-room-traj0"),
    SequenceSpec("living_room_traj1_frei_png", "livingRoom1.gt.freiburg", "living-room-traj1"),
    SequenceSpec("living_room_traj2_frei_png", "livingRoom2.gt.freiburg", "living-room-traj2"),
    SequenceSpec("living_room_traj3_frei_png", "livingRoom3.gt.freiburg", "living-room-traj3"),
    SequenceSpec("traj0_frei_png", "traj0.gt.freiburg", "office-traj0"),
    SequenceSpec("traj1_frei_png", "traj1.gt.freiburg", "office-traj1"),
    SequenceSpec("traj2_frei_png", "traj2.gt.freiburg", "office-traj2"),
    SequenceSpec("traj3_frei_png", "traj3.gt.freiburg", "office-traj3"),
)

SEQUENCE_NAMES: Tuple[str, ...] = tuple(spec.name for spec in ALL_SEQS)

# The dataset page publishes K with a negative fy and warns that projections
# break without it. That applies to the POVRay-native rendering convention,
# whose image rows run bottom-up. The TUM-compatible PNGs converted here are
# stored top-down like every other dataset in this repo, so fy is positive.
ICL_CAMERA = rgbd.PinholeCamera(
    focal=(481.20, 480.00),
    principal=(319.50, 239.50),
    size=(640, 480),
    depth_scale_factor=5000.0,
)

SEGMENT_LENGTHS: Tuple[float, ...] = (1, 2, 3, 5, 7.5, 10, 15, 20, 25, 35, 45)

_FRAME_PATH = re.compile(r"^(?:rgb|depth)/\d+\.png$")


def archive_name(sequence: str) -> str:
    """Return the source archive filename for one sequence."""
    return f"{sequence}.tar.gz"


def _spec(name: str) -> SequenceSpec:
    for spec in ALL_SEQS:
        if spec.name == name:
            return spec
    raise ConversionError(
        f"unknown sequence '{name}'; known: {', '.join(SEQUENCE_NAMES)}"
    )


def _selected_sequences(sequences: Optional[Sequence[str]]) -> List[SequenceSpec]:
    if sequences is None:
        return list(ALL_SEQS)
    selected = list(sequences)
    if not selected:
        raise ConversionError("no sequences selected")
    unknown = [name for name in selected if name not in SEQUENCE_NAMES]
    if unknown:
        raise ConversionError(
            f"unknown sequence(s): {', '.join(sorted(unknown))}; known: {', '.join(SEQUENCE_NAMES)}"
        )
    duplicated = sorted({name for name in selected if selected.count(name) > 1})
    if duplicated:
        raise ConversionError(f"duplicate sequence(s): {', '.join(duplicated)}")
    chosen = set(selected)
    # Keep the published trajectory order regardless of the order requested.
    return [spec for spec in ALL_SEQS if spec.name in chosen]


def required_archives(sequences: Optional[Sequence[str]] = None) -> List[str]:
    """Return the source archives needed to convert the selected sequences."""
    return [archive_name(spec.name) for spec in _selected_sequences(sequences)]


def ground_truth_name(sequence: str) -> str:
    """Return the pose filename bundled in one sequence's archive."""
    return _spec(sequence).ground_truth


def _required_text(path: Path, sequence: str) -> str:
    if not path.is_file():
        raise ConversionError(f"{sequence}: missing {path.name} in the extracted archive")
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flip_world_y(
    translation: Sequence[float], quaternion: Sequence[float]
) -> Tuple[List[float], List[float]]:
    """Re-express a pose from ICL-NUIM's y-up world in a y-down camera world.

    ICL-NUIM renders through POVRay, whose world has y pointing up, and its
    published K carries the matching negative fy. The TUM-compatible PNGs are
    stored top-down, so the images are in the usual y-down optical frame while
    the poses are not. Reflecting the world about the XZ plane reconciles them:
    with ``S = diag(1, -1, 1)``, ``R' = S R S`` and ``t' = S t``. In quaternion
    terms that reflection is ``(x, y, z, w) -> (-x, y, -z, w)``, which keeps the
    pose in quaternion form so interpolation still works.

    Skipping this leaves the ground truth mirrored against the imagery. It is not
    a subtle error: RGB-D odometry on office traj0 scores 40.39% ATE and
    23.48 deg/m ARE against the published poses, and 1.51% and 0.30 deg/m against
    the reflected ones.
    """
    x, y, z, w = quaternion
    return [translation[0], -translation[1], translation[2]], [-x, y, -z, w]


def read_indexed_trajectory(text: str, source: str) -> List[rgbd.TrajectoryRow]:
    """Parse a ``.gt.freiburg`` pose file, whose first column is a frame index.

    The layout is TUM's ``<key> tx ty tz qx qy qz qw``, so the shared reader
    handles it; only the meaning of the first column differs. Returning the
    index scaled to the same nanosecond space as the synthesized frame
    timestamps lets the shared ground-truth writer match poses to frames
    exactly, with no interpolation. Poses are also reflected into the camera
    world frame; see :func:`flip_world_y`.
    """
    rows = rgbd.read_tum_trajectory(text, source)
    scaled = []
    for key, translation, quaternion in rows:
        index, remainder = divmod(key, rgbd.NANOSECONDS_PER_SECOND)
        if remainder:
            raise ConversionError(f"{source}: frame index must be an integer, got {key / 1e9}")
        flipped_translation, flipped_quaternion = flip_world_y(translation, quaternion)
        scaled.append((index * NANOSECONDS_PER_FRAME, flipped_translation, flipped_quaternion))
    return scaled


def read_associations(text: str, source: str) -> Dict[int, Tuple[str, str]]:
    """Parse ``associations.txt`` into ``{frame index: (colour path, depth path)}``.

    The file states the dataset's own colour-to-depth correspondence as
    ``<depth key> depth/<n>.png <colour key> rgb/<n>.png``. Both keys are the
    same frame number, and this reads them rather than assuming it.
    """
    associations: Dict[int, Tuple[str, str]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        columns = stripped.split()
        if len(columns) != 4:
            raise ConversionError(
                f"{source} line {line_number}: expected 4 columns, got {len(columns)}"
            )
        try:
            depth_key = int(columns[0])
            color_key = int(columns[2])
        except ValueError as exc:
            raise ConversionError(
                f"{source} line {line_number}: frame keys must be integers"
            ) from exc
        if depth_key != color_key:
            raise ConversionError(
                f"{source} line {line_number}: colour key {color_key} does not match "
                f"depth key {depth_key}"
            )
        if depth_key in associations:
            raise ConversionError(f"{source} line {line_number}: duplicate frame {depth_key}")
        for path in (columns[1], columns[3]):
            if not _FRAME_PATH.match(path):
                raise ConversionError(
                    f"{source} line {line_number}: unexpected media path {path!r}"
                )
        associations[depth_key] = (columns[3], columns[1])
    if not associations:
        raise ConversionError(f"{source}: no associations")
    return associations


def convert_sequence(sequence_root: Path, spec: SequenceSpec, output_dir: Path) -> Dict[str, object]:
    """Convert one extracted ICL-NUIM sequence and return its metadata."""
    trajectory = read_indexed_trajectory(
        _required_text(sequence_root / spec.ground_truth, spec.name),
        f"{spec.name}/{spec.ground_truth}",
    )
    associations = read_associations(
        _required_text(sequence_root / ASSOCIATIONS_FILE, spec.name),
        f"{spec.name}/{ASSOCIATIONS_FILE}",
    )

    # Pose coverage is the limiting stream: the office scenes render from index 1
    # but their poses start at 2, so the first frame has nothing to score against.
    pairs: List[rgbd.FramePair] = []
    unposed = []
    for timestamp, _, _ in trajectory:
        index = timestamp // NANOSECONDS_PER_FRAME
        media = associations.get(index)
        if media is None:
            unposed.append(index)
            continue
        color_path, depth_path = media
        pairs.append((timestamp, color_path, timestamp, depth_path))
    if unposed:
        raise ConversionError(
            f"{spec.name}: {ASSOCIATIONS_FILE} has no entry for {len(unposed)} posed "
            f"frame(s), first {unposed[0]}"
        )
    if not pairs:
        raise ConversionError(f"{spec.name}: no frame has a pose")
    missing = [
        path for _, color, _, depth in pairs for path in (color, depth)
        if not (sequence_root / path).is_file()
    ]
    if missing:
        raise ConversionError(f"{spec.name}: missing {len(missing)} media file(s), first {missing[0]}")

    sequence_dir = output_dir / spec.name
    if sequence_dir.is_symlink():
        raise ConversionError(f"{spec.name}: refusing to replace symlinked output directory")
    if sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    color_output = sequence_dir / rgbd.COLOR_DIR
    depth_output = sequence_dir / rgbd.DEPTH_DIR
    color_output.mkdir(parents=True)
    depth_output.mkdir()

    color_names = []
    depth_names = []
    for frame_index, (_, color_source, _, depth_source) in enumerate(pairs):
        name = f"{frame_index:06d}.png"
        shutil.copyfile(sequence_root / color_source, color_output / name)
        shutil.copyfile(sequence_root / depth_source, depth_output / name)
        color_names.append(name)
        depth_names.append(name)

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
            rgbd.edex_document(ICL_CAMERA, len(pairs), color_names[0], depth_names[0]),
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    first_index = pairs[0][0] // NANOSECONDS_PER_FRAME
    return {
        "sequence": spec.name,
        "title": spec.title,
        "source_archive": archive_name(spec.name),
        "source_ground_truth": spec.ground_truth,
        "source_counts": {
            "associated_frames": len(associations),
            "ground_truth_poses": len(trajectory),
        },
        "converted_counts": {
            "frames": len(pairs),
            "ground_truth_poses": len(pairs),
        },
        "first_source_frame_index": first_index,
        "frame_rate_hz": FRAME_RATE_HZ,
    }


def _config_entries(specs: Sequence[SequenceSpec]) -> List[List[Tuple[str, object]]]:
    entries = []
    for spec in specs:
        for use_slam in (False, True):
            entries.append(
                rgbd.reporter_sequence_entry(
                    sequence_folder=spec.name,
                    sequence_title=f"{spec.title}-{'SLAM' if use_slam else 'ODOM'}",
                    use_slam=use_slam,
                )
            )
    return entries


def _write_configs(output_dir: Path, specs: Sequence[SequenceSpec]) -> List[str]:
    """Write the reporter config, returning its name.

    One config for both suites: ICL-NUIM runs whole in smoke and full, so there
    is no subset config and its KPI row stays comparable between the two. The
    name sets the KPI prefix, which the collector takes from everything before
    the first hyphen, so ``icl_nuim-`` yields ``ICL_NUIM`` where the retired
    ``icl-nuim.cfg`` yielded just ``ICL``.
    """
    configs = {f"{DATASET_ID}-rgbd_slam.cfg": _config_entries(specs)}
    for name, entries in configs.items():
        (output_dir / name).write_text(
            rgbd.format_reporter_config(entries, f"{DATASET_ID}/", SEGMENT_LENGTHS),
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
            "license": SOURCE_LICENSE,
            "citation": SOURCE_CITATION,
            "archives": archives,
        },
        "camera": {
            "distortion_model": "pinhole",
            "focal": list(ICL_CAMERA.focal),
            "principal": list(ICL_CAMERA.principal),
            "size": list(ICL_CAMERA.size),
            "depth_scale_factor": ICL_CAMERA.depth_scale_factor,
            "depth_encoding": "uint16 png",
            "focal_y_sign": "positive for the TUM-compatible PNGs; the POVRay-native format uses -480",
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
    and returns the directory holding its ``rgb/`` and ``depth/`` folders. It is
    injected so archive safety stays in ``prepare.py`` and tests can convert from
    a plain directory.
    """
    selected = _selected_sequences(sequences)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    missing = [
        archive_name(spec.name)
        for spec in selected
        if not (raw_dir / archive_name(spec.name)).is_file()
    ]
    if missing:
        raise ConversionError(f"missing archive(s) in {raw_dir}: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_metadata = []
    sequence_metadata = []
    for spec in selected:
        archive = raw_dir / archive_name(spec.name)
        archive_metadata.append({"name": archive.name, "sha256": _sha256(archive)})
        print(f"Converting {spec.name} …")
        # Extract beside the output so a small /tmp cannot fail the conversion,
        # and delete each extraction before starting the next sequence.
        staging = output_dir.parent / f".icl_extract_{spec.name}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            sequence_root = extract_sequence(archive, staging)
            sequence_metadata.append(convert_sequence(sequence_root, spec, output_dir))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    config_names = _write_configs(output_dir, selected)
    metadata = _dataset_metadata(archive_metadata, config_names, sequence_metadata)
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
