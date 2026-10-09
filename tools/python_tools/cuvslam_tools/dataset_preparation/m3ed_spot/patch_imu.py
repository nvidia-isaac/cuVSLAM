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

"""One-time upgrade of a converter-v1 M3ED SPOT dataset to v2 by adding the IMU.

Adds ``IMU.jsonl`` and the EDEX ``imu`` section to every sequence of an
already converted dataset, reading only ``/ovc/imu`` and the camera calibration
from the source rather than the images. The result is byte-identical to what
``convert_m3ed_spot`` v2 writes for the same frames, because both call the same
functions.

The patch refuses a sequence whose cameras no longer match the source
calibration: that tarball is not a v1 conversion of this source, and only a
full provision can bring it up to date.

Re-running it on a patched dataset changes nothing, and ``--report`` records
how many files changed so the caller can skip the upload.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from cuvslam_tools.dataset_preparation import rgbd
from cuvslam_tools.dataset_preparation.common import PreparationError, run_preparation
from cuvslam_tools.dataset_preparation.m3ed_spot import convert_m3ed_spot
from cuvslam_tools.dataset_preparation.m3ed_spot.prepare import _select_source

ConversionError = convert_m3ed_spot.ConversionError


def _write_if_changed(path: Path, text: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def frame_span(sequence_dir: Path) -> Tuple[int, int, int]:
    """Return the frame count and the first and last frame timestamps."""
    path = sequence_dir / rgbd.FRAME_METADATA_FILE
    if not path.is_file():
        raise ConversionError(f"missing {path}")
    stamps = [
        json.loads(line)["cams"][0]["timestamp"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not stamps:
        raise ConversionError(f"{path} lists no frames")
    return len(stamps), stamps[0], stamps[-1]


def patch_sequence(sequence_dir: Path, data_handle) -> Dict[str, object]:
    """Add the IMU to one converted sequence and return what changed."""
    sequence_dir = Path(sequence_dir)
    frames, first_ns, last_ns = frame_span(sequence_dir)

    edex_path = sequence_dir / rgbd.EDEX_FILE
    document = json.loads(edex_path.read_text(encoding="utf-8"))
    header = document[0]
    if header.get("frame_end") != frames - 1:
        raise ConversionError(
            f"{edex_path}: frame_end {header.get('frame_end')} does not match the {frames} "
            f"frames in {rgbd.FRAME_METADATA_FILE}"
        )

    left = convert_m3ed_spot.read_camera_calibration(data_handle, "left")
    right = convert_m3ed_spot.read_camera_calibration(data_handle, "right")
    expected = convert_m3ed_spot.edex_document(left, right, frames)[0]["cameras"]
    if header.get("cameras") != expected:
        raise ConversionError(
            f"{edex_path}: cameras differ from the source calibration; this dataset is not a "
            "conversion of the current source, so re-provision it instead of patching"
        )

    imu = convert_m3ed_spot.read_imu(data_handle)
    lines = convert_m3ed_spot.imu_lines(imu, first_ns, last_ns)
    changed: List[str] = []
    if _write_if_changed(sequence_dir / convert_m3ed_spot.IMU_FILE, "\n".join(lines) + "\n"):
        changed.append(convert_m3ed_spot.IMU_FILE)
    header["imu"] = convert_m3ed_spot.imu_document(left, imu)
    if _write_if_changed(edex_path, convert_m3ed_spot.render_edex(document)):
        changed.append(rgbd.EDEX_FILE)
    return {"changed": changed, "imu": imu, "imu_samples": len(lines)}


def _add_imu_counts(entry: Dict[str, object], result: Dict[str, object]) -> None:
    convert_m3ed_spot.add_imu_metadata(entry, result["imu"], result["imu_samples"])


def patch(
    root: Path, open_sequence, sequences: Optional[Sequence[str]] = None
) -> Dict[str, object]:
    """Patch every sequence of the dataset at ``root`` and return a report."""
    root = Path(root)
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise ConversionError(f"missing {metadata_path}; is {root} an M3ED SPOT dataset root?")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = metadata.get("converter_version")
    if version not in (1, 2):
        raise ConversionError(f"{metadata_path}: converter_version {version!r} is not 1 or 2")

    entries = {entry["sequence"]: entry for entry in metadata.get("sequences", [])}
    selected = list(sequences) if sequences is not None else list(entries)
    unknown = [sequence for sequence in selected if sequence not in entries]
    if unknown:
        raise ConversionError(f"sequences not in {metadata_path}: {', '.join(unknown)}")

    changed: List[str] = []
    for sequence in selected:
        published = convert_m3ed_spot.source_name(sequence)
        print(f"Patching {sequence} from {published} …", flush=True)
        with open_sequence(published, "data") as (handle, _):
            result = patch_sequence(root / sequence, handle)
        print(
            f"  {result['imu_samples']} IMU samples ({result['imu'].dropped_samples} dropouts removed "
            f"from the source), changed: {result['changed'] or 'nothing'}"
        )
        changed.extend(f"{sequence}/{name}" for name in result["changed"])
        _add_imu_counts(entries[sequence], result)

        state = convert_m3ed_spot.read_state(root / sequence)
        if state is not None and isinstance(state.get("metadata"), dict):
            before = json.dumps(state, sort_keys=True)
            _add_imu_counts(state["metadata"], result)
            if json.dumps(state, sort_keys=True) != before:
                convert_m3ed_spot.write_state(
                    root / sequence, state.get("frame_limit"), state["metadata"]
                )
                changed.append(f"{sequence}/.conversion_state.json")

    metadata["converter_version"] = convert_m3ed_spot._CONVERTER_VERSION
    metadata["imu"] = convert_m3ed_spot.imu_metadata()
    if _write_if_changed(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n"):
        changed.append("dataset_metadata.json")
    return {"root": str(root), "sequences": selected, "changed_files": changed}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="patch_m3ed_spot_imu",
        description="Add the IMU to an extracted converter-v1 M3ED SPOT dataset, in place.",
    )
    parser.add_argument("--root", type=Path, required=True, help="The dataset root, holding dataset_metadata.json.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Read local _data.h5 files from here instead of the public bucket.",
    )
    parser.add_argument("--sequences", nargs="+", metavar="SEQUENCE", help="Patch only these sequences.")
    parser.add_argument("--report", type=Path, default=None, help="Write the JSON report here.")
    args = parser.parse_args(argv)

    def run() -> Path:
        opener, source = _select_source(args.raw_dir)
        print(f"Source : {source}")
        print(f"Root   : {args.root}")
        try:
            report = patch(args.root, opener, args.sequences)
        except ConversionError as exc:
            raise PreparationError(str(exc)) from exc
        print(f"{len(report['changed_files'])} file(s) changed")
        if args.report is not None:
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return args.root

    return run_preparation(run)


if __name__ == "__main__":
    raise SystemExit(main())
