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

"""Download the TUM RGB-D freiburg3 sequences and convert them to cuVSLAM data."""

import argparse
import tarfile
from pathlib import Path, PurePosixPath
from typing import List, Optional, Sequence

from cuvslam_tools.dataset_preparation.common import (
    PreparationError,
    add_common_arguments,
    dataset_file,
    require_nonempty_files,
    resolve_output_dir,
    resolve_raw_dir,
    run_download_script,
    run_preparation,
)

from . import convert_tum

DATASET_NAME = convert_tum.DATASET_ID
DOWNLOAD_SCRIPT = "download_tum.sh"

DATASET_ARTIFACTS = (
    "dataset_metadata.json",
    "tum-rgbd_slam.cfg",
)
SEQUENCE_ARTIFACTS = (
    "stereo.edex",
    "frame_metadata.jsonl",
    "gt.txt",
    "00/000000.png",
    "01/000000.png",
)


def _check_member_path(member_path: str, description: str) -> None:
    """Reject an archive path that is absolute or escapes the extraction directory."""
    path = PurePosixPath(member_path)
    if path.is_absolute() or ".." in path.parts:
        raise PreparationError(f"unsafe {description} in archive: {member_path}")


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract a tar archive, rejecting any member that points outside destination."""
    # tarfile's own 'data' filter only exists on newer interpreters, so validate the
    # member paths here as well.
    extract_options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                _check_member_path(member.name, "member path")
                if member.islnk() or member.issym():
                    _check_member_path(member.linkname, "link target")
            tar.extractall(destination, **extract_options)
    except tarfile.TarError as exc:
        raise PreparationError(f"failed to extract {archive}: {exc}") from exc


def extract_sequence(archive: Path, destination: Path) -> Path:
    """Extract one sequence archive and return the directory holding its files."""
    extract_archive(archive, destination)
    # Each TUM archive holds exactly one top-level directory named after the
    # sequence, but locate it rather than assume, so a renamed archive fails here
    # instead of midway through conversion.
    candidates = [entry for entry in destination.iterdir() if entry.is_dir()]
    if len(candidates) != 1:
        names = ", ".join(sorted(entry.name for entry in candidates)) or "none"
        raise PreparationError(
            f"{archive.name}: expected one top-level directory in the archive, found: {names}"
        )
    return candidates[0]


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sequences: Optional[Sequence[str]] = None,
    force_download: bool = False,
    download_only: bool = False,
) -> Path:
    """Download the required TUM archives, convert them, and return the prepared root.

    Only the archives holding the selected sequences are downloaded.
    ``sequences`` defaults to all 15 evaluated freiburg3 sequences. Returns the
    raw directory when ``download_only`` is set, otherwise the converted root.
    """
    raw_dir = resolve_raw_dir(raw_dir, DATASET_NAME)
    output_dir = resolve_output_dir(output_dir)
    # Only an omitted selection means "all 15"; an explicitly empty one is an
    # error the converter reports, not a request to convert everything.
    selected = list(sequences) if sequences is not None else None

    print(f"Raw dir    : {raw_dir}")
    print(f"Output dir : {output_dir}")
    print()

    try:
        archives = convert_tum.required_archives(selected)
    except convert_tum.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    download_arguments = [str(raw_dir)]
    if force_download:
        download_arguments.append("--force")
    if selected is not None:
        for archive in archives:
            download_arguments.extend(["--archive", archive])
    run_download_script(dataset_file(__file__, DOWNLOAD_SCRIPT), download_arguments)

    if download_only:
        return raw_dir

    dataset_dir = output_dir / DATASET_NAME
    print()
    print(f"Converting TUM RGB-D data to {dataset_dir} …")
    try:
        convert_tum.convert(
            raw_dir, dataset_dir, selected, extract_sequence=extract_sequence
        )
    except convert_tum.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    require_nonempty_files(dataset_dir, DATASET_ARTIFACTS, "converter")
    for sequence in selected or convert_tum.ALL_SEQS:
        require_nonempty_files(dataset_dir / sequence, SEQUENCE_ARTIFACTS, "converter")

    print()
    print(f"done — portable TUM RGB-D dataset ready at {dataset_dir}")
    return dataset_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the TUM RGB-D dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_tum",
        description=(
            "Download and convert the 15 evaluated TUM RGB-D freiburg3 sequences "
            "to portable cuVSLAM EDEX data."
        ),
    )
    add_common_arguments(parser, DATASET_NAME, label="TUM")
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=convert_tum.ALL_SEQS,
        metavar="SEQUENCE",
        help=(
            "Convert an explicit sequence subset, such as "
            "rgbd_dataset_freiburg3_long_office_household. The default is all 15 sequences."
        ),
    )
    args = parser.parse_args(argv)

    return run_preparation(
        lambda: prepare(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            sequences=args.sequences,
            force_download=args.force_download,
            download_only=args.download_only,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
