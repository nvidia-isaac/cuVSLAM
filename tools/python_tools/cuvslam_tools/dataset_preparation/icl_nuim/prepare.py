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

"""Download the ICL-NUIM TUM-compatible sequences and convert them to cuVSLAM data."""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from cuvslam_tools.dataset_preparation.common import (
    PreparationError,
    add_common_arguments,
    dataset_file,
    extract_tar_archive,
    require_nonempty_files,
    resolve_output_dir,
    resolve_raw_dir,
    run_download_script,
    run_preparation,
)

from . import convert_icl_nuim

DATASET_NAME = convert_icl_nuim.DATASET_ID
DOWNLOAD_SCRIPT = "download_icl_nuim.sh"

DATASET_ARTIFACTS = (
    "dataset_metadata.json",
    "icl_nuim-rgbd_slam.cfg",
)
SEQUENCE_ARTIFACTS = (
    "stereo.edex",
    "frame_metadata.jsonl",
    "gt.txt",
    "00/000000.png",
    "01/000000.png",
)


def extract_sequence(archive: Path, destination: Path) -> Path:
    """Extract one sequence archive and return the directory holding rgb/ and depth/.

    ICL-NUIM archives hold ``rgb/`` and ``depth/`` at the archive root with no
    enclosing directory, unlike TUM. Locate the pair rather than assume either
    layout, so a repackaged archive fails here instead of midway through
    conversion.
    """
    extract_tar_archive(archive, destination)
    candidates = [destination] + [entry for entry in destination.iterdir() if entry.is_dir()]
    roots = [
        candidate
        for candidate in candidates
        if (candidate / "rgb").is_dir() and (candidate / "depth").is_dir()
    ]
    if len(roots) != 1:
        found = ", ".join(sorted(entry.name for entry in destination.iterdir())) or "none"
        raise PreparationError(
            f"{archive.name}: expected one directory holding both rgb/ and depth/, found: {found}"
        )
    return roots[0]


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sequences: Optional[Sequence[str]] = None,
    force_download: bool = False,
    download_only: bool = False,
) -> Path:
    """Download the required ICL-NUIM sources, convert them, and return the prepared root.

    Only the archives and pose files for the selected sequences are downloaded.
    ``sequences`` defaults to all eight published trajectories. Returns the raw
    directory when ``download_only`` is set, otherwise the converted root.
    """
    raw_dir = resolve_raw_dir(raw_dir, DATASET_NAME)
    output_dir = resolve_output_dir(output_dir)
    # Only an omitted selection means "all eight"; an explicitly empty one is an
    # error the converter reports, not a request to convert everything.
    selected = list(sequences) if sequences is not None else None

    print(f"Raw dir    : {raw_dir}")
    print(f"Output dir : {output_dir}")
    print()

    try:
        wanted = convert_icl_nuim.required_archives(selected)
    except convert_icl_nuim.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    download_arguments = [str(raw_dir)]
    if force_download:
        download_arguments.append("--force")
    if selected is not None:
        for archive in wanted:
            download_arguments.extend(["--archive", archive])
    run_download_script(dataset_file(__file__, DOWNLOAD_SCRIPT), download_arguments)

    if download_only:
        return raw_dir

    dataset_dir = output_dir / DATASET_NAME
    print()
    print(f"Converting ICL-NUIM data to {dataset_dir} …")
    try:
        convert_icl_nuim.convert(
            raw_dir, dataset_dir, selected, extract_sequence=extract_sequence
        )
    except convert_icl_nuim.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    require_nonempty_files(dataset_dir, DATASET_ARTIFACTS, "converter")
    for sequence in selected or convert_icl_nuim.SEQUENCE_NAMES:
        require_nonempty_files(dataset_dir / sequence, SEQUENCE_ARTIFACTS, "converter")

    print()
    print(f"done — portable ICL-NUIM dataset ready at {dataset_dir}")
    return dataset_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the ICL-NUIM dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_icl_nuim",
        description=(
            "Download and convert the eight ICL-NUIM living-room and office "
            "trajectories to portable cuVSLAM EDEX data."
        ),
    )
    add_common_arguments(parser, DATASET_NAME, label="ICL-NUIM")
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=convert_icl_nuim.SEQUENCE_NAMES,
        metavar="SEQUENCE",
        help=(
            "Convert an explicit sequence subset, such as traj2_frei_png. "
            "The default is all eight trajectories. A subset run rewrites "
            "dataset_metadata.json and the reporter config to describe only the "
            "selected sequences, but leaves any previously converted sequence "
            "directories in place, so an output directory reused this way holds "
            "sequences the config no longer lists."
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
