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

"""Check for the CODa sequence archives and convert them to portable cuVSLAM data.

CODa is not redistributable: the archives are downloaded by hand after accepting
the dataset license, so the "download" step only verifies that they are in place
and prints the registration walkthrough when they are not.
"""

import argparse
from pathlib import Path
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

from . import convert_coda

DATASET_NAME = convert_coda.DATASET_ID
DOWNLOAD_SCRIPT = "download_coda.sh"

DATASET_ARTIFACTS = (
    "dataset_metadata.json",
    f"{DATASET_NAME}-vio_slam.cfg",
)


def _sequence_artifacts(sequence: str) -> List[str]:
    return [
        "stereo.edex",
        f"00/{sequence}.0.00001.png",
        f"01/{sequence}.1.00001.png",
    ]


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sequences: Optional[Sequence[str]] = None,
    force_download: bool = False,
    download_only: bool = False,
) -> Path:
    """Check for the CODa archives, convert them, and return the prepared root.

    ``sequences`` defaults to every sequence whose archive is present in
    ``raw_dir``, because CODa is fetched one sequence at a time by hand. Returns
    the raw directory when ``download_only`` is set, otherwise the converted root.
    """
    raw_dir = resolve_raw_dir(raw_dir, DATASET_NAME)
    output_dir = resolve_output_dir(output_dir)
    # Only an omitted selection means "everything on disk"; an explicitly empty
    # one is an error the converter reports.
    selected = list(sequences) if sequences is not None else None

    print(f"Raw dir    : {raw_dir}")
    print(f"Output dir : {output_dir}")
    print()

    try:
        archives = convert_coda.required_archives(selected)
    except convert_coda.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    download_arguments = [str(raw_dir)]
    if force_download:
        download_arguments.append("--force")
    for archive in archives:
        download_arguments.extend(["--archive", archive])
    run_download_script(dataset_file(__file__, DOWNLOAD_SCRIPT), download_arguments)

    if download_only:
        return raw_dir

    dataset_dir = output_dir / DATASET_NAME
    print()
    print(f"Converting CODa data to {dataset_dir} …")
    try:
        metadata = convert_coda.convert(raw_dir, dataset_dir, selected)
    except convert_coda.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    require_nonempty_files(dataset_dir, DATASET_ARTIFACTS, "converter")
    for entry in metadata["sequences"]:
        sequence = str(entry["sequence"])
        require_nonempty_files(dataset_dir / sequence, _sequence_artifacts(sequence), "converter")

    print()
    print(f"done — portable CODa dataset ready at {dataset_dir}")
    return dataset_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the CODa dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_coda",
        description=(
            "Convert manually downloaded CODa sequence archives to portable cuVSLAM EDEX data."
        ),
    )
    add_common_arguments(
        parser,
        DATASET_NAME,
        label="CODa",
        force_download_help="Accepted for parity with the other datasets; CODa archives are never downloaded.",
        download_only_help="Check that the archives are in place but skip conversion.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=convert_coda.ALL_SEQS,
        metavar="SEQUENCE",
        help="Convert an explicit sequence subset, such as 0 1 2. The default is every archive present.",
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
