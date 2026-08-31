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

"""Download the official EuRoC MAV bundles and convert them to portable cuVSLAM data."""

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

from . import convert_euroc

DATASET_NAME = "euroc"
DOWNLOAD_SCRIPT = "download_euroc.sh"

DATASET_ARTIFACTS = (
    "dataset_metadata.json",
    "euroc-vio.cfg",
    "euroc-slam.cfg",
    "euroc-vio_slam.cfg",
)
SEQUENCE_ARTIFACTS = (
    "stereo.edex",
    "frame_metadata.jsonl",
    "IMU.jsonl",
    "gt.txt",
    "00/l.000000.png",
    "01/r.000000.png",
)


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sequences: Optional[Sequence[str]] = None,
    force_download: bool = False,
    download_only: bool = False,
) -> Path:
    """Download the required EuRoC bundles, convert them, and return the prepared root.

    Only the bundles that contain the selected sequences are downloaded. ``sequences``
    defaults to all 11 official sequences. Returns the raw directory when
    ``download_only`` is set, otherwise the converted EuRoC root.
    """
    raw_dir = resolve_raw_dir(raw_dir, DATASET_NAME)
    output_dir = resolve_output_dir(output_dir)
    # Only an omitted selection means "all 11"; an explicitly empty one is an error
    # the converter reports, not a request to convert everything.
    selected = list(sequences) if sequences is not None else None

    print(f"Raw dir    : {raw_dir}")
    print(f"Output dir : {output_dir}")
    print()

    try:
        archives = convert_euroc.required_archives(selected)
    except convert_euroc.ConversionError as exc:
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
    print(f"Converting EuRoC data to {dataset_dir} …")
    try:
        convert_euroc.convert(raw_dir, dataset_dir, selected)
    except convert_euroc.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    require_nonempty_files(dataset_dir, DATASET_ARTIFACTS, "converter")
    for sequence in selected or convert_euroc.ALL_SEQS:
        require_nonempty_files(
            dataset_dir / sequence,
            SEQUENCE_ARTIFACTS,
            "converter",
        )

    print()
    print(f"done — portable EuRoC dataset ready at {dataset_dir}")
    return dataset_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the EuRoC MAV dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_euroc",
        description="Download and convert the 11 official EuRoC MAV sequences to portable cuVSLAM EDEX data.",
    )
    add_common_arguments(parser, DATASET_NAME, label="EuRoC")
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=convert_euroc.ALL_SEQS,
        metavar="SEQUENCE",
        help="Convert an explicit sequence subset, such as MH_01_easy. The default is all 11 sequences.",
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
