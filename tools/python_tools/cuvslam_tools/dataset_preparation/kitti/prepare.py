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

"""Download KITTI odometry archives and convert them to cuVSLAM format."""

import argparse
from pathlib import Path
from typing import List, Optional

from cuvslam_tools.dataset_preparation.common import (
    PreparationError,
    add_common_arguments,
    dataset_file,
    resolve_output_dir,
    resolve_raw_dir,
    run_download_script,
    run_preparation,
)

from . import convert_kitti

DATASET_NAME = "kitti"
DOWNLOAD_SCRIPT = "download_kitti.sh"


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force_download: bool = False,
    download_only: bool = False,
) -> Path:
    """Download the KITTI odometry archives, convert them, and return the prepared root.

    Returns the raw directory when ``download_only`` is set, otherwise the converted root.
    """
    raw_dir = resolve_raw_dir(raw_dir, DATASET_NAME)
    output_dir = resolve_output_dir(output_dir)

    print(f"Raw dir    : {raw_dir}")
    print(f"Output dir : {output_dir}")
    print()

    download_arguments = [str(raw_dir)]
    if force_download:
        download_arguments.append("--force")
    run_download_script(dataset_file(__file__, DOWNLOAD_SCRIPT), download_arguments)

    if download_only:
        return raw_dir

    print()
    try:
        convert_kitti.convert(raw_dir, output_dir)
    except convert_kitti.ConversionError as exc:
        raise PreparationError(str(exc)) from exc
    return output_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the KITTI odometry dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_kitti",
        description="Download KITTI odometry archives and convert them to cuVSLAM format.",
    )
    add_common_arguments(parser, DATASET_NAME, label="KITTI")
    args = parser.parse_args(argv)

    return run_preparation(
        lambda: prepare(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            force_download=args.force_download,
            download_only=args.download_only,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
