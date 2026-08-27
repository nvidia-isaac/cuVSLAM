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

"""Download the TUM RGB-D freiburg3 archive and lay out the sequence for the examples.

This is provisioning only: it extracts the archive and copies the rig calibration
into place. It does not convert the dataset to the cuVSLAM reporter format.
"""

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath
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

DATASET_NAME = "tum"
DOWNLOAD_SCRIPT = "download_tum.sh"
RIG_FILE = "freiburg3_rig.yaml"
SEQUENCE_NAME = "rgbd_dataset_freiburg3_long_office_household"


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


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force_download: bool = False,
    download_only: bool = False,
) -> Path:
    """Download and lay out the TUM RGB-D sequence, and return the prepared root.

    Returns the raw directory when ``download_only`` is set, otherwise the dataset
    root holding the extracted sequence and its rig calibration.
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

    dataset_dir = output_dir / DATASET_NAME
    sequence_dir = dataset_dir / SEQUENCE_NAME
    archive = raw_dir / f"{SEQUENCE_NAME}.tgz"

    print()
    print(f"Extracting {archive.name} …")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    extract_archive(archive, dataset_dir)

    if not sequence_dir.is_dir():
        raise PreparationError(f"expected {sequence_dir} after extraction")

    print("Copying rig calibration …")
    shutil.copyfile(dataset_file(__file__, RIG_FILE), sequence_dir / RIG_FILE)

    print()
    print(f"done — sequence ready at {sequence_dir}")
    # The dataset root is returned, not the sequence: provisioning archives this
    # directory, and the reporter expects sequence folders directly beneath the
    # dataset mount.
    return dataset_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the TUM RGB-D dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_tum",
        description="Download and lay out the TUM RGB-D freiburg3 long_office_household dataset.",
    )
    add_common_arguments(
        parser,
        DATASET_NAME,
        label="TUM",
        download_only_help="Download archives but skip dataset layout.",
    )
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
