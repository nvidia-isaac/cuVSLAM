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

"""Shared TartanGround download implementation."""

import argparse
import os
import sys
from typing import Optional

from cuvslam_tools.dataset_preparation.common import PreparationError

DEFAULT_DATA_ROOT = "dataset/tartan_ground/"

VARIANTS = {
    "multicamera": {
        "modality": ["image", "meta"],
        "camera_name": [
            "lcam_front",
            "rcam_front",
            "lcam_left",
            "rcam_left",
            "lcam_right",
            "rcam_right",
            "lcam_back",
            "rcam_back",
            "lcam_top",
            "rcam_top",
            "lcam_bottom",
            "rcam_bottom",
        ],
    },
    "multisensor": {
        "modality": ["image", "depth", "imu", "meta"],
        "camera_name": ["lcam_front", "lcam_back"],
    },
}


def download_tartan_ground(variant: str, data_root: str = DEFAULT_DATA_ROOT) -> None:
    """Download one supported TartanGround variant using the tartanair package."""
    variant_config = VARIANTS[variant]

    try:
        import tartanair as ta
    except ImportError as exc:
        raise PreparationError(
            "the tartanair package is required to download TartanGround data; install it with "
            "'pip install tartanair'. It only works on x86_64: on aarch64 it fails at import, so "
            "download on an x86_64 machine and transfer the data to the target."
        ) from exc

    os.makedirs(data_root, exist_ok=True)
    ta.init(data_root)
    ta.download_ground(
        env=["OldTownFall"],
        version=["anymal"],
        modality=variant_config["modality"],
        traj=["P2000"],
        camera_name=variant_config["camera_name"],
        unzip=True,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for TartanGround download helpers."""
    parser = argparse.ArgumentParser(description="Download a supported TartanGround dataset variant.")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="multisensor",
        help="Dataset variant to download.",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Directory passed to tartanair.init. Defaults to dataset/tartan_ground/ relative to cwd.",
    )
    args = parser.parse_args(argv)

    try:
        download_tartan_ground(args.variant, args.data_root)
    except PreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
