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

"""Download TartanGround data and convert its stereo pairs to EDEX.

The bundled converter handles the classic TartanAir layout (``image_left``/
``image_right`` plus ``pose_left.txt``/``pose_right.txt``) and rewrites each
sequence in place. Sequences are therefore staged under the output directory
first, either directly for classic layouts or by mapping TartanGround
``lcam_*``/``rcam_*`` stereo pairs into the classic layout, so the raw download
is preserved.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional

from cuvslam_tools.dataset_preparation.common import (
    PreparationError,
    add_common_arguments,
    resolve_output_dir,
    resolve_raw_dir,
    run_preparation,
)

from .download import VARIANTS, download_tartan_ground
from .stage_sequences import stage_sequences

DATASET_NAME = "tartan"
DEFAULT_VARIANT = "multisensor"

# The converter writes its results in place but still requires both output folders,
# so it is pointed at fresh paths inside the converted directory.
_UNUSED_GT_FOLDER = ".gt_unused"
_UNUSED_EDEX_FOLDER = ".edex_unused"


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    variant: str = DEFAULT_VARIANT,
    force_download: bool = False,
    download_only: bool = False,
) -> Path:
    """Download one TartanGround variant, convert its stereo pairs, and return the root.

    Returns the raw directory when ``download_only`` is set, otherwise the converted
    directory holding the staged and converted sequences.
    """
    if variant not in VARIANTS:
        raise PreparationError(f"unknown variant '{variant}' (expected {'|'.join(sorted(VARIANTS))})")

    raw_dir = resolve_raw_dir(raw_dir, DATASET_NAME)
    output_dir = resolve_output_dir(output_dir)
    sequence_root = raw_dir / "dataset" / "tartan_ground"
    converted_dir = output_dir / DATASET_NAME / variant

    print(f"Raw dir    : {raw_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Variant    : {variant}")
    print()

    if not download_only and not force_download and converted_dir.exists():
        raise PreparationError(f"{converted_dir} already exists; use --force-download to overwrite")

    if force_download and sequence_root.exists():
        shutil.rmtree(sequence_root)

    print(f"Downloading TartanGround ({variant}) into {sequence_root} …")
    download_tartan_ground(variant, str(sequence_root))

    if download_only:
        return raw_dir

    if force_download and converted_dir.exists():
        shutil.rmtree(converted_dir)

    if not sequence_root.is_dir():
        raise PreparationError(f"no downloaded TartanGround data under {sequence_root}")

    print()
    print("Staging converter-compatible sequences …")
    staged_sequences = stage_sequences(sequence_root, converted_dir)
    if not staged_sequences:
        raise PreparationError(
            f"no convertible TartanAir or TartanGround stereo sequences found under {sequence_root}; "
            "need classic image_left/image_right plus pose_left.txt/pose_right.txt, or TartanGround "
            "image_lcam_*/image_rcam_* plus pose_lcam_*/pose_rcam_* pairs"
        )
    for sequence in staged_sequences:
        print(f"  - {sequence}")

    print("Converting to edex …")
    # Imported here so --download-only and --help do not require the converter's
    # numerical dependencies.
    from .dataset_converter.convert import convert_sequences

    convert_sequences(
        str(converted_dir),
        str(converted_dir / _UNUSED_GT_FOLDER),
        str(converted_dir / _UNUSED_EDEX_FOLDER),
    )

    if next(converted_dir.rglob("cfg.edex"), None) is None:
        raise PreparationError(f"conversion produced no cfg.edex under {converted_dir}")

    print()
    print(f"done — converted sequences (00/01 images, gt.txt, cfg.edex) under {converted_dir}")
    return converted_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the TartanGround dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_tartan",
        description=(
            "Download TartanGround data and convert TartanGround stereo pairs or "
            "classic TartanAir-layout sequences to EDEX."
        ),
    )
    add_common_arguments(
        parser,
        DATASET_NAME,
        label="TartanGround",
        force_download_help="Remove existing download/conversion output first.",
        download_only_help="Download but skip conversion.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default=DEFAULT_VARIANT,
        help="TartanGround download variant. Both variants include metadata; use multicamera for EDEX conversion.",
    )
    args = parser.parse_args(argv)

    return run_preparation(
        lambda: prepare(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            variant=args.variant,
            force_download=args.force_download,
            download_only=args.download_only,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
