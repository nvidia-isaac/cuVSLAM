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

"""Download the public TartanAir V1 Hard zips and convert them to the stereo evaluation."""

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

from . import convert_tartanair_v1 as convert

DATASET_NAME = convert.DATASET_ID
DOWNLOAD_SCRIPT = "download_tartanair_v1.sh"

SEQUENCE_ARTIFACTS = (
    convert.EDEX_FILE,
    convert.GROUND_TRUTH_FILE,
    "00/000000.png",
    "01/000000.png",
    "depth_00/000000.png",
)


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force_download: bool = False,
    download_only: bool = False,
    envs: Optional[Sequence[str]] = None,
    workers: Optional[int] = None,
) -> Path:
    """Download the pinned zips, convert them, and return the dataset root.

    ``envs`` limits a partial local run to some environments; the reporter
    configs then list only their sequences. Returns the raw directory when
    ``download_only`` is set.
    """
    raw_dir = resolve_raw_dir(raw_dir, DATASET_NAME)
    output_dir = resolve_output_dir(output_dir)
    try:
        chosen = convert.selected(envs)
    except convert.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    print(f"Raw dir    : {raw_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Selection  : {len(chosen)} environments, {sum(map(len, chosen.values()))} trajectories")
    print()

    download_arguments = [str(raw_dir)]
    if force_download:
        download_arguments.append("--force")
    for env in envs or ():
        download_arguments.extend(["--env", env])
    run_download_script(dataset_file(__file__, DOWNLOAD_SCRIPT), download_arguments)

    if download_only:
        return raw_dir

    dataset_dir = output_dir / DATASET_NAME
    print()
    print(f"Converting TartanAir V1 Hard to {dataset_dir} …")
    try:
        convert.convert(raw_dir, dataset_dir, envs, workers)
    except convert.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    require_nonempty_files(dataset_dir, ("dataset_metadata.json", *convert.config_subsets(chosen)), "converter")
    for env, trajectories in chosen.items():
        for trajectory in trajectories:
            require_nonempty_files(dataset_dir / env / trajectory, SEQUENCE_ARTIFACTS, "converter")

    print()
    print(f"done — TartanAir V1 Hard stereo dataset ready at {dataset_dir}")
    return dataset_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the TartanAir V1 Hard dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_tartanair_v1",
        description="Download the public TartanAir V1 Hard zips and convert them to the stereo evaluation.",
    )
    add_common_arguments(parser, DATASET_NAME, label="TartanAir V1")
    parser.add_argument(
        "--env",
        action="append",
        default=None,
        choices=convert.ENVIRONMENTS,
        help="Limit to one environment; repeatable. For partial local runs only: "
             "the reporter configs then list just these sequences.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Conversion processes.")
    args = parser.parse_args(argv)

    return run_preparation(
        lambda: prepare(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            force_download=args.force_download,
            download_only=args.download_only,
            envs=args.env,
            workers=args.workers,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
