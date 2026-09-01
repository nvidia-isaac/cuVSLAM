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

"""Shared building blocks for the per-dataset ``prepare.py`` modules.

Every dataset exposes the same contract::

    def prepare(...) -> Path    # download when required, convert, validate
    def main(argv=None) -> int  # parse arguments and call prepare()

``prepare()`` raises :class:`PreparationError` for expected failures. ``main()``
turns those into a nonzero exit code and a concise stderr message.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

DEFAULT_RAW_DIR_TEMPLATE = "./datasets/{dataset}/raw"
DEFAULT_OUTPUT_DIR = "./datasets/converted"


class PreparationError(RuntimeError):
    """Raised when a dataset cannot be downloaded, converted, or validated."""


def default_raw_dir(dataset_name: str) -> Path:
    """Default raw-data directory, relative to the current working directory."""
    return Path.cwd() / "datasets" / dataset_name / "raw"


def default_output_dir() -> Path:
    """Default prepared-data directory, relative to the current working directory."""
    return Path.cwd() / "datasets" / "converted"


def resolve_raw_dir(raw_dir: Optional[Path], dataset_name: str) -> Path:
    """Return the caller-supplied raw directory or the dataset default."""
    return Path(raw_dir) if raw_dir is not None else default_raw_dir(dataset_name)


def resolve_output_dir(output_dir: Optional[Path]) -> Path:
    """Return the caller-supplied output directory or the shared default."""
    return Path(output_dir) if output_dir is not None else default_output_dir()


def dataset_file(module_file: str, name: str) -> Path:
    """Return a data file shipped next to a dataset's ``prepare.py``."""
    return Path(module_file).resolve().with_name(name)


def add_common_arguments(
    parser: argparse.ArgumentParser,
    dataset_name: str,
    label: Optional[str] = None,
    force_download_help: str = "Re-download archives even when they already exist.",
    download_only_help: str = "Download archives but skip conversion.",
) -> None:
    """Add the four arguments every dataset preparation command accepts."""
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            f"Directory for raw {label or dataset_name} data. "
            f"Default: {DEFAULT_RAW_DIR_TEMPLATE.format(dataset=dataset_name)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for prepared data. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--force-download", action="store_true", help=force_download_help)
    parser.add_argument("--download-only", action="store_true", help=download_only_help)


def run_download_script(script: Path, arguments: Sequence[str] = ()) -> None:
    """Run a dataset ``download_*.sh`` script and raise on failure."""
    if not script.is_file():
        raise PreparationError(f"download script not found: {script}")

    # Keep our own progress output ordered relative to the script's output.
    sys.stdout.flush()
    completed = subprocess.run(["bash", str(script), *arguments], check=False)
    if completed.returncode != 0:
        raise PreparationError(f"{script.name} failed with exit code {completed.returncode}")


def require_nonempty_files(root: Path, relative_paths: Sequence[str], producer: str) -> None:
    """Raise when an expected artifact is missing or empty after conversion."""
    for relative_path in relative_paths:
        artifact = root / relative_path
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise PreparationError(f"{producer} did not produce {artifact}")


def run_preparation(prepare: Callable[[], Path]) -> int:
    """Call ``prepare`` and translate expected failures into a CLI exit code."""
    try:
        prepare()
    except (PreparationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
