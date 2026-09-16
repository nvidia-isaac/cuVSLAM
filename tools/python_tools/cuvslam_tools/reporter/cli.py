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

"""Console entry point for generating cuVSLAM dataset reports."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from cuvslam_tools.tracker.cli import add_tracker_arguments


def add_reporter_backend_arguments(parser: argparse.ArgumentParser) -> None:
    """Add backend selection shared by reporter and validator."""
    parser.add_argument(
        "--tracker_backend",
        choices=["python", "api_launcher"],
        default="python",
        help="Tracking implementation. Python is in-process; api_launcher runs the C++ executable.",
    )
    parser.add_argument(
        "--api_launcher_path",
        default="",
        help="C++ launcher executable; otherwise use CUVSLAM_BUILD_DIR/bin or PATH.",
    )


def parse_reporter_arguments(
    parser: argparse.ArgumentParser, argv: Optional[list[str]] = None
) -> tuple[argparse.Namespace, list[str]]:
    """Parse reporter arguments and reserve trailing `--` values for the C++ launcher."""
    full_argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in full_argv:
        separator = full_argv.index("--")
        parser_argv = full_argv[:separator]
        launcher_args = full_argv[separator + 1 :]
    else:
        parser_argv = full_argv
        launcher_args = []
    args = parser.parse_args(parser_argv)
    if launcher_args and args.tracker_backend != "api_launcher":
        parser.error("arguments after -- are supported only with --tracker_backend=api_launcher")
    args.api_launcher_args = launcher_args
    return args, full_argv


def _resolve_config_path(test_config: str, datasets_root: str):
    """Resolve a reporter config path from command-line input."""
    config_path = Path(test_config)
    if config_path.is_file():
        return config_path

    if not config_path.is_absolute():
        dataset_config_path = Path(datasets_root) / test_config
        if dataset_config_path.is_file():
            return dataset_config_path

    raise FileNotFoundError(f"Reporter config not found: {test_config}")


def run_report(args: argparse.Namespace) -> str:
    """Run tracking for one reporter config and generate report outputs."""
    try:
        from cuvslam_tools.reporter.execution import run_parallel_tracking
        from cuvslam_tools.reporter.generate_report import generate_report, save_stats_to_json
    except ModuleNotFoundError as exc:
        if exc.name == "cuvslam":
            raise RuntimeError("cuvslam_reporter requires the cuVSLAM Python binding to be installed") from exc
        raise

    datasets_root = args.datasets_root or os.environ.get("CUVSLAM_DATASETS")
    if not datasets_root:
        raise ValueError("Provide --datasets_root or set CUVSLAM_DATASETS")

    config_path = _resolve_config_path(args.test_config, datasets_root)
    with config_path.open() as f:
        reporter_config = json.load(f)

    config_name = Path(config_path.name).stem
    output_root = args.output_root or os.environ.get("CUVSLAM_OUTPUT")
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)
    else:
        if not output_root:
            raise ValueError("Provide --output_root, --output_dir, or set CUVSLAM_OUTPUT")
        args.output_dir = os.path.join(
            output_root,
            config_name,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

    try:
        args.segment_lengths = reporter_config["segment_lengths"]
    except KeyError as exc:
        raise ValueError("Reporter config missing required key: segment_lengths") from exc
    stats = run_parallel_tracking(reporter_config, args, datasets_root, max_workers=args.max_workers)
    save_stats_to_json(stats, args.output_dir)
    report_comments = getattr(args, "report_comments", sys.argv[1:])
    generate_report(args.output_dir, report_comments, stats, generate_pdf=args.pdf, config_name=config_name)
    return args.output_dir


def main(argv: Optional[list[str]] = None) -> int:
    """Parse reporter CLI arguments and run the report workflow."""
    parser = argparse.ArgumentParser(prog="cuvslam_reporter")
    parser.add_argument("--test_config", type=str, required=True, help="Reporter config file.")
    parser.add_argument("--datasets_root", type=str, default="", help="Root directory containing datasets.")
    parser.add_argument("--output_root", type=str, default="", help="Root directory for report outputs.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of sequence workers.")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF report in addition to HTML.")
    add_reporter_backend_arguments(parser)
    add_tracker_arguments(parser)

    args, full_argv = parse_reporter_arguments(parser, argv)
    args.report_comments = full_argv
    try:
        run_report(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
