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

"""Console entry point for multi-dataset cuVSLAM validation runs."""

import argparse
import fnmatch
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

from cuvslam_tools.datasets_validator.combine_results import combine_report_stats
from cuvslam_tools.reporter.cli import add_reporter_backend_arguments, parse_reporter_arguments, run_report
from cuvslam_tools.tracker.cli import add_tracker_arguments


def _resolve_validation_config(path: str):
    """Resolve a validation config from a file path or packaged config name."""
    config_path = Path(path)
    if config_path.is_absolute() and config_path.exists():
        return config_path
    if config_path.exists():
        return config_path

    package_config = resources.files("cuvslam_tools.datasets_validator").joinpath("configs", path)
    if package_config.is_file():
        return package_config

    raise FileNotFoundError(f"Validation config not found: {path}")


def _load_validation_config(path: str) -> dict:
    """Load and validate a YAML validation config mapping."""
    import yaml

    config_path = _resolve_validation_config(path)
    try:
        with config_path.open() as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Validation config is not valid YAML: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Validation config must be a YAML mapping: {config_path}")
    return config


def _is_trivial_change(config: dict, base_ref: str) -> bool:
    """Return whether all changed files match configured trivial-change globs."""
    patterns = config.get("trivial_change_paths", [])
    if not patterns:
        return False

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Could not determine trivial changes; running validation: {exc}", file=sys.stderr)
        return False
    changed_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not changed_files:
        return False

    for path in changed_files:
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            return False
    print(f"Skipping validation: all changed files match trivial-change rules: {patterns}")
    return True


def _check_metrics(rows: list[dict], checks: list[dict]) -> None:
    """Validate combined report rows against configured metric bounds."""
    for check in checks:
        metric = check["metric"]
        if not rows:
            raise ValueError(f"{metric} missing from rows: <none>")
        missing = [
            row.get("sequence_name", row.get("report_dir", "<unknown>"))
            for row in rows
            if metric not in row or row[metric] in ("", None)
        ]
        if missing:
            raise ValueError(f"{metric} missing from rows: {', '.join(missing)}")
        values = [float(row[metric]) for row in rows]

        if "min_value" in check and min(values) < float(check["min_value"]):
            raise ValueError(f"{metric} min check failed: {min(values)} < {check['min_value']}")
        if "max_value" in check and max(values) > float(check["max_value"]):
            raise ValueError(f"{metric} max check failed: {max(values)} > {check['max_value']}")


def main(argv: Optional[list[str]] = None) -> int:
    """Run validator configs, combine report stats, and enforce metric checks."""
    parser = argparse.ArgumentParser(prog="cuvslam_validator")
    parser.add_argument("--validation_config", required=True, help="Datasets validator config file.")
    parser.add_argument("--datasets_root", default="", help="Root directory containing datasets.")
    parser.add_argument("--output_root", required=True, help="Root directory for validation outputs.")
    parser.add_argument("--skip_if_trivial_change", action="store_true", help="Skip when changed files are trivial.")
    parser.add_argument("--base_ref", default="origin/main", help="Base ref for trivial-change detection.")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum reporter sequence workers.")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF reports.")
    add_reporter_backend_arguments(parser)
    add_tracker_arguments(parser)
    args, full_argv = parse_reporter_arguments(parser, argv)

    try:
        config = _load_validation_config(args.validation_config)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.skip_if_trivial_change and config.get("allow_skip_for_trivial_changes"):
        if _is_trivial_change(config, args.base_ref):
            return 0

    validation_output_root = args.output_root
    report_dirs = []
    for reporter_config in config["reporter_configs"]:
        args.test_config = reporter_config
        args.output_dir = ""
        args.output_root = os.path.join(validation_output_root, Path(reporter_config).stem)
        args.report_comments = [
            f"validation_config={args.validation_config}",
            f"reporter_config={reporter_config}",
            *full_argv,
        ]
        try:
            report_dirs.append(run_report(args))
        except RuntimeError as exc:
            parser.error(str(exc))

    combined_report = config.get("combined_report", {})
    output_csv = os.path.join(validation_output_root, combined_report.get("table", "validation-summary.csv"))
    rows = combine_report_stats(report_dirs, output_csv)
    try:
        _check_metrics(rows, config.get("metric_checks", []))
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote validation summary to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
