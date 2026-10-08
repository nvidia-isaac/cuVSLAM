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

"""Console entry point for evaluating cuVSLAM Visual Place Recognition."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from cuvslam_tools.tracker.cli import _str2bool, add_tracker_arguments


def _resolve_config_path(test_config: str, datasets_root: str) -> Path:
    """Resolve a VPR reporter config path from command-line input."""
    config_path = Path(test_config)
    if config_path.is_file():
        return config_path

    if not config_path.is_absolute():
        dataset_config_path = Path(datasets_root) / test_config
        if dataset_config_path.is_file():
            return dataset_config_path

    raise FileNotFoundError(f"VPR reporter config not found: {test_config}")


def run_vpr_report(args: argparse.Namespace) -> str:
    """Evaluate every configured (mode, pair) combination and generate report outputs."""
    from cuvslam_tools.vpr_reporter.execution import (
        DEFAULT_SUCCESS_RADIUS_M,
        DEFAULT_VPR_MODES,
        evaluate_pairs,
        normalize_modes,
        pair_from_args,
        parse_config,
    )
    from cuvslam_tools.vpr_reporter.generate_report import generate_report, save_stats_to_json

    if args.frame_limit < 0:
        raise ValueError("--frame_limit must be zero (all frames) or positive")
    if args.query_stride < 1:
        raise ValueError("--query_stride must be a positive integer")

    datasets_root = args.datasets_root or os.environ.get("CUVSLAM_DATASETS", "")
    config_modes = None
    config_radius = None

    if args.test_config:
        if args.map_sequence or args.query_sequence:
            raise ValueError("Pass either --test_config or --map_sequence/--query_sequence, not both")
        config_path = _resolve_config_path(args.test_config, datasets_root)
        with config_path.open() as f:
            parsed = parse_config(json.load(f), datasets_root)
        pairs = parsed["pairs"]
        config_modes = parsed["vpr_modes"]
        config_radius = parsed["success_radius_m"]
        config_name = config_path.stem
    else:
        if not args.map_sequence or not args.query_sequence:
            raise ValueError("Provide --test_config, or both --map_sequence and --query_sequence")
        pairs = [pair_from_args(args, datasets_root)]
        config_name = "vpr"

    # An explicit flag wins over the config, which wins over the built-in default.
    modes = normalize_modes(args.vpr_modes.split(",")) if args.vpr_modes else (config_modes or list(DEFAULT_VPR_MODES))
    if args.success_radius is not None:
        success_radius_m = args.success_radius
    else:
        success_radius_m = config_radius if config_radius is not None else DEFAULT_SUCCESS_RADIUS_M
    if success_radius_m <= 0:
        raise ValueError("--success_radius must be positive")

    output_root = args.output_root or os.environ.get("CUVSLAM_OUTPUT")
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)
    else:
        if not output_root:
            raise ValueError("Provide --output_root, --output_dir, or set CUVSLAM_OUTPUT")
        args.output_dir = os.path.join(output_root, config_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    stats = evaluate_pairs(pairs, modes, args, success_radius_m, args.output_dir)
    save_stats_to_json(stats, args.output_dir)
    report_comments = getattr(args, "report_comments", sys.argv[1:])
    generate_report(args.output_dir, report_comments, stats, generate_pdf=args.pdf,
                    config_name=config_name, success_radius_m=success_radius_m)
    return args.output_dir


def main(argv: Optional[list[str]] = None) -> int:
    """Parse VPR reporter CLI arguments and run the evaluation workflow."""
    parser = argparse.ArgumentParser(
        prog="cuvslam_vpr_reporter",
        description="Evaluate Visual Place Recognition: build a map from one sequence, "
                    "recognize places in another, and score both against KITTI-format ground truth.",
    )
    parser.add_argument("--test_config", type=str, default="", help="VPR reporter config file.")
    parser.add_argument("--datasets_root", type=str, default="", help="Root directory containing datasets.")
    parser.add_argument("--output_root", type=str, default="", help="Root directory for report outputs.")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF report in addition to HTML.")
    parser.add_argument("--map_sequence", type=str, default="",
                        help="EDEX sequence the place recognition map is built from, without --test_config.")
    parser.add_argument("--query_sequence", type=str, default="",
                        help="EDEX sequence whose frames are recognized in that map, without --test_config.")
    parser.add_argument("--pair_title", type=str, default="",
                        help="Title of the sequence pair; defaults to the two folder names.")
    parser.add_argument("--vpr_modes", type=str, default=None,
                        help="Comma-separated place recognition backends: Simple, DBoW2, AnyLoc "
                             "(default: the config's vpr_modes, else Simple).")
    parser.add_argument("--success_radius", type=float, default=None,
                        help="Distance in meters within which a recognized place counts as correct "
                             "(default: the config's success_radius_m, else 10.0).")
    parser.add_argument("--vpr_score_threshold", type=float, default=0.0,
                        help="Minimum similarity for a match; 0 selects the backend default.")
    parser.add_argument("--vpr_model_path", type=str, default="",
                        help="Model file the backend needs; AnyLoc reads a DINOv2 ONNX model from here.")
    parser.add_argument("--frame_limit", type=int, default=0,
                        help="Stop each sequence after this many frames; 0 replays all of them.")
    parser.add_argument("--query_stride", type=int, default=1,
                        help="Query every Nth frame of the query sequence; every frame is still tracked.")
    parser.add_argument("--query_tracking", type=_str2bool, default=True,
                        help="Track the query sequence while recognizing places in it (default, and what a robot "
                             "localizing against a saved map does; the loaded map is read only, so the query "
                             "session's own keyframes never enter the search either way). false skips odometry "
                             "during the query pass, a diagnostic that times and scores place recognition alone.")
    parser.add_argument("--keep_vpr_maps", action="store_true",
                        help="Keep the temporary map folders instead of deleting them; each holds the whole SLAM "
                             "map of its combination, landmark database included, so they are large.")
    add_tracker_arguments(parser)

    args = parser.parse_args(argv)
    args.report_comments = sys.argv[1:] if argv is None else argv
    try:
        run_vpr_report(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
