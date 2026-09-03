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

"""Execution helpers for running reporter sequences in parallel."""

import argparse
import copy
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cuvslam_tools.tracker.runner import Stat


LOGGER = logging.getLogger(__name__)
CUVSLAM_REPORTER_DEPENDENCY_ERROR = "cuvslam_reporter requires the cuVSLAM Python binding to be installed"


def _warn_unsupported_config_fields(config: dict[str, Any]) -> None:
    """Warn about legacy reporter config fields that the Python tool ignores."""
    if config.get("write_cache"):
        warnings.warn(
            "Ignoring reporter config field write_cache=true; EDEX result cache export is not supported.",
            stacklevel=2,
        )
    if config.get("use_icp_scaling"):
        warnings.warn(
            "Ignoring reporter config field use_icp_scaling=true; ICP scale correction is not supported.",
            stacklevel=2,
        )
    if config.get("use_cuda"):
        warnings.warn(
            "Ignoring reporter config field use_cuda=true; use --use_gpu for the Python API.",
            stacklevel=2,
        )
    if "edex_folder" in config:
        warnings.warn(
            "Ignoring unsupported legacy config field edex_folder; use dataset_folder + sequence_folder.",
            stacklevel=2,
        )


def _warn_unsupported_sequence_fields(sequence: dict[str, Any]) -> None:
    """Warn about legacy sequence fields that the Python tracker does not use."""
    if sequence.get("use_gt_scale"):
        warnings.warn(
            f"Ignoring use_gt_scale for {sequence.get('sequence_title', '<unnamed>')}; not supported.",
            stacklevel=2,
        )
    if sequence.get("start_end_residual"):
        warnings.warn(
            f"Ignoring start_end_residual for {sequence.get('sequence_title', '<unnamed>')}; not supported.",
            stacklevel=2,
        )
    if sequence.get("precompute_2d_tracks"):
        warnings.warn(
            f"Ignoring precompute_2d_tracks for {sequence.get('sequence_title', '<unnamed>')}.",
            stacklevel=2,
        )
    if sequence.get("precompute_key_frames"):
        warnings.warn(
            f"Ignoring precompute_key_frames for {sequence.get('sequence_title', '<unnamed>')}.",
            stacklevel=2,
        )
    if "multicam_mode" in sequence or "multicam_setup" in sequence:
        warnings.warn(
            "Ignoring per-sequence multicam_mode/multicam_setup; use CLI --multicam_mode.",
            stacklevel=2,
        )


def _resolve_sequence_gt_path(sequence: dict[str, Any]) -> Optional[str]:
    """Return a usable ground-truth path from a reporter sequence entry."""
    gt_file_path = sequence.get("gt_file_path")
    if not gt_file_path:
        if gt_file_path == "":
            warnings.warn(
                f"Ignoring empty gt_file_path for {sequence.get('sequence_title', '<unnamed>')}.",
                stacklevel=2,
            )
        return None
    return gt_file_path


def _resolve_dataset_base(datasets_root: str, dataset_folder: str) -> str:
    """Resolve the directory that contains sequence folders."""
    datasets_root = os.path.normpath(datasets_root)
    dataset_folder = os.path.normpath(dataset_folder)
    if dataset_folder in ("", "."):
        return datasets_root

    configured_base = os.path.normpath(os.path.join(datasets_root, dataset_folder))
    if os.path.isdir(configured_base):
        return configured_base

    if os.path.basename(datasets_root) == os.path.basename(dataset_folder):
        return datasets_root

    return configured_base


def _load_track():
    """Load the tracker entry point lazily so config validation does not require cuVSLAM."""
    try:
        from cuvslam_tools.tracker.runner import track
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        if getattr(exc, "name", None) == "cuvslam" or "cuvslam" in str(exc):
            raise RuntimeError(CUVSLAM_REPORTER_DEPENDENCY_ERROR) from exc
        raise
    return track


# Checked only where present, so a disabled stub missing the required keys still passes.
_SEQUENCE_FLAGS = ("enable", "use_slam", "gt_from_shuttle")


def _validate_sequence_flags(index: int, sequence: dict) -> None:
    """Reject sequence flags that are not real booleans.

    bool("false") is True, so a quoted flag would switch on the very thing it
    says to switch off.
    """
    for key in _SEQUENCE_FLAGS:
        if key in sequence and not isinstance(sequence[key], bool):
            raise ValueError(f"Reporter config sequence_cfgs[{index}] key {key} must be a boolean")


def _validate_reporter_config(reporter_config: dict) -> tuple[str, list[dict]]:
    """Validate reporter config shape and return dataset and sequence entries."""
    if not isinstance(reporter_config, dict):
        raise ValueError("Reporter config must be a JSON object")

    for key in ("dataset_folder", "sequence_cfgs"):
        if key not in reporter_config:
            raise ValueError(f"Reporter config missing required key: {key}")

    dataset_folder = reporter_config["dataset_folder"]
    if not isinstance(dataset_folder, str):
        raise ValueError("Reporter config key dataset_folder must be a string")

    sequence_cfgs = reporter_config["sequence_cfgs"]
    if not isinstance(sequence_cfgs, list):
        raise ValueError("Reporter config key sequence_cfgs must be a list")

    for index, sequence in enumerate(sequence_cfgs):
        if not isinstance(sequence, dict):
            raise ValueError(f"Reporter config sequence_cfgs[{index}] must be an object")
        _validate_sequence_flags(index, sequence)
        if sequence.get("enable") is False:
            continue
        for key in ("sequence_title", "sequence_folder"):
            if key not in sequence:
                raise ValueError(
                    f"Reporter config sequence_cfgs[{index}] missing required key: {key}"
                )
            if not isinstance(sequence[key], str):
                raise ValueError(f"Reporter config sequence_cfgs[{index}] key {key} must be a string")

    return dataset_folder, sequence_cfgs


def process_sequence(sequence: dict, args: argparse.Namespace, datasets_root: str, dataset_folder: str) -> "Stat":
    """Process a single reporter sequence entry."""
    track = _load_track()

    _warn_unsupported_sequence_fields(sequence)
    dataset_base = _resolve_dataset_base(datasets_root, dataset_folder)
    args_copy = copy.deepcopy(args)
    args_copy.sequence_title = sequence["sequence_title"]
    args_copy.dataset = os.path.join(dataset_base, sequence["sequence_folder"])
    args_copy.config_path = os.path.join(args_copy.dataset, sequence.get("edex_file", "stereo.edex"))
    args_copy.gt_path = _resolve_sequence_gt_path(sequence)
    args_copy.gt_from_shuttle = sequence.get("gt_from_shuttle", False)
    args_copy.camera_ids = sequence.get("cameras")
    args_copy.use_slam = sequence.get("use_slam", False)
    if "repeat_type" in sequence:
        args_copy.repeat_type = (sequence["repeat_type"] or "none").lower()
    if "sequence_num_repeats" in sequence:
        args_copy.num_loops = sequence["sequence_num_repeats"]

    tracker_results = track(args_copy)
    return tracker_results.stat


def run_parallel_tracking(
    reporter_config: dict,
    args: argparse.Namespace,
    datasets_root: str,
    max_workers: Optional[int] = None,
) -> list["Stat"]:
    """Run tracking on all enabled sequences in a reporter config."""
    dataset_folder, sequence_cfgs = _validate_reporter_config(reporter_config)
    _warn_unsupported_config_fields(reporter_config)

    if max_workers is None:
        max_workers = 12
        print(f"Using default max_workers={max_workers} for GPU workloads (use --max_workers to override)")
    else:
        if max_workers < 1:
            raise ValueError("--max_workers must be a positive integer")
        print(f"Using max_workers={max_workers} (user specified)")

    stats = []
    failed_sequences = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_sequence = {}
        for sequence in sequence_cfgs:
            if sequence.get("enable") is False:
                continue
            future = executor.submit(process_sequence, sequence, args, datasets_root, dataset_folder)
            future_to_sequence[future] = sequence

        for future, sequence in future_to_sequence.items():
            sequence_name = sequence.get("sequence_title") or sequence.get("sequence_folder") or "<unknown>"
            try:
                stat = future.result()
            except RuntimeError as exc:
                if str(exc) == CUVSLAM_REPORTER_DEPENDENCY_ERROR:
                    raise
                LOGGER.exception("Tracking failed for sequence %s", sequence_name)
                failed_sequences.append(sequence_name)
                continue
            except Exception:
                LOGGER.exception("Tracking failed for sequence %s", sequence_name)
                failed_sequences.append(sequence_name)
                continue
            if stat is None:
                LOGGER.warning("Tracking returned no stats for sequence %s", sequence_name)
                failed_sequences.append(sequence_name)
                continue
            stats.append(stat)

    if future_to_sequence and not stats:
        raise RuntimeError(
            "Tracking failed for all enabled sequences: " + ", ".join(failed_sequences)
        )
    if failed_sequences:
        raise RuntimeError(
            "Tracking failed for sequences: " + ", ".join(failed_sequences)
        )

    return stats
