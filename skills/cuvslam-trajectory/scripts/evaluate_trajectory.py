#!/usr/bin/env python3

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

"""Evaluate timestamped poses against GT, independently of cuVSLAM (requires NumPy)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


@dataclass
class Trajectory:
    timestamps: np.ndarray  # int64 nanoseconds, never epoch seconds in float64
    poses: np.ndarray  # T_world_from_sensor, homogeneous 4x4 matrices


def timestamp_ns(value: str, unit: str = "s") -> int:
    """Parse decimal timestamps without losing precision through epoch floats."""
    try:
        number = Decimal(value) * (10**9 if unit == "s" else 1)
        if not number.is_finite():
            raise ValueError("non-finite timestamp")
        result = int(number.to_integral_value())
        if not -(2**63) < result < 2**63:
            raise ValueError("timestamp outside int64 nanosecond range")
        return result
    except InvalidOperation as exc:
        raise ValueError(f"invalid timestamp: {value}") from exc


def quaternion_matrix(xyzw: np.ndarray) -> np.ndarray:
    """Active, right-handed rotation; q and -q represent the same rotation."""
    norm = np.linalg.norm(xyzw)
    if not np.isfinite(norm) or abs(norm - 1) > 1e-3:
        raise ValueError(f"quaternion norm {norm} is not near 1")
    x, y, z, w = xyzw / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def rigid_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.size != 16:
        raise ValueError("rigid transform needs 16 row-major values")
    matrix = matrix.reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("transform must be finite SE(3), with no reflection or scale")
    return matrix


def inverse(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def load_trajectory(path: Path, fmt: str, times_path: Path | None = None) -> Trajectory:
    """Read TUM (seconds/xyzw), EuRoC CSV (ns/wxyz), or KITTI + times.txt."""
    timestamps, poses = [], []
    rows = [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if fmt == "kitti":
        if times_path is None:
            raise ValueError("KITTI input requires its --*-timestamps times.txt (seconds)")
        times = [line.strip() for line in times_path.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if len(times) != len(rows):
            raise ValueError("KITTI pose count differs from timestamp count")
    elif times_path is not None:
        raise ValueError("separate timestamps are supported only for KITTI input")
    for index, line in enumerate(rows):
        fields = line.split(",") if fmt == "euroc" else line.split()
        pose = np.eye(4)
        if fmt == "kitti":
            if len(fields) != 12:
                raise ValueError(f"pose {index + 1}: KITTI needs 12 columns")
            pose[:3] = np.asarray(fields, dtype=float).reshape(3, 4)
            rigid_matrix(pose)
            stamp = timestamp_ns(times[index])
        else:
            if (fmt == "tum" and len(fields) != 8) or (fmt == "euroc" and len(fields) != 17):
                raise ValueError(f"pose {index + 1}: expected 8 TUM or 17 EuRoC columns")
            values = np.asarray(fields[1:], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"pose {index + 1}: non-finite value")
            pose[:3, 3] = values[:3]
            quaternion = values[3:7] if fmt == "tum" else values[[4, 5, 6, 3]]
            pose[:3, :3] = quaternion_matrix(quaternion)
            stamp = timestamp_ns(fields[0], "ns" if fmt == "euroc" else "s")
        if timestamps and stamp <= timestamps[-1]:
            raise ValueError(f"pose {index + 1}: timestamps must be strictly increasing")
        timestamps.append(stamp)
        poses.append(pose)
    if len(poses) < 2:
        raise ValueError("evaluation requires at least two poses per trajectory")
    return Trajectory(np.asarray(timestamps, dtype=np.int64), np.asarray(poses))


def associate(estimate: np.ndarray, gt: np.ndarray, tolerance_ns: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Greedy minimum-|dt|, one-to-one association within the strict common interval."""
    if tolerance_ns < 0:
        raise ValueError("timestamp tolerance must be nonnegative")
    start, end = max(int(estimate[0]), int(gt[0])), min(int(estimate[-1]), int(gt[-1]))
    if end <= start:
        raise ValueError("trajectories have no overlapping interval")
    est_ids = np.flatnonzero((estimate >= start) & (estimate <= end))
    gt_ids = np.flatnonzero((gt >= start) & (gt <= end))
    gt_times = gt[gt_ids]
    candidates = []
    for i in est_ids:
        stamp = int(estimate[i])
        lo = np.searchsorted(gt_times, stamp - tolerance_ns, side="left")
        hi = np.searchsorted(gt_times, stamp + tolerance_ns, side="right")
        for j in gt_ids[lo:hi]:
            candidates.append((abs(stamp - int(gt[j])), int(i), int(j)))
    used_est, used_gt, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_est and j not in used_gt:
            used_est.add(i)
            used_gt.add(j)
            pairs.append((i, j))
    pairs.sort()
    if len(pairs) < 2:
        raise ValueError("fewer than two timestamp matches; check clocks, units, offset, and tolerance")
    matched_est, matched_gt = np.asarray(pairs, dtype=int).T
    if np.any(np.diff(matched_gt) <= 0):
        raise ValueError("nearest matches cross in time; lower tolerance or resample explicitly")
    return matched_est, matched_gt, {
        "overlap_start_ns": start, "overlap_end_ns": end,
        "estimate_total": len(estimate), "gt_total": len(gt), "evaluated_pairs": len(pairs),
        "estimate_outside_overlap": len(estimate) - len(est_ids),
        "gt_outside_overlap": len(gt) - len(gt_ids),
        "estimate_unmatched_in_overlap": len(est_ids) - len(pairs),
        "gt_unmatched_in_overlap": len(gt_ids) - len(pairs),
        "estimate_excluded": len(estimate) - len(pairs), "gt_excluded": len(gt) - len(pairs),
    }


def pose_errors(estimate: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.linalg.norm(estimate[:, :3, 3] - gt[:, :3, 3], axis=1)
    relative = np.swapaxes(gt[:, :3, :3], 1, 2) @ estimate[:, :3, :3]
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    # atan2 is stable near both zero and pi, including rotations differing only by roundoff.
    skew = np.stack([relative[:, 2, 1] - relative[:, 1, 2],
                     relative[:, 0, 2] - relative[:, 2, 0],
                     relative[:, 1, 0] - relative[:, 0, 1]], axis=1)
    sine = np.linalg.norm(skew, axis=1) / 2
    return translation, np.degrees(np.arctan2(sine, cosine))


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def fit_alignment(estimate: np.ndarray, gt: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """Fit a fixed SE(3), with orientations or non-collinear positions determining rotation."""
    if len(estimate) < 3:
        raise ValueError("alignment needs at least three matched poses in the selected window")
    est_p, gt_p = estimate[:, :3, 3], gt[:, :3, 3]
    if method == "initial-poses":
        moment = np.mean(gt[:, :3, :3] @ np.swapaxes(estimate[:, :3, :3], 1, 2), axis=0)
    else:
        moment = (gt_p - gt_p.mean(axis=0)).T @ (est_p - est_p.mean(axis=0)) / len(est_p)
    u, singular, vt = np.linalg.svd(moment)
    if method == "initial-poses":
        # A vanishing direction or reflection makes the orientation mean unreliable.
        if singular[-1] < 1e-6 or np.linalg.det(moment) <= 0:
            raise ValueError("ambiguous orientation alignment; inspect frames and initial window")
    elif singular[0] < 1e-12 or singular[1] / singular[0] < 1e-6:
        raise ValueError("degenerate position alignment (stationary/collinear); use orientations or another window")
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    transform = np.eye(4)
    transform[:3, :3] = u @ correction @ vt
    transform[:3, 3] = gt_p.mean(axis=0) - transform[:3, :3] @ est_p.mean(axis=0)
    translation, rotation = pose_errors(transform @ estimate, gt)
    return transform, {
        "singular_values": singular.tolist(),
        "fit_translation_rmse_m": rmse(translation), "fit_rotation_rmse_deg": rmse(rotation),
    }


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sensor_matrix(path: Path) -> np.ndarray:
    import yaml  # Optional: only needed when reading sensor.yaml.

    config = yaml.safe_load(path.read_text())
    block = config["T_BS"]
    if block["rows"] != 4 or block["cols"] != 4:
        raise ValueError("T_BS must be 4x4")
    return rigid_matrix(block["data"])


def frame_conversion(args: argparse.Namespace) -> tuple[np.ndarray, list[dict], list[str]]:
    """Return T_estimate_sensor_from_gt_sensor, for right multiplication of estimates."""
    sources, warnings = [], []
    yaml_paths = (args.estimate_sensor_yaml, args.gt_sensor_yaml)
    if args.allow_missing_extrinsics and args.estimate_frame == args.gt_frame:
        raise ValueError("missing-extrinsics fallback requires different physical frames")
    if args.estimate_from_gt and any(yaml_paths):
        raise ValueError("choose a JSON transform or a pair of sensor YAML files")
    if args.allow_missing_extrinsics and (args.estimate_from_gt or any(yaml_paths)):
        raise ValueError("missing-extrinsics fallback cannot be combined with calibrated conversion")
    if args.estimate_from_gt:
        transform = rigid_matrix(json.loads(args.estimate_from_gt.read_text()))
        sources.append(file_record(args.estimate_from_gt))
    elif any(yaml_paths):
        if not all(yaml_paths):
            raise ValueError("both sensor YAML files are required, with the same T_BS reference frame")
        transform = inverse(sensor_matrix(yaml_paths[0])) @ sensor_matrix(yaml_paths[1])
        sources.extend(file_record(path) for path in yaml_paths)
    elif args.estimate_frame == args.gt_frame:
        transform = np.eye(4)
    elif args.allow_missing_extrinsics:
        if args.alignment not in ("initial-poses", "initial-positions"):
            raise ValueError("missing-extrinsics fallback requires an explicit initial-window alignment")
        transform = np.eye(4)
        warnings.append("PROVISIONAL: physical frames differ and extrinsics are unavailable. Initial-window "
                        "world alignment cannot generally recover a missing camera/body extrinsic; "
                        "lever-arm and orientation errors remain confounded with tracking error.")
    else:
        raise ValueError("different physical frames require extrinsics; see --allow-missing-extrinsics for a provisional result")
    return transform, sources, warnings


def evaluate(args: argparse.Namespace) -> tuple[dict, list[tuple]]:
    estimate = load_trajectory(args.estimate, args.estimate_format, args.estimate_timestamps)
    gt = load_trajectory(args.ground_truth, args.gt_format, args.gt_timestamps)
    offset = timestamp_ns(args.time_offset)
    # Do not let int64 wrap if a supplied offset is implausibly large.
    shifted = [int(t) + offset for t in estimate.timestamps]
    if min(shifted) <= -(2**63) or max(shifted) >= 2**63:
        raise ValueError("time offset overflows int64 timestamps")
    estimate.timestamps = np.asarray(shifted, dtype=np.int64)
    tolerance = timestamp_ns(args.max_time_difference)
    conversion, calibration, warnings = frame_conversion(args)
    estimate.poses = estimate.poses @ conversion
    est_ids, gt_ids, counts = associate(estimate.timestamps, gt.timestamps, tolerance)
    matched_est, matched_gt = estimate.poses[est_ids], gt.poses[gt_ids]
    matched_times = estimate.timestamps[est_ids]
    window_ns = timestamp_ns(args.alignment_window)
    if window_ns <= 0:
        raise ValueError("alignment window must be positive")
    fit_ids = np.array([], dtype=int)
    diagnostics = {}
    transform = np.eye(4)
    if args.alignment != "none":
        fit_ids = (np.arange(len(est_ids)) if args.alignment == "all-positions" else
                   np.flatnonzero(matched_times - matched_times[0] <= window_ns))
        transform, diagnostics = fit_alignment(matched_est[fit_ids], matched_gt[fit_ids], args.alignment)
    elif args.estimate_world != args.gt_world:
        raise ValueError("--alignment none requires the same declared world frame")
    translation, rotation = pose_errors(transform @ matched_est, matched_gt)
    # Use all native GT samples between matched endpoints, not just downsampled matches.
    distance = float(np.linalg.norm(np.diff(gt.poses[gt_ids[0]:gt_ids[-1] + 1, :3, 3], axis=0), axis=1).sum())
    ate = rmse(translation)
    if distance <= 1e-12:
        warnings.append("Normalized ATE is undefined because GT traveled distance is zero.")
    reasons = {
        "initial-poses": "Initial pose orientations fix the world rotation even without translational excitation; "
                         "initial positions set the origin without fitting later drift.",
        "initial-positions": "Initial positions set the world frame without fitting later drift; "
                             "non-collinearity is required.",
        "all-positions": "Global position least squares measures error after fitting the entire evaluated interval; "
                         "this may hide some drift relative to initial-window alignment.",
        "none": "Inputs are declared to use the same world frame; no alignment is applied.",
    }
    report = {
        "schema_version": 1, "status": "provisional" if args.allow_missing_extrinsics and warnings else "ok",
        "inputs": {
            "estimate": {**file_record(args.estimate), "format": args.estimate_format},
            "ground_truth": {**file_record(args.ground_truth), "format": args.gt_format},
            "timestamp_files": [file_record(p) for p in (args.estimate_timestamps, args.gt_timestamps) if p],
            "evaluator": file_record(Path(__file__)),
        },
        "runtime": {"python": sys.version, "numpy": np.__version__},
        "frames": {
            "pose_convention": "T_world_from_sensor, right-handed active rotations, positions in meters",
            "quaternions": "TUM xyzw; EuRoC CSV wxyz; normalized within 1e-3 norm tolerance",
            "estimate_sensor": args.estimate_frame, "gt_sensor": args.gt_frame,
            "estimate_world": args.estimate_world, "gt_world": args.gt_world,
            "evaluation_sensor": args.gt_frame if not args.allow_missing_extrinsics else "unresolved (see warnings)",
            "estimate_from_gt": conversion.tolist(), "calibration_sources": calibration,
            "conversion_rule": "T_W_E @ T_E_G; then T_Wgt_West @ converted_estimate",
        },
        "association": {
            "method": "all candidate pairs sorted by (absolute dt, estimate index, GT index); greedy one-to-one; "
                      "strict overlap; no interpolation; reject crossing matches",
            "estimate_time_offset_ns": offset, "max_difference_ns": tolerance,
            "actual_max_difference_ns": max(abs(int(estimate.timestamps[i]) - int(gt.timestamps[j]))
                                            for i, j in zip(est_ids, gt_ids)),
            **counts,
        },
        "alignment": {
            "method": args.alignment, "scale": 1.0, "rationale": args.alignment_reason or reasons[args.alignment],
            "requested_initial_window_seconds": window_ns / 1e9 if args.alignment.startswith("initial") else None,
            "fit_pose_count": len(fit_ids), "fit_start_ns": int(matched_times[fit_ids[0]]) if len(fit_ids) else None,
            "fit_end_ns": int(matched_times[fit_ids[-1]]) if len(fit_ids) else None,
            "fit_poses_included_in_metrics": True, "gt_world_from_estimate_world": transform.tolist(),
            **diagnostics,
        },
        "metrics": {
            "ate_rmse_m": ate, "are_rmse_deg": rmse(rotation),
            "normalized_ate_rmse_percent": 100 * ate / distance if distance > 1e-12 else None,
            "gt_traveled_distance_m": distance,
        },
        "metric_definitions": {
            "ate_rmse_m": "sqrt(mean(||p_est_aligned - p_gt||^2))",
            "are_rmse_deg": "sqrt(mean(angle(R_gt.T @ R_est_aligned)^2)), angles in [0,180] degrees",
            "normalized_ate_rmse_percent": "100 * ate_rmse_m / gt_traveled_distance_m; null for zero distance",
            "gt_traveled_distance_m": "sum of distances between ALL consecutive native GT positions from "
                                      "first to last matched GT timestamp, in the GT physical frame",
        },
        "evaluated_interval": {"first_gt_ns": int(gt.timestamps[gt_ids[0]]),
                               "last_gt_ns": int(gt.timestamps[gt_ids[-1]])},
        "warnings": warnings,
        "command": [sys.executable, *sys.argv],
    }
    errors = [(int(i), int(j), int(estimate.timestamps[i]), int(gt.timestamps[j]),
               int(estimate.timestamps[i]) - int(gt.timestamps[j]), float(t), float(r))
              for i, j, t, r in zip(est_ids, gt_ids, translation, rotation)]
    return report, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--estimate-format", choices=("tum", "euroc", "kitti"), default="tum")
    parser.add_argument("--gt-format", choices=("tum", "euroc", "kitti"), default="tum")
    parser.add_argument("--estimate-timestamps", type=Path, help="KITTI times.txt, seconds")
    parser.add_argument("--gt-timestamps", type=Path, help="KITTI times.txt, seconds")
    parser.add_argument("--estimate-frame", required=True, help="physical frame, e.g. cam0")
    parser.add_argument("--gt-frame", required=True, help="physical frame, e.g. imu0")
    parser.add_argument("--estimate-world", required=True, help="world frame and axis convention")
    parser.add_argument("--gt-world", required=True, help="world frame and axis convention")
    parser.add_argument("--estimate-from-gt", type=Path, help="JSON 4x4 T_E_G: maps GT sensor points to estimate sensor")
    parser.add_argument("--estimate-sensor-yaml", type=Path, help="T_BS for estimate sensor; requires PyYAML")
    parser.add_argument("--gt-sensor-yaml", type=Path, help="T_BS for GT sensor, SAME reference B as estimate YAML")
    parser.add_argument("--allow-missing-extrinsics", action="store_true", help="explicit provisional initial-window fallback")
    parser.add_argument("--time-offset", default="0", help="seconds ADDED to estimate times; no automatic clock fitting")
    parser.add_argument("--max-time-difference", default="0.01", help="association tolerance, seconds (default: 0.01)")
    parser.add_argument("--alignment", choices=("initial-poses", "initial-positions", "all-positions", "none"),
                        default="initial-poses")
    parser.add_argument("--alignment-window", default="1.0", help="seconds from first matched estimate, inclusive; min 3 pairs")
    parser.add_argument("--alignment-reason", help="rationale for the selected alignment")
    parser.add_argument("--output-dir", type=Path, required=True, help="write evaluation.json and errors.csv; never overwrite")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.output_dir = args.output_dir.expanduser().resolve()
        outputs = {"report": args.output_dir / "evaluation.json", "errors": args.output_dir / "errors.csv"}
        if any(path.exists() for path in outputs.values()):
            raise ValueError("evaluation output already exists; choose a new --output-dir")
        report, errors = evaluate(args)
        report["outputs"] = {key: str(path) for key, path in outputs.items()}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with outputs["errors"].open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["estimate_index", "gt_index", "estimate_timestamp_ns", "gt_timestamp_ns",
                             "dt_ns", "translation_error_m", "rotation_error_deg"])
            writer.writerows(errors)
        with outputs["report"].open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, ImportError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Trajectory evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
