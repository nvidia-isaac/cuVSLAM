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

"""Collect, compare, aggregate, and render cuVSLAM evaluation KPIs.

Runner-local port of the OSMO osmo_reporter (full_kpis_report.py). The KPI math
(ATE / ARE / Kabsch / Losts / FPS, per dataset, ODOM + SLAM) is unchanged; the
OSMO-specific output paths and the Slack webhook notification are removed so the
script runs on the GitHub Actions gpu runner with only the standard library.

The ``collect`` command converts cuvslam_app stats into machine-readable raw and
report JSON. The ``render`` and ``aggregate`` commands are the only owners of
Markdown KPI tables for PR and nightly CI publication.
"""

from argparse import ArgumentParser
import datetime
import glob
import json
import os
import re
from statistics import fmean, median, pstdev

REQUIRED_METRICS = ["ATE", "ARE", "Kabsch", "TrackingLosts", "Failed", "FPS"]
# Integer counts: shown as min–max across configurations, and never flagged
# by the rolling baseline for a change of one.
COUNT_METRICS = {"TrackingLosts", "Failed"}
REPORT_SCHEMA_VERSION = 3

# Per-sequence failure checks, overridable in the KPI config's "defaults" and
# per dataset prefix under "datasets". A failed sequence is
# counted in the report; with exclude_failed it is also left out of the
# accuracy metrics, so a few divergent trajectories cannot swing the means.
DEFAULT_SEQUENCE_CHECKS = {"max_ate_pct": 10.0, "max_lost_frame_pct": 1.0, "exclude_failed": False}
EXCLUDABLE_METRICS = ("ATE", "ARE", "Kabsch")
FAILURE_MISSING = "missing"

# Regression check against known levels in the KPI history. Many SLAM KPIs flip
# between a few discrete values run to run, so a spread around one median
# cannot tell a flip from a regression. Instead, a level is known when at least
# baseline_level_min_runs of the last baseline_window runs lie within
# max(baseline_level_pct % of the median, LEVEL_MAD_FACTOR * MAD) of it, the
# earliest at least baseline_level_min_age_days old. A KPI regresses when, in
# every configuration of a group (the two Ubuntu builds of one CUDA version run
# identical computations), it is off every known level and worse than all of
# them: a flip lands on a known level, and a
# one-off jump in one build is not confirmed by the other. A new level stays
# flagged until it is old enough to count as known. Counts use a tolerance of
# at least one. FPS depends on the runner, so it is only checked where the
# collector was told the hardware is dedicated (--flag-fps).
DEFAULT_BASELINE = {"baseline_window": 14, "baseline_level_pct": 5.0, "baseline_level_min_runs": 2,
                    "baseline_level_min_age_days": 7}
MIN_BASELINE_RUNS = 3
# The level tolerance also scales with the history's spread (median absolute
# deviation), so a KPI that jitters continuously, rather than flipping between
# discrete values, finds its ordinary values on known levels.
LEVEL_MAD_FACTOR = 2.0
COUNT_MIN_BAND = 1.0
HIGHER_IS_BETTER = {"FPS"}

DATASET_DISPLAY_ALIASES = {"TARTAN_FLAKY": "TARTAN_F"}
METRIC_UNITS = {"ATE": "%", "ARE": "º/m", "Kabsch": "", "TrackingLosts": "", "Failed": "", "FPS": "Hz"}


def display_dataset_key(key):
    """TARTAN_FLAKY-MCAM_ODOM -> TARTAN_F-MCAM_ODOM (display only)."""
    for full, short in DATASET_DISPLAY_ALIASES.items():
        if key.startswith(full + "-"):
            return short + key[len(full):]
    return key


def get_unit(metric):
    for unit_key, unit_value in METRIC_UNITS.items():
        if unit_key in metric:
            return unit_value
    return ""


def get_display_name(metric):
    """Get display name for metric (e.g., 'TrackingLosts' -> 'Losts')."""
    display_names = {
        "TrackingLosts": "Losts",
    }
    return display_names.get(metric, metric)


def parse_all_stats_json(json_path):
    """Parse the all_stats.json file.

    Returns:
        list: List of stat dictionaries
    """
    try:
        with open(json_path, 'r') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        print(f'Warning: failed to parse JSON file {json_path}: {e}')
        return None


# One KPI type per odometry mode. Keep in step with ODOMETRY_MODE_TYPES in
# tools/python_tools/cuvslam_tools/dataset_registry.py, which names the keys a
# suite is expected to produce: a mismatch reads as a missing KPI. A type must
# not contain an underscore, because parse_kpi_key splits keys on it.
ODOMETRY_MODE_TYPES = {
    'multicamera': 'MCAM',
    'mono': 'MONO',
    'inertial': 'VIO',
    'rgbd': 'RGBD',
    'multisensor': 'MSF',
}


def odometry_mode_to_type(odometry_mode):
    """Convert odometry_mode string to dataset type.

    Args:
        odometry_mode: String like "OdometryMode.Multicamera". The enum qualifier
            is optional and matching is case-insensitive, so command-line values
            like "multicamera" map the same way.

    Returns:
        str: Dataset type (MCAM, MONO, VIO, RGBD, MSF)

    Raises:
        ValueError: If the mode is unrecognized. Defaulting would file the run
            under another mode's KPI keys and overwrite them.
    """
    # Exact lookup on the unqualified name, not substring containment: a value
    # such as "notmultisensor" has to stay unmapped so the caller skips the run
    # instead of recording it as MSF.
    normalized = str(odometry_mode).rsplit('.', 1)[-1].lower()
    if normalized in ODOMETRY_MODE_TYPES:
        return ODOMETRY_MODE_TYPES[normalized]
    raise ValueError(
        f"unknown odometry_mode {odometry_mode!r}; expected one of {', '.join(ODOMETRY_MODE_TYPES)}"
    )


def load_expected_keys(path):
    """KPI keys a run is expected to produce, one per line (dataset_registry kpi-keys)."""
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _non_negative(value, where):
    if isinstance(value, bool) or safe_float(value) is None or float(value) < 0:
        raise ValueError(f"{where} must be a non-negative number")
    return float(value)


def _validated_tolerance(spec, where):
    if not isinstance(spec, dict) or len(set(spec) & {"tol_pct", "tol_abs"}) != 1 or set(spec) - {"tol_pct", "tol_abs"}:
        raise ValueError(f"{where} must be an object with exactly one of tol_pct or tol_abs")
    return {name: _non_negative(value, f"{where}.{name}") for name, value in spec.items()}


def _validated_expected(entry, where):
    if not isinstance(entry, dict):
        return {"value": _non_negative(entry, where)}
    if "value" not in entry:
        raise ValueError(f"{where} must be a number or an object with a value")
    tolerance = {name: value for name, value in entry.items() if name != "value"}
    validated = {"value": _non_negative(entry["value"], f"{where}.value")}
    if tolerance:
        validated.update(_validated_tolerance(tolerance, where))
    return validated


def _validated_settings(spec, where, *, is_defaults):
    """Validate one "defaults" or per-dataset settings object."""
    if not isinstance(spec, dict):
        raise ValueError(f"{where} must be a JSON object")
    allowed = (set(DEFAULT_SEQUENCE_CHECKS) | set(DEFAULT_BASELINE) | {"tolerances"}
               | (set() if is_defaults else {"expected"}))
    unknown = sorted(set(spec) - allowed - {"_comment"})
    if unknown:
        raise ValueError(f"{where} has unknown keys: {', '.join(unknown)}")
    settings = {}
    for key, value in spec.items():
        if key == "_comment":
            continue
        if key == "exclude_failed":
            if not isinstance(value, bool):
                raise ValueError(f"{where}.{key} must be a boolean")
            settings[key] = value
        elif key in ("baseline_window", "baseline_level_min_runs", "baseline_level_min_age_days"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{where}.{key} must be a positive integer")
            settings[key] = value
        elif key in DEFAULT_SEQUENCE_CHECKS or key in DEFAULT_BASELINE:
            settings[key] = _non_negative(value, f"{where}.{key}")
        else:
            if not isinstance(value, dict):
                raise ValueError(f"{where}.{key} must be a JSON object")
            validate = _validated_tolerance if key == "tolerances" else _validated_expected
            settings[key] = {name: validate(item, f"{where}.{key}.{name}") for name, item in value.items()}
    return settings


def load_kpi_config(path):
    """Load the KPI config file: {"defaults": {...}, "datasets": {PREFIX: {...}}}.

    Both levels hold the sequence checks (max_ate_pct, max_lost_frame_pct,
    exclude_failed) and per-metric drift "tolerances"; a dataset also holds
    calibrated "expected" values keyed <METRIC>_<TYPE>_<MODE>. An empty path or
    an unreadable file yields the built-in sequence checks and no drift data.
    A malformed file raises, because the sequence checks change KPI values.
    """
    config = {"defaults": dict(DEFAULT_SEQUENCE_CHECKS, **DEFAULT_BASELINE, tolerances={}), "datasets": {}}
    if not path:
        return config
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except OSError as e:
        print(f"Warning: failed to load KPI config {path}: {e}; using built-in sequence checks")
        return config
    if not isinstance(data, dict):
        raise ValueError(f"KPI config {path} must be a JSON object")
    unknown = sorted(set(data) - {"_comment", "defaults", "datasets"})
    if unknown:
        raise ValueError(f"KPI config {path} has unknown keys: {', '.join(unknown)}")
    config["defaults"].update(_validated_settings(data.get("defaults", {}), "defaults", is_defaults=True))
    datasets = data.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"datasets in {path} must be a JSON object")
    config["datasets"] = {
        name: _validated_settings(spec, f"datasets.{name}", is_defaults=False) for name, spec in datasets.items()
    }
    return config


def resolve_sequence_checks(kpi_config, dataset_name):
    overrides = kpi_config["datasets"].get(dataset_name, {})
    return {key: overrides.get(key, kpi_config["defaults"][key]) for key in DEFAULT_SEQUENCE_CHECKS}


def resolve_baseline_settings(kpi_config, dataset_name):
    overrides = kpi_config["datasets"].get(dataset_name, {})
    return {key: overrides.get(key, kpi_config["defaults"][key]) for key in DEFAULT_BASELINE}


def load_history(history_dir, run_id, count):
    """The last count readable kpi_<run>.json history files as [(run, {key: value})], oldest first.

    The file for run_id itself is skipped, so a rerun does not compare against itself.
    """
    paths = sorted(glob.glob(os.path.join(history_dir, "kpi_[0-9]*.json")))
    paths = [p for p in paths if os.path.basename(p) != f"kpi_{run_id}.json"]
    history = []
    for path in reversed(paths):
        if len(history) == count:
            break
        try:
            data = load_json_object(path, "KPI history")
        except (OSError, ValueError) as e:
            print(f"Warning: skipping unreadable KPI history {path}: {e}")
            continue
        history.append((os.path.basename(path)[len("kpi_"):-len(".json")], data))
    return history[::-1]


def build_baseline(current, history, kpi_config, *, flag_fps=False):
    """Per-key history series for the regression check, stored in the report."""
    prefixes = {parse_kpi_key(key)[0].split("-")[0] for key in current if parse_kpi_key(key)}
    return {
        "runs": [run for run, _ in history],
        "values": {key: [safe_float(data.get(key)) for _, data in history] for key in sorted(current)},
        "settings": {prefix: resolve_baseline_settings(kpi_config, prefix) for prefix in sorted(prefixes)},
        "flag_fps": flag_fps,
    }


def run_date(run):
    """The date of a history run named by its RUN_ID (YYYY-MM-DD), or None."""
    try:
        return datetime.date.fromisoformat(str(run)[:10])
    except ValueError:
        return None


def history_values(baseline, key, window):
    """The last window available (run date, value) pairs of one key; missing runs are skipped."""
    runs = baseline.get("runs", [])
    values = baseline.get("values", {}).get(key, [])
    pairs = [(run_date(run), value) for run, value in zip(runs, values) if value is not None]
    return pairs[-window:]


def classify_levels(metric, tonight, reference, today, settings):
    """Compare tonight's values of one group against the known levels of its history.

    tonight holds one value per configuration of the group; reference pools the
    (run date, value) history of all of them. Returns None while no level is
    known yet, else a dict whose status is "regressed" (every value off all
    known levels and worse than all of them), "better" (the same, but better),
    or "known".
    """
    if len(reference) < MIN_BASELINE_RUNS:
        return None
    center = median(value for _, value in reference)
    spread = median(abs(value - center) for _, value in reference)
    tolerance = max(settings["baseline_level_pct"] / 100.0 * abs(center), LEVEL_MAD_FACTOR * spread)
    if metric in COUNT_METRICS:
        tolerance = max(tolerance, COUNT_MIN_BAND)

    def near(value):
        return [date for date, other in reference if abs(other - value) <= tolerance]

    def known(value):
        dates = near(value)
        if len(dates) < settings["baseline_level_min_runs"]:
            return False
        dated = [date for date in dates if date is not None]
        return not dated or (today - min(dated)).days >= settings["baseline_level_min_age_days"]

    levels = [value for _, value in reference if known(value)]
    if not levels:
        return None
    higher_is_better = metric in HIGHER_IS_BETTER
    worst = min(levels) if higher_is_better else max(levels)
    best = max(levels) if higher_is_better else min(levels)
    status = "known"
    if not any(known(value) for value in tonight):
        if all((value < worst - tolerance) if higher_is_better else (value > worst + tolerance) for value in tonight):
            status = "regressed"
        elif all((value > best + tolerance) if higher_is_better else (value < best - tolerance) for value in tonight):
            status = "better"
    first_seen = sorted(date for value in tonight for date in near(value) if date is not None)
    return {"status": status, "values": list(tonight), "worst": worst, "best": best,
            "since": first_seen[0] if first_seen else today}


def resolve_tolerance(kpi_config, dataset_name, metric):
    overrides = kpi_config["datasets"].get(dataset_name, {}).get("tolerances", {})
    return overrides.get(metric, kpi_config["defaults"]["tolerances"].get(metric, {}))


def calibrated_values(kpi_config):
    """{full KPI key: {"value": ..., optional tolerance}} across all datasets."""
    return {
        f"{name}_{suffix}": entry
        for name, settings in kpi_config["datasets"].items()
        for suffix, entry in settings.get("expected", {}).items()
    }


def sequence_failure(stat, checks, *, check_ate=True):
    """Return the failure kind for one sequence run, or None if it passed.

    Kinds: no_frames, lost_frames, no_ate, ate.
    """
    n_frames = safe_float(stat.get("n_frames")) or 0
    if n_frames <= 0:
        return "no_frames"
    losts = max(safe_float(stat.get("num_tracking_losts")) or 0, 0)
    if 100.0 * losts / n_frames > checks["max_lost_frame_pct"]:
        return "lost_frames"
    if not check_ate:
        return None
    ate = safe_float(stat.get("gt_av_translation_error"))
    if ate is None:
        return "no_ate"
    if ate > checks["max_ate_pct"]:
        return "ate"
    return None


def safe_float(value):
    """Best-effort float conversion. Returns None for None, NaN, or non-numeric
    values (e.g. a malformed string in the committed ranges file)."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # drop NaN


def evaluate_drift(kpis_dict, kpi_config, expected_keys=None):
    """Compare each KPI against its expected value +/- tolerance.

    Checks every key in expected_keys (the keys a run should produce; the keys
    this run produced if None) and every calibrated key. The tolerance is the
    calibrated entry's own, else the dataset's for that metric, else the
    default's.

    Soft check only: returns a list of (key, status, detail) rows and never
    raises. Statuses: WITHIN (in range), DRIFT (out of range), SKIPPED
    (uncalibrated/malformed), MISSING (no value this run).
    """
    calibrated = calibrated_values(kpi_config)
    keys = set(kpis_dict if expected_keys is None else expected_keys) | set(calibrated)
    rows = []
    for key in sorted(keys):
        try:
            if key not in kpis_dict:
                rows.append((key, "MISSING", "no value produced this run"))
                continue
            actual = safe_float(kpis_dict[key])
            if actual is None:
                rows.append((key, "SKIPPED", f"non-numeric actual value: {kpis_dict[key]!r}"))
                continue
            entry = calibrated.get(key, {})
            raw_expected = entry.get("value")
            parsed = parse_kpi_key(key)
            spec = resolve_tolerance(kpi_config, parsed[0].split("-")[0], parsed[1]) if parsed else {}
            if "tol_pct" in entry or "tol_abs" in entry:
                spec = {name: entry[name] for name in ("tol_pct", "tol_abs") if name in entry}
            expected = safe_float(raw_expected)
            if expected is None:
                detail = (f"uncalibrated (actual={actual:.4g})" if raw_expected is None
                          else f"non-numeric expected={raw_expected!r} (actual={actual:.4g})")
                rows.append((key, "SKIPPED", detail))
                continue
            if spec.get("tol_abs") is not None:
                tol = safe_float(spec.get("tol_abs"))
            else:
                tol_pct = safe_float(spec.get("tol_pct"))
                tol = None if tol_pct is None else abs(expected) * tol_pct / 100.0
            tol = abs(tol) if tol is not None else 0.0
            low, high = expected - tol, expected + tol
            status = "WITHIN" if low <= actual <= high else "DRIFT"
            rows.append((key, status, f"actual={actual:.4g} expected={expected:.4g} range=[{low:.4g}, {high:.4g}]"))
        except Exception as exc:
            rows.append((key, "SKIPPED", f"error evaluating spec {calibrated.get(key)!r}: {exc}"))
    return rows


def summarize_mode(stats, dataset_type, checks):
    """Compute one mode's KPIs and per-sequence check results.

    Losts and FPS always cover every sequence. With exclude_failed, the
    accuracy metrics cover only the passing ones, unless none passed.
    """
    runs = []
    for stat in stats:
        n_frames = int(safe_float(stat.get("n_frames")) or 0)
        losts = max(int(safe_float(stat.get("num_tracking_losts")) or 0), 0)
        runs.append({
            "sequence": stat.get("sequence_title", ""),
            "n_frames": n_frames,
            "tracking_losts": losts,
            "lost_frame_pct": 100.0 * losts / n_frames if n_frames > 0 else None,
            "ate": safe_float(stat.get("gt_av_translation_error")),
            "failure": sequence_failure(stat, checks, check_ate=dataset_type != "MONO"),
        })

    accuracy_stats = stats
    excluded = []
    if checks["exclude_failed"]:
        passing = [stat for stat, run in zip(stats, runs) if run["failure"] is None]
        if passing:
            accuracy_stats = passing
            excluded = [run["sequence"] for run in runs if run["failure"] is not None]

    def mean(field, subset):
        values = [value for value in (safe_float(s.get(field)) for s in subset) if value is not None]
        return fmean(values) if values else None

    kpis = {
        "ATE": mean("gt_av_translation_error", accuracy_stats),
        "ARE": mean("gt_av_rotation_error", accuracy_stats),
        "Kabsch": mean("gt_simple_error", accuracy_stats),
        "FPS": mean("average_fps", stats),
        "Failed": sum(1 for run in runs if run["failure"] is not None),
        "TrackingLosts": sum(
            s.get("num_tracking_losts", 0) for s in stats if s.get("num_tracking_losts", -1) >= 0
        ),
    }
    return kpis, {"checks": checks, "excluded": excluded, "runs": runs}


def process_dataset_folder(dataset_folder_path, kpi_config=None):
    """Process a dataset folder to extract metrics for ODOM and SLAM.

    Args:
        dataset_folder_path: Path to dataset folder (e.g., kitti-vio_slam_gt)
        kpi_config: Result of load_kpi_config; built-in sequence checks if None.

    Returns:
        tuple: (flat KPI dict, {row key: sequence check block}), or None.
    """
    if kpi_config is None:
        kpi_config = load_kpi_config("")
    dataset_name = os.path.basename(dataset_folder_path).split('-')[0].upper()

    timestamped_folders = glob.glob(os.path.join(dataset_folder_path, '*'))
    timestamped_folders = [f for f in timestamped_folders if os.path.isdir(f)]

    if not timestamped_folders:
        print(f'Warning: no timestamped folders found in {dataset_folder_path}')
        return None

    latest_folder = max(timestamped_folders, key=os.path.getmtime)
    stats_folder = os.path.join(latest_folder, 'stats')

    if not os.path.exists(stats_folder):
        print(f'Warning: stats folder not found in {latest_folder}')
        return None

    out_dict = {}

    all_stats_json = os.path.join(stats_folder, 'all_stats.json')
    if not os.path.exists(all_stats_json):
        print(f'Warning: all_stats.json not found in {stats_folder}')
        return None

    all_stats = parse_all_stats_json(all_stats_json)
    if not all_stats:
        print(f'Warning: failed to parse all_stats.json in {stats_folder}')
        return None

    # The mode is the only trustworthy source of the KPI type. Guessing it from
    # the folder name mislabels every config whose name mentions another mode,
    # such as kitti-vio_slam_gt run as multicamera.
    if 'odometry_mode' not in all_stats[0]:
        print(f'Warning: odometry_mode not found in {all_stats_json}; cannot name this run\'s KPI keys')
        return None

    try:
        dataset_type = odometry_mode_to_type(all_stats[0]['odometry_mode'])
    except ValueError as exc:
        print(f'Warning: {exc} in {all_stats_json}')
        return None
    print(f'  Detected dataset type: {dataset_type} (from odometry_mode: {all_stats[0]["odometry_mode"]})')

    checks = resolve_sequence_checks(kpi_config, dataset_name)
    sequences = {}
    for mode in ("ODOM", "SLAM"):
        mode_stats = [s for s in all_stats if mode in s.get('sequence_title', '').upper()]
        if not mode_stats:
            continue
        kpis, block = summarize_mode(mode_stats, dataset_type, checks)
        for metric, value in kpis.items():
            out_dict[f"{dataset_name}_{metric}_{dataset_type}_{mode}"] = value
        sequences[f"{dataset_name}-{dataset_type}_{mode}"] = block

    return out_dict, sequences


def parse_kpi_key(key):
    """Return (dataset row key, metric), or None for an unknown key."""
    parts = key.split("_")
    if len(parts) < 4:
        return None
    mode = parts[-1]
    dataset_type = parts[-2]
    metric = parts[-3]
    dataset_name = "_".join(parts[:-3])
    if metric not in REQUIRED_METRICS:
        return None
    return f"{dataset_name}-{dataset_type}_{mode}", metric


def format_metric(metric, value):
    """Format one configuration's KPI value."""
    if value == "NA":
        return value
    if metric in COUNT_METRICS:
        return str(int(value))
    if metric == "FPS":
        return f"{float(value):.1f}"
    return f"{float(value):.4f}"


def format_diff(metric, value):
    """Format one configuration's change from the previous run, with its sign."""
    if metric in COUNT_METRICS:
        return f"{int(value):+d}"
    if metric == "FPS":
        return f"{float(value):+.1f}"
    return f"{float(value):+.4f}"


def organize_data(data, required_metrics=REQUIRED_METRICS, prev_data=None):
    """Organize flat KPI dictionaries into dataset rows for a single config."""
    if prev_data is not None and not isinstance(prev_data, dict):
        raise ValueError(f"previous KPI data must be a dictionary, got {type(prev_data).__name__}")

    organized_data = {}
    for key in sorted(data):
        parsed = parse_kpi_key(key)
        if parsed is None:
            continue
        dataset_key, metric = parsed
        metrics = organized_data.setdefault(dataset_key, {})
        value = safe_float(data[key])
        if value is None:
            metrics[metric] = "NA"
            metrics["diff " + metric] = "NA"
            continue

        if metric in COUNT_METRICS:
            metrics[metric] = int(value)
        elif metric == "FPS":
            metrics[metric] = round(value, 1)
        else:
            metrics[metric] = round(value, 4)

        if "MONO" in dataset_key and metric == "ATE":
            metrics["diff " + metric] = "NA"
        elif prev_data is not None and safe_float(prev_data.get(key)) is not None:
            difference = value - safe_float(prev_data[key])
            metrics["diff " + metric] = int(difference) if metric in COUNT_METRICS else round(difference, 4)
        else:
            metrics["diff " + metric] = "NA"

    for dataset_key, metrics in organized_data.items():
        for metric in required_metrics:
            metrics.setdefault(metric, "NA")
            if "MONO" in dataset_key and metric == "ATE":
                metrics[metric] = "NA"

    return organized_data


def column_title(metric, *, aggregate=False):
    title = get_display_name(metric)
    unit = get_unit(metric)
    if unit:
        title += f", {unit}"
    if aggregate:
        title += " (min–max)" if metric in COUNT_METRICS else " (mean ± σ)"
    return title


def create_table(organized_data, required_metrics=REQUIRED_METRICS, *, config=None, aggregate=False, diffs=True,
                 extra=None):
    """Render an already-organized KPI dictionary as Markdown.

    With diffs, each value that has a change from the previous run gets it on a
    second line of the same cell, keeping the table one column per KPI. An
    unchanged count gets no second line. extra, if given, is (column title,
    {evaluation row: cell}) placed right after Failed.
    """
    headers = (["Config"] if config else []) + ["Evaluation"]
    for metric in required_metrics:
        headers.append(column_title(metric, aggregate=aggregate))
        if extra and metric == "Failed":
            headers.append(extra[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]

    for dataset in sorted(organized_data):
        metrics = organized_data[dataset]
        missing = [metric for metric in required_metrics if metric not in metrics]
        if missing:
            raise ValueError(f"{dataset} is missing table columns: {', '.join(missing)}")
        cells = ([config] if config else []) + [display_dataset_key(dataset)]
        for metric in required_metrics:
            cell = str(metrics[metric]) if aggregate else format_metric(metric, metrics[metric])
            diff = metrics.get("diff " + metric, "NA")
            if diffs and diff != "NA":
                text = str(diff)
                if not (metric in COUNT_METRICS and text in ("+0", "+0/+0")):
                    cell += f"<br><sub>Δ {text}</sub>"
            cells.append(cell)
            if extra and metric == "Failed":
                cells.append(extra[1].get(dataset, "NA"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def classify_sequences(config_reports):
    """Classify each sequence across configurations.

    A sequence that fails, or is missing, in every configuration is broken; one
    that fails in some is flaky. Returns {dataset row: {"broken": [...],
    "flaky": [...], "excluded": bool}}, each list entry a dict with the
    sequence name, the failed count, and the failed runs.
    """
    configs = [config for config, _ in config_reports]
    runs_by_row = {}
    excluding_rows = set()
    for config, report in config_reports:
        for row, block in report["sequences"].items():
            if block.get("checks", {}).get("exclude_failed"):
                excluding_rows.add(row)
            for run in block.get("runs", []):
                runs_by_row.setdefault(row, {}).setdefault(run["sequence"], {})[config] = run

    classified = {}
    for row, sequences in runs_by_row.items():
        result = {"broken": [], "flaky": [], "excluded": row in excluding_rows}
        for name in sorted(sequences):
            by_config = sequences[name]
            failed = []
            for config in configs:
                run = by_config.get(config)
                if run is None:
                    failed.append({"config": config, "failure": FAILURE_MISSING})
                elif run.get("failure"):
                    failed.append(dict(run, config=config))
            if not failed:
                continue
            entry = {"sequence": name, "failed": len(failed), "runs": failed}
            result["broken" if len(failed) == len(configs) else "flaky"].append(entry)
        classified[row] = result
    return classified


def describe_failed_runs(runs):
    """One-line summary of why a sequence's runs failed, e.g. "ATE 41.8–52.9%"."""
    parts = []
    kinds = {run["failure"] for run in runs}
    ates = [run["ate"] for run in runs if run["failure"] == "ate" and run.get("ate") is not None]
    if ates:
        low, high = min(ates), max(ates)
        parts.append(f"ATE {low:.1f}%" if round(low, 1) == round(high, 1) else f"ATE {low:.1f}–{high:.1f}%")
    lost = [run["lost_frame_pct"] for run in runs
            if run["failure"] == "lost_frames" and run.get("lost_frame_pct") is not None]
    if lost:
        parts.append(f"lost {max(lost):.1f}% of frames")
    labels = {"no_frames": "no frames", "no_ate": "no ATE", FAILURE_MISSING: "no result"}
    parts.extend(labels[kind] for kind in sorted(kinds) if kind in labels)
    return ", ".join(parts)


def render_failed_sequences(classified, config_count):
    """Markdown list of broken and flaky sequences, or "" if there are none."""
    lines = []
    for row in sorted(classified):
        for status in ("broken", "flaky"):
            for entry in classified[row][status]:
                label = status if config_count > 1 else "failed"
                if config_count > 1:
                    label += f" {entry['failed']}/{config_count}"
                excluded = " (excluded from accuracy KPIs)" if classified[row]["excluded"] else ""
                lines.append(
                    f"- `{display_dataset_key(row)}` {entry['sequence']}: {label}, "
                    f"{describe_failed_runs(entry['runs'])}{excluded}"
                )
    if not lines:
        return ""
    return "\n<details><summary>Failed sequences</summary>\n\n" + "\n".join(lines) + "\n\n</details>\n"


def sequence_check_note(classified):
    rows = sorted({display_dataset_key(row).split("-")[0] for row, c in classified.items() if c["excluded"]})
    if not rows:
        return ""
    return f" Failed sequences are excluded from ATE, ARE and Kabsch for {', '.join(rows)}."


def dataset_sort_key(folder):
    name = os.path.basename(folder).lower()
    if "mono" in name:
        return (0, name)
    if "stereo" in name and "vio" not in name:
        return (1, name)
    if "vio" in name:
        return (2, name)
    if "rgbd" in name:
        return (3, name)
    return (4, name)


def collect_kpis(stat_folder, kpi_config=None):
    """Compute a flat KPI dictionary and per-row sequence checks from a cuvslam_app stats directory."""
    if not os.path.isdir(stat_folder):
        raise ValueError(f"Stat folder does not exist: {stat_folder}")
    dataset_folders = sorted(
        (
            os.path.join(stat_folder, name)
            for name in os.listdir(stat_folder)
            if os.path.isdir(os.path.join(stat_folder, name))
        ),
        key=dataset_sort_key,
    )
    if not dataset_folders:
        raise ValueError(f"No dataset folders found in: {stat_folder}")

    kpis = {}
    sequences = {}
    key_source = {}
    print(f"Processing {len(dataset_folders)} dataset folders...")
    for dataset_folder in dataset_folders:
        folder_name = os.path.basename(dataset_folder)
        print(f"  Processing: {folder_name}")
        processed = process_dataset_folder(dataset_folder, kpi_config)
        if not processed:
            continue
        result, result_sequences = processed
        if not result:
            continue
        # The dataset prefix is the first hyphen-delimited token of the reporter
        # config name, so two configs that share it produce identical keys and one
        # would silently replace the other's whole report.
        collisions = sorted(set(result) & set(kpis))
        if collisions:
            sources = ", ".join(sorted({key_source[key] for key in collisions}))
            raise ValueError(
                f"KPI key collision: '{folder_name}' produces keys already produced by {sources}: "
                f"{', '.join(collisions)}. Give one of the reporter configs a distinct dataset "
                "prefix, keeping underscores inside it (tartan_flaky-... not tartan-flaky-...)."
            )
        kpis.update(result)
        sequences.update(result_sequences)
        key_source.update(dict.fromkeys(result, folder_name))
    if not kpis:
        raise ValueError("output KPI JSON is empty; check the input stats format")
    return kpis, sequences


def load_json_object(path, description):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return data


def write_json(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def write_text(path, text):
    if path == "-":
        print(text, end="")
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def build_report(run_id, current, previous=None, kpi_config=None, sequences=None, expected_keys=None,
                 baseline=None):
    drift = []
    if kpi_config is not None:
        drift = [
            {"key": key, "status": status, "detail": detail}
            for key, status, detail in evaluate_drift(current, kpi_config, expected_keys)
        ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "current": current,
        "previous": previous,
        "drift": drift,
        "sequences": sequences or {},
        "baseline": baseline or {"runs": [], "values": {}, "settings": {}},
    }


def load_report(path):
    report = load_json_object(path, "KPI report")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported KPI report schema in {path}: "
            f"{report.get('schema_version')!r} (expected {REPORT_SCHEMA_VERSION})"
        )
    if not isinstance(report.get("current"), dict):
        raise ValueError(f"KPI report current field must be an object: {path}")
    if report.get("previous") is not None and not isinstance(report["previous"], dict):
        raise ValueError(f"KPI report previous field must be an object or null: {path}")
    if not isinstance(report.get("drift"), list):
        raise ValueError(f"KPI report drift field must be an array: {path}")
    if not isinstance(report.get("sequences"), dict):
        raise ValueError(f"KPI report sequences field must be an object: {path}")
    if not isinstance(report.get("baseline"), dict):
        raise ValueError(f"KPI report baseline field must be an object: {path}")
    return report


def stored_baseline_settings(baseline, dataset_name):
    """A report's baseline settings for one dataset, with defaults for keys older reports lack."""
    stored = baseline.get("settings", {}).get(dataset_name)
    return {**DEFAULT_BASELINE, **stored} if stored else None


def config_group(config):
    """The configurations that run identical computations share a group: one per CUDA version."""
    match = re.search(r"cuda(\d+\.\d+)", config)
    return f"CUDA {match.group(1)}" if match else config


def report_date(report):
    return run_date(report.get("run_id")) or datetime.date.today()


def group_level_results(key, members, settings, today):
    """Known-level results per group for one key; members are (config, report) pairs."""
    metric = parse_kpi_key(key)[1]
    groups = {}
    for config, report in members:
        groups.setdefault(config_group(config), []).append(report)
    results = []
    for group, reports in sorted(groups.items()):
        tonight = [safe_float(report["current"].get(key)) for report in reports]
        if any(value is None for value in tonight):
            continue
        reference = [pair for report in reports
                     for pair in history_values(report["baseline"], key, settings["baseline_window"])]
        result = classify_levels(metric, tonight, reference, today, settings)
        if result:
            results.append(dict(result, group=group, configs=len(reports)))
    return results


def baseline_results(config_reports):
    """Per-key Δ from the baseline median and known-level regressions, for one or more configurations.

    With several configurations, Δ compares the mean (min/max for counts) with
    the median of the same reduction over the history runs every configuration
    has; regressions are judged per group, never on the reduced values.
    """
    reference = config_reports[0][1]
    flag_fps = all(report["baseline"].get("flag_fps", False) for _, report in config_reports)
    run_lists = [report["baseline"].get("runs", []) for _, report in config_reports]
    common = [run for run in run_lists[0] if all(run in runs for runs in run_lists[1:])]
    today = report_date(reference)
    results = []
    for key in sorted(reference["current"]):
        parsed = parse_kpi_key(key)
        settings = stored_baseline_settings(reference["baseline"], parsed[0].split("-")[0]) if parsed else None
        currents = [safe_float(report["current"].get(key)) for _, report in config_reports]
        if not settings or any(value is None for value in currents):
            continue
        metric = parsed[1]
        per_run = []
        for run in common:
            values = []
            for (_, report), runs in zip(config_reports, run_lists):
                history = report["baseline"].get("values", {}).get(key, [])
                index = runs.index(run)
                values.append(history[index] if index < len(history) else None)
            if all(value is not None for value in values):
                per_run.append(values)
        per_run = per_run[-settings["baseline_window"]:]
        if len(per_run) < MIN_BASELINE_RUNS:
            continue
        if metric in COUNT_METRICS:
            delta = (f"{round(min(currents) - median(min(v) for v in per_run)):+d}/"
                     f"{round(max(currents) - median(max(v) for v in per_run)):+d}")
            if len(config_reports) == 1:
                delta = delta.split("/")[1]
        else:
            delta = format_diff(metric, fmean(currents) - median(fmean(v) for v in per_run))
        levels = [] if metric == "FPS" and not flag_fps else group_level_results(key, config_reports, settings, today)
        results.append({"key": key, "delta": delta, "levels": levels,
                        "regressed": any(level["status"] == "regressed" for level in levels)})
    return results


def apply_baseline_deltas(organized, results):
    """Set each cell's Δ to its change from the baseline median, ⚠ on a regression."""
    for metrics in organized.values():
        for metric in REQUIRED_METRICS:
            metrics["diff " + metric] = "NA"
    for result in results:
        row, metric = parse_kpi_key(result["key"])
        if row in organized and organized[row].get(metric) != "NA":
            organized[row]["diff " + metric] = result["delta"] + (" ⚠" if result["regressed"] else "")
    return organized


def _level_lines(results, status):
    lines = []
    for result in results:
        row, metric = parse_kpi_key(result["key"])
        for level in result["levels"]:
            if level["status"] != status:
                continue
            bound = level["worst"] if status == "regressed" else level["best"]
            values = ", ".join(format_metric(metric, value) for value in level["values"])
            lines.append(
                f"- `{display_dataset_key(row)}` {get_display_name(metric)}, {level['group']}: {values} vs "
                f"{'worst' if status == 'regressed' else 'best'} known level {format_metric(metric, bound)} "
                f"(new level since {level['since']})"
            )
    return lines


def render_regressions(results):
    """Regressed KPIs (the cells marked ⚠) and, collapsed, accuracy KPIs at a new better level."""
    if not results:
        return "\n_Regression check: not enough KPI history yet._\n"
    regressed = _level_lines(results, "regressed")
    text = (f"\n**Regressions** ({len(regressed)}, marked ⚠ above):\n\n" + "\n".join(regressed) + "\n"
            if regressed else f"\n_Regression check: no regressions in {len(results)} KPIs._\n")
    better = _level_lines([r for r in results if parse_kpi_key(r["key"])[1] in EXCLUDABLE_METRICS], "better")
    if better:
        text += "\n<details><summary>New better levels</summary>\n\n" + "\n".join(better) + "\n\n</details>\n"
    return text


def baseline_note(config_reports):
    """One sentence describing Δ and ⚠ with the settings in use."""
    settings = {}
    for _, report in config_reports:
        settings.update(report["baseline"].get("settings", {}))
    if not settings:
        return "Δ needs KPI history"
    s = {**DEFAULT_BASELINE, **next(iter(settings.values()))}
    scope = ("in both builds of a CUDA version" if len(config_reports) > 1 else "in this configuration")
    fps = "" if all(r["baseline"].get("flag_fps") for _, r in config_reports) else "; FPS is not checked here"
    return (f"Δ is the change from the median of the last {s['baseline_window']} runs. ⚠ marks a KPI that, {scope}, "
            f"is more than {s['baseline_level_pct']:g}% off every level seen before (in at least "
            f"{s['baseline_level_min_runs']} runs, first at least {s['baseline_level_min_age_days']} days ago) and "
            f"worse than all of them{fps}")


def render_report(report, config):
    results = baseline_results([(config, report)])
    organized = apply_baseline_deltas(organize_data(report["current"]), results)
    classified = classify_sequences([(config, report)])
    settings_note = baseline_note([(config, report)]) + ", against main's nightly history"
    note = f"{settings_note}. Failed counts sequences that failed.{sequence_check_note(classified)}"
    return (create_table(organized, config=config) + f"\n_{note}_\n"
            + render_regressions(results) + render_failed_sequences(classified, 1))


def render_drift(report):
    lines = ["KPI drift check (soft, informational only)"]
    for row in report["drift"]:
        lines.append(f"  [{row['status']:7}] {row['key']}: {row['detail']}")
    n_drift = sum(row["status"] == "DRIFT" for row in report["drift"])
    n_calibrated = sum(row["status"] in ("WITHIN", "DRIFT") for row in report["drift"])
    if report["drift"] and n_calibrated == 0:
        lines.append("  (no calibrated KPIs yet; seed expected values in the ranges file)")
    elif n_drift:
        lines.append(f"  {n_drift} KPI(s) outside expected range (not failing the job).")
    return "\n".join(lines) + "\n"


def require_numeric(value, key, config):
    number = safe_float(value)
    if number is None:
        raise ValueError(f"non-numeric KPI {key!r} for configuration {config!r}: {value!r}")
    return number


def format_distribution(metric, values):
    if metric in COUNT_METRICS:
        low, high = int(min(values)), int(max(values))
        return str(low) if low == high else f"{low}–{high}"
    mean = fmean(values)
    deviation = pstdev(values)
    if metric == "FPS":
        return f"{mean:.1f} ± {deviation:.1f}"
    return f"{mean:.4f} ± {deviation:.4f}"


def aggregate_reports(config_reports):
    """Aggregate matching KPI reports across build configurations."""
    if not config_reports:
        raise ValueError("at least one configuration report is required")

    reference_config, reference_report = config_reports[0]
    reference_keys = set(reference_report["current"])
    for config, report in config_reports[1:]:
        keys = set(report["current"])
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(
                f"KPI keys differ for {config!r} vs {reference_config!r}; "
                f"missing={missing}, extra={extra}"
            )

    organized = {}
    for key in sorted(reference_keys):
        parsed = parse_kpi_key(key)
        if parsed is None:
            continue
        dataset_key, metric = parsed
        metrics = organized.setdefault(dataset_key, {})
        if "MONO" in dataset_key and metric == "ATE":
            metrics[metric] = "NA"
            metrics["diff " + metric] = "NA"
            continue

        # null means the run had no valid measurement for this KPI.
        if any(report["current"][key] is None for _, report in config_reports):
            metrics[metric] = "NA"
            metrics["diff " + metric] = "NA"
            continue
        current_values = [
            require_numeric(report["current"][key], key, config) for config, report in config_reports
        ]
        metrics[metric] = format_distribution(metric, current_values)

    for metrics in organized.values():
        for metric in REQUIRED_METRICS:
            metrics.setdefault(metric, "NA")
            metrics.setdefault("diff " + metric, "NA")
    return organized


def render_aggregate_report(config_reports, *, values_only=False):
    """Render the cross-configuration table, with its notes below it.

    values_only drops the Δ lines, the Broken / flaky column, the regression
    list, and the failed-sequence list, for release notes.
    """
    organized = aggregate_reports(config_reports)
    classified = classify_sequences(config_reports)
    count = len(config_reports)
    configs = ", ".join(f"`{config}`" for config, _ in config_reports)
    note = (
        f"Across {count} configuration{'s' if count != 1 else ''}: {configs}. Values are mean ± population σ; "
        "Losts and Failed are min–max. Failed counts the sequences that failed in a configuration"
    )
    if values_only:
        note += "." + sequence_check_note(classified)
        return create_table(organized, aggregate=True, diffs=False) + f"\n_{note}_\n"
    results = baseline_results(config_reports)
    apply_baseline_deltas(organized, results)
    settings_note = baseline_note(config_reports)
    note += (
        "; of those, a broken sequence failed in every configuration and a flaky one in some. "
        f"{settings_note}. For Losts and Failed, Δ is of min/max."
        + sequence_check_note(classified)
    )
    broken_flaky = (
        "Broken / flaky",
        {row: f"{len(c['broken'])} / {len(c['flaky'])}" for row, c in classified.items()},
    )
    return (create_table(organized, aggregate=True, extra=broken_flaky) + f"\n_{note}_\n"
            + render_regressions(results)
            + render_failed_sequences(classified, count))


def parse_config_report(value):
    if "=" not in value:
        raise ValueError(f"configuration input must have CONFIG=PATH form: {value!r}")
    config, path = value.split("=", 1)
    if not config or not path:
        raise ValueError(f"configuration input must have CONFIG=PATH form: {value!r}")
    return config, load_report(path)


def collect_command(args):
    print("=============================\nKPI collector is up!")
    kpi_config = load_kpi_config(args.baseline_ranges)
    current, sequences = collect_kpis(args.stat_folder, kpi_config)
    previous = load_json_object(args.prev_kpi, "previous KPI data") if args.prev_kpi else None
    expected_keys = load_expected_keys(args.expected_keys) if args.expected_keys else None
    baseline = None
    if args.history and os.path.isdir(args.history):
        windows = [kpi_config["defaults"]["baseline_window"]] + [
            settings["baseline_window"] for settings in kpi_config["datasets"].values() if "baseline_window" in settings
        ]
        history = load_history(args.history, args.run_id, max(windows))
        print(f"Rolling baseline: {len(history)} KPI history run(s) from {args.history}")
        baseline = build_baseline(current, history, kpi_config, flag_fps=args.flag_fps)
    report = build_report(args.run_id, current, previous, kpi_config, sequences, expected_keys, baseline)
    write_json(args.out_kpi_json, current)
    write_json(args.out_report_json, report)
    print(f"Raw KPI JSON saved at {args.out_kpi_json}")
    print(f"KPI report JSON saved at {args.out_report_json}")


def render_command(args):
    write_text(args.output, render_report(load_report(args.report_json), args.config))


def aggregate_command(args):
    config_reports = [parse_config_report(value) for value in args.input]
    configs = [config for config, _ in config_reports]
    if len(set(configs)) != len(configs):
        raise ValueError(f"configuration names must be unique: {configs}")
    write_text(args.output, render_aggregate_report(config_reports, values_only=args.values_only))


def drift_command(args):
    write_text(args.output, render_drift(load_report(args.report_json)))


def build_argument_parser():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="collect stats into raw and report JSON")
    collect.add_argument("-s", "--stat_folder", required=True)
    collect.add_argument("-j", "--out_kpi_json", required=True)
    collect.add_argument("-r", "--out_report_json", required=True)
    collect.add_argument("-d", "--run_id", default="")
    collect.add_argument("-k", "--prev_kpi", default="")
    collect.add_argument("-b", "--baseline_ranges", default="")
    collect.add_argument("--flag-fps", action="store_true",
                         help="also check FPS for regressions; only meaningful on dedicated hardware")
    collect.add_argument("-H", "--history", default="",
                         help="KPI history directory of kpi_<run>.json files for the rolling-baseline check")
    collect.add_argument("-e", "--expected_keys", default="",
                         help="file listing the KPI keys the run should produce, one per line; "
                              "keys missing from the run are reported as MISSING")
    collect.set_defaults(func=collect_command)

    render = commands.add_parser("render", help="render one configuration report as Markdown")
    render.add_argument("-r", "--report_json", required=True)
    render.add_argument("-c", "--config", default="")
    render.add_argument("-o", "--output", default="-")
    render.set_defaults(func=render_command)

    aggregate = commands.add_parser("aggregate", help="aggregate configuration reports as Markdown")
    aggregate.add_argument("-i", "--input", action="append", required=True, metavar="CONFIG=PATH")
    aggregate.add_argument("-o", "--output", default="-")
    aggregate.add_argument("--values-only", action="store_true",
                           help="omit diffs and broken/flaky sequence counts (release notes)")
    aggregate.set_defaults(func=aggregate_command)

    drift = commands.add_parser("drift", help="render the soft drift report as text")
    drift.add_argument("-r", "--report_json", required=True)
    drift.add_argument("-o", "--output", default="-")
    drift.set_defaults(func=drift_command)
    return parser


def main():
    args = build_argument_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
