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
import glob
import json
import os
from statistics import fmean, pstdev

NO_DIFF_METRICS = {"FPS"}
REQUIRED_METRICS = ["ATE", "ARE", "Kabsch", "TrackingLosts", "FPS"]
REPORT_SCHEMA_VERSION = 1

DATASET_DISPLAY_ALIASES = {"TARTAN_FLAKY": "TARTAN_F"}
METRIC_UNITS = {"ATE": "%", "ARE": "º/m", "Kabsch": "", "TrackingLosts": "", "FPS": "Hz"}


def display_dataset_key(key):
    """TARTAN_FLAKY-STEREO_ODOM -> TARTAN_F-STEREO_ODOM (display only)."""
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
        "diff TrackingLosts": "diff Losts"
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


def odometry_mode_to_type(odometry_mode):
    """Convert odometry_mode string to dataset type.

    Args:
        odometry_mode: String like "OdometryMode.Multicamera". Matching is
            case-insensitive, so command-line values like "multicamera" map the
            same way. None or non-string values fall back to STEREO.

    Returns:
        str: Dataset type (MONO, STEREO, VIO, RGBD)
    """
    normalized = str(odometry_mode).lower()
    if 'multicamera' in normalized:
        return 'STEREO'
    elif 'mono' in normalized:
        return 'MONO'
    elif 'inertial' in normalized:
        return 'VIO'
    elif 'rgbd' in normalized:
        return 'RGBD'
    else:
        return 'STEREO'


def load_baseline_ranges(path):
    """Load the committed baseline-ranges json.

    Returns the dict of per-KPI range specs (the "kpis" block, or the top-level
    dict if no "kpis" key), or an empty dict on any error. Never raises.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load baseline ranges {path}: {e}")
        return {}
    if not isinstance(data, dict):
        print(f"Warning: baseline ranges {path} is not a json object; skipping drift check")
        return {}
    ranges = data.get("kpis", data)
    return ranges if isinstance(ranges, dict) else {}


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


def evaluate_drift(kpis_dict, ranges):
    """Compare each computed KPI against its expected value +/- tolerance.

    Soft check only: returns a list of (key, status, detail) rows and never
    raises, even on malformed baseline entries. Statuses: WITHIN (in range),
    DRIFT (out of range), SKIPPED (uncalibrated/malformed), MISSING (no value
    this run).
    """
    rows = []
    for key in sorted(ranges):
        try:
            spec = ranges[key] if isinstance(ranges[key], dict) else {}
            if key not in kpis_dict:
                rows.append((key, "MISSING", "no value produced this run"))
                continue
            actual = safe_float(kpis_dict[key])
            if actual is None:
                rows.append((key, "SKIPPED", f"non-numeric actual value: {kpis_dict[key]!r}"))
                continue
            raw_expected = spec.get("expected")
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
            rows.append((key, "SKIPPED", f"error evaluating spec {ranges.get(key)!r}: {exc}"))
    return rows


def process_dataset_folder(dataset_folder_path):
    """Process a dataset folder to extract metrics for ODOM and SLAM.

    Args:
        dataset_folder_path: Path to dataset folder (e.g., kitti-vio_slam_gt)

    Returns:
        dict: Dictionary with dataset metrics
    """
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

    if all_stats and 'odometry_mode' in all_stats[0]:
        dataset_type = odometry_mode_to_type(all_stats[0]['odometry_mode'])
        print(f'  Detected dataset type: {dataset_type} (from odometry_mode: {all_stats[0]["odometry_mode"]})')
    else:
        print(f'  Warning: odometry_mode not found in JSON, falling back to folder name parsing')
        folder_name = os.path.basename(dataset_folder_path).lower()
        if 'mono' in folder_name:
            dataset_type = 'MONO'
        elif 'vio' in folder_name or 'imu' in folder_name:
            dataset_type = 'VIO'
        elif 'rgbd' in folder_name or 'depth' in folder_name:
            dataset_type = 'RGBD'
        else:
            dataset_type = 'STEREO'

    odom_stats = [s for s in all_stats if 'ODOM' in s.get('sequence_title', '').upper()]
    slam_stats = [s for s in all_stats if 'SLAM' in s.get('sequence_title', '').upper()]

    if odom_stats:
        avg_translation_error = sum(s.get('gt_av_translation_error', 0) for s in odom_stats) / len(odom_stats)
        avg_rotation_error = sum(s.get('gt_av_rotation_error', 0) for s in odom_stats) / len(odom_stats)
        avg_kabsch = sum(s.get('gt_simple_error', 0) for s in odom_stats) / len(odom_stats)
        avg_fps = sum(s.get('average_fps', 0) for s in odom_stats) / len(odom_stats)
        total_tracking_losts = sum(s.get('num_tracking_losts', 0) for s in odom_stats if s.get('num_tracking_losts', -1) >= 0)

        out_dict[f"{dataset_name}_ATE_{dataset_type}_ODOM"] = avg_translation_error
        out_dict[f"{dataset_name}_ARE_{dataset_type}_ODOM"] = avg_rotation_error
        out_dict[f"{dataset_name}_Kabsch_{dataset_type}_ODOM"] = avg_kabsch
        out_dict[f"{dataset_name}_FPS_{dataset_type}_ODOM"] = avg_fps
        out_dict[f"{dataset_name}_TrackingLosts_{dataset_type}_ODOM"] = total_tracking_losts

    if slam_stats:
        avg_translation_error = sum(s.get('gt_av_translation_error', 0) for s in slam_stats) / len(slam_stats)
        avg_rotation_error = sum(s.get('gt_av_rotation_error', 0) for s in slam_stats) / len(slam_stats)
        avg_kabsch = sum(s.get('gt_simple_error', 0) for s in slam_stats) / len(slam_stats)
        avg_fps = sum(s.get('average_fps', 0) for s in slam_stats) / len(slam_stats)
        total_tracking_losts = sum(s.get('num_tracking_losts', 0) for s in slam_stats if s.get('num_tracking_losts', -1) >= 0)

        out_dict[f"{dataset_name}_ATE_{dataset_type}_SLAM"] = avg_translation_error
        out_dict[f"{dataset_name}_ARE_{dataset_type}_SLAM"] = avg_rotation_error
        out_dict[f"{dataset_name}_Kabsch_{dataset_type}_SLAM"] = avg_kabsch
        out_dict[f"{dataset_name}_FPS_{dataset_type}_SLAM"] = avg_fps
        out_dict[f"{dataset_name}_TrackingLosts_{dataset_type}_SLAM"] = total_tracking_losts

    return out_dict


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


def format_metric(metric, value, *, aggregate=False):
    """Format one scalar KPI or one aggregate diff."""
    if value == "NA":
        return value
    base_metric = metric.removeprefix("diff ")
    if base_metric == "TrackingLosts":
        return f"{float(value):.2f}" if aggregate else str(int(value))
    if base_metric == "FPS":
        return f"{float(value):.1f}"
    return f"{float(value):.4f}"


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
        value = data[key]

        if metric == "TrackingLosts":
            metrics[metric] = int(value)
        elif metric == "FPS":
            metrics[metric] = round(value, 1)
        else:
            metrics[metric] = round(value, 4)

        if metric in NO_DIFF_METRICS:
            continue
        if "MONO" in dataset_key and metric == "ATE":
            metrics["diff " + metric] = "NA"
        elif prev_data is not None and key in prev_data:
            difference = value - prev_data[key]
            metrics["diff " + metric] = int(difference) if metric == "TrackingLosts" else round(difference, 4)
        else:
            metrics["diff " + metric] = "NA"

    for dataset_key, metrics in organized_data.items():
        for metric in required_metrics:
            metrics.setdefault(metric, "NA")
            if "MONO" in dataset_key and metric == "ATE":
                metrics[metric] = "NA"

    return organized_data


def table_columns(required_metrics=REQUIRED_METRICS):
    diffable = [metric for metric in required_metrics if metric not in NO_DIFF_METRICS]
    nondiff = [metric for metric in required_metrics if metric in NO_DIFF_METRICS]
    return diffable + ["diff " + metric for metric in diffable] + nondiff


def column_title(metric, *, aggregate=False):
    title = get_display_name(metric)
    unit = get_unit(metric)
    if unit:
        title += f", {unit}"
    if aggregate:
        title += " (mean)" if metric.startswith("diff ") else " (mean ± σ)"
    return title


def create_table(organized_data, required_metrics=REQUIRED_METRICS, *, config=None, aggregate=False):
    """Render an already-organized KPI dictionary as Markdown."""
    columns = table_columns(required_metrics)
    headers = (["Config"] if config else []) + ["Dataset"] + [
        column_title(metric, aggregate=aggregate) for metric in columns
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]

    for dataset in sorted(organized_data):
        metrics = organized_data[dataset]
        if not all(metric in metrics for metric in columns):
            missing = [metric for metric in columns if metric not in metrics]
            raise ValueError(f"{dataset} is missing table columns: {', '.join(missing)}")
        cells = ([config] if config else []) + [display_dataset_key(dataset)]
        cells.extend(
            str(metrics[metric]) if aggregate else format_metric(metric, metrics[metric]) for metric in columns
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


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


def collect_kpis(stat_folder):
    """Compute a flat KPI dictionary from a cuvslam_app stats directory."""
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
    key_source = {}
    print(f"Processing {len(dataset_folders)} dataset folders...")
    for dataset_folder in dataset_folders:
        folder_name = os.path.basename(dataset_folder)
        print(f"  Processing: {folder_name}")
        result = process_dataset_folder(dataset_folder)
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
        key_source.update(dict.fromkeys(result, folder_name))
    if not kpis:
        raise ValueError("output KPI JSON is empty; check the input stats format")
    return kpis


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


def build_report(run_id, current, previous=None, baseline_ranges=None):
    drift = []
    if baseline_ranges:
        drift = [
            {"key": key, "status": status, "detail": detail}
            for key, status, detail in evaluate_drift(current, baseline_ranges)
        ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "current": current,
        "previous": previous,
        "drift": drift,
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
    return report


def render_report(report, config):
    organized = organize_data(report["current"], prev_data=report["previous"])
    return create_table(organized, config=config)


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
    mean = fmean(values)
    deviation = pstdev(values)
    if metric == "TrackingLosts":
        return f"{mean:.2f} ± {deviation:.2f}"
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

        current_values = [
            require_numeric(report["current"][key], key, config) for config, report in config_reports
        ]
        metrics[metric] = format_distribution(metric, current_values)

        if metric in NO_DIFF_METRICS:
            continue
        previous_complete = all(
            report["previous"] is not None and key in report["previous"] for _, report in config_reports
        )
        if previous_complete:
            previous_values = [
                require_numeric(report["previous"][key], key, config) for config, report in config_reports
            ]
            metrics["diff " + metric] = format_metric(
                metric, fmean(current_values) - fmean(previous_values), aggregate=True
            )
        else:
            metrics["diff " + metric] = "NA"

    for metrics in organized.values():
        for metric in REQUIRED_METRICS:
            metrics.setdefault(metric, "NA")
            if metric not in NO_DIFF_METRICS:
                metrics.setdefault("diff " + metric, "NA")
    return organized


def render_aggregate_report(config_reports):
    organized = aggregate_reports(config_reports)
    count = len(config_reports)
    note = (
        f"_Aggregated across {count} configuration{'s' if count != 1 else ''}. "
        "KPI values are mean ± population σ; diffs compare the current and previous aggregated means._\n\n"
    )
    return note + create_table(organized, aggregate=True)


def parse_config_report(value):
    if "=" not in value:
        raise ValueError(f"configuration input must have CONFIG=PATH form: {value!r}")
    config, path = value.split("=", 1)
    if not config or not path:
        raise ValueError(f"configuration input must have CONFIG=PATH form: {value!r}")
    return config, load_report(path)


def collect_command(args):
    print("=============================\nKPI collector is up!")
    current = collect_kpis(args.stat_folder)
    previous = load_json_object(args.prev_kpi, "previous KPI data") if args.prev_kpi else None
    ranges = load_baseline_ranges(args.baseline_ranges) if args.baseline_ranges else {}
    report = build_report(args.run_id, current, previous, ranges)
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
    write_text(args.output, render_aggregate_report(config_reports))


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
    collect.set_defaults(func=collect_command)

    render = commands.add_parser("render", help="render one configuration report as Markdown")
    render.add_argument("-r", "--report_json", required=True)
    render.add_argument("-c", "--config", default="")
    render.add_argument("-o", "--output", default="-")
    render.set_defaults(func=render_command)

    aggregate = commands.add_parser("aggregate", help="aggregate configuration reports as Markdown")
    aggregate.add_argument("-i", "--input", action="append", required=True, metavar="CONFIG=PATH")
    aggregate.add_argument("-o", "--output", default="-")
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
