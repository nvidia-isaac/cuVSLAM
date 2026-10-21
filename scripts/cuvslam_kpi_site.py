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

"""Build the static KPI trend site published next to the docs on GitHub Pages.

Each --config SLUG=PATH adds one CI configuration. PATH is either a KPI history
directory of kpi_<run>.json files or a single kpi_<run>.report.json, whose
baseline block already carries the preceding nightlies. The output directory
receives a copy of --docs-dir (the gh-pages tree, unchanged) plus kpi/index.html
and kpi/data.json.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import glob
import math
import os
from pathlib import Path
import shutil
import sys

from cuvslam_kpi_report import (COUNT_METRICS, COUNT_MIN_BAND, DEFAULT_BASELINE, HIGHER_IS_BETTER, LEVEL_MAD_FACTOR,
                                METRIC_UNITS, MIN_BASELINE_RUNS, REQUIRED_METRICS, config_group, get_display_name,
                                load_json_object, load_kpi_config, parse_kpi_key, resolve_baseline_settings,
                                safe_float, stored_baseline_settings, write_json)

SITE_SUBDIR = "kpi"
SCHEMA_VERSION = 3
SITE_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kpi_site")
PAGE_SOURCE = os.path.join(SITE_SOURCE_DIR, "index.html")
DEFAULT_EXCLUSIONS = os.path.join(SITE_SOURCE_DIR, "exclusions.json")
# An eval is retired when no config produced a value for it in its last few nights with results; a margin of
# several nights keeps one broken nightly from retiring everything.
RETIRED_AFTER_RUNS = 3


def load_history_dir(path):
    """{run: {key: value}} from every readable kpi_<run>.json in a history directory."""
    runs = {}
    for file in sorted(glob.glob(os.path.join(path, "kpi_[0-9]*.json"))):
        name = os.path.basename(file)
        if name.endswith(".report.json"):
            continue
        try:
            runs[name[len("kpi_"):-len(".json")]] = load_json_object(file, "KPI history")
        except (OSError, ValueError) as e:
            print(f"Warning: skipping unreadable KPI history {file}: {e}", file=sys.stderr)
    return runs


def load_report_file(path):
    """({run: {key: value}}, {dataset: baseline settings}) from a report.

    The runs are the baseline series plus the current run; the settings are the
    ones the report was judged with, filled in the way the report check fills them.
    """
    report = load_json_object(path, "KPI report")
    baseline = report.get("baseline") or {}
    runs = {}
    for index, run in enumerate(baseline.get("runs", [])):
        runs[run] = {key: series[index] for key, series in baseline.get("values", {}).items() if index < len(series)}
    runs[report["run_id"]] = report["current"]
    settings = {}
    for dataset in baseline.get("settings", {}):
        stored = stored_baseline_settings(baseline, dataset)
        if stored:
            settings[dataset] = {key: stored[key] for key in DEFAULT_BASELINE}
    return runs, settings


def load_config_runs(path):
    """({run: {key: value}}, stored baseline settings or None for a history directory)."""
    if os.path.isdir(path):
        return load_history_dir(path), None
    return load_report_file(path)


def load_exclusions(path):
    """Validated entries of the exclusions file: {"dataset", "types", "runs", "reason"} each."""
    data = load_json_object(path, "KPI site exclusions")
    entries = data.get("exclusions", [])
    if not isinstance(entries, list):
        raise ValueError(f"exclusions in {path} must be a list")
    for index, entry in enumerate(entries):
        where = f"{path} exclusions[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"dataset", "types", "runs", "reason"}:
            raise ValueError(f"{where} must have exactly dataset, types, runs, and reason")
        if not isinstance(entry["dataset"], str) or not entry["reason"]:
            raise ValueError(f"{where} needs a dataset name and a reason")
        for field in ("types", "runs"):
            if not entry[field] or not all(isinstance(item, str) for item in entry[field]):
                raise ValueError(f"{where} {field} must be a non-empty list of strings")
    return entries


def apply_exclusions(configs, entries):
    """Drop the excluded values from every config, in place; warn about entries that match nothing."""
    for entry in entries:
        matched = 0
        for runs in configs.values():
            for run in entry["runs"]:
                values = runs.get(run, {})
                for key in list(values):
                    parsed = parse_kpi_key(key)
                    if not parsed:
                        continue
                    dataset, mode = parsed[0].split("-", 1)
                    if dataset == entry["dataset"] and mode.rsplit("_", 1)[0] in entry["types"]:
                        del values[key]
                        matched += 1
        if not matched:
            print(f"Warning: exclusion for {entry['dataset']} {entry['types']} on {entry['runs']} matches no value",
                  file=sys.stderr)


def retired_keys(data_configs):
    """KPI keys that no config has a value for in its last RETIRED_AFTER_RUNS nights with any value."""
    active = set()
    for config in data_configs.values():
        nights = [index for index in range(len(config["runs"]))
                  if any(series[index] is not None for series in config["values"].values())]
        for index in nights[-RETIRED_AFTER_RUNS:]:
            active.update(key for key, series in config["values"].items() if series[index] is not None)
    return {key for config in data_configs.values() for key in config["values"]} - active


def finite_float(value):
    """safe_float, with infinities also dropped, since JSON cannot represent them."""
    result = safe_float(value)
    return result if result is None or math.isfinite(result) else None


def effective_settings(datasets, stored, kpi_config_path):
    """Per-dataset baseline settings for the page, matching what the reports were judged with.

    stored maps dataset to the settings stored in report inputs. The KPI config
    covers datasets without stored settings, and, when given, must agree with
    the stored ones, so the page never judges with settings the report did not use.
    """
    resolved = {}
    if kpi_config_path:
        kpi_config = load_kpi_config(kpi_config_path)
        resolved = {dataset: resolve_baseline_settings(kpi_config, dataset) for dataset in datasets}
    settings = {}
    for dataset in datasets:
        if dataset in stored and dataset in resolved and stored[dataset] != resolved[dataset]:
            raise ValueError(f"baseline settings for {dataset} in the reports ({stored[dataset]}) differ from "
                             f"{kpi_config_path} ({resolved[dataset]})")
        if dataset in stored:
            settings[dataset] = stored[dataset]
        elif dataset in resolved:
            settings[dataset] = resolved[dataset]
        else:
            raise ValueError(f"no baseline settings for {dataset}: no report input stores them; pass --kpi-config")
    return settings


def level_group(slug):
    """The regression-check group of a <platform>-cuda<X.Y.Z>-ubuntu<V> config.

    Each platform's nightly report is judged on its own, and within one, the
    builds of a CUDA version form a group (config_group).
    """
    return f"{slug.split('-', 1)[0]} {config_group(slug)}"


def build_data(configs, stored_settings, kpi_config_path):
    """The kpi/data.json document: per-config run dates and value series, plus KPI metadata."""
    data_configs = {}
    keys = set()
    for slug, runs in configs.items():
        dates = sorted(runs)
        config_keys = sorted({key for values in runs.values() for key in values if parse_kpi_key(key)})
        keys.update(config_keys)
        data_configs[slug] = {
            "group": level_group(slug),
            # run_eval.sh passes --flag-fps on aarch64 only: Jetson runners are dedicated, x86 ones shared.
            "flag_fps": not slug.startswith("x86_64-"),
            "runs": dates,
            "values": {key: [finite_float(runs[run].get(key)) for run in dates] for key in config_keys},
        }
    retired = retired_keys(data_configs)
    kpis = {}
    for key in sorted(keys):
        row, metric = parse_kpi_key(key)
        dataset, mode = row.split("-", 1)
        kpis[key] = {"dataset": dataset, "mode": mode, "metric": metric, "retired": key in retired}
    datasets = sorted({entry["dataset"] for entry in kpis.values()})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": {
            metric: {
                "label": get_display_name(metric),
                "unit": METRIC_UNITS.get(metric, ""),
                "higher_is_better": metric in HIGHER_IS_BETTER,
                "count": metric in COUNT_METRICS,
            }
            for metric in REQUIRED_METRICS
        },
        "baseline": {
            "min_runs": MIN_BASELINE_RUNS,
            "level_mad_factor": LEVEL_MAD_FACTOR,
            "count_min_band": COUNT_MIN_BAND,
            "settings": effective_settings(datasets, stored_settings, kpi_config_path),
        },
        "kpis": kpis,
        "configs": data_configs,
    }


def parse_config_arg(value):
    slug, sep, path = value.partition("=")
    if not sep or not slug or not path:
        raise ValueError(f"--config expects SLUG=PATH, got {value!r}")
    return slug, path


def build_site(args):
    configs = {}
    stored_settings = {}
    for value in args.config:
        slug, path = parse_config_arg(value)
        if slug in configs:
            raise ValueError(f"duplicate --config slug {slug!r}")
        configs[slug], settings = load_config_runs(path)
        print(f"{slug}: {len(configs[slug])} runs from {path}")
        for dataset, entry in (settings or {}).items():
            if stored_settings.setdefault(dataset, entry) != entry:
                raise ValueError(f"report inputs disagree on the baseline settings for {dataset}: "
                                 f"{stored_settings[dataset]} and {entry} ({path})")
    if args.exclusions:
        apply_exclusions(configs, load_exclusions(args.exclusions))
    data = build_data(configs, stored_settings, args.kpi_config)

    if args.docs_dir:
        docs_dir, output = Path(args.docs_dir).resolve(), Path(args.output).resolve()
        if output.is_relative_to(docs_dir):
            raise ValueError(f"output {args.output} is inside the docs directory {args.docs_dir}")
    if os.path.exists(args.output):
        raise ValueError(f"output directory {args.output} already exists")
    if args.docs_dir:
        if os.path.exists(os.path.join(args.docs_dir, SITE_SUBDIR)):
            raise ValueError(f"{args.docs_dir} already has a {SITE_SUBDIR}/ entry; it would be overwritten")
        shutil.copytree(args.docs_dir, args.output, ignore=shutil.ignore_patterns(".git"))
    site_dir = os.path.join(args.output, SITE_SUBDIR)
    os.makedirs(site_dir)
    shutil.copyfile(PAGE_SOURCE, os.path.join(site_dir, "index.html"))
    write_json(os.path.join(site_dir, "data.json"), data)
    print(f"Wrote {site_dir}")


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", required=True, metavar="SLUG=PATH",
                        help="CI configuration slug and its KPI history directory or report JSON")
    parser.add_argument("--kpi-config",
                        help="KPI config JSON (scripts/kpi_baseline_ranges.json) for the regression check; required "
                             "for history directories, and checked against the settings stored in reports")
    parser.add_argument("--exclusions", default=DEFAULT_EXCLUSIONS,
                        help="JSON list of KPI values to leave out, such as an eval's broken first night "
                             "(default: %(default)s; pass an empty string for none)")
    parser.add_argument("--docs-dir", help="gh-pages tree to publish unchanged next to kpi/")
    parser.add_argument("--output", required=True, help="new directory to write the site into")
    args = parser.parse_args()
    try:
        build_site(args)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
