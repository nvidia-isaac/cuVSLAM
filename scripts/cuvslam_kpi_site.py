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
import os
import shutil
import sys

from cuvslam_kpi_report import (COUNT_METRICS, COUNT_MIN_BAND, HIGHER_IS_BETTER, LEVEL_MAD_FACTOR, METRIC_UNITS,
                                MIN_BASELINE_RUNS, REQUIRED_METRICS, config_group, get_display_name, load_json_object,
                                load_kpi_config, parse_kpi_key, resolve_baseline_settings, safe_float, write_json)

SITE_SUBDIR = "kpi"
SCHEMA_VERSION = 2
PAGE_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kpi_site", "index.html")


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
    """{run: {key: value}} from a report's baseline series plus its current run."""
    report = load_json_object(path, "KPI report")
    baseline = report.get("baseline") or {}
    runs = {}
    for index, run in enumerate(baseline.get("runs", [])):
        runs[run] = {key: series[index] for key, series in baseline.get("values", {}).items() if index < len(series)}
    runs[report["run_id"]] = report["current"]
    return runs


def load_config_runs(path):
    if os.path.isdir(path):
        return load_history_dir(path)
    return load_report_file(path)


def level_group(slug):
    """The regression-check group of a <platform>-cuda<X.Y.Z>-ubuntu<V> config.

    Each platform's nightly report is judged on its own, and within one, the
    builds of a CUDA version form a group (config_group).
    """
    return f"{slug.split('-', 1)[0]} {config_group(slug)}"


def build_data(configs, kpi_config):
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
            "values": {key: [safe_float(runs[run].get(key)) for run in dates] for key in config_keys},
        }
    kpis = {}
    for key in sorted(keys):
        row, metric = parse_kpi_key(key)
        dataset, mode = row.split("-", 1)
        kpis[key] = {"dataset": dataset, "mode": mode, "metric": metric}
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
            "settings": {dataset: resolve_baseline_settings(kpi_config, dataset) for dataset in datasets},
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
    for value in args.config:
        slug, path = parse_config_arg(value)
        if slug in configs:
            raise ValueError(f"duplicate --config slug {slug!r}")
        configs[slug] = load_config_runs(path)
        print(f"{slug}: {len(configs[slug])} runs from {path}")
    kpi_config = load_kpi_config(args.kpi_config)

    if os.path.exists(args.output):
        raise ValueError(f"output directory {args.output} already exists")
    if args.docs_dir:
        if os.path.exists(os.path.join(args.docs_dir, SITE_SUBDIR)):
            raise ValueError(f"{args.docs_dir} already has a {SITE_SUBDIR}/ entry; it would be overwritten")
        shutil.copytree(args.docs_dir, args.output, ignore=shutil.ignore_patterns(".git"))
    site_dir = os.path.join(args.output, SITE_SUBDIR)
    os.makedirs(site_dir)
    shutil.copyfile(PAGE_SOURCE, os.path.join(site_dir, "index.html"))
    write_json(os.path.join(site_dir, "data.json"), build_data(configs, kpi_config))
    print(f"Wrote {site_dir}")


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", required=True, metavar="SLUG=PATH",
                        help="CI configuration slug and its KPI history directory or report JSON")
    parser.add_argument("--kpi-config", help="KPI config JSON (scripts/kpi_baseline_ranges.json) for baseline bands")
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
