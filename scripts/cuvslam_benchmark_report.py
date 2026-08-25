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

"""Convert GoogleTest benchmark XML properties into JSON and Markdown reports."""

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED_PROPERTIES = (
    "iterations",
    "cpu_ns_per_iteration",
    "gpu_ns_per_iteration",
    "speedup",
)


def _command(arguments: list[str]) -> str:
    if shutil.which(arguments[0]) is None:
        return ""
    result = subprocess.run(arguments, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def collect_metadata(repo_root: Path) -> dict:
    tegra_path = Path("/etc/nv_tegra_release")
    source_status = _command(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"])
    return {
        "config": os.environ.get("BENCHMARK_CONFIG", "local"),
        "git_sha": os.environ.get("GITHUB_SHA") or _command(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "source_modified": bool(source_status),
        "tegra_release": tegra_path.read_text().strip() if tegra_path.exists() else "",
        "jetpack": _command(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]),
        "cuda_version": os.environ.get("CUDA_VERSION", ""),
        "ubuntu_version": os.environ.get("UBUNTU_VERSION", ""),
        "nvpmodel": _command(["nvpmodel", "-q"]),
        "jetson_clocks": _command(["jetson_clocks", "--show"]),
        "nvidia_smi": _command(["nvidia-smi", "-L"]),
        "hostname": platform.node(),
        "kernel": platform.release(),
    }


def _parse_positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _parse_positive_float(value: str, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _test_properties(testcase: ET.Element) -> dict[str, str]:
    properties = testcase.find("properties")
    if properties is None:
        return {}
    return {item.get("name", ""): item.get("value", "") for item in properties.findall("property")}


def parse_gtest_xml(xml_path: Path, expected_count: int = 0) -> dict:
    root = ET.parse(xml_path).getroot()
    benchmarks = []

    for testcase in root.findall(".//testcase"):
        if testcase.get("status") == "notrun" or testcase.get("result") in {"skipped", "suppressed"}:
            continue

        suite = testcase.get("classname", "")
        name = testcase.get("name", "")
        full_name = f"{suite}.{name}".strip(".")
        properties = _test_properties(testcase)
        missing = [key for key in REQUIRED_PROPERTIES if not properties.get(key)]
        metric_error = ""
        metrics = {}

        if missing:
            metric_error = f"missing properties: {', '.join(missing)}"
        else:
            try:
                metrics = {
                    "iterations": _parse_positive_int(properties["iterations"], "iterations"),
                    "cpu_ns_per_iteration": _parse_positive_int(
                        properties["cpu_ns_per_iteration"], "cpu_ns_per_iteration"
                    ),
                    "gpu_ns_per_iteration": _parse_positive_int(
                        properties["gpu_ns_per_iteration"], "gpu_ns_per_iteration"
                    ),
                    "speedup": _parse_positive_float(properties["speedup"], "speedup"),
                }
            except ValueError as error:
                metric_error = str(error)

        failures = testcase.findall("failure") + testcase.findall("error")
        failure_message = "; ".join(
            filter(
                None,
                (
                    failure.get("message") or (failure.text or "").strip()
                    for failure in failures
                ),
            )
        )
        if metric_error:
            status = "error"
        elif failures:
            status = "fail"
        else:
            status = "pass"

        benchmarks.append(
            {
                "name": full_name,
                "status": status,
                "elapsed_seconds": float(testcase.get("time", "0") or 0),
                **metrics,
                "failure_message": failure_message,
                "metric_error": metric_error,
            }
        )

    benchmarks.sort(key=lambda benchmark: benchmark["name"])
    passed = sum(benchmark["status"] == "pass" for benchmark in benchmarks)
    failed = sum(benchmark["status"] == "fail" for benchmark in benchmarks)
    errors = sum(benchmark["status"] == "error" for benchmark in benchmarks)
    count_matches = expected_count == 0 or len(benchmarks) == expected_count

    return {
        "schema_version": 1,
        "summary": {
            "expected_count": expected_count,
            "total": len(benchmarks),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "complete": count_matches and errors == 0,
        },
        "benchmarks": benchmarks,
    }


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _markdown_text(value: object) -> str:
    return _compact(value).replace("|", "\\|")


def render_markdown(report: dict) -> str:
    metadata = report.get("metadata", {})
    summary = report["summary"]
    config = _markdown_text(metadata.get("config", "unknown"))
    expected = summary["expected_count"]
    expected_text = f"/{expected}" if expected else ""
    revision = metadata.get("git_sha", "")
    if revision and metadata.get("source_modified"):
        revision = f"{revision}-modified"

    lines = [
        f"### {config}",
        "",
        (
            f"{summary['total']}{expected_text} benchmarks reported: "
            f"{summary['passed']} passed, {summary['failed']} failed, {summary['errors']} invalid."
        ),
        "",
    ]

    metadata_items = [
        ("Commit", revision),
        ("Jetson", metadata.get("tegra_release")),
        ("JetPack", metadata.get("jetpack")),
        ("CUDA", metadata.get("cuda_version")),
        ("Ubuntu", metadata.get("ubuntu_version")),
        ("GPU", metadata.get("nvidia_smi")),
        ("Kernel", metadata.get("kernel")),
        ("Power mode", metadata.get("nvpmodel")),
        ("Clocks", metadata.get("jetson_clocks")),
    ]
    visible_metadata = [(label, _markdown_text(value)) for label, value in metadata_items if _compact(value)]
    if visible_metadata:
        lines.extend(f"- {label}: `{value}`" for label, value in visible_metadata)
        lines.append("")

    lines.extend(
        [
            "| Status | Test | CPU ms/iteration | GPU ms/iteration | Speedup | Test time (s) |",
            "|:------:|------|-----------------:|-----------------:|--------:|--------------:|",
        ]
    )

    icons = {"pass": "PASS", "fail": "FAIL", "error": "INVALID"}
    for benchmark in report["benchmarks"]:
        cpu_ms = benchmark.get("cpu_ns_per_iteration")
        gpu_ms = benchmark.get("gpu_ns_per_iteration")
        speedup = benchmark.get("speedup")
        lines.append(
            "| {status} | {name} | {cpu} | {gpu} | {speedup} | {elapsed:.3f} |".format(
                status=icons[benchmark["status"]],
                name=_markdown_text(benchmark["name"]),
                cpu=f"{cpu_ms / 1_000_000:.6f}" if cpu_ms is not None else "—",
                gpu=f"{gpu_ms / 1_000_000:.6f}" if gpu_ms is not None else "—",
                speedup=f"{speedup:.3f}x" if speedup is not None else "—",
                elapsed=benchmark["elapsed_seconds"],
            )
        )

    details = [
        (benchmark["name"], benchmark["failure_message"] or benchmark["metric_error"])
        for benchmark in report["benchmarks"]
        if benchmark["failure_message"] or benchmark["metric_error"]
    ]
    if details:
        lines.extend(["", "<details><summary>Benchmark failures and invalid results</summary>", ""])
        lines.extend(f"- **{_markdown_text(name)}:** {_markdown_text(message)}" for name, message in details)
        lines.extend(["", "</details>"])

    lines.extend(["", "_Informational soak-period results; benchmark failures do not gate nightly._", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser("metadata", help="Collect host benchmark metadata")
    metadata_parser.add_argument("--repo-root", type=Path, required=True)
    metadata_parser.add_argument("--output", type=Path, required=True)

    render_parser = subparsers.add_parser("render", help="Render GoogleTest benchmark results")
    render_parser.add_argument("--xml", type=Path, required=True, help="GoogleTest XML produced by cuda_modules_test")
    render_parser.add_argument("--metadata", type=Path, required=True, help="JSON file containing runner metadata")
    render_parser.add_argument("--output-json", type=Path, required=True)
    render_parser.add_argument("--output-markdown", type=Path, required=True)
    render_parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "metadata":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(collect_metadata(args.repo_root), indent=2, sort_keys=True) + "\n")
        return 0

    report = parse_gtest_xml(args.xml, args.expected_count)
    report["metadata"] = json.loads(args.metadata.read_text())

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(render_markdown(report))
    return 0 if report["summary"]["complete"] else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ET.ParseError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
