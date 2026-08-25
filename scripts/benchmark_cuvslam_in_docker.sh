#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/benchmark_cuvslam_in_docker.sh <build_output_dir>"
  echo "  Run after build_cuvslam_in_docker.sh. Expects build/ in the output dir."
  exit 1
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR=$(realpath "$1")
EXPECTED_BENCHMARKS=13

if [ ! -d "$OUTPUT_DIR/build" ]; then
  echo "Error: $OUTPUT_DIR/build not found."
  echo "Run './scripts/build_cuvslam_in_docker.sh Release $OUTPUT_DIR' first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -f \
  "$OUTPUT_DIR/cpp-benchmark-output.log" \
  "$OUTPUT_DIR/cpp-benchmark-results.json" \
  "$OUTPUT_DIR/cpp-benchmark-results.xml" \
  "$OUTPUT_DIR/benchmark-summary.md"

export REPO_ROOT
python3 - "$OUTPUT_DIR/cpp-benchmark-metadata.json" <<'PY'
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(arguments):
    if shutil.which(arguments[0]) is None:
        return ""
    result = subprocess.run(arguments, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


tegra_path = Path("/etc/nv_tegra_release")
source_status = command(["git", "-C", os.environ["REPO_ROOT"], "status", "--porcelain", "--untracked-files=no"])
metadata = {
    "config": os.environ.get("BENCHMARK_CONFIG", "local"),
    "git_sha": os.environ.get("GITHUB_SHA") or command(["git", "-C", os.environ["REPO_ROOT"], "rev-parse", "HEAD"]),
    "source_modified": bool(source_status),
    "tegra_release": tegra_path.read_text().strip() if tegra_path.exists() else "",
    "jetpack": command(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]),
    "cuda_version": os.environ.get("CUDA_VERSION", ""),
    "ubuntu_version": os.environ.get("UBUNTU_VERSION", ""),
    "nvpmodel": command(["nvpmodel", "-q"]),
    "jetson_clocks": command(["jetson_clocks", "--show"]),
    "nvidia_smi": command(["nvidia-smi", "-L"]),
    "hostname": platform.node(),
    "kernel": platform.release(),
}
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

set +e
docker run --runtime=nvidia --gpus all --rm $TTY_FLAG \
  -v "$REPO_ROOT:/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  cuvslam:local bash -c '
    set -euo pipefail
    trap "chown -R $HOST_UID:$HOST_GID /output" EXIT
    benchmark=/output/build/bin/cuda_modules_test
    if [ ! -x "$benchmark" ]; then
      echo "Error: $benchmark not found or not executable." >&2
      exit 1
    fi
    "$benchmark" \
      --gtest_filter="*SpeedUp*:*Speedup*" \
      --gtest_random_seed=42 \
      --gtest_color=no \
      --gtest_output=xml:/output/cpp-benchmark-results.xml \
      2>&1 | tee /output/cpp-benchmark-output.log
  '
benchmark_status=$?
set -e

report_status=0
if [ -f "$OUTPUT_DIR/cpp-benchmark-results.xml" ]; then
  python3 "$REPO_ROOT/scripts/cuvslam_benchmark_report.py" \
    --xml "$OUTPUT_DIR/cpp-benchmark-results.xml" \
    --metadata "$OUTPUT_DIR/cpp-benchmark-metadata.json" \
    --output-json "$OUTPUT_DIR/cpp-benchmark-results.json" \
    --output-markdown "$OUTPUT_DIR/benchmark-summary.md" \
    --expected-count "$EXPECTED_BENCHMARKS" || report_status=$?
else
  echo "Error: benchmark run did not produce cpp-benchmark-results.xml." >&2
  report_status=2
fi

if [ "$benchmark_status" -ne 0 ]; then
  echo "Benchmark tests failed; preserving the failure in the generated report." >&2
fi
exit "$report_status"
