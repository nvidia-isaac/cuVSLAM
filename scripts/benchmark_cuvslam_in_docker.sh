#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/benchmark_cuvslam_in_docker.sh <build_output_dir>"
  echo "  Run after build_cuvslam_in_docker.sh. Expects build/ in the output dir."
  exit 1
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR=$(realpath "$1")

annotate() {
  if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    echo "::$1::$2"
  else
    echo "${1^}: $2" >&2
  fi
}

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

python3 "$REPO_ROOT/scripts/cuvslam_benchmark_report.py" metadata \
  --repo-root "$REPO_ROOT" \
  --output "$OUTPUT_DIR/cpp-benchmark-metadata.json"

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

set +e
docker run --runtime=nvidia --gpus all --rm --label "cuvslam.runner=${RUNNER_NAME:-local}" $TTY_FLAG \
  -v "$REPO_ROOT:/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  cuvslam:local bash -c '
    set -euo pipefail
    trap "chown -R $HOST_UID:$HOST_GID /output" EXIT
    cd /output/build
    GTEST_FILTER="*SpeedUp*:*Speedup*" \
      GTEST_RANDOM_SEED=42 \
      GTEST_COLOR=no \
      GTEST_OUTPUT=xml:/output/cpp-benchmark-results.xml \
      ctest -R "^cuda_modules_test$" -V 2>&1 | tee /output/cpp-benchmark-output.log
  '
benchmark_status=$?
set -e

report_status=0
if [ -f "$OUTPUT_DIR/cpp-benchmark-results.xml" ]; then
  python3 "$REPO_ROOT/scripts/cuvslam_benchmark_report.py" render \
    --xml "$OUTPUT_DIR/cpp-benchmark-results.xml" \
    --metadata "$OUTPUT_DIR/cpp-benchmark-metadata.json" \
    --output-json "$OUTPUT_DIR/cpp-benchmark-results.json" \
    --output-markdown "$OUTPUT_DIR/benchmark-summary.md" || report_status=$?
else
  annotate error "The benchmark container exited with status $benchmark_status without writing \
cpp-benchmark-results.xml; see cpp-benchmark-output.log in the benchmark artifact."
  report_status=1
fi

if [ "$benchmark_status" -ne 0 ] && [ "$report_status" -eq 0 ]; then
  echo "The benchmark container exited with status $benchmark_status; failures are recorded in the benchmark report."
fi
exit "$report_status"
