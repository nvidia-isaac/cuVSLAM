#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/eval_cuvslam_in_docker.sh <build_output_dir>"
  echo "  Run after build_cuvslam_in_docker.sh and stage_eval_datasets.sh."
  echo ""
  echo "Environment variables:"
  echo "  RUNNER_STORAGE_ROOT  KPI history on runner storage (repo variable)"
  echo "  RUNNER_LOCAL_DATASETS_ROOT  Local extract root (default \$HOME/.cache/cuvslam)"
  echo "  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  From AWS_S3_* repository secrets in CI"
  echo "  KPI_HISTORY_DIR  Override KPI path (default \$RUNNER_STORAGE_ROOT/cuvslam-ci/kpi-history)"
  echo "  RUN_ID, MAX_WORKERS, EVAL_WRITE_HISTORY"
  exit 1
fi

OUTPUT_DIR=$(realpath "$1")

if [ ! -d "$OUTPUT_DIR/build" ]; then
  echo "Error: $OUTPUT_DIR/build not found."
  echo "Run './scripts/build_cuvslam_in_docker.sh Release $OUTPUT_DIR' first."
  exit 1
fi

"$(dirname "$(realpath "$0")")/check_eval_prerequisites.sh"

KPI_HISTORY_DIR="${KPI_HISTORY_DIR:-$RUNNER_STORAGE_ROOT/cuvslam-ci/kpi-history}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d)}"
RUNNER_LOCAL_DATASETS_ROOT="${RUNNER_LOCAL_DATASETS_ROOT:-${HOME:-/tmp}/.cache/cuvslam}"
DATASET_MOUNT_SRC="$RUNNER_LOCAL_DATASETS_ROOT/datasets/vslam"

if [ -z "${MAX_WORKERS:-}" ]; then
  worker_ceiling=12
  nproc_n=$(nproc)
  mem_cap=$(awk '/MemAvailable/ { print int($2/1024/1024/4) }' /proc/meminfo)
  if [ "$mem_cap" -lt 1 ]; then
    mem_cap=1
  fi
  MAX_WORKERS=$nproc_n
  if [ "$mem_cap" -lt "$MAX_WORKERS" ]; then
    MAX_WORKERS=$mem_cap
  fi
  if [ "$worker_ceiling" -lt "$MAX_WORKERS" ]; then
    MAX_WORKERS=$worker_ceiling
  fi
  echo "Eval parallelism: max_workers=$MAX_WORKERS (nproc=$nproc_n, mem-capped=${mem_cap} at 4GB/worker, ceiling=${worker_ceiling})"
fi

if [ "${EVAL_WRITE_HISTORY:-}" = "true" ] || \
   { [ "${GITHUB_ACTIONS:-}" = "true" ] && \
     { [ "${GITHUB_EVENT_NAME:-}" = "schedule" ] || [ "${GITHUB_EVENT_NAME:-}" = "workflow_dispatch" ]; }; }; then
  WRITE_HISTORY=true
else
  WRITE_HISTORY=false
fi

HISTORY_MOUNT=()
if [ "$WRITE_HISTORY" = "true" ]; then
  mkdir -p "$KPI_HISTORY_DIR"
  HISTORY_MOUNT=(-v "$KPI_HISTORY_DIR:/kpi-history")
elif [ -d "$KPI_HISTORY_DIR" ]; then
  HISTORY_MOUNT=(-v "$KPI_HISTORY_DIR:/kpi-history:ro")
else
  echo "Note: KPI history dir not found ($KPI_HISTORY_DIR); running without diff-vs-previous."
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

docker run --runtime=nvidia --gpus all --rm $TTY_FLAG \
  -v "$(pwd):/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output" \
  -v "$DATASET_MOUNT_SRC:/datasets:ro" \
  "${HISTORY_MOUNT[@]}" \
  -e OUTPUT_DIR=/output \
  -e DATASETS_ROOT=/datasets \
  -e KPI_HISTORY=/kpi-history \
  -e EVAL_WRITE_HISTORY="$WRITE_HISTORY" \
  -e RUN_ID="$RUN_ID" \
  -e MAX_WORKERS="$MAX_WORKERS" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  cuvslam:local /cuvslam/scripts/run_eval.sh
