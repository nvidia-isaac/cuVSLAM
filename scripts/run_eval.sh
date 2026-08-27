#!/bin/bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/output}"
DATASETS_ROOT="${DATASETS_ROOT:-/datasets}"
KPI_HISTORY="${KPI_HISTORY:-/kpi-history}"
WRITE_HISTORY="${EVAL_WRITE_HISTORY:-true}"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d)}"
MAX_WORKERS="${MAX_WORKERS:-12}"

CHOWN_TARGETS=("$OUTPUT_DIR")
[ "$WRITE_HISTORY" = "true" ] && CHOWN_TARGETS+=("$KPI_HISTORY")
trap 'chown -R "${HOST_UID:-0}:${HOST_GID:-0}" "${CHOWN_TARGETS[@]}" 2>/dev/null || true' EXIT

if [ ! -d "$OUTPUT_DIR/build" ]; then
  echo "Error: $OUTPUT_DIR/build not found."
  echo "Run scripts/build_cuvslam_in_docker.sh Release $OUTPUT_DIR first."
  exit 1
fi

EVAL_STATS="$OUTPUT_DIR/eval/stats"
mkdir -p "$EVAL_STATS"
[ "$WRITE_HISTORY" = "true" ] && mkdir -p "$KPI_HISTORY"

echo "=== Installing cuvslam python bindings against $OUTPUT_DIR/build ==="
CUVSLAM_BUILD_DIR="$OUTPUT_DIR/build" SKBUILD_BUILD_DIR=/tmp/skbuild \
  pip install /cuvslam/python/

echo "=== Installing cuvslam tools ==="
(
  python_tools_install_src="$(mktemp -d)"
  trap 'rm -rf "$python_tools_install_src"' EXIT
  cp -a /cuvslam/tools/python_tools/. "$python_tools_install_src/"
  pip install "${python_tools_install_src}[pdf]"
)

# Evaluation records come from the dataset registry, not a local array. The
# registry is standard library only, so PYTHONPATH is enough here; this runs
# before the tools package is installed below.
dataset_registry() {
  PYTHONPATH="/cuvslam/tools/python_tools${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m cuvslam_tools.dataset_registry "$@"
}

dataset_registry validate

DATASETS=()
while IFS= read -r record; do
  DATASETS+=("$record")
done < <(dataset_registry eval-records)

if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "Error: the dataset registry lists no evaluation records." >&2
  exit 1
fi

echo "=== Datasets present under $DATASETS_ROOT ==="
if [ -d "$DATASETS_ROOT" ]; then
  find "$DATASETS_ROOT" -mindepth 1 -maxdepth 1 -printf '  %f\n' 2>/dev/null | sort \
    || ls -1 "$DATASETS_ROOT" 2>/dev/null | sed 's/^/  /'
else
  echo "  (datasets root does not exist: $DATASETS_ROOT)"
fi

requested=()
missing=()
for record in "${DATASETS[@]}"; do
  IFS=$'\t' read -r name label _cfg _flags <<< "$record"
  requested+=("$label")
  [ -d "$DATASETS_ROOT/$name" ] || missing+=("$label -> $DATASETS_ROOT/$name")
done

echo "=== Requested datasets (${#requested[@]}): ${requested[*]} ==="
if [ "${#missing[@]}" -gt 0 ]; then
  echo "ERROR: ${#missing[@]} requested dataset(s) not present under $DATASETS_ROOT:" >&2
  for m in "${missing[@]}"; do echo "  MISSING: $m" >&2; done
  echo "Run ./scripts/stage_eval_datasets.sh or the Provision dataset workflow." >&2
  exit 1
fi
echo "All requested datasets present; proceeding with eval."

mkdir -p /sequences

export CUVSLAM_DATASETS=/sequences
export CUVSLAM_OUTPUT="$EVAL_STATS"

cd /cuvslam/tools/cuvslam_app

for record in "${DATASETS[@]}"; do
  IFS=$'\t' read -r name label test_config app_flags <<< "$record"

  # The mount name equals the dataset ID, which is also the "dataset_folder"
  # recorded in the shipped reporter config.
  ln -sfn "$DATASETS_ROOT/$name" "/sequences/$name"

  echo "=== Running cuVSLAM eval on $label ($test_config) ==="
  # shellcheck disable=SC2086
  python3 cuvslam_app.py \
    $app_flags \
    --test_config="$test_config" \
    --max_workers="$MAX_WORKERS" \
    --pdf
done

PREV_KPI=""
if [ -d "$KPI_HISTORY" ]; then
  PREV_KPI=$(ls -1 "$KPI_HISTORY"/kpi_[0-9]*.json 2>/dev/null | sort | tail -1 || true)
fi

KPI_JSON="$OUTPUT_DIR/eval/kpi_${RUN_ID}.json"
KPI_REPORT_JSON="$OUTPUT_DIR/eval/kpi_${RUN_ID}.report.json"
KPI_ARGS=(
  collect
  -s "$CUVSLAM_OUTPUT"
  -j "$KPI_JSON"
  -r "$KPI_REPORT_JSON"
  -d "$RUN_ID"
)
if [ -n "$PREV_KPI" ]; then
  echo "Using previous KPI history: $PREV_KPI"
  KPI_ARGS+=(-k "$PREV_KPI")
else
  echo "No previous KPI history found, starting fresh"
fi

BASELINE_RANGES="/cuvslam/scripts/kpi_baseline_ranges.json"
if [ -f "$BASELINE_RANGES" ]; then
  echo "Using baseline ranges for drift check: $BASELINE_RANGES"
  KPI_ARGS+=(-b "$BASELINE_RANGES")
fi

python3 /cuvslam/scripts/cuvslam_kpi_report.py "${KPI_ARGS[@]}"

# Keep the old outputs until CI has switched to the report JSON. A follow-up
# script-only change can remove them without another workflow migration.
python3 /cuvslam/scripts/cuvslam_kpi_report.py render \
  -r "$KPI_REPORT_JSON" \
  -o "${KPI_JSON}.table"
python3 /cuvslam/scripts/cuvslam_kpi_report.py drift \
  -r "$KPI_REPORT_JSON" \
  -o "${KPI_JSON}.drift"

if [ "$WRITE_HISTORY" = "true" ]; then
  # The S3-backed history mount has no rename(2), so publish the KPI JSON with a
  # direct copy (no atomic rename available on this mount).
  cp -f "$KPI_JSON" "$KPI_HISTORY/kpi_${RUN_ID}.json"
  echo "Persisted KPI history: $KPI_HISTORY/kpi_${RUN_ID}.json"
  if [ -f "$BASELINE_RANGES" ]; then
    cp "$BASELINE_RANGES" "$KPI_HISTORY/kpi_baseline_ranges.json"
    echo "Deployed baseline ranges: $KPI_HISTORY/kpi_baseline_ranges.json"
  fi
else
  echo "Read-only KPI history: baseline not modified (diff-only against existing history)."
fi

echo "=== Eval complete. KPI report data: ${KPI_REPORT_JSON} ==="
