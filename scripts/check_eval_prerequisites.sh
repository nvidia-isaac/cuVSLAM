#!/bin/bash
set -euo pipefail

: "${RUNNER_STORAGE_ROOT:?Set repository variable RUNNER_STORAGE_ROOT for KPI history on runner storage.}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
source "$SCRIPT_DIR/datasets_config.sh"

LOCAL_DATASETS_DIR="${RUNNER_LOCAL_DATASETS_ROOT:-${HOME:-/tmp}/.cache/cuvslam}/datasets/vslam"

have_aws=false
if command -v aws >/dev/null 2>&1; then
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    have_aws=true
  elif aws sts get-caller-identity >/dev/null 2>&1; then
    have_aws=true
  fi
fi

resolve_eval_suite_args
dataset_registry validate "${EVAL_SUITE_ARGS[@]}"

# Captured rather than piped into the loop: a failing or empty listing would
# otherwise skip the body and leave cache_ok true, reporting success. The suite
# must match staging, or this reports a warm cache for datasets the run skips.
if ! eval_datasets="$(dataset_registry list --eval "${EVAL_SUITE_ARGS[@]}")" || [ -z "$eval_datasets" ]; then
  echo "Error: the dataset registry lists no evaluation datasets for this suite." >&2
  exit 1
fi

cache_ok=true
while read -r name; do
  dest="$LOCAL_DATASETS_DIR/$name"
  if [ ! -d "$dest" ] || [ -z "$(find "$dest" -type f ! -name '.s3_etag' -print -quit 2>/dev/null)" ]; then
    cache_ok=false
    break
  fi
done <<< "$eval_datasets"

if $have_aws; then
  echo "Eval prerequisites OK (S3 tarball staging; KPI history under $RUNNER_STORAGE_ROOT)"
elif $cache_ok; then
  echo "Note: no AWS CLI with credentials here; using the staged dataset cache at $LOCAL_DATASETS_DIR." >&2
  echo "Eval prerequisites OK (cached datasets; KPI history under $RUNNER_STORAGE_ROOT)"
else
  echo "Error: eval datasets missing at $LOCAL_DATASETS_DIR, and no AWS CLI with credentials to stage them." >&2
  echo "Run ./scripts/stage_eval_datasets.sh with AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY set" \
    "(in CI, from the AWS_S3_RO_* repository secrets)." >&2
  exit 1
fi
