#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
source "$SCRIPT_DIR/datasets_config.sh"

export AWS_DEFAULT_REGION
RUNNER_LOCAL_DATASETS_ROOT="${RUNNER_LOCAL_DATASETS_ROOT:-${HOME:-/tmp}/.cache/cuvslam}"
LOCAL_DATASETS_DIR="$RUNNER_LOCAL_DATASETS_ROOT/datasets/vslam"
FORCE_RESTAGE="${FORCE_RESTAGE:-false}"

_s3_path="${S3_DATASETS_BUCKET#s3://}"
S3_BUCKET="${_s3_path%%/*}"
S3_KEY_PREFIX="${_s3_path#*/}"

mkdir -p "$LOCAL_DATASETS_DIR"

have_aws=false
if command -v aws >/dev/null 2>&1; then
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    have_aws=true
  elif aws sts get-caller-identity >/dev/null 2>&1; then
    have_aws=true
  fi
fi

stage_one() {
  local name="$1"
  local s3_key="${S3_KEY_PREFIX}/${name}.tar"
  local s3_uri="s3://${S3_BUCKET}/${s3_key}"
  local dest="$LOCAL_DATASETS_DIR/$name"
  local etag_file="$dest/.s3_etag"
  local tmp_tar
  tmp_tar="$(mktemp "${TMPDIR:-/tmp}/cuvslam-${name}.XXXXXX.tar")"

  remote_etag=""
  if $have_aws; then
    remote_etag="$(aws s3api head-object --bucket "$S3_BUCKET" --key "$s3_key" --query ETag --output text 2>/dev/null || true)"
  fi

  cache_has_files=false
  if [ -d "$dest" ] && [ -n "$(find "$dest" -type f ! -name '.s3_etag' -print -quit 2>/dev/null)" ]; then
    cache_has_files=true
  fi

  if [ "$FORCE_RESTAGE" != "true" ] && $cache_has_files; then
    if ! $have_aws; then
      echo "Using cached dataset $name at $dest (no AWS credentials; cannot verify freshness)"
      return 0
    fi
    if [ -n "$remote_etag" ] && [ -f "$etag_file" ] && [ "$(cat "$etag_file")" = "$remote_etag" ]; then
      echo "Using cached dataset $name at $dest (etag match)"
      return 0
    fi
  fi

  if ! $have_aws; then
    echo "Error: dataset $name not in local cache ($dest) and AWS credentials are unset." >&2
    exit 1
  fi

  echo "Staging $name from $s3_uri -> $dest"
  aws s3 cp "$s3_uri" "$tmp_tar" --no-progress
  rm -rf "$dest"
  mkdir -p "$dest"
  tar -xf "$tmp_tar" -C "$dest"
  rm -f "$tmp_tar"
  if [ -n "$remote_etag" ]; then
    echo "$remote_etag" > "$etag_file"
  fi
  echo "  staged $(find "$dest" -type f ! -name '.s3_etag' | wc -l) files under $dest"
}

for name in "${EVAL_DATASET_NAMES[@]}"; do
  stage_one "$name"
done

echo "Dataset staging complete: $LOCAL_DATASETS_DIR"
