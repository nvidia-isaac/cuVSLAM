#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
source "$SCRIPT_DIR/datasets_config.sh"

export AWS_DEFAULT_REGION
RUNNER_LOCAL_DATASETS_ROOT="${RUNNER_LOCAL_DATASETS_ROOT:-${HOME:-/tmp}/.cache/cuvslam}"
LOCAL_DATASETS_DIR="$RUNNER_LOCAL_DATASETS_ROOT/datasets/vslam"
FORCE_RESTAGE="${FORCE_RESTAGE:-false}"

S3_BUCKET="$(s3_dataset_bucket)"

mkdir -p "$LOCAL_DATASETS_DIR"

echo "Dataset cache root: $LOCAL_DATASETS_DIR (RUNNER_LOCAL_DATASETS_ROOT=$RUNNER_LOCAL_DATASETS_ROOT)"

have_aws=false
if command -v aws >/dev/null 2>&1; then
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    have_aws=true
  elif aws sts get-caller-identity >/dev/null 2>&1; then
    have_aws=true
  fi
fi

# One greppable line per staged dataset. The step duration is already visible in
# the Actions UI, but that covers every dataset at once, so report the per-dataset
# figures the UI cannot show: payload size, file count, and the throughput that
# makes two runs comparable at a glance.
report_staging_profile() {
  local name="$1" bytes="$2" total_seconds="$3" file_count="$4"
  local mib="unknown" total_rate="unknown"
  if [[ "$bytes" =~ ^[0-9]+$ ]]; then
    mib=$((bytes / 1048576))
    [ "$total_seconds" -gt 0 ] && total_rate=$((mib / total_seconds))
  fi
  echo "staging profile: dataset=$name mib=$mib total_s=$total_seconds" \
    "files=$file_count total_mib_s=$total_rate"
}

stage_one() {
  local name="$1"
  local s3_key s3_uri
  s3_key="$(s3_dataset_key "$name")"
  s3_uri="$(s3_tarball_uri "$name")"
  local dest="$LOCAL_DATASETS_DIR/$name"
  local etag_file="$dest/.s3_etag"

  remote_etag=""
  remote_bytes=""
  if $have_aws; then
    # ETag and size come from one request so the staging profile can report
    # throughput without stat(2) on a local copy.
    head_output="$(aws s3api head-object --bucket "$S3_BUCKET" --key "$s3_key" \
      --query '[ETag,ContentLength]' --output text 2>/dev/null || true)"
    if [ -n "$head_output" ]; then
      remote_etag="$(printf '%s' "$head_output" | cut -f1)"
      remote_bytes="$(printf '%s' "$head_output" | cut -f2)"
    fi
  fi

  cache_has_files=false
  if [ -d "$dest" ] && [ -n "$(find "$dest" -type f ! -name '.s3_etag' -print -quit 2>/dev/null)" ]; then
    cache_has_files=true
  fi

  cached_etag="none"
  [ -f "$etag_file" ] && cached_etag="$(cat "$etag_file")"
  echo "Cache check $name: have_aws=$have_aws cache_has_files=$cache_has_files" \
    "force_restage=$FORCE_RESTAGE remote_etag=${remote_etag:-none} cached_etag=$cached_etag"

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
  # Extract as the object arrives: peak disk is one expanded dataset instead of
  # the archive plus its expansion, which matters most for the largest corpora.
  # Extraction lands in a sibling directory and is swapped in only on success,
  # so a failed transfer leaves any existing cache intact.
  local staging_dir="${dest}.partial"
  rm -rf "$staging_dir"
  mkdir -p "$staging_dir"
  local stream_start=$SECONDS
  # pipefail is set at the top of this script, so a failed download or a
  # truncated archive fails the pipeline rather than yielding a partial dataset.
  if ! aws s3 cp "$s3_uri" - --no-progress | tar -xf - -C "$staging_dir"; then
    rm -rf "$staging_dir"
    echo "Error: staging $name failed; existing cache at $dest is unchanged." >&2
    exit 1
  fi
  local stream_seconds=$((SECONDS - stream_start))

  rm -rf "$dest"
  mv "$staging_dir" "$dest"
  if [ -n "$remote_etag" ]; then
    echo "$remote_etag" > "$etag_file"
  fi
  local file_count
  file_count="$(find "$dest" -type f ! -name '.s3_etag' | wc -l)"
  echo "  staged $file_count files under $dest"
  report_staging_profile "$name" "$remote_bytes" "$stream_seconds" "$file_count"
}

for name in "${EVAL_DATASET_NAMES[@]}"; do
  stage_one "$name"
done

echo "Dataset staging complete: $LOCAL_DATASETS_DIR"
