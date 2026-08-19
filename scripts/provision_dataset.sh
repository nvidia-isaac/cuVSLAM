#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
source "$SCRIPT_DIR/datasets_config.sh"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <dataset>" >&2
  echo "  Datasets: ${PROVISIONABLE_DATASETS[*]}" >&2
  exit 1
fi

DATASET="$1"

export AWS_DEFAULT_REGION
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"
DRY_RUN="${DRY_RUN:-false}"

if ! is_provisionable_dataset "$DATASET"; then
  echo "Error: '$DATASET' is not a provisionable dataset (expected: ${PROVISIONABLE_DATASETS[*]})." >&2
  exit 1
fi

if [ "$DRY_RUN" != "true" ] && ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI not found. This script runs inside the cuvslam-ci:local image." >&2
  exit 1
fi

download_args=()
[ "$FORCE_DOWNLOAD" = "true" ] && download_args+=(--force-download)

if [ -n "${PROVISION_WORK_DIR:-}" ]; then
  WORK_DIR="$PROVISION_WORK_DIR"
else
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cuvslam-provision.XXXXXX")"
  trap 'rm -rf "$WORK_DIR"' EXIT
fi

WORK_DIR="${WORK_DIR%/}"
case "$WORK_DIR" in
  *..*) echo "Error: PROVISION_WORK_DIR must not contain '..': '$WORK_DIR'" >&2; exit 1 ;;
  /*/*) : ;;
  *) echo "Error: PROVISION_WORK_DIR must be an absolute path with at least two components: '$WORK_DIR'" >&2; exit 1 ;;
esac
case "$WORK_DIR/" in
  /bin/*|/boot/*|/dev/*|/etc/*|/lib/*|/lib64/*|/proc/*|/root/*|/run/*|/sbin/*|/srv/*|/sys/*|/usr/*|/var/*|/home/*)
    echo "Error: refusing to use a system path as PROVISION_WORK_DIR: '$WORK_DIR'" >&2; exit 1 ;;
esac

raw_dir="$WORK_DIR/raw"
converted_dir="$WORK_DIR/converted"
tarball="$WORK_DIR/${DATASET}.tar"
s3_tarball="$(s3_tarball_uri "$DATASET")"

prepare_rel="$(dataset_prepare_script "$DATASET")"
upload_subdir="$(dataset_upload_subdir "$DATASET")"
upload_src="$converted_dir${upload_subdir:+/$upload_subdir}"
prepare_script="$REPO_ROOT/$prepare_rel"

if [ ! -f "$prepare_script" ]; then
  echo "Error: dataset preparation script not found: $prepare_rel" >&2
  exit 1
fi

if [ "$DRY_RUN" != "true" ]; then
  echo "=== Verifying AWS credentials for S3 upload ==="
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    echo "Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for upload." >&2
    exit 1
  fi
  cred_rc=0
  caller_arn="$(aws sts get-caller-identity --query Arn --output text 2>&1)" || cred_rc=$?
  if [ "$cred_rc" -ne 0 ]; then
    echo "Error: AWS credential check failed (sts get-caller-identity, exit $cred_rc):" >&2
    echo "$caller_arn" >&2
    echo "Set secrets AWS_S3_ACCESS_KEY_ID and AWS_S3_SECRET_ACCESS_KEY to credentials authorized for $S3_DATASETS_BUCKET." >&2
    exit 1
  fi
  echo "AWS credentials OK ($caller_arn)"
fi

rm -rf "$raw_dir" "$converted_dir" "$tarball"
mkdir -p "$raw_dir" "$converted_dir"

bash "$prepare_script" \
  --raw-dir "$raw_dir" \
  --output-dir "$converted_dir" \
  "${download_args[@]}"

file_count="$(find "$upload_src" -type f | wc -l)"
if [ "$file_count" -eq 0 ]; then
  echo "Error: $upload_src has no files; refusing to upload an empty tarball." >&2
  exit 1
fi

echo "=== Creating tarball from $upload_src ==="
echo "Archiving ${file_count} files"
sync
tar_rc=0
tar -C "$upload_src" -cf "$tarball" \
  --warning=no-file-changed \
  --checkpoint=1000 \
  --checkpoint-action=echo='tar checkpoint %d' \
  --totals \
  . || tar_rc=$?
if [ "$tar_rc" -ne 0 ]; then
  echo "Error: tar failed with exit code $tar_rc" >&2
  exit 1
fi
ls -lh "$tarball"

if [ "$DRY_RUN" = "true" ]; then
  echo "DRY_RUN: skipping upload to $s3_tarball"
  exit 0
fi

echo "=== Uploading to $s3_tarball ==="
aws s3 cp "$tarball" "$s3_tarball" --no-progress
aws s3 ls "$s3_tarball" --summarize
echo "Provision complete: $s3_tarball"
