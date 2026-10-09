#!/bin/bash
# One-time upgrade of the m3ed_spot tarball in S3 from converter v1 to v2 by
# adding the IMU, without re-reading the source images. Branch-only; not meant
# for main. Downloads the tarball, patches it in place, and uploads it back
# only if a file changed, keeping a server-side copy of the original.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
source "$SCRIPT_DIR/datasets_config.sh"

DATASET=m3ed_spot
export AWS_DEFAULT_REGION
DRY_RUN="${DRY_RUN:-false}"

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI not found. This script runs inside the cuvslam-ci:local image." >&2
  exit 1
fi
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  echo "Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set." >&2
  exit 1
fi
echo "AWS credentials: $(aws sts get-caller-identity --query Arn --output text)"

if [ -n "${PATCH_WORK_DIR:-}" ]; then
  WORK_DIR="${PATCH_WORK_DIR%/}"
else
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cuvslam-patch.XXXXXX")"
  trap 'rm -rf "$WORK_DIR"' EXIT
fi
case "$WORK_DIR" in
  *..*) echo "Error: PATCH_WORK_DIR must not contain '..': '$WORK_DIR'" >&2; exit 1 ;;
  /*/*) : ;;
  *) echo "Error: PATCH_WORK_DIR must be an absolute path with at least two components" >&2; exit 1 ;;
esac

bucket="$(s3_dataset_bucket)"
key="$(s3_dataset_key "$DATASET")"
s3_tarball="$(s3_tarball_uri "$DATASET")"
root="$WORK_DIR/root"
tarball="$WORK_DIR/${DATASET}.tar"
report="$WORK_DIR/report.json"
rm -rf "$root" "$tarball" "$report"
mkdir -p "$root"

etag_before="$(aws s3api head-object --bucket "$bucket" --key "$key" --query ETag --output text)"
echo "=== Downloading $s3_tarball (ETag $etag_before) ==="
aws s3 cp "$s3_tarball" - --no-progress | tar -xf - -C "$root"
echo "Extracted $(find "$root" -type f | wc -l) files"

echo "=== Installing cuvslam tools ==="
(
  install_src="$(mktemp -d)"
  trap 'rm -rf "$install_src"' EXIT
  cp -a "$CUVSLAM_REPO_ROOT/tools/python_tools/." "$install_src/"
  pip install --no-cache-dir "$install_src"
)

echo "=== Patching ==="
python3 -m cuvslam_tools.dataset_preparation.m3ed_spot.patch_imu --root "$root" --report "$report"
changed="$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["changed_files"]))' "$report")"
if [ "$changed" -eq 0 ]; then
  echo "Nothing changed; $s3_tarball is already up to date, skipping upload."
  exit 0
fi
echo "$changed file(s) changed"

echo "=== Creating tarball ==="
sync
tar -C "$root" -cf "$tarball" --checkpoint=100000 --checkpoint-action=echo='%T' --totals .
ls -lh "$tarball"

if [ "$DRY_RUN" = "true" ]; then
  echo "DRY_RUN: skipping upload to $s3_tarball"
  exit 0
fi

etag_now="$(aws s3api head-object --bucket "$bucket" --key "$key" --query ETag --output text)"
if [ "$etag_now" != "$etag_before" ]; then
  echo "Error: $s3_tarball changed while patching (ETag $etag_before -> $etag_now); not overwriting." >&2
  exit 1
fi

backup="${s3_tarball%.tar}.pre-imu-$(date -u +%Y%m%d%H%M%S).tar"
echo "=== Backing up the original to $backup ==="
# The default copy also copies tags, which needs s3:GetObjectTagging; the
# provisioning key does not have it.
aws s3 cp "$s3_tarball" "$backup" --copy-props metadata-directive --no-progress

echo "=== Uploading to $s3_tarball ==="
aws s3 cp "$tarball" "$s3_tarball" --no-progress
aws s3 ls "$s3_tarball" --summarize
echo "Patch complete: $s3_tarball (original kept at $backup)"
