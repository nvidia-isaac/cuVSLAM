#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
  echo "Usage: ./scripts/verify_pycuvslam_wheel_in_docker.sh <build_output_dir> [<expected_version> <expected_git_sha>]"
  echo "  Run after build_pycuvslam_in_docker.sh. Installs the repaired wheel into a"
  echo "  fresh environment and imports cuvslam, verifying the wheel filename is valid"
  echo "  (pip-installable) and the auditwheel-repaired extension loads with the"
  echo "  excluded CUDA libraries resolved from the system."
  echo "  When expected_version and expected_git_sha are provided, also verifies that"
  echo "  get_version() identifies that clean source revision without '-modified'."
  exit 1
fi

OUTPUT_DIR=$(realpath "$1")
EXPECTED_VERSION="${2:-}"
EXPECTED_GIT_SHA="${3:-}"

if [ "$#" -eq 3 ] && [ -z "$EXPECTED_VERSION" ]; then
  echo "Error: expected_version must not be empty when provenance verification is requested." >&2
  exit 1
fi
if [ "$#" -eq 3 ] && [[ ! "$EXPECTED_GIT_SHA" =~ ^[0-9a-fA-F]{40,64}$ ]]; then
  echo "Error: expected_git_sha must be a full hexadecimal Git object ID." >&2
  exit 1
fi

shopt -s nullglob
WHEELS=("$OUTPUT_DIR"/wheel/*.whl)
shopt -u nullglob

if [ "${#WHEELS[@]}" -eq 0 ]; then
  echo "Error: no wheel found in $OUTPUT_DIR/wheel."
  echo "Run './scripts/build_pycuvslam_in_docker.sh $OUTPUT_DIR' first."
  exit 1
elif [ "${#WHEELS[@]}" -gt 1 ]; then
  echo "Error: expected exactly one wheel in $OUTPUT_DIR/wheel, found ${#WHEELS[@]}:"
  printf '  %s\n' "${WHEELS[@]}"
  echo "Remove stale wheels (or rebuild into a clean output dir) before rerunning."
  exit 1
fi

WHEEL_NAME=$(basename "${WHEELS[0]}")

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

# --network host so pip can resolve the wheel's declared runtime deps (pyyaml).
# --system-site-packages keeps numpy/scipy from the image available so this stays a
# wheel-install/load smoke test rather than a full dependency-completeness audit.
docker run --runtime=nvidia --gpus all --rm --label "cuvslam.runner=${RUNNER_NAME:-local}" $TTY_FLAG --network host \
  --user "$(id -u):$(id -g)" --group-add video -e HOME=/tmp \
  -v "$OUTPUT_DIR:/output:ro" \
  -e WHEEL_NAME="$WHEEL_NAME" \
  -e EXPECTED_VERSION="$EXPECTED_VERSION" \
  -e EXPECTED_GIT_SHA="$EXPECTED_GIT_SHA" \
  cuvslam:local bash -c '
    set -euo pipefail
    python3 -m venv --system-site-packages /tmp/wheel_venv
    . /tmp/wheel_venv/bin/activate
    pip install --no-cache-dir "/output/wheel/$WHEEL_NAME"
    cd /tmp
    python3 - <<PY
import os

import cuvslam

version_info = cuvslam.get_version()
actual_version = version_info[0]
expected_version = os.environ.get("EXPECTED_VERSION", "")
expected_git_sha = os.environ.get("EXPECTED_GIT_SHA", "").lower()

if expected_version:
    expected_prefix = f"{expected_version}+"
    if not actual_version.startswith(expected_prefix):
        raise SystemExit(
            f"Expected runtime version prefix {expected_prefix}, got {actual_version}"
        )

    revision = actual_version[len(expected_prefix):]
    if revision.endswith("-modified"):
        raise SystemExit(f"Expected a clean runtime version, got {actual_version}")
    if len(revision) < 7 or not expected_git_sha.startswith(revision.lower()):
        raise SystemExit(
            f"Runtime revision {revision} does not identify Git SHA {expected_git_sha}"
        )

print("cuvslam wheel import OK, version:", version_info)
PY
  '
