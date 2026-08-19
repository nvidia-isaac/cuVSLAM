#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
  echo "Usage: ./scripts/verify_pycuvslam_wheel_in_docker.sh <build_output_dir> [<expected_version> <expected_git_sha>]"
  echo "  Run after build_pycuvslam_in_docker.sh. Installs the repaired wheel into a"
  echo "  fresh environment and imports cuvslam, verifying the wheel filename is valid"
  echo "  (pip-installable) and the auditwheel-repaired extension loads with the"
  echo "  excluded CUDA libraries resolved from the system."
  echo "  Then reinstalls it with its cu12/cu13 extra in a clean environment and verifies that"
  echo "  the excluded CUDA libraries also resolve from the nvidia-* pip packages alone."
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

# Wheels are versioned <version>+cu12/<version>+cu13 by build_pycuvslam_in_docker.sh; the same tag names the extra
# that provides the CUDA math libraries excluded from the wheel. Empty for wheels built without the tag.
CUDA_EXTRA=$(echo "$WHEEL_NAME" | grep -oE "\+cu[0-9]+" | tr -d "+" || true)
if [ -z "$CUDA_EXTRA" ]; then
  echo "Note: $WHEEL_NAME carries no +cuNN version tag; skipping the pip-provided CUDA libraries check."
fi

# The extras target x86_64 systems without a CUDA Toolkit. The aarch64 entries in the wheel matrix are Jetson, where
# CUDA comes from JetPack and the aarch64 nvidia-* wheels are server-ARM (SBSA) builds, so installing those over a
# Tegra CUDA is not the configuration this check certifies. Decide that from the architecture, before touching pip: a
# resolver failure is what a broken extras declaration looks like too, and that must fail the verification.
HOST_ARCH=$(uname -m)
if [ -n "$CUDA_EXTRA" ] && [ "$HOST_ARCH" != "x86_64" ]; then
  echo "SKIPPED: no pip-provided CUDA libraries check on $HOST_ARCH; the $CUDA_EXTRA extra targets x86_64 systems."
  CUDA_EXTRA=""
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

# --network host so pip can resolve the wheel's declared runtime deps (pyyaml).
# --system-site-packages keeps numpy/scipy from the image available so this stays a
# wheel-install/load smoke test rather than a full dependency-completeness audit.
docker run --runtime=nvidia --gpus all --rm $TTY_FLAG --network host \
  --user "$(id -u):$(id -g)" --group-add video -e HOME=/tmp \
  -v "$OUTPUT_DIR:/output:ro" \
  -e WHEEL_NAME="$WHEEL_NAME" \
  -e CUDA_EXTRA="$CUDA_EXTRA" \
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

    if [ -z "$CUDA_EXTRA" ]; then
      exit 0
    fi

    # The wheel does not bundle cuBLAS/cuSOLVER/cuSPARSE, and this image provides them system-wide, so the check
    # above cannot tell a wheel that declares them from one that silently depends on the CUDA Toolkit being
    # installed. Install the wheel with its CUDA extra into a clean environment instead, and require that the
    # libraries actually loaded are the pip-provided ones.
    echo "--- Verifying the [$CUDA_EXTRA] extra provides the CUDA math libraries ---"
    python3 -m venv /tmp/wheel_venv_cuda_extra
    . /tmp/wheel_venv_cuda_extra/bin/activate
    # A resolution failure here means the extra does not describe an installable set of packages, which is the very
    # thing this stage exists to catch, so it fails the verification.
    pip install --no-cache-dir "/output/wheel/$WHEEL_NAME[$CUDA_EXTRA]"
    cd /tmp
    python3 - <<PY
import os

# Imported for its side effect: the preloading this stage verifies happens while cuvslam is imported.
import cuvslam
from cuvslam import _cuda_libs

# Require the set the package itself declares it needs rather than a copy of it, and resolve it the same way the
# import-time preload does. A component listed in _cuda_libs.CUDA_COMPONENTS with no matching entry in the extras of
# pyproject.toml is precisely the drift this stage exists to catch, and the two lists stay independent that way.
nvidia_root = _cuda_libs.nvidia_root()
with open("/proc/self/maps") as maps_file:
    maps = maps_file.read()

unresolved = [name for name in _cuda_libs.CUDA_COMPONENTS
              if os.path.join(nvidia_root, name, "lib") not in maps]
if unresolved:
    raise SystemExit(
        "cuvslam did not load " + ", ".join(unresolved) + " from " + nvidia_root +
        ": the CUDA extra does not cover every CUDA library excluded from the wheel"
    )

print("cuvslam wheel import OK with CUDA libraries from", nvidia_root)
PY
  '
