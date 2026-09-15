#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
  echo "Usage: ./scripts/verify_pycuvslam_wheel_in_docker.sh <build_output_dir> [<expected_version> <expected_git_sha>]"
  echo "  Run after build_pycuvslam_in_docker.sh. Validates the three-package wheel"
  echo "  set and runs the Python suite against the repaired CUDA backend wheel."
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

if [ "${#WHEELS[@]}" -ne 3 ]; then
  echo "Error: expected one backend, one common, and one metadata wheel in $OUTPUT_DIR/wheel; found ${#WHEELS[@]}."
  echo "Run './scripts/build_pycuvslam_in_docker.sh $OUTPUT_DIR' first."
  printf '  %s\n' "${WHEELS[@]}"
  exit 1
fi

shopt -s nullglob
BACKEND_WHEELS=("$OUTPUT_DIR"/wheel/cuvslam_cu12-*.whl "$OUTPUT_DIR"/wheel/cuvslam_cu13-*.whl)
COMMON_WHEELS=("$OUTPUT_DIR"/wheel/cuvslam_common-*.whl)
META_WHEELS=("$OUTPUT_DIR"/wheel/cuvslam-[0-9]*.whl)
shopt -u nullglob

if [ "${#BACKEND_WHEELS[@]}" -ne 1 ] ||
   [ "${#COMMON_WHEELS[@]}" -ne 1 ] ||
   [ "${#META_WHEELS[@]}" -ne 1 ]; then
  echo "Error: malformed cuVSLAM wheel set:" >&2
  printf '  %s\n' "${WHEELS[@]}" >&2
  exit 1
fi

BACKEND_NAME=$(basename "${BACKEND_WHEELS[0]}")
if [[ "$BACKEND_NAME" == cuvslam_cu12-* ]]; then
  CUDA_MAJOR=12
elif [[ "$BACKEND_NAME" == cuvslam_cu13-* ]]; then
  CUDA_MAJOR=13
else
  echo "Error: cannot determine CUDA major from $BACKEND_NAME." >&2
  exit 1
fi
if [ "$CUDA_MAJOR" = "13" ]; then
  INSTALL_NAME=$(basename "${META_WHEELS[0]}")
else
  INSTALL_NAME="$BACKEND_NAME"
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

# Validate metadata, wheel contents, ownership, and ELF policy in the build image.
docker run --runtime=nvidia --gpus all --rm $TTY_FLAG \
  --user "$(id -u):$(id -g)" --group-add video -e HOME=/tmp \
  -v "$(pwd):/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output:ro" \
  -e CUDA_MAJOR="$CUDA_MAJOR" \
  cuvslam:local bash -c '
    set -euo pipefail
    wheels=(/output/wheel/*.whl)
    twine check --strict "${wheels[@]}"
    for wheel in "${wheels[@]}"; do
      if [[ "$(basename "$wheel")" == cuvslam-[0-9]*.whl ]]; then
        check-wheel-contents --ignore W007,W008 "$wheel"
      else
        check-wheel-contents "$wheel"
      fi
    done
    backend=(/output/wheel/cuvslam_cu"${CUDA_MAJOR}"-*.whl)
    auditwheel show "${backend[0]}"
    python3 /cuvslam/python/packaging/build_packages.py verify-ownership "${wheels[@]}"

    python3 -m wheel unpack --dest /tmp/verify-wheel "${backend[0]}"
    libcuvslam=(/tmp/verify-wheel/*/cuvslam/cu"${CUDA_MAJOR}"/libcuvslam.so)
    if [ "${#libcuvslam[@]}" -ne 1 ]; then
      echo "Error: backend wheel does not contain exactly one libcuvslam.so." >&2
      exit 1
    fi
    expected_rpath=$(python3 /cuvslam/python/packaging/build_packages.py backend-rpath \
      --source-root /cuvslam --cuda-major "$CUDA_MAJOR")
    actual_rpath=$(patchelf --print-rpath "${libcuvslam[0]}")
    IFS=: read -ra actual_entries <<< "$actual_rpath"
    for entry in "${actual_entries[@]}"; do
      case "$entry" in
        \$ORIGIN*) ;;
        *)
          echo "Error: libcuvslam.so contains a non-relative RUNPATH entry: $entry" >&2
          exit 1
          ;;
      esac
    done
    IFS=: read -ra expected_entries <<< "$expected_rpath"
    for entry in "${expected_entries[@]}"; do
      case ":$actual_rpath:" in
        *":$entry:"*) ;;
        *)
          echo "Error: libcuvslam.so RUNPATH is missing $entry: $actual_rpath" >&2
          exit 1
          ;;
      esac
    done
  '

if [ "$(uname -m)" = "x86_64" ]; then
  if [[ "$BACKEND_NAME" == *-cp310-* ]]; then
    VERIFY_IMAGE=python:3.10-slim
  else
    VERIFY_IMAGE=python:3.12-slim
  fi
else
  # Jetson release wheels intentionally resolve CUDA from JetPack.
  VERIFY_IMAGE=cuvslam:local
fi

PIP_CACHE_ROOT="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
mkdir -p "$PIP_CACHE_ROOT"

# --network host lets pip resolve the backend's declared PyYAML and CUDA
# dependencies. The x86 image has no CUDA toolkit, so a missing Requires-Dist or
# package-relative RUNPATH cannot be masked by the build environment.
docker run --runtime=nvidia --gpus all --rm $TTY_FLAG --network host \
  --user "$(id -u):$(id -g)" --group-add video -e HOME=/tmp \
  -v "$(pwd):/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output:ro" \
  -v "$PIP_CACHE_ROOT:/pip-cache" \
  -e BACKEND_NAME="$BACKEND_NAME" \
  -e CUDA_MAJOR="$CUDA_MAJOR" \
  -e INSTALL_NAME="$INSTALL_NAME" \
  -e EXPECTED_VERSION="$EXPECTED_VERSION" \
  -e EXPECTED_GIT_SHA="$EXPECTED_GIT_SHA" \
  -e PIP_CACHE_DIR=/pip-cache \
  "$VERIFY_IMAGE" bash -c '
    set -euo pipefail
    python3 -m venv /tmp/wheel_venv
    . /tmp/wheel_venv/bin/activate
    pip install --find-links /output/wheel "/output/wheel/$INSTALL_NAME"
    pip install -r /cuvslam/python/test/requirements.txt
    pip check
    cd /tmp
    python3 - <<PY
import os
from pathlib import Path

import cuvslam

version_info = cuvslam.get_version()
actual_version = version_info[0]
expected_version = os.environ.get("EXPECTED_VERSION", "")
expected_git_sha = os.environ.get("EXPECTED_GIT_SHA", "").lower()

if expected_version and cuvslam.__version__ != expected_version:
    raise SystemExit(
        f"Expected package version {expected_version}, got {cuvslam.__version__}"
    )

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

cuvslam.warm_up_gpu()

if os.uname().machine == "x86_64":
    maps = Path("/proc/self/maps").read_text()
    for soname in ("libcudart", "libcublas", "libcusolver", "libcusparse"):
        matching = [line for line in maps.splitlines() if soname in line]
        if not matching:
            raise SystemExit(f"{soname} was not loaded")
        if not any("site-packages/nvidia/" in line for line in matching):
            raise SystemExit(
                f"{soname} did not resolve from a declared pip dependency: {matching}"
            )

print("cuvslam wheel import OK, version:", version_info)
PY
    python3 -m unittest discover -v -s /cuvslam/python/test --locals
  '

if [ "$(uname -m)" = "x86_64" ] && [[ "$BACKEND_NAME" == *-abi3-* ]]; then
  docker run --runtime=nvidia --gpus all --rm $TTY_FLAG --network host \
    --user "$(id -u):$(id -g)" --group-add video -e HOME=/tmp \
    -v "$OUTPUT_DIR:/output:ro" \
    -v "$PIP_CACHE_ROOT:/pip-cache" \
    -e BACKEND_NAME="$BACKEND_NAME" \
    -e INSTALL_NAME="$INSTALL_NAME" \
    -e PIP_CACHE_DIR=/pip-cache \
    python:3.13-slim bash -c '
      set -euo pipefail
      python3 -m venv /tmp/wheel_venv
      . /tmp/wheel_venv/bin/activate
      pip install --find-links /output/wheel "/output/wheel/$INSTALL_NAME"
      pip check
      cd /tmp
      python3 -c "import cuvslam; cuvslam.warm_up_gpu()"
    '
fi
