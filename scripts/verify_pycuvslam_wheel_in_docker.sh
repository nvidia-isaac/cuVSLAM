#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ] && [ "$#" -ne 3 ]; then
  echo "Usage: ./scripts/verify_pycuvslam_wheel_in_docker.sh <build_output_dir> [<expected_version> <expected_git_sha>]"
  echo "  Run after build_pycuvslam_in_docker.sh. Installs each repaired wheel into a"
  echo "  fresh environment of the Python version in its tag (abi3 wheels also into"
  echo "  Python 3.14) and imports cuvslam, verifying the wheel filename is valid"
  echo "  (pip-installable) and the auditwheel-repaired extension loads with the"
  echo "  excluded CUDA libraries resolved from the system. Before installing, runs the"
  echo "  PyPI metadata check (twine check --strict) and, for abi3 wheels, abi3audit."
  echo "  The abi3 wheel also builds and loads an AnyLoc engine, with the TensorRT of"
  echo "  its tensorrt extra in place of the image's (Orin's wheel: JetPack's)."
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
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

# --network host so uv can fetch the Python versions and the wheels' runtime deps (pyyaml).
docker run --runtime=nvidia --gpus all --rm --label "cuvslam.runner=${RUNNER_NAME:-local}" $TTY_FLAG --network host \
  --user "$(id -u):$(id -g)" --group-add video -e HOME=/tmp \
  -v "$OUTPUT_DIR:/output:ro" \
  -e EXPECTED_VERSION="$EXPECTED_VERSION" \
  -e EXPECTED_GIT_SHA="$EXPECTED_GIT_SHA" \
  cuvslam:local bash -c '
    set -euo pipefail
    cd /tmp
    for wheel in /output/wheel/*.whl; do
      name=$(basename "$wheel")
      twine check --strict "$wheel"
      pythons=$(echo "$name" | sed -E "s/^[^-]+-[^-]+-cp3([0-9]+)-.*/3.\1/")
      case "$name" in
        *-abi3-*) abi3audit "$wheel"; pythons="$pythons 3.14" ;;
      esac
      for py in $pythons; do
        echo "=== $name on Python $py"
        venv="/tmp/venv-${name%.whl}-$py"
        uv venv -q --managed-python --python "$py" "$venv"
        uv pip install -q --python "$venv" "$wheel"
        "$venv/bin/python" - <<PY
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
      done
    done
  '

# The abi3 wheel's AnyLoc engine, with the TensorRT of its tensorrt extra once the image's is gone.
# Orin's wheel has no such extra and keeps JetPack's TensorRT.
docker run --runtime=nvidia --gpus all --rm --label "cuvslam.runner=${RUNNER_NAME:-local}" $TTY_FLAG --network host \
  -v "$OUTPUT_DIR:/output:ro" \
  cuvslam:local bash -c '
    set -euo pipefail
    cd /tmp
    for wheel in /output/wheel/*-abi3-*.whl; do
      name=$(basename "$wheel")
      requirement="${name%%-*} @ file://$wheel"
      if [[ "$name" != cuvslam_cu12-*_aarch64.whl ]]; then
        rm -f /usr/lib/*-linux-gnu/libnvinfer* /usr/lib/*-linux-gnu/libnvonnxparser*
        ldconfig
        if ldconfig -p | grep -E "libnv(infer|onnxparser)"; then
          echo "The image still has TensorRT" >&2
          exit 1
        fi
        requirement="${name%%-*}[tensorrt] @ file://$wheel"
      fi
      echo "=== $requirement: AnyLoc engine"
      venv="/tmp/venv-anyloc-${name%.whl}"
      uv venv -q --managed-python --python 3.12 "$venv"
      uv pip install -q --python "$venv" "$requirement"
      "$venv/bin/python" - <<PY
import cuvslam

cuvslam.build_anyloc_engine("/tmp/anyloc.engine")
cuvslam.VprVocabularyBuilder(cuvslam.Slam.VprSettings(model_path="/tmp/anyloc.engine"))
print("AnyLoc engine from the wheel OK")
PY
    done
  '
