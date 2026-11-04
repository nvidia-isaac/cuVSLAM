#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/build_pycuvslam_in_docker.sh <build_output_dir>"
  echo "  Run after build_cuvslam_in_docker.sh. Expects build/ in the output dir."
  echo ""
  echo "Environment variables (optional):"
  echo "  CUDA_VERSION  CUDA version used for build (default: 12.6.3)"
  exit 1
fi

OUTPUT_DIR=$(realpath "$1")
CUDA_VERSION="${CUDA_VERSION:-12.6.3}"

if [ ! -d "$OUTPUT_DIR/build" ]; then
  echo "Error: $OUTPUT_DIR/build not found."
  echo "Run './scripts/build_cuvslam_in_docker.sh Release $OUTPUT_DIR' first."
  exit 1
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

docker run --runtime=nvidia --gpus all --rm --label "cuvslam.runner=${RUNNER_NAME:-local}" $TTY_FLAG \
  --user "$(id -u):$(id -g)" --group-add video -e HOME=/tmp \
  -v "$(pwd):/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output" \
  -e CUDA_VERSION="$CUDA_VERSION" \
  cuvslam:local bash -c '
    set -euo pipefail
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)

    if [ "$CUDA_MAJOR" = "12" ]; then
      EXCLUDES="--exclude libcusolver.so.11 --exclude libcublas.so.12 --exclude libcublasLt.so.12 --exclude libcusparse.so.12 --exclude libnvJitLink.so.12 --exclude libnvinfer.so.10"
    elif [ "$CUDA_MAJOR" = "13" ]; then
      EXCLUDES="--exclude libcusolver.so.12 --exclude libcublas.so.13 --exclude libcublasLt.so.13 --exclude libcusparse.so.12 --exclude libnvJitLink.so.13 --exclude libnvinfer.so.10"
    else
      echo "Error: unknown CUDA major version $CUDA_MAJOR, cannot determine auditwheel exclusions"
      exit 1
    fi

    cp -r /cuvslam /tmp/cuvslam_src
    python3 - /tmp/cuvslam_src/python/pyproject.toml "$CUDA_MAJOR" <<"PY"
import sys
from pathlib import Path

path, major = Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
replacements = [("name = \"cuvslam\"", f"name = \"cuvslam-cu{major}\"")]
if major != "12":
    replacements.append(("==12.*; platform_machine != \"aarch64\"", f"=={major}.*"))
for old, new in replacements:
    if text.count(old) != 1:
        raise SystemExit(f"Expected exactly one {old!r} in {path}")
    text = text.replace(old, new)
path.write_text(text)
PY

    CUVSLAM_BUILD_DIR=/output/build pip wheel --no-deps \
      -w /tmp/wheels /tmp/cuvslam_src/python/

    mkdir -p /output/wheel
    for whl in /tmp/wheels/*.whl; do
      auditwheel repair "$whl" -w /output/wheel $EXCLUDES
    done
  '
