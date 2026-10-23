#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/test_pycuvslam_in_docker.sh <build_output_dir>"
  echo "  Run after build_cuvslam_in_docker.sh. Expects build/ in the output dir."
  exit 1
fi

OUTPUT_DIR=$(realpath "$1")

if [ ! -d "$OUTPUT_DIR/build" ]; then
  echo "Error: $OUTPUT_DIR/build not found."
  echo "Run './scripts/build_cuvslam_in_docker.sh Release $OUTPUT_DIR' first."
  exit 1
fi

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

docker run --runtime=nvidia --gpus all --rm --label "cuvslam.runner=${RUNNER_NAME:-local}" $TTY_FLAG \
  -v "$(pwd):/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  cuvslam:local bash -c '
    set -euo pipefail
    trap "chown -R $HOST_UID:$HOST_GID /output" EXIT
    CUVSLAM_BUILD_DIR=/output/build SKBUILD_BUILD_DIR=/tmp/skbuild \
      pip install /cuvslam/python/
    python3 -m unittest discover -v -s /cuvslam/python/test 2>&1 | \
      tee /output/python-test-output.txt
  '
