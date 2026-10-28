#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./scripts/build_docs_in_docker.sh <build_output_dir>"
  echo "  Run after build_cuvslam_in_docker.sh. Expects build/ in the output dir."
  echo ""
  echo "  Builds Doxygen (C++ API) and Sphinx (Python API) HTML docs directly"
  echo "  into <build_output_dir>/docs/{cpp,python}."
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

# No GPU is required: Doxygen and Sphinx only read sources and docstrings, and
# the pip install links the prebuilt libs rather than compiling CUDA. Runs as
# root so the docs build can install the cuvslam package for Sphinx autodoc;
# output artifacts are chowned back to the host user on exit.
docker run --rm --label "cuvslam.runner=${RUNNER_NAME:-local}" $TTY_FLAG \
  -v "$(pwd):/cuvslam:ro" \
  -v "$OUTPUT_DIR:/output" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  cuvslam:local bash -c '
    set -euo pipefail
    trap "chown -R $HOST_UID:$HOST_GID /output" EXIT

    rm -rf /output/docs
    mkdir -p /output/docs

    # C++ API docs (Doxygen). Override OUTPUT_DIRECTORY so HTML is written
    # straight to /output/docs/cpp (HTML_OUTPUT=cpp is set in Doxyfile.in), no
    # copy afterwards. Config is piped in, which works from any cwd because the
    # Doxyfile INPUT paths are absolute. doxygen hangs occasionally: timeout.
    { cat /output/build/doc/Doxyfile
      echo "OUTPUT_DIRECTORY=/output/docs"
    } | timeout 120 doxygen -

    # Python API docs (Sphinx) written directly to /output/docs/python.
    # autodoc needs cuvslam importable; -W turns doc warnings into errors.
    CUVSLAM_BUILD_DIR=/output/build SKBUILD_BUILD_DIR=/tmp/skbuild \
      pip install /cuvslam/python/
    sphinx-build -W -b html /cuvslam/python/docs /output/docs/python
  '
