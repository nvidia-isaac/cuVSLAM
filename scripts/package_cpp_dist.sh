#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <build-output-dir> <archive-path>" >&2
  exit 1
fi

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(realpath "$SCRIPT_DIR/..")
BUILD_OUTPUT=$(realpath "$1")
ARCHIVE_PATH=$(realpath -m "$2")

declare -a sources=(
  "$BUILD_OUTPUT/build/bin/libcuvslam.so"
  "$BUILD_OUTPUT/build/bin/cuvslam_api_launcher"
  "$REPO_ROOT/libs/cuvslam/cuvslam2.h"
  "$REPO_ROOT/libs/cuvslam/cuvslam_gpu.h"
  "$REPO_ROOT/libs/cuvslam/ground_constraint2.h"
  "$REPO_ROOT/libs/cuvslam/vpr_vocabulary_builder2.h"
  "$REPO_ROOT/LICENSE"
)

for source in "${sources[@]}"; do
  if [ ! -f "$source" ]; then
    echo "Required C++ distribution file is missing: $source" >&2
    exit 1
  fi
done

staging_dir=$(mktemp -d)
trap 'rm -rf "$staging_dir"' EXIT

mkdir -p "$staging_dir/bin" "$staging_dir/include/cuvslam" "$(dirname "$ARCHIVE_PATH")"
install -m 0755 "${sources[0]}" "$staging_dir/bin/libcuvslam.so"
install -m 0755 "${sources[1]}" "$staging_dir/bin/cuvslam_api_launcher"
install -m 0644 "${sources[2]}" "$staging_dir/include/cuvslam/cuvslam2.h"
install -m 0644 "${sources[3]}" "$staging_dir/include/cuvslam/cuvslam_gpu.h"
install -m 0644 "${sources[4]}" "$staging_dir/include/cuvslam/ground_constraint2.h"
install -m 0644 "${sources[5]}" "$staging_dir/include/cuvslam/vpr_vocabulary_builder2.h"
install -m 0644 "${sources[6]}" "$staging_dir/LICENSE"

tar -czf "$ARCHIVE_PATH" -C "$staging_dir" .

expected_manifest=$(
  printf '%s\n' \
    LICENSE \
    bin/cuvslam_api_launcher \
    bin/libcuvslam.so \
    include/cuvslam/cuvslam2.h \
    include/cuvslam/cuvslam_gpu.h \
    include/cuvslam/ground_constraint2.h \
    include/cuvslam/vpr_vocabulary_builder2.h
)
actual_manifest=$(tar -tzf "$ARCHIVE_PATH" | sed 's#^\./##' | awk 'NF && !/\/$/' | LC_ALL=C sort)

if [ "$actual_manifest" != "$expected_manifest" ]; then
  echo "Unexpected C++ distribution manifest in $ARCHIVE_PATH:" >&2
  printf '%s\n' "$actual_manifest" >&2
  exit 1
fi

echo "Created $ARCHIVE_PATH"
