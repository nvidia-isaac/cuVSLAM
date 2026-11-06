#!/usr/bin/env bash

set -euo pipefail

EXPECTED_COMMIT="e31f4cc1d41a329a01946e5fe63669f8b15da677"
EXPECTED_NITROS_PACKAGE="ros-humble-isaac-ros-nitros"
EXPECTED_NITROS_VERSION="3.2.5-0jammy"
SDK_HEADER="/opt/ros/humble/share/isaac_ros_nitros/cuvslam/include/cuvslam.h"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PATCH_FILE="$INTEGRATION_DIR/patches/isaac_ros_visual_slam_v3_2_15_imu_timestamp.patch"
SOURCE_FILE="isaac_ros_visual_slam/src/impl/visual_slam_impl.cpp"
PATCH_MARKER="ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1"

usage() {
  echo "Usage: $0 [--source-only] /path/to/isaac_ros_visual_slam" >&2
}

source_only=0
if [[ "${1:-}" == "--source-only" ]]; then
  source_only=1
  shift
fi

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

repo_input="$(cd -- "$1" && pwd)"

if ! repo="$(git -C "$repo_input" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "[STOP] Not a Git worktree: $repo_input" >&2
  exit 1
fi

verify_sdk_contract() {
  local installed_version
  local verification_output

  if ! command -v dpkg-query >/dev/null 2>&1 || ! command -v dpkg >/dev/null 2>&1; then
    echo "[STOP] dpkg and dpkg-query are required for the SDK compatibility check." >&2
    exit 1
  fi

  if ! installed_version="$(dpkg-query -W -f='${Version}' "$EXPECTED_NITROS_PACKAGE" 2>/dev/null)"; then
    echo "[STOP] Required package is not installed: $EXPECTED_NITROS_PACKAGE" >&2
    exit 1
  fi

  if [[ "$installed_version" != "$EXPECTED_NITROS_VERSION" ]]; then
    echo "[STOP] Unsupported NITROS package version." >&2
    echo "expected=$EXPECTED_NITROS_VERSION" >&2
    echo "actual  =$installed_version" >&2
    exit 1
  fi

  if [[ ! -f "$SDK_HEADER" ]]; then
    echo "[STOP] cuVSLAM SDK header not found: $SDK_HEADER" >&2
    exit 1
  fi

  if ! verification_output="$(dpkg --verify "$EXPECTED_NITROS_PACKAGE" 2>&1)"; then
    echo "[STOP] Unable to verify installed NITROS package files:" >&2
    echo "$verification_output" >&2
    exit 1
  fi
  if [[ -n "$verification_output" ]]; then
    echo "[STOP] Installed NITROS package files differ from package metadata:" >&2
    echo "$verification_output" >&2
    exit 1
  fi

  if ! grep -Fq \
      '* @param[in] timestamp timestamp is in nanoseconds and should always increment' \
      "$SDK_HEADER" || \
    ! grep -Fq \
      'CUVSLAM_Status CUVSLAM_RegisterImuMeasurement(CUVSLAM_TrackerHandle tracker, int64_t timestamp,' \
      "$SDK_HEADER"
  then
    echo "[STOP] Installed cuVSLAM header does not contain the required timestamp contract." >&2
    exit 1
  fi
}

if [[ "$source_only" -eq 0 ]]; then
  verify_sdk_contract
fi

actual_commit="$(git -C "$repo" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "[STOP] Unsupported NVIDIA revision." >&2
  echo "expected=$EXPECTED_COMMIT" >&2
  echo "actual  =$actual_commit" >&2
  exit 1
fi

if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]; then
  echo "[STOP] NVIDIA checkout has tracked changes; verification will not modify it." >&2
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "[STOP] Patch not found: $PATCH_FILE" >&2
  exit 1
fi

if [[ ! -f "$repo/$SOURCE_FILE" ]]; then
  echo "[STOP] Expected NVIDIA source file not found: $repo/$SOURCE_FILE" >&2
  exit 1
fi

patch_applied=0
cleanup() {
  if [[ "$patch_applied" -eq 1 ]]; then
    git -C "$repo" apply --reverse "$PATCH_FILE"
    patch_applied=0
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

git -C "$repo" apply --check "$PATCH_FILE"
git -C "$repo" apply "$PATCH_FILE"
patch_applied=1

target="$repo/$SOURCE_FILE"

grep -Fq \
  'sequencer(node.imu_buffer_size_, 1e6 * node.imu_jitter_threshold_ms_,' \
  "$target"
grep -Fq \
  'node.image_buffer_size_, 1e6 * node.image_jitter_threshold_ms_),' \
  "$target"
grep -Fq 'cuvslam_handle, imu_ts, &imu_measurement);' "$target"
grep -Fq "$PATCH_MARKER" "$target"

if grep -Fq 'cuvslam_handle, latest_ts, &imu_measurement);' "$target"; then
  echo "[STOP] The incorrect image timestamp call is still present." >&2
  exit 1
fi

git -C "$repo" diff --check

git -C "$repo" apply --reverse "$PATCH_FILE"
patch_applied=0

if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]; then
  echo "[STOP] Verification did not restore the NVIDIA checkout." >&2
  exit 1
fi

trap - EXIT INT TERM
echo "[PASS] Patch matches NVIDIA v3.2-15, both corrections and the runtime marker are present, and the checkout was restored."
