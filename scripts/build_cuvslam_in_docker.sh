#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: ./scripts/build_cuvslam_in_docker.sh <CMAKE_BUILD_TYPE = Release | RelWithDebInfo> [output_dir]"
  echo ""
  echo "Environment variables (optional):"
  echo "  CUDA_VERSION    CUDA version for base image (default: 12.6.3)"
  echo "  UBUNTU_VERSION  Ubuntu version for base image (default: 24.04)"
  echo "  BASE_IMAGE      Override base Docker image (e.g. NGC Jetson image)"
  echo "  UBUNTU_PORTS_MIRROR  Optional Ubuntu Ports mirror URL for Jetson builds"
  echo "  EXTRA_CMAKE_ARGS  Additional CMake arguments"
  echo "  CUVSLAM_REQUIRE_CLEAN_SOURCE  Fail if tracked source files differ from HEAD"
  echo "  BUILD_JOBS      Parallel build jobs, passed as --jobs to build_release.sh"
  echo "                  (default: min(ceil(nproc/2), MemAvailable/4GB))"
  exit 1
fi

BUILD_TYPE=$1
OUTPUT_DIR=${2:-$(pwd)/build_docker}
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
mkdir -p "$OUTPUT_DIR"

DOCKERFILE=$(dirname "$(realpath "$0")")/Dockerfile
DOCKER_BUILD_ARGS=""
[ -n "${CUDA_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg CUDA_VERSION=$CUDA_VERSION"
[ -n "${UBUNTU_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_VERSION=$UBUNTU_VERSION"
[ -n "${BASE_IMAGE:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg BASE_IMAGE=$BASE_IMAGE"
[ -n "${UBUNTU_PORTS_MIRROR:-}" ] \
  && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_PORTS_MIRROR=$UBUNTU_PORTS_MIRROR"
docker build -f "$DOCKERFILE" . --network host $DOCKER_BUILD_ARGS --tag cuvslam:local

INSTALL_DIR="/output"
DOCKER_VOLUMES="-v $(pwd):/cuvslam:ro -v $OUTPUT_DIR:$INSTALL_DIR"
DOCKER_USER="--user $(id -u):$(id -g) --group-add video -e HOME=/tmp"

TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"

# Cap parallelism at half the available CPUs to leave power and SMT headroom,
# and at ~4GB of available memory per compile job. Override by setting BUILD_JOBS.
if [ -z "${BUILD_JOBS:-}" ]; then
  nproc_n=$(nproc)
  cpu_cap=$(( (nproc_n + 1) / 2 ))
  mem_cap=$(awk '/MemAvailable/ { print int($2/1024/1024/4) }' /proc/meminfo)
  if [ "$mem_cap" -lt 1 ]; then
    mem_cap=1
  fi
  BUILD_JOBS=$cpu_cap
  if [ "$mem_cap" -lt "$BUILD_JOBS" ]; then
    BUILD_JOBS=$mem_cap
  fi
  echo "Build parallelism: --jobs=$BUILD_JOBS (nproc=$nproc_n, CPU cap=$cpu_cap, memory cap=$mem_cap at 4GB/job)"
fi

JOBS_ARG=()
if [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  JOBS_ARG=( "--jobs=$BUILD_JOBS" )
else
  echo "Error: invalid BUILD_JOBS='$BUILD_JOBS' (must be a positive integer)" >&2
  exit 1
fi

docker run --runtime=nvidia --gpus all --rm $TTY_FLAG $DOCKER_USER $DOCKER_VOLUMES \
  -e CUVSLAM_SRC_DIR=/cuvslam \
  -e CUVSLAM_DST_DIR=$INSTALL_DIR/build \
  -e CUVSLAM_REQUIRE_CLEAN_SOURCE="${CUVSLAM_REQUIRE_CLEAN_SOURCE:-false}" \
  -e EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS:-}" \
  cuvslam:local /cuvslam/build_release.sh --build_type="$BUILD_TYPE" "${JOBS_ARG[@]}"
