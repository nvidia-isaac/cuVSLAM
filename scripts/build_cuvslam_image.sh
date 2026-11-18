#!/bin/bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "Usage: ./scripts/build_cuvslam_image.sh"
  echo ""
  echo "  Builds the cuvslam:local image from scripts/Dockerfile, using the current directory as"
  echo "  the build context. build_cuvslam_in_docker.sh runs it too; running it earlier on its own"
  echo "  lets that later build reuse the cached layers."
  echo ""
  echo "Environment variables (optional):"
  echo "  CUDA_VERSION    CUDA version for base image (default: 12.6.3)"
  echo "  UBUNTU_VERSION  Ubuntu version for base image (default: 24.04)"
  echo "  BASE_IMAGE      Override base Docker image (e.g. NGC Jetson image)"
  echo "  UBUNTU_PORTS_MIRROR  Optional Ubuntu Ports mirror URL for Jetson builds"
  exit 1
fi

DOCKERFILE=$(dirname "$(realpath "$0")")/Dockerfile
DOCKER_BUILD_ARGS=""
[ -n "${CUDA_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg CUDA_VERSION=$CUDA_VERSION"
[ -n "${UBUNTU_VERSION:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_VERSION=$UBUNTU_VERSION"
[ -n "${BASE_IMAGE:-}" ] && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg BASE_IMAGE=$BASE_IMAGE"
[ -n "${UBUNTU_PORTS_MIRROR:-}" ] \
  && DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg UBUNTU_PORTS_MIRROR=$UBUNTU_PORTS_MIRROR"
docker build -f "$DOCKERFILE" . --network host $DOCKER_BUILD_ARGS --tag cuvslam:local
