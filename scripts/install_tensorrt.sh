#!/bin/bash
# TensorRT without libnvinfer-dev, which adds GBs of static libs; Jetson images have no CUDA repo
set -euo pipefail

TENSORRT_VERSION=10.16.1.11
JETSON_TENSORRT_VERSION=10.3.0.30-1+cuda12.5

if grep -rqs "compute/cuda/repos" /etc/apt/sources.list.d; then
  case "${CUDA_VERSION%%.*}" in
    12) trt="${TENSORRT_VERSION}-1+cuda12.9" ;;
    13) trt="${TENSORRT_VERSION}-1+cuda13.2" ;;
    *) echo "No TensorRT build for CUDA ${CUDA_VERSION}" >&2; exit 1 ;;
  esac
else
  curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc -o /usr/share/keyrings/jetson-ota.asc
  echo "deb [signed-by=/usr/share/keyrings/jetson-ota.asc] https://repo.download.nvidia.com/jetson/common r36.4 main" \
    > /etc/apt/sources.list.d/jetson-common.list
  trt="${JETSON_TENSORRT_VERSION}"
fi
apt-get update
apt-get install -y --no-install-recommends libnvinfer10="$trt" libnvinfer-headers-dev="$trt"
apt-mark hold libnvinfer10 libnvinfer-headers-dev
so=$(dpkg -L libnvinfer10 | grep -m1 '/libnvinfer\.so\.[0-9]*$')
ln -s "$(basename "$so")" "${so%.*}"
apt-get clean
rm -rf /var/lib/apt/lists/*
