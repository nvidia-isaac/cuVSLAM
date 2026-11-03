ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

FROM ${BASE_IMAGE}

ARG CUDA_VERSION
ARG BASE_IMAGE
RUN echo "Base image: ${BASE_IMAGE:-default}" && \
    echo "CUDA_VERSION arg: ${CUDA_VERSION:-not set}" && \
    echo "Actual CUDA:" && (nvcc --version 2>/dev/null || echo "nvcc not found")

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_PORTS_MIRROR=""

COPY scripts/apt-get-update-retry.sh /usr/local/bin/apt-get-update-retry

COPY scripts/ubuntu-ports-mirror.conf /tmp/ubuntu-ports-mirror.conf

RUN if [ -n "${UBUNTU_PORTS_MIRROR}" ]; then \
      set -eux; \
      find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) \
          -exec sed -Ei "s#https?://ports\\.ubuntu\\.com/ubuntu-ports#${UBUNTU_PORTS_MIRROR}#g" {} +; \
      install -m 0644 /tmp/ubuntu-ports-mirror.conf /etc/apt/apt.conf.d/99-nvidia-ubuntu-ports-mirror; \
    fi && \
    rm -f /tmp/ubuntu-ports-mirror.conf

RUN apt-get-update-retry && apt-get install -y --no-install-recommends \
       g++ gcc git git-lfs cmake ccache patch patchelf curl \
       ca-certificates \
       python3 python3-pip python3-dev python3-venv \
       doxygen graphviz \
    && git lfs install --system \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get-update-retry && apt-get install -y --no-install-recommends \
       libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get-update-retry && apt-get install -y --no-install-recommends \
       libpango-1.0-0 libpangocairo-1.0-0 libpangoft2-1.0-0 \
       libgdk-pixbuf-2.0-0 libffi8 shared-mime-info fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

COPY scripts/requirements.txt /tmp/ci-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ci-requirements.txt && \
    rm /tmp/ci-requirements.txt

COPY python/tests/requirements.txt /tmp/test-requirements.txt
RUN pip install --no-cache-dir -r /tmp/test-requirements.txt && \
    rm /tmp/test-requirements.txt

RUN mkdir -p /cuvslam
WORKDIR /cuvslam
