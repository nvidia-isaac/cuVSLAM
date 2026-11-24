# Docker Build for PyCuVSLAM with RealSense Support

Multi-architecture (aarch64/x86_64) Docker setup for PyCuVSLAM with RealSense camera support

## Available Docker Images

| File | Ubuntu | CUDA | Python | librealsense |
|------|--------|------|--------|--------------|
| `Dockerfile.realsense-cu12` | 22.04 | 12.6 | 3.12 (uv-managed venv in `/opt/venv`) | v2.57.6 |
| `Dockerfile.realsense-cu13` | 24.04 | 13.0 | 3.12 | v2.57.6 |

## Quick Start

### Build

All commands should be run from the **repository root** (the Dockerfiles build cuVSLAM & PyCuVSLAM from sources).

```bash
# Ubuntu 22.04 + CUDA 12
docker build -f docker/Dockerfile.realsense-cu12 -t pycuvslam:realsense-cu12 .

# Ubuntu 24.04 + CUDA 13
docker build -f docker/Dockerfile.realsense-cu13 -t pycuvslam:realsense-cu13 .
```

### Run

```bash
# Ubuntu 22.04 + CUDA 12 (default)
./docker/run_docker.sh

# Ubuntu 24.04 + CUDA 13
./docker/run_docker.sh 24
```

## Minimum Driver Versions

| Image | CUDA | Minimum NVIDIA Driver |
|-------|------|-----------------------|
| `Dockerfile.realsense-cu12` | 12.6 | >= 560 |
| `Dockerfile.realsense-cu13` | 13.0 | >= 580 |

Check your driver version with `nvidia-smi`.

## Jetson Orin Support

On Jetson Orin devices (aarch64) with CUDA 12, the run script automatically mounts the host's CUDA installation into the container.

The script:
- Detects Jetson/aarch64 architecture automatically
- Mounts the host's `/usr/local/cuda-12.6` directory read-only
- Uses CUDA headers from the host for compilation

> **Note:** On Jetson devices with old CUDA (e.g. JetPack 6.0 with CUDA 12.2), you may need to set
> `NVIDIA_DISABLE_REQUIRE=1` when running the container, since the container's
> CUDA version (12.6) is newer than the host's. Add `-e NVIDIA_DISABLE_REQUIRE=1`
> to your `docker run` command.

## Key Features

### Multi-Architecture Support
- Both Dockerfiles work on x86_64 and aarch64 (Jetson)
- Docker automatically pulls the correct architecture


### RealSense Integration
- Builds librealsense from source with Python bindings
- Uses stable release branch (v2.57.6)
- Includes udev rules for RealSense camera access
- Sets up proper Python path for `pyrealsense2` import

### GUI Support
- Includes all necessary X11 and GUI libraries for rerun-sdk
- Proper X11 forwarding setup for GUI applications
- Support for RealSense camera visualization

### Automatic Build and Installation
- Builds cuVSLAM from source during `docker build`
- Installs PyCuVSLAM Python bindings into the image
- Installs all Python dependencies from requirements.txt
- Ready to run examples immediately after container startup

## Dependencies Included

- Build tools (cmake, make, gcc)
- Python development headers
- USB and graphics libraries
- X11 and GUI libraries for rerun-sdk
- CUDA development environment

## Usage Example

cuVSLAM and PyCuVSLAM are pre-built inside the image.

After starting the container, run stereo tracking with RealSense camera:

```bash
python3 examples/realsense/run_stereo.py
```
