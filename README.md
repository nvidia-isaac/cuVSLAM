# cuVSLAM: CUDA-Accelerated Visual Odometry and Mapping

![Demo](examples/assets/tutorial_multicamera_edex.gif)


### [Latest release](https://github.com/nvidia-isaac/cuVSLAM/releases/latest) | [ArXiv paper](https://www.arxiv.org/abs/2506.04359) | [Python API](https://nvidia-isaac.github.io/cuVSLAM/python/) | [C++ API](https://nvidia-isaac.github.io/cuVSLAM/cpp/) | [ROS2](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)

## Overview

cuVSLAM is the library by NVIDIA, providing various Visual Tracking Camera modes and Simultaneous Localization and Mapping (SLAM) capabilities. Leveraging CUDA acceleration and a rich set of features, cuVSLAM delivers highly accurate, computationally efficient, real-time performance.

**cuVSLAM works out of the box.** All internal algorithms use carefully engineered defaults or adapt automatically to the environment and motion profile — no parameter tuning is required or expected. To get started, provide two things: accurate camera calibration (intrinsics and extrinsics) and a sensor configuration (which tracking mode matches your rig). Everything else is handled internally.

![Overview](examples/assets/pycuvslam_overview.jpg)

## Table of Contents

- [Tracking modes](#tracking-modes)
- [Using cuVSLAM](#using-cuvslam)
- [Performance](#performance)
- [Install PyCuVSLAM](#install-pycuvslam)
- [Build cuVSLAM](#build-cuvslam)
- [FAQ](#faq)
- [Development](#development)
- [Feedback](#feedback)
- [License](#license)
- [Citation](#citation)

# Tracking modes

cuVSLAM's tracker supports several odometry modes selected via `Odometry::Config::odometry_mode`
(C++) / `cuvslam.Odometry.OdometryMode` (Python). Modes differ in which sensors they require, which
ones they can additionally fuse, and how they handle scale. Pick the mode that matches the most
informative sensor set on your rig:

| Mode | Required sensors | Optional sensors | Mode-specific settings | When to use |
|------|------------------|------------------|------------------------|-------------|
| `Mono` | 1 camera | — | — | Single-camera tracking; cheapest setup but scale-ambiguous. |
| `RGBD` | 1 RGB-D camera (aligned RGB + depth) | — | `RGBDSettings` | Single depth-aligned camera (e.g. RealSense, TUM RGB-D). |
| `Multicamera` | ≥2 cameras with at least one overlapping pair (a stereo pair) | up to 32 cameras total | — | Stereo or multi-stereo rigs. Most accurate purely-visual mode. |
| `Inertial` | 1 stereo pair + 1 IMU | — | — | Stereo VIO. Adds robustness to brief visual failures. |
| `Multisensor` | ≥1 RGB-D camera **or** ≥1 overlapping camera pair; cuNLS-enabled build | Additional RGB/RGB-D cameras, 0 or 1 IMU | `MultisensorSettings` | Any-mix RGB / RGB-D rigs with optional IMU, including one RGB-D camera with IMU. |

Notes:
- `Multisensor` is the only mode that requires a cuNLS-enabled build — see
  [cuNLS](#cunls). All other modes work with the default build.
- IMU fusion is available in `Inertial` (always on) and `Multisensor` (auto-enabled when
  `Rig::imus` is non-empty).
- For a runnable Multisensor walkthrough on a multi-RGB-D + IMU rig, see
  [examples/multisensor/](examples/multisensor/README.md). For the full per-field API reference, see the
  [C++](https://nvidia-isaac.github.io/cuVSLAM/cpp/) or
  [Python](https://nvidia-isaac.github.io/cuVSLAM/python/) docs.

## Multisensor mode

Multisensor tracking is experimental: tracking may be inaccurate or fail for some sensor configurations and scenes.
The current implementation supports pinhole cameras only.

# Using cuVSLAM

The quickest way to get started is to [install PyCuVSLAM from a pre-built wheel](#install-from-wheels)
and explore the [examples](examples/).

## ROS2 Support

To use cuVSLAM in a ROS2 environment:
* [Isaac ROS cuVSLAM GitHub](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)
* [Isaac ROS cuVSLAM Documentation](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/index.html)
* [Isaac ROS cuVSLAM User Manual](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html)

## cuVSLAM Documentation

- [cuVSLAM Technical Report](https://www.arxiv.org/abs/2506.04359)
- [PyCuVSLAM API Documentation](https://nvidia-isaac.github.io/cuVSLAM/python/)
- [cuVSLAM C++ API Documentation](https://nvidia-isaac.github.io/cuVSLAM/cpp/)
- Version-matched offline API documentation is available as `cuvslam-docs-<version>.tar.gz` on the
  [latest release](https://github.com/nvidia-isaac/cuVSLAM/releases/latest). Extract it and open
  `python/index.html` or `cpp/index.html`.

# Performance

cuVSLAM is a highly optimized visual tracking library validated across numerous public datasets and popular robotic camera setups. For detailed benchmarking and validation results, please refer to our [technical report](https://arxiv.org/html/2506.04359v3#S3).

<img src="examples/assets/cuvslam_performance.png" alt="cuVSLAM performance" width="800" />

The accuracy and robustness of cuVSLAM can be influenced by several factors. If you experience performance issues, please check your system against these common causes:

- **Hardware Overload**: Hardware overload can negatively impact visual tracking, resulting in dropped frames or insufficient computational resources for cuVSLAM. Disable intensive visualization or image-saving operations to improve performance. For expected performance metrics on Jetson embedded platforms, see our [technical report](https://arxiv.org/html/2506.04359v3#A1.F13)

- **Intrinsic and Extrinsic Calibration**: Accurate camera calibration is crucial. Ensure your calibration parameters are precise. For more details, refer to our guide on [image undistortion](examples/euroc/README.md#distortion-models). If you're new to calibration, consider working with [experienced vendors](https://nvidia-isaac-ros.github.io/v/release-3.2/getting_started/hardware_setup/sensors/amr_extrinsic_calibration.html#extrinsic-calibration-of-sensors-in-custom-locations)

- **Synchronization and Timestamps**: Accurate synchronization significantly impacts cuVSLAM performance. Make sure multi-camera images are captured simultaneously—ideally through hardware synchronization—and verify correct relative timestamps across cameras. Refer to our [multi-camera hardware assembly guide](examples/realsense/multicamera_hardware_assembly.md) for building a rig with synchronized RealSense cameras

- **Frame Rate**: Frame rate significantly affects performance. The ideal frame rate depends on translational and rotational velocities. Typically, 30 FPS is suitable for most "human-speed" motions. Adjust accordingly for faster movements

- **Resolution**: Image resolution matters. VGA resolution or higher is recommended. cuVSLAM efficiently handles relatively high-resolution images due to CUDA acceleration

- **Image Quality**: Ensure good image quality by using suitable lenses, correct exposure, and proper white balance to avoid clipping large image regions. For significant distortion or external objects within the camera's field of view, please refer to our guide on [static masking](examples/kitti/README.md#static-masks)

- **Motion Blur**: Excessive motion blur can negatively impact tracking. Ensure that exposure times are short enough to minimize motion blur. If avoiding motion blur isn't feasible, consider increasing the frame rate or try the following [Mono-Depth](examples/realsense/README.md#running-monocular-depth-visual-odometry) or [Stereo Inertial](examples/realsense/README.md#running-stereo-inertial-odometry) tracking modes

See [Troubleshooting](#troubleshooting)

# Install PyCuVSLAM

PyCuVSLAM is the Python wrapper (bindings) for the cuVSLAM library.

## Install from Wheels

Pre-built wheels are available on the [cuVSLAM releases page](https://github.com/nvidia-isaac/cuVSLAM/releases)
for the following configurations:

| Target | Ubuntu | Python wheel tag | CUDA wheel tag | Architecture |
|--------|--------|------------------|----------------|--------------|
| Desktop/server | 22.04 | `cp310` (Python 3.10) | `cu12`, `cu13` | x86_64 |
| Desktop/server | 24.04 | `cp312-abi3` (Python 3.12+) | `cu12`, `cu13` | x86_64 |
| Jetson Orin | 22.04 (JetPack 6.x) | `cp310` (Python 3.10) | `cu12` | aarch64 |
| Jetson Thor | 24.04 (JetPack 7.x) | `cp312-abi3` (Python 3.12+) | `cu13` | aarch64 |

Only the combinations listed above are provided as pre-built wheels. The `cp312-abi3` wheels use Python's stable ABI
and are compatible with Python 3.12 and later. Other Python, CUDA, or Jetson combinations require an
[installation from source](#install-from-source).

**Prerequisite**: [CUDA Toolkit 12 or 13](https://developer.nvidia.com/cuda/toolkit) must be installed separately
(not included in the wheels). Its major version must match the wheel's `cu12` or `cu13` tag.

Official wheels include cuNLS support for `Multisensor` mode; no separate cuNLS installation is required.

To install (virtual environment is recommended):

1. Go to the [latest release](https://github.com/nvidia-isaac/cuVSLAM/releases/latest).
2. Download the wheel matching the table above. On Jetson, also match the device family: Orin uses `cu12`/`cp310`;
   Thor uses `cu13`/`cp312-abi3`.
3. Install with pip:

```bash
pip install cuvslam-*.whl
```

If a pre-built wheel is not available for your system, see [Install from Source](#install-from-source) below.

## Install from Source

*Note*: cuVSLAM must be [built](#build-cuvslam) before installing from source.

To install PyCuVSLAM from repository (virtual environment is recommended):

```bash
CUVSLAM_BUILD_DIR=<path-to-cuvslam-build> pip install python/
```
`CUVSLAM_BUILD_DIR` is required for build script to find `libcuvslam.so`.
**Warning**: Due to scikit-build-core limitations, bindings must be reinstalled after rebuilding libcuvslam.

# Build cuVSLAM

## Pre-built Libraries

Pre-built C++ SDK archives are available on the
[latest release](https://github.com/nvidia-isaac/cuVSLAM/releases/latest):

| Target | Ubuntu | CUDA | Archive slug |
|--------|--------|------|--------------|
| Desktop/server | 22.04 or 24.04 | 12.6.3 or 13.2.0 | `x86_64-cuda<version>-ubuntu<version>` |
| Jetson Orin | 22.04 (JetPack 6.x) | 12.6.3 | `orin-cuda12.6.3-ubuntu22.04` |
| Jetson Thor | 24.04 (JetPack 7.x) | 13.0.1 | `thor-cuda13.0.1-ubuntu24.04` |

The desktop/server row represents four archives covering both Ubuntu versions with both listed CUDA versions. Select
the archive matching your target, Ubuntu version, and CUDA version. Each archive contains `bin/libcuvslam.so`,
`bin/cuvslam_api_launcher`, and the public headers under `include/cuvslam/`.

For Python usage, [pre-built wheels](#install-from-wheels) are the recommended approach.

## Building from Source

### Requirements

* Ubuntu 22+ (22.04 & 24.04 tested) x86_64/aarch64 (Desktop/Laptop & [Nvidia Jetson Orin/Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/))
* [CUDA Toolkit 12 or 13](https://developer.nvidia.com/cuda/toolkit), [Jetpack 6.1/6.2/7.0/7.1](https://docs.nvidia.com/jetson/)
* `apt update && apt install g++ cmake git git-lfs python3-dev`
    * git + git-lfs to clone this repository
    * CMake 3.19+, gcc
    * Python 3.9+ (for python bindings, examples and some tools)

### Build on local x86

In the repository root build C++ code using one of two ways:
1. Build manually:
```
cmake -S . -B build
cmake --build build --parallel $(nproc)
```
With CMake 4.x, add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` during configure if an external dependency reports that
compatibility with CMake < 3.5 has been removed:
```
cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```
2. Set source & build paths for `build_release.sh` and run it.

   **Important**: Before running `build_release.sh` set paths using one of the options:
   1. Set paths in `~/.bashrc` (or equivalent shell login script), which will be useful when switching between cuvslam branches:
      ```bash
      export CUVSLAM_SRC_DIR=<path-to-cuvslam-src>
      export CUVSLAM_DST_DIR=<path-to-cuvslam-build>
      ```
   2. Update SRC & DST paths in `build_release.sh`

### CMake options

All flags have defaults; override with `-DFLAG=VALUE`.

| Flag | Default | Purpose |
|------|---------|---------|
| `USE_CUDA` | ON | CUDA acceleration |
| `USE_CUNLS` | ON | cuNLS (CUDA nonlinear least squares); requires `USE_CUDA` |
| `USE_LMDB` | ON | LMDB map database |
| `USE_RERUN` | OFF | Rerun SDK visualization |
| `USE_NVTX` | OFF | NVIDIA NVTX profiling |
| `TREAT_WARNINGS_AS_ERRORS` | OFF | Strict warning policy |

Build types: `Release` (default), `Debug`, `RelWithDebInfo`, `MinSizeRel`. Do not mix types in the same build directory.

### cuNLS

**cuNLS** (CUDA nonlinear least squares) is enabled by default (`USE_CUNLS=ON`) and is built from
source via `FetchContent`, like the other external dependencies — no separate install or path is
required. The source is pinned in `cmake/ext/cunls.cmake`; cuNLS downloads its own dependencies
(spdlog is shared with cuVSLAM, cuDSS is fetched as a prebuilt archive) at configure time, so a
network connection and the CUDA Toolkit are the only prerequisites. The resulting static archive
is bundled into `libcuvslam`.

cuNLS source and license information are published in the
[nvidia-isaac/cuNLS repository](https://github.com/nvidia-isaac/cuNLS). Release wheels and C++ packages built with the
default configuration include cuNLS support and do not require a separate runtime package.

`USE_CUDA` must be `ON`. To build without cuNLS (and disable the multisensor odometry mode that
depends on it), configure with `-DUSE_CUNLS=OFF`.

### Build on Jetson (aarch64)

JetPack only ships CUDA runtime libraries by default. Install the dev packages on the Jetson before building:
```bash
# JetPack 6.x (CUDA 12)
sudo apt-get install libcublas-dev-12-6 libcusolver-dev-12-6
# JetPack 7.x (CUDA 13)
sudo apt-get install libcublas-dev-13-0 libcusolver-dev-13-0
```

For building natively, follow the same steps as [Build on local x86](#build-on-local-x86).
For building remotely via SSH:
```bash
./copy_to_remote.sh <jetson-host>
ssh <jetson-host> 'export CUVSLAM_SRC_DIR=~/cuvslam/src CUVSLAM_DST_DIR=~/cuvslam/build && ~/cuvslam/src/build_release.sh [options]'
./copy_from_remote.sh <jetson-host>
```

To speed up the build, target your specific GPU architecture instead of building for all, e.g.:
```bash
./build_release.sh --cuda_arch=87   # 87 for Orin Nano/NX/AGX
```
Omit `--cuda_arch` to build for all architectures (default). To detect your GPU's SM version:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. 8.7 -> use 87
```

### Enable rerun visualizer for C++ code

1. Create virtual environment and install rerun SDK:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install rerun-sdk==0.33.1
   ```
2. Specify virtual env with `CUVSLAM_TOOLS_PYENV` environment variable
   (defaults to `.venv` in repository root).
3. Update `build_release.sh` and set `USE_RERUN` to `ON`
4. Run any tool from tools folder

C++ unit tests disable Rerun by default, even in builds configured with `USE_RERUN=ON`. To enable the viewer for
interactive test debugging, ensure `rerun` is on `PATH` and set `RERUN=1`:

```bash
cd <build-dir>
RERUN=1 ctest --output-on-failure
```

# FAQ

**Q**: What Python versions are supported by PyCuVSLAM?

**A**: Pre-built wheels use `cp310` for Python 3.10 on Ubuntu 22.04 and `cp312-abi3` for Python 3.12 or later on
Ubuntu 24.04. Jetson wheels are limited to Orin with `cu12`/`cp310` and Thor with `cu13`/`cp312-abi3`; see the
[wheel table](#install-from-wheels). When built from source, PyCuVSLAM supports Python 3.9 and later.


# Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)



# Feedback

Are you having problems running cuVSLAM or PyCuVSLAM? Do you have any suggestions? We'd love to hear your feedback in the [issues](https://github.com/nvidia-isaac/cuVSLAM/issues) tab.

# License

This project is licensed under the NVIDIA Community License, for details refer to the [LICENSE](LICENSE) file.

# Citation

If you find this work useful in your research, please consider citing:
```bibtex
@article{korovko2025cuvslam,
      title={cuVSLAM: CUDA accelerated visual odometry and mapping},
      author={Alexander Korovko and Dmitry Slepichev and Alexander Efitorov and Aigul Dzhumamuratova and Viktor Kuznetsov and Hesam Rabeti and Joydeep Biswas and Soha Pouya},
      year={2025},
      eprint={2506.04359},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.04359},
}
```
