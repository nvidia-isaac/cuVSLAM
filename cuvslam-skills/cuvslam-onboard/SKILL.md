---
name: cuvslam-onboard
description: >
  Build, install, and run NVIDIA cuVSLAM and PyCuVSLAM from source or wheels.
  Covers environment setup, dataset preparation, and running examples for all
  tracking modes (stereo, mono, mono-depth, stereo-inertial, multi-camera, and
  Multisensor) and SLAM (mapping, localization, loop closure). Use when asked to: build cuVSLAM,
  install PyCuVSLAM, set up cuVSLAM environment, run cuVSLAM examples, prepare
  KITTI/EuRoC/TUM datasets, run visual odometry, set up live camera tracking
  (RealSense/ZED/OAK-D/Orbbec), run cuVSLAM in Docker, or use cuVSLAM C++ tools.
---

# cuVSLAM Onboarding

Build, install, and run NVIDIA cuVSLAM — CUDA-accelerated visual odometry and SLAM.

- **Repo:** https://github.com/nvidia-isaac/cuVSLAM
- **Python API docs:** https://nvidia-isaac.github.io/cuVSLAM/python/
- **C++ API docs:** https://nvidia-isaac.github.io/cuVSLAM/cpp/
- **Technical report:** https://arxiv.org/abs/2506.04359

For tracking/pose accuracy issues, see the `cuvslam-troubleshoot` skill instead.

For dataset-specific walkthroughs (EuRoC calibration, KITTI SLAM, TUM depth settings, multi-camera EDEX extraction), read `references/dataset-guides.md`.

For live camera setup (RealSense, ZED, OAK-D, Orbbec), read `references/live-cameras.md`.

---

## Agent Interaction Guidelines

Before executing any setup step, ask the user for missing paths. Do not assume defaults.

- **Cloning the repo:** Always ask — "Where would you like to clone the cuVSLAM repository? (e.g. `~/cuVSLAM`)"
- **Dataset preparation:** Always ask — "Where would you like to save the [KITTI/EuRoC/TUM/etc.] dataset? (e.g. `~/datasets/kitti`)"
- **Virtual environment:** If not already set up, ask — "Where would you like to create the Python virtual environment? (e.g. `~/cuVSLAM/.venv`)"

Once the user provides a path, use it consistently throughout all subsequent commands in that session. Do not re-ask for the same path.

---

## 1. Requirements

- Ubuntu 22+ (x86_64 or aarch64/Jetson)
- CUDA Toolkit 12 or 13 (https://developer.nvidia.com/cuda/toolkit)
- System packages: `apt update && apt install g++ cmake git git-lfs python3-dev`
- CMake 3.19+, Python 3.9+

## 2. Install PyCuVSLAM (Quickest Path)

Pre-built wheels from https://github.com/nvidia-isaac/cuVSLAM/releases/latest:

| Target | Ubuntu | Python wheel tag | CUDA wheel tag | Arch |
|--------|--------|------------------|----------------|------|
| Desktop/server | 22.04 | `cp310` (Python 3.10) | `cu12`, `cu13` | x86_64 |
| Desktop/server | 24.04 | `cp312-abi3` (Python 3.12+) | `cu12`, `cu13` | x86_64 |
| Jetson Orin | 22.04 (JetPack 6.x) | `cp310` (Python 3.10) | `cu12` | aarch64 |
| Jetson Thor | 24.04 (JetPack 7.x) | `cp312-abi3` (Python 3.12+) | `cu13` | aarch64 |

Only these combinations are provided. The installed CUDA major must match the wheel's `cu12` or `cu13` tag; use a
source build for other combinations.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install cuvslam-<version>.whl
pip install -r examples/requirements.txt   # rerun-sdk, numpy, etc.
```

## 3. Build from Source

### Clone

> **Ask the user:** "Where would you like to clone the cuVSLAM repository?" before running these commands. Use their answer as `<install-dir>`.

```bash
git clone https://github.com/nvidia-isaac/cuVSLAM.git <install-dir>
cd <install-dir>
```

### Build C++ library

```bash
cmake -S . -B build
cmake --build build --parallel $(nproc)
```

CMake options:
- `-DUSE_RERUN=ON` — enable Rerun visualization for C++ tools
- `-DCUVSLAM_BUILD_SHARED_LIB=TRUE` — build shared library (default)
- `-DUSE_CUDA=ON` — use CUDA (default)
- `-DUSE_CUNLS=ON` — enable Multisensor mode (default; requires CUDA)

### Install PyCuVSLAM from source

After building C++:

```bash
CUVSLAM_BUILD_DIR=$(pwd)/build pip install -e python/
```

**Warning:** Reinstall PyCuVSLAM after every C++ rebuild (scikit-build-core limitation).

### Build on Jetson (remote ARM)

```bash
./copy_to_remote.sh <jetson-host>
ssh <jetson-host> 'CUVSLAM_SRC_DIR="<install-dir>"; CUVSLAM_DST_DIR="$CUVSLAM_SRC_DIR/build"; export CUVSLAM_SRC_DIR CUVSLAM_DST_DIR; "$CUVSLAM_SRC_DIR/build_release.sh"'
./copy_from_remote.sh <jetson-host>
```

### Docker (with RealSense support)

```bash
# Ubuntu 22.04 + CUDA 12
docker build -f docker/Dockerfile.realsense-cu12 -t pycuvslam:realsense-cu12 .
./docker/run_docker.sh

# Ubuntu 24.04 + CUDA 13
docker build -f docker/Dockerfile.realsense-cu13 -t pycuvslam:realsense-cu13 .
./docker/run_docker.sh 24
```

Minimum drivers: CUDA 12 → driver ≥560, CUDA 13 → driver ≥580.

## 4. Tracking Modes

cuVSLAM supports these visual tracking modes:

| Mode | Enum | Use case |
|------|------|----------|
| Stereo | `OdometryMode.Multicamera` (0) | Default. Two+ synchronized cameras |
| Stereo-Inertial | `OdometryMode.Inertial` (1) | Stereo + IMU for robustness |
| Mono-Depth (RGB-D) | `OdometryMode.RGBD` (2) | Monocular + depth image |
| Monocular | `OdometryMode.Mono` (3) | Single camera (no scale) |
| Multisensor | `OdometryMode.Multisensor` (4) | At least one RGB-D camera or one overlapping camera pair, with optional IMU. Requires cuNLS and currently supports pinhole cameras only. Configure via `MultisensorSettings`. See `examples/multisensor/`. |

## 5. Run Examples — Public Datasets

### Environment setup (common to all examples)

```bash
source .venv/bin/activate    # if using venv
cd examples
pip install -r requirements.txt
```

### KITTI (Stereo Odometry) — quickest demo

> **Ask the user:** "Where would you like to save the KITTI dataset? (e.g. `~/datasets/kitti`)" before downloading.
> Then create a symlink so the example script can find it: `ln -s <dataset-dir> examples/kitti/dataset`

```bash
cd examples/kitti
# Download: http://www.cvlibs.net/datasets/kitti/eval_odometry.php (grayscale, 22GB)
# Unzip so <dataset-dir>/sequences/00/image_0/*.png exists
# Symlink dataset into the example directory:
ln -s <dataset-dir> dataset
python3 track_kitti.py
```

SLAM with mapping + localization:
```bash
python3 track_kitti_slam.py     # maps, saves trajectory + map/data.mdb
```

### EuRoC (Stereo-Inertial)

> **Ask the user:** "Where would you like to save the EuRoC dataset? (e.g. `~/datasets/euroc`)" before downloading.
> Then create a symlink: `ln -s <dataset-dir> examples/euroc/dataset`

```bash
cd examples/euroc
# Download MH_01_easy from https://doi.org/10.3929/ethz-b-000690084
# Extract mav0/ to <dataset-dir>/mav0/
ln -s <dataset-dir> dataset
cp sensor_cam0.yaml dataset/mav0/cam0/sensor_recalibrated.yaml
cp sensor_cam1.yaml dataset/mav0/cam1/sensor_recalibrated.yaml
cp sensor_imu0.yaml dataset/mav0/imu0/sensor_recalibrated.yaml
python3 track_euroc.py
```

### TUM RGB-D (Mono-Depth)

> **Ask the user:** "Where would you like to save the TUM RGB-D dataset? (e.g. `~/datasets/tum`)" before downloading.

```bash
cd examples/tum
mkdir -p <dataset-dir>
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz -O <dataset-dir>/fr3.tgz
tar -xzf <dataset-dir>/fr3.tgz -C <dataset-dir> && rm <dataset-dir>/fr3.tgz
ln -s <dataset-dir> dataset
cp freiburg3_rig.yaml dataset/rgbd_dataset_freiburg3_long_office_household/
python3 track_tum.py
```

### Multi-Camera (Tartan Ground, 6 stereo pairs)

```bash
cd examples/multicamera_edex
pip install tartanair     # x86_64 only
python3 download_tartan.py
python3 track_multicamera_tartan.py
```

### Multisensor (Tartan Ground, RGB-D + optional IMU)

Multisensor tracking is experimental and may be inaccurate or fail for some sensor configurations and scenes.

```bash
cd examples/multisensor
pip install tartanair     # x86_64 only
python3 download_tartan.py
python3 track_multisensor_tartan.py
python3 track_multisensor_tartan.py --no-imu
```

## 6. Run Examples — Live Cameras

See `references/live-cameras.md` for detailed setup per camera.
This table lists ready-made Python examples shipped in the repository. An em dash means no
ready-made example exists; it does not mean the hardware integration is unsupported.

| Camera | Stereo | VIO | RGB-D | Multi-cam | Multisensor |
|--------|--------|-----|-------|-----------|--------------------------|
| RealSense | `run_stereo.py` | `run_vio.py` | `run_rgbd.py` | `run_multicamera.py` | `run_multisensor.py` |
| ZED | `run_stereo.py` | — | `run_rgbd.py` | — | — |
| OAK-D | `run_stereo.py` | — | — | — | — |
| Orbbec | `run_stereo.py` | — | `run_rgbd.py` | — | — |

## 7. C++ API

### EuRoC C++ example

```bash
cmake -S . -B build
cmake --build build --target track_euroc
./build/bin/track_euroc /path/to/euroc/mav0
```

With Rerun: `cmake -S . -B build -DUSE_RERUN=ON && cmake --build build --target track_euroc`

### C++ tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `tracker` | CLI image-sequence tracking | `./bin/tracker config.cfg` |
| `cuvslam_api_launcher` | Track, save map, localize | `./bin/cuvslam_api_launcher -dataset=<edex>` |
| `undistort` | Remove lens distortion | `./bin/undistort in.png calib.edex out.png` |
| `result_visualizer` | Visualize EDEX trajectories | `python3 tools/edex/result_visualizer/result_visualizer.py result.edex` |
| `bag2edex` | Convert ROS2 bag → EDEX | `python3 bag_to_edex.py <bag> <out.edex>` |

## 8. SLAM Workflow

1. **Map:** Run tracker with SLAM config to collect map
2. **Save:** Map stored as `map/data.mdb` (LMDB)
3. **Localize:** Load saved map, provide initial pose hint, call `tracker.slam.localize_in_map()`

```python
odom_cfg = cuvslam.Odometry.Config(...)
slam_cfg = cuvslam.Slam.Config(sync_mode=True)  # sync for reproducibility
tracker = cuvslam.Tracker(cuvslam.Rig(...), odom_cfg, slam_cfg)

odom_pose, slam_pose = tracker.track(...)
slam = tracker.slam

# Save map
def map_saved(success):
    print(f"Map save {'succeeded' if success else 'failed'}")

slam.save_map("map/", map_saved)

# Later: localize
loc_settings = cuvslam.Slam.LocalizationSettings()

def localization_started():
    print("Localization started")

def localization_finished(pose, error_message):
    if pose is not None:
        print(f"Localized pose: {pose}")
    else:
        print(f"Localization failed: {error_message}")

slam.localize_in_map(
    "map/",
    timestamp,
    pose_hint,
    images,
    loc_settings,
    localization_started,
    localization_finished,
)
```

See `examples/kitti/track_kitti_slam.py` for the complete workflow.

## 9. Advanced Features

- **Static masks:** Crop robot body / distorted edges via `camera.border_top/bottom/left/right`
- **Dynamic masks:** Real-time segmentation masks as PyTorch GPU tensors — see `examples/kitti/track_kitti_masks.py`
- **Distortion models:** pinhole, fisheye, brown, polynomial — see `references/dataset-guides.md`
- **Debug dump:** Set `config.debug_dump_directory` to capture EDEX + images for offline analysis
- **Rerun visualization:** All Python examples use Rerun; C++ needs `-DUSE_RERUN=ON`

## 10. ROS 2 Integration

Isaac ROS cuVSLAM wraps the C++ API as a ROS 2 node:
- GitHub: https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam
- Docs: https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/index.html
- Supports Humble on Jetson (JetPack 6.1/6.2) and x86_64 (Ubuntu 22.04+)
