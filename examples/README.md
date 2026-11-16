## Prerequisites

### Install PyCuVSLAM

The easiest way to get started is to install a pre-built wheel from the
[cuVSLAM releases page](https://github.com/nvidia-isaac/cuVSLAM/releases).
See the [root README](../README.md#install-from-wheels) for details.

Alternatively, you can [build cuVSLAM from source](../README.md#build-cuvslam) and
[install PyCuVSLAM from source](../README.md#install-from-source).

### Environment Setup (Optional)

#### Using Venv

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Using Docker

See the [Docker README](../docker/README.md) for building and running PyCuVSLAM in a Docker container with RealSense camera support.

### Install Example Dependencies

```bash
pip install -r requirements.txt
```

The shared example requirements pin `numpy==2.2.4`. If you install an additional camera SDK Python wrapper, make sure it
is compatible with NumPy 2.x and does not downgrade NumPy. For ZED examples, see
[ZED SDK Installation](zed/README.md#zed-sdk-installation).

## Examples and Guides

Explore various examples to quickly get started with cuVSLAM using [Python API (PyCuVSLAM)](#visual-tracking-mode-examples) or [C++ API](#c-api-examples).

> **Note:** All example scripts use the [Rerun viewer](https://rerun.io/docs/reference/viewer/overview) for interactive visualization of trajectories, camera frames, and mapped features. The Rerun UI will remain open after an example script finishes, allowing you to replay, inspect, and interact with the visualized session data.

### Visual Tracking Mode Examples

- **Monocular Visual Odometry**
    - [EuRoC Dataset](euroc/README.md)

- **Monocular-Depth Visual Odometry**
    - [TUM Dataset](tum/README.md#running-monocular-depth-odometry)
    - [RealSense Live Camera](realsense/README.md#running-monocular-depth-visual-odometry)
    - [ZED Live Camera](zed/live/README.md#running-monocular-depth-visual-odometry)
    - [Orbbec Live Camera](orbbec/README.md#running-monocular-depth-visual-odometry)

- **Stereo Visual Odometry**
    - [KITTI Dataset](kitti/README.md#running-pycuvslam-stereo-visual-odometry)
    - [RealSense Live Camera](realsense/README.md#running-stereo-visual-odometry)
    - [ZED Live Camera](zed/live/README.md#running-stereo-visual-odometry)
    - [OAK-D Live Camera](oak-d/README.md#running-stereo-visual-odometry)
    - [Orbbec Live Camera](orbbec/README.md#running-stereo-visual-odometry)

- **Stereo Visual-Inertial Odometry**
    - [EuRoC Dataset](euroc/README.md#running-stereo-inertial-odometry)
    - [RealSense Live Camera](realsense/README.md#running-stereo-inertial-odometry)

- **Multi-Camera Stereo Visual Odometry**
    - [Tartan Ground Dataset](multicamera_edex/README.md#tartan-ground-dataset)
    - [R2B Galileo Dataset](multicamera_edex/README.md#r2b-galileo-dataset)
    - [RealSense Live Camera](realsense/README.md#running-multicamera-odometry)

- **Multisensor Odometry (any-mix RGB / RGB-D, optional IMU)**
    - [Tartan Ground Dataset](multisensor/README.md)
    - [RealSense Live Camera](realsense/README.md#running-multisensor-odometry)

### SLAM Examples

- [**Visual Mapping, Localization, and Map Saving/Loading**](kitti/README.md#slam-mapping-collecting-storing-loading-and-localization)
- [**Global Localization without a Guess Pose**](kitti/README.md#global-localization-without-a-guess-pose)

### Advanced Features and Guides

- **Distorted Images**
    - [EuRoC Dataset](euroc/README.md#distortion-models)
    - [OAK-D Live Camera](oak-d/README.md#running-stereo-visual-odometry)
    - [ZED Live Camera](zed/live/README.md#using-distorted-images)

- **Image Masking**
    - [Static Masks](tum/README.md#masking-regions-to-prevent-feature-selection)
    - [Dynamic Masks](kitti/README.md#dynamic-masks-with-pytorch-tensors)

- [**PyTorch GPU Tensor Handling**](kitti/README.md#example-real-time-car-segmentation)

- [**ZED Camera live recording and offline tracking SVO2**](zed/recording/README.md)

- [**RealSense Multi-Camera Assembly Guide**](realsense/multicamera_hardware_assembly.md)

- [**Nvblox live 3D reconstruction**](https://nvidia-isaac.github.io/nvblox/v0.0.9/pages/torch_examples_realsense.html#realsense-live-example)

### C++ API Examples

- [**EuRoC VIO and SLAM (C++)**](euroc/cpp/README.md)
