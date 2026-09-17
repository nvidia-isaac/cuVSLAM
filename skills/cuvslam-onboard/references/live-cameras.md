# Live Camera Setup

Detailed setup for each camera platform supported by PyCuVSLAM.

## Table of Contents
- [RealSense](#realsense)
- [ZED](#zed)
- [OAK-D](#oak-d)
- [Orbbec](#orbbec)

---

## RealSense

**Supported modes:** Stereo, Stereo-Inertial (VIO), Mono-Depth (RGB-D), Multi-Camera, Multisensor

### Install librealsense

Skip if using Docker (`docker/Dockerfile.realsense-cu12`).

Build from source with Python bindings (tested with v2.57.6):
```bash
git clone --branch v2.57.6 https://github.com/realsenseai/librealsense.git
cd librealsense
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=true \
  -DBUILD_PYTHON_BINDINGS=true \
  -DCHECK_FOR_UPDATES=false \
  -DPYTHON_EXECUTABLE=$(which python3)
cmake --build build --parallel $(nproc)
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/Release
```

Test: run a simple camera example from librealsense Python wrappers.

### Run examples

```bash
cd examples/realsense
python3 run_stereo.py        # Stereo odometry
python3 run_vio.py           # Stereo-inertial (requires IMU calibration)
python3 run_rgbd.py          # Mono-depth (enable IR emitter, use RGB not IR for tracking)
python3 run_multicamera.py   # Multi-camera (needs frame_nano_rig.yaml with your extrinsics)
python3 run_multisensor.py   # Multisensor RGB-D + IMU (requires cuNLS; pinhole only)
```

### IMU calibration for VIO

Update noise parameters in `camera_utils.py` for your specific camera:
```python
IMU_GYROSCOPE_NOISE_DENSITY = ...
IMU_GYROSCOPE_RANDOM_WALK = ...
IMU_ACCELEROMETER_NOISE_DENSITY = ...
IMU_ACCELEROMETER_RANDOM_WALK = ...
```
Use [kalibr IMU noise model](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model) to calculate these.

### Multi-camera rig

See `examples/realsense/multicamera_hardware_assembly.md` for building a synchronized multi-RealSense rig. Update `frame_nano_rig.yaml` with your extrinsics and camera serial numbers.

### Notes
- **Jetson VIO:** Real-time inertial odometry via Python API on Jetson may not be reliable. Use the C++ ROS 2 node instead.
- **USB troubleshooting:** If camera fails to start, disconnect and reconnect USB.
- **Multi-camera sync:** Takes ~15 seconds for multiple RealSense cameras to synchronize timestamps.

---

## ZED

**Supported modes:** Stereo, Mono-Depth (RGB-D), Mono-Depth+Stereo

### Install ZED SDK

1. Install C++ SDK: follow official docs for [Linux x86](https://www.stereolabs.com/docs/development/zed-sdk/linux) or [Jetson](https://www.stereolabs.com/docs/development/zed-sdk/jetson)
2. When prompted for Python API, press `n` (install manually in venv instead)
3. Install Python API in your venv:
```bash
pip install requests==2.32.5
python3 /usr/local/zed/get_python_api.py
```

### Run examples

```bash
cd examples/zed/live
python3 run_stereo.py    # Stereo (rectified by default)
python3 run_rgbd.py      # Mono-depth
```

### Distorted images

Set `RAW = True` in `run_stereo.py` to use distorted images. Tested on ZED2 with brown distortion model ($k_1, k_2, p_1, p_2, k_3$). For other ZED models, verify distortion coefficients in `camera_utils.py`.

### Mono-Depth + Stereo mode

Set `RUN_STEREO = True` in `run_rgbd.py`. Combines depth and stereo input at the solver level (not post-processing). Good for cameras without IR emitter.

### ZED recording and offline tracking

```bash
cd examples/zed/recording
cmake -S . -B build
cmake --build build
./build/record_from_zed    # Ctrl+C to stop; produces .svo2 file
python3 track_svo.py       # Offline tracking with Rerun visualization
```
Records at VGA 100fps, ~1GB per 15min. Supports long recordings (>1 hour).

---

## OAK-D

**Supported modes:** Stereo

**Requirement:** Global shutter sensor (essential for cuVSLAM).

### Install DepthAI

Install [DepthAI-Core V3 SDK](https://github.com/luxonis/depthai-core/tree/main) with Python bindings. Ensure udev rules are configured per the [official script](https://raw.githubusercontent.com/luxonis/depthai-python/main/docs/install_dependencies.sh).

### Run

```bash
cd examples/oak-d
python3 run_stereo.py
```

Tested on OAK-D W Pro. Distortion models and camera frame conventions may differ for other OAK-D models.

### Static masks for unrectified images

Unrectified fisheye images have distortion near edges. Apply border masks:
```python
cam = cuvslam.Camera()
cam.border_top = 20
cam.border_bottom = 30
cam.border_left = 30
cam.border_right = 50
```

### Notes
- If FPS is low, check Luxonis [optimization guide](https://docs.luxonis.com/software-v3/depthai/tutorials/optimizing)
- Stereo tracker requires stable FPS and synchronized pairs

---

## Orbbec

**Supported modes:** Stereo, Mono-Depth (RGB-D)

### Install OrbbecSDK

Install [OrbbecSDK V2 Python wrapper](https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html#online-installation). Configure udev rules per [official guide](https://github.com/orbbec/OrbbecSDK#environment-setup).

Tested on Orbbec 335L.

### Run

```bash
cd examples/orbbec
python3 run_stereo.py   # Stereo odometry
python3 run_rgbd.py     # Mono-depth (enable IR emitter, use RGB not IR)
```

### Notes
- Some Orbbec models (e.g., 336) use rolling shutter RGB sensor → poor visual tracking. Use global shutter sensor.
- For RGB-D: ensure depth is aligned with RGB camera. Enable IR emitter for better depth quality but avoid using IR images for visual tracking (emitter pattern creates artificial features).
- Mono-depth mode requires significantly more GPU resources than stereo. See [hardware scalability](https://arxiv.org/html/2506.04359v3#A1.F13) for Jetson guidance.
