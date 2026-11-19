# Tutorial: Running PyCuVSLAM Stereo Visual Odometry on a Realsense Stereo Camera

This tutorial walks you through how to operate live PyCuVSLAM tracking on a Realsense stereo camera

## Set Up the PyCuVSLAM Environment

Refer to the [Installation Guide](../README.md#prerequisites) for detailed environment setup instructions

## Setting Up librealsense

> **Note:** skip this section if using PyCuVSLAM with [Docker](../README.md#using-docker); only Docker setup was tested on Jetson

The official [librealsense Python installation guide](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md) is recommended. We highly suggest building both librealsense and its Python wrapper from source:

1. Follow the build instructions for the [librealsense library](https://github.com/realsenseai/librealsense/blob/master/doc/installation.md)
   > **Note:** Examples are tested with `v2.57.6` librealsense version, please use `git clone --branch v2.57.6 https://github.com/realsenseai/librealsense.git` instead of the command in the instructions.

2. Compile librealsense2 with Python bindings (note the flags):
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON -DBUILD_EXAMPLES=true -DBUILD_PYTHON_BINDINGS=true  -DCHECK_FOR_UPDATES=false -DPYTHON_EXECUTABLE=$(which python3)
   cmake --build build --parallel $(nproc)
   ```

3. Once the build and installation are finished, add the build directory to your path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:~/librealsense/build/Release
   ```

4. Test your setup by running a [simple camera example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py) to verify that everything was installed correctly

## Running Stereo Visual Odometry

>**USB camera troubleshooting:** occasionally, USB camera may fail to start. If this occurs, briefly disconnect and reconnect it from the USB port

Go to the `examples/realsense` folder and run stereo visual odometry:
```bash
python3 run_stereo.py
```
You should see the following interactive visualization in rerun:
![Visualization Example](../assets/tutorial_realsense_stereo.gif)

## Running Stereo Inertial Odometry

>**Note**: real-time inertial odometry via the **Python API on the Jetson** platform may **not** be **reliable**. If you need to run cuVSLAM with a RealSense camera on Jetson in this mode, please use the [C++ ROS2 node](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/tutorial_realsense.html) instead, or consider using any other PyCuVSLAM modes with a RealSense live camera from this tutorial

This example demonstrates how to run PyCuVSLAM stereo inertial odometry using Realsense stereo cameras with integrated IMU. Accurate inertial odometry heavily depends on correct [IMU noise parameters](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model), so a calibration specific to your IMU is required.

To achieve optimal performance, you must [calculate the IMU noise parameters](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model) for your particular camera and update these parameters accordingly in [camera_utils.py](camera_utils.py):

```
IMU_GYROSCOPE_NOISE_DENSITY = ...
IMU_GYROSCOPE_RANDOM_WALK = ...
IMU_ACCELEROMETER_NOISE_DENSITY = ...
IMU_ACCELEROMETER_RANDOM_WALK = ...
```

Run stereo visual inertial odometry:

```bash
python3 run_vio.py
```

After a brief initialization period, a gravity vector (shown as a bold red line pointing downward) should appear in the 3D visualization window:

![Visualization Example](../assets/tutorial_realsense_vio.gif)

> **Note**: Although inertial odometry can handle short-term invalid frame observations (e.g., due to motion blur or featureless surfaces), for Realsense cameras we recommend increasing the camera frame rate to minimize these artifacts and using the default visual odometry mode. For challenging long-term visual conditions, consider using a multi-camera setup.
>

## Running Multicamera Odometry

PyCuVSLAM supports visual odometry using multiple stereo hardware-synchronized cameras. This allows robust and continuous tracking even when some cameras in the rig become fully occluded, as illustrated below. Unlike the single-stereo camera case, where extrinsics are directly obtained from the camera, the multicamera setup requires extrinsic parameters to be manually defined by the user.

In the provided example, we use a [yaml description](frame_nano_rig.yaml) file for the Jetson Nano camera rig from the [Multi-RealSense Hardware Assembly Guide](multicamera_hardware_assembly.md). Use this guide to assemble your own multicamera rig, making sure to update the calibration parameters and camera serial numbers (S/N) in the YAML file to match your specific hardware setup.

Once your multicamera setup is assembled and parameters in yaml are updated, you can run multicamera odometry by executing:

```bash
python3 run_multicamera.py
```

It takes approximately 15 seconds for multiple RealSense cameras to synchronize their timestamps. After this warm-up period, PyCuVSLAM will start receiving images from all cameras, and you should see the following visualization displaying left images from all cameras along with the 3D trajectory:

![Visualization Example](../assets/tutorial_realsense_multicamera.gif)

## Running Monocular-Depth Visual Odometry

Monocular-Depth Visual Odometry requires pixel-to-pixel correspondence between camera and depth images. For RealSense cameras, please ensure depth images are aligned with the RGB camera.

We recommend enabling the RealSense infrared emitter for improved depth image quality. Note that when the emitter is enabled, infrared images from stereo cameras will contain artificial features from the emitter pattern. Therefore, avoid using infrared images for visual tracking in this mode. Instead, use the RGB camera as your visual data source.

To run monocular-depth visual odometry, execute:

```bash
python3 run_rgbd.py
```

You should see the following rerun visualization, displaying the camera trajectory along with the input RGB and depth images:

![Visualization Example](../assets/tutorial_realsense_rgbd.gif)

## Running Multisensor Odometry

Multisensor odometry uses the cuNLS-based `OdometryMode.Multisensor` pipeline. It requires at least one RGB-D camera or
one overlapping camera pair; RGB-D cameras currently must use the pinhole model, while cameras without depth can use
any distortion model. With a RealSense D455, the simplest live setup
is RGB-D + IMU: the RGB stream provides visual features, the aligned depth stream provides metric depth, and the
integrated IMU is registered through the same serialized tracker call path as the stereo inertial example.

To run RGB-D + IMU multisensor odometry, execute:

```bash
python3 run_multisensor.py
```

For a stereo + IMU multisensor smoke test using the same IR streams as the stereo inertial example, set
`USE_MULTISENSOR_MODE = True` in `run_vio.py` and run:

```bash
python3 run_vio.py
```

The multi-camera example offers the same switch: set `USE_MULTISENSOR_MODE = True` in `run_multicamera.py` to track
all stereo pairs in Multisensor mode.

> **Note:** Multisensor mode requires a cuNLS-enabled PyCuVSLAM build. Official release wheels include cuNLS, and the
> default source build enables it. If cuVSLAM was configured with `-DUSE_CUNLS=OFF`, these multisensor examples are
> unavailable.
