# Tutorial: Running PyCuVSLAM Stereo-Inertial Odometry on the EuRoC MAV Dataset

This tutorial demonstrates how to run PyCuVSLAM Stereo-Inertial Visual Odometry using the EuRoC MAV dataset with unrectified images

## Visual Tracking Modes

PyCuVSLAM supports multiple visual tracking modes. You can specify the desired tracking mode through the `cuvslam.Odometry.Config` object when initializing visual tracking. Tracking modes can be set either by using enumeration values or directly using their respective names:

* **Stereo**: Visual tracking using stereo cameras. This mode can be extended to multiple stereo cameras (*PyCuVSLAM default mode*, set as `OdometryMode(0)` or `OdometryMode.Multicamera`)

* **Stereo-Inertial**: Visual-inertial tracking using stereo cameras combined with IMU data (set as `OdometryMode(1)` or `OdometryMode.Inertial`)

* **Mono-Depth (RGB-D)**: Visual tracking using a monocular camera and depth images (set as `OdometryMode(2)` or `OdometryMode.RGBD`)

* **Monocular**: Visual tracking using a monocular camera. This mode provides accurate camera rotation estimation but does not estimate scale. (set as `OdometryMode(3)` or `OdometryMode.Mono`)

* **Multisensor**: Unified mode that accepts at least one RGB-D camera or one overlapping camera pair, with optional additional RGB/RGB-D cameras and one IMU (set as `OdometryMode(4)` or `OdometryMode.Multisensor`). Requires a cuNLS-enabled build, currently supports pinhole cameras only, and is configured via `MultisensorSettings`. See [examples/multisensor/](../multisensor/README.md) for a runnable walkthrough.

PyCuVSLAM exercises three modes on the EuRoC MAV dataset — Stereo (`Multicamera`), Stereo-Inertial (`Inertial`), and Monocular (`Mono`) — because *Mono-Depth* has no aligned depth stream (see the [TUM-RGBD dataset example](../tum/README.md)) and *Multisensor* has its own [example](../multisensor/README.md). You can try different tracking modes by modifying the following line in `track_euroc.py`:

```python
euroc_tracking_mode = cuvslam.Odometry.OdometryMode(1)
```

## Distortion Models

PyCuVSLAM supports several distortion models. Each model is specified by name along with a corresponding list of coefficients during camera initialization (`cuvslam.Camera(cuvslam.Distortion(...))`). Supported models include:

- **Pinhole**: Assumes no distortion and requires 0 coefficients. This is the default model (`Distortion.Model.Pinhole` or `Distortion.Model(0)`)
- **Fisheye (Equidistant)**: Uses 4 distortion coefficients (`Distortion.Model.Fisheye` or `Distortion.Model(1)`)
- **Brown**: Distortion model consisting of 3 radial and 2 tangential coefficients: $k_1, k_2, k_3, p_1, p_2$ (`Distortion.Model.Brown` or `Distortion.Model(2)`)
- **Polynomial**: Distortion model with 8 coefficients: $k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6$ (`Distortion.Model.Polynomial` or `Distortion.Model(3)`)

The example provided in this repository uses the **Brown** model for the original dataset calibration and the **Fisheye** model for [updated calibrations](./sensor_cam1.yaml) provided in repository. To achieve results similar to those shown in the [cuVSLAM technical report](https://arxiv.org/html/2506.04359v3#S3.T2), use the recalibrated camera and imu parameters.

> **Note**: Ensure the correct number and order of distortion coefficients when initializing your cameras. If you experience poor tracking performance with unrectified cameras, consider testing with `OdometryMode.Mono`. This mode typically yields smoother trajectories and accurate rotational poses when camera parameters are correct

## Set Up the PyCuVSLAM Environment

Refer to the [Installation Guide](../README.md#prerequisites) for detailed environment setup instructions

## Dataset Setup

1. Download the EuRoC MH_01_easy dataset:

    1. Go to https://doi.org/10.3929/ethz-b-000690084
    2. Download "Machine Hall Datasets (ZIP, 12096.15 MB)"
    3. Extract `/machine_hall/MH_01_easy/MH_01_easy.zip` from the downloaded archive (`machine_hall.zip`)
    4. Extract `mav0` to `dataset/` from `MH_01_easy.zip`

2. Copy the calibration files:

    ```bash
    cd examples/euroc
    cp sensor_cam0.yaml dataset/mav0/cam0/sensor_recalibrated.yaml
    cp sensor_cam1.yaml dataset/mav0/cam1/sensor_recalibrated.yaml
    cp sensor_imu0.yaml dataset/mav0/imu0/sensor_recalibrated.yaml
    ```

3. Ensure the dataset path is correctly set at the beginning of the visual tracking script.

## Running Stereo Inertial Odometry

In the `examples/euroc` folder run:

```bash
python3 track_euroc.py
```

You should see the following visualization in Rerun. In Visual-Inertial mode, the red arrow pointing downward represents the gravity vector estimated by cuVSLAM during inertial tracking:

![Visualization Example](../assets/tutorial_euroc.gif)

> **Note**:
> - If you experience poor Stereo-Inertial tracking, first validate that Mono tracking and Stereo Visual tracking perform correctly with the same intrinsic and extrinsic camera parameters
> - If the gravity vector is consistently misaligned (not pointing downward), please double-check and update your IMU extrinsics matrix
