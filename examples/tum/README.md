# Tutorial: Running PyCuVSLAM on the TUM RGB-D Dataset

This tutorial demonstrates how to run PyCuVSLAM Mono-Depth Visual Odometry on the TUM RGB-D dataset.

PyCuVSLAM supports both monocular and mono-depth tracking modes. However, there are significant differences between these two modes under the hood. You can read more in the [cuVSLAM Technical Report](https://arxiv.org/html/2506.04359v3#S2). This tutorial focuses exclusively on Mono-Depth tracking. For instructions on Mono tracking, please see the [EuroC dataset example](../euroc/README.md).

## Depth Settings

Running PyCuVSLAM in Mono-Depth tracking mode requires synchronized rectified depth-image pairs with pixel-to-pixel correspondence:

- **Images**: must be grayscale or RGB with format `uint8`
- **Depth maps**: must be represented as `uint16`

To initialize the PyCuVSLAM tracker, set `odometry_mode` to `Odometry.OdometryMode.RGBD` and configure depth settings via `Odometry.RGBDSettings` with the following parameters:

- `depth_scale_factor`: defines the mapping between pixel values and real-world depth values
- `depth_camera_id`: the camera index in the camera rig providing the visual image corresponding to the depth map
- `enable_depth_stereo_tracking`: enables the use of the visual camera from the monocular-depth setup as part of a stereo pair, providing additional input data to the RGBD tracker. For more details, see the example for the [ZED camera](../zed/live/README.md#monocular-depth--stereo-visual-odometry)

## Masking Regions to Prevent Feature Selection

Depth maps captured by real cameras often exhibit reduced quality near image edges. This degradation can arise due to depth-image undistortion, area of projected IR pattern and other limitations. To enhance visual tracking accuracy, we recommend applying static masks at image borders to prevent cuVSLAM from selecting unreliable features in these regions

Each border (top, bottom, left, right) can have independently specified mask thicknesses, allowing flexible exclusion of problematic regions. In the schematic example below, transparent red regions illustrate areas where masks are applied:

![tum_mask_example](../assets/tutorial_tum_mask.png)

You can define these mask borders individually for each camera instance with pixel values:

```python
cam = cuvslam.Camera()
cam.border_top = 20
cam.border_bottom = 20
cam.border_left = 10
cam.border_right = 50
```

## Set Up the PyCuVSLAM Environment

Refer to the [Installation Guide](../README.md#prerequisites) for detailed environment setup instructions

## Dataset Setup

1. Download the TUM RGB-D freiburg3 dataset by executing the following commands:

    ```bash
    cd examples/tum
    mkdir -p dataset
    wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz -O dataset/rgbd_dataset_freiburg3_long_office_household.tgz
    tar -xzf dataset/rgbd_dataset_freiburg3_long_office_household.tgz -C dataset
    rm dataset/rgbd_dataset_freiburg3_long_office_household.tgz
    ```
2. Copy the calibration file:

    ```bash
    cp freiburg3_rig.yaml dataset/rgbd_dataset_freiburg3_long_office_household/freiburg3_rig.yaml
    ```
> **Note**: PyCuVSLAM requires rectified images for monocular-depth visual odometry, which is why the freiburg3 dataset is used. The camera calibration is provided in [freiburg3_rig.yaml](freiburg3_rig.yaml), corresponding to the original calibration parameters available on the [dataset page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect)

3. Ensure the dataset path is correctly set at the beginning of the visual tracking script

## Running Monocular-Depth Odometry

To start tracking on the TUM RGB-D dataset, execute the following command:

```bash
python3 track_tum.py
```

> **Note**: PyCuVSLAM expects image-depth pairs for monocular-depth visual odometry. The original dataset provides lists of image and depth files with their timestamps, which are aligned to image-depth pairs corresponding to a common timestamp in [dataset_utils.py](dataset_utils.py).


You should see the following interactive visualization in rerun:

![Visualization Example](../assets/tutorial_tum.gif)
