# Tutorial: Running PyCuVSLAM Visual Odometry on ZED Stereo Camera

This tutorial demonstrates how to perform live PyCuVSLAM tracking using stereo images and depth data from a ZED stereo
camera

## Setting Up the Environment

Refer to the [Installation ZED SDK](../README.md#zed-sdk-installation) for instructions on installing and configuring
all required dependencies.

## Camera Resolution and FPS

Both scripts open the camera at `RESOLUTION = sl.RESOLUTION.AUTO`, letting the ZED SDK pick the camera's native mode
(HD720 on USB ZED, HD1200 on ZED X). The default frame rate is 60 FPS for `run_stereo.py` and 30 FPS for
`run_rgbd.py`. To pick a specific mode, edit `RESOLUTION` at the top of the script &mdash; see
[Stereolabs' resolution and FPS matrix](https://www.stereolabs.com/docs/video/camera-controls#selecting-a-resolution)
for the supported (camera, mode, FPS) combinations.

If the FPS you set isn't supported at the chosen resolution, the ZED SDK falls back to the highest supported value for
that mode and prints a warning &mdash; tracking still runs.

## Running Stereo Visual Odometry

### Using Distorted Images

The ZED camera can provide either rectified or distorted stereo images. **By default, rectified images are used for
visual odometry.** If you need to use distorted images, update the following flag at the top
of [run_stereo.py](run_stereo.py):

```
RAW = True
```

> **Tip:** Rectified images are generally preferred for most visual odometry applications, as they simplify downstream
> processing. Only use distorted images if you have a specific requirement

This script has been developed and validated for the ZED2 camera using distortion coefficients in the
format: $[k_1, k_2, p_1, p_2, k_3]$. If you plan to use a different ZED camera model with distorted images, please
validate the distortion coefficients and model. You may need to update [camera_utils.py](camera_utils.py) to ensure
compatibility with your hardware.

#### Running the Stereo Visual Odometry Script

To start stereo visual odometry, run:

```bash
python3 run_stereo.py
```

After starting, you should see a visualization similar to the following:
![Visualization Example](../../assets/tutorial_zed_stereo.gif)

> **Performance Note:** PyCuVSLAM stereo tracker requires reliably synchronized stereo pairs and a stable frame rate. If
> you see warnings about low FPS, try reducing the camera resolution or decreasing the frame rate in your camera settings

## Running Monocular-Depth Visual Odometry

Monocular-Depth Visual Odometry requires pixel-to-pixel alignment between the camera image and its corresponding depth
image. For ZED cameras, the depth image is aligned with the left camera image by default.

PyCuVSLAM expects the depth image in `uint16` format, while the camera images (either RGB or grayscale) should be in
`uint8` format. A scale factor is used to convert pixel values to meters. In the provided script, depth is output in
millimeters (`sl.UNIT.MILLIMETER`) with a scale factor of 1000. If you need different depth units, modify these settings
in [run_rgbd.py](run_rgbd.py).

By default, the script uses the `sl.DEPTH_MODE.PERFORMANCE` depth mode of ZED SDK. For details on other depth modes and
GPU acceleration, refer to the [official ZED documentation](https://www.stereolabs.com/docs/depth-sensing/using-depth).

> **Tracking stability:** If the trajectory shows sudden jumps or the script reports frame drops at the default 30 FPS,
> reduce `FPS` near the top of [run_rgbd.py](run_rgbd.py) from `30` to `15`. The lower frame rate can help the RGB-D
> pipeline process frames consistently on systems that cannot keep up at 30 FPS.

To run monocular-depth visual odometry, execute:

```bash
python3 run_rgbd.py
```

You should see a visualization showing the camera trajectory along with the input RGB and depth images:

![ZED Mono-Depth Example](../../assets/tutorial_zed_rgbd.gif)

---

### Monocular-Depth + Stereo Visual Odometry

For stereo depth cameras without an IR emitter (which can interfere with stereo image capture), you can add the stereo
camera input to the RGBD tracker. In this mode, cuVSLAM still utilizes a single visual tracker
`cuvslam.Odometry.OdometryMode.RGBD`, predicting the camera pose based on both image-depth and stereo-image input data.
This approach combines the inputs at the solver level, rather than simply blending the results of two separate trackers
at postprocessing stage.

To enable Mono-Depth + Stereo visual tracking, set the `RUN_STEREO = True` flag at the top
of [run_rgbd.py](run_rgbd.py), then run:

```bash
python3 run_rgbd.py
```

You should see a visualization displaying the camera trajectory along with the input stereo RGB and depth images:

![ZED Mono-Depth+Stereo Example](../../assets/tutorial_zed_rgbd_stereo.gif)

When using the Mono-Depth + Stereo tracker, ensure that `depth_camera_id` corresponds to the index of the camera aligned
with the depth image:

```python
rgbd_settings = vslam.Odometry.RGBDSettings()
rgbd_settings.depth_scale_factor = 1000
rgbd_settings.depth_camera_id = 0
rgbd_settings.enable_depth_stereo_tracking = RUN_STEREO
```

This index should match the position of the left camera in the list of cameras passed to `cuvslam.Rig()`:

```python
rig.cameras = [left_camera, right_camera]
```
