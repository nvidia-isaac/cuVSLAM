# Tutorial: PyCuVSLAM Multisensor Odometry (multi RGB-D + IMU)

> **Experimental:** Tracking may be inaccurate or fail for some sensor configurations and scenes.

This tutorial demonstrates how to run PyCuVSLAM in **Multisensor** odometry
mode, which solves a single tightly-coupled cuNLS step over any mix of plain
RGB cameras, RGB-D cameras, and one optional IMU. The example uses two
RGB-D cameras (`lcam_front`, `lcam_back`) plus a synthetic IMU from the
[TartanGround dataset](https://tartanair.org/tartanground/).

## Requirements

- Use an official release wheel, which includes cuNLS, or build from source with `USE_CUNLS=ON`.
- Configure at least one RGB-D camera in `depth_camera_ids`, or provide at least one camera pair with overlapping
  frustums. A single RGB-D camera is valid, with or without an IMU.
- Use pinhole cameras. Other camera models are not supported by the current solver.
- Align each depth image pixel-for-pixel with the RGB image at the same camera index. PyCuVSLAM accepts 2D `uint16`
  depth; the C++ API accepts `UINT16` or `FLOAT32`.
- Configure no more than one IMU. Serialize image and IMU calls in non-decreasing timestamp order; camera frame
  timestamps must be strictly increasing.

Construction raises `ValueError` when the rig or settings are invalid. A valid tracker may return a
`PoseEstimate` with `world_from_rig=None` while initializing or after tracking loss. Configured depth streams may be
omitted from an individual frame after a sensor drop, but every supplied depth must match a configured camera index.

## Set Up the PyCuVSLAM Environment

Refer to the [Installation Guide](../README.md#prerequisites) for detailed
environment setup instructions.

## Download Dataset

Install the [tartanair](https://tartanair.org/installation.html) package and
run the download script (see also
[TartanGround download docs](https://tartanair.org/examples.html#download-tartanground)):

> **Note**: The `tartanair` package only works on **x86_64**. On aarch64
> (e.g. Jetson) it fails at import due to an upstream numba compatibility
> bug. Download the dataset on an x86_64 machine and transfer it to the
> target device.

```bash
pip install tartanair
python3 download_tartan.py
```

> **Troubleshooting**: If the download fails (for example, with a connection
> timeout to `airlab-share-02.andrew.cmu.edu`), the PyPI version may be
> outdated. Install the latest version directly from GitHub (you may also
> need to fix tartanair API calls in the download script):
> ```bash
> pip install --force-reinstall git+https://github.com/castacks/tartanairpy.git
> ```

The download fetches `image`, `depth`, and `imu` modalities for two cameras
(`lcam_front`, `lcam_back`) on the `OldTownFall` / `Data_anymal` / `P2000`
sequence. Expect a few GB of zips and a comparable amount on disk after
unzip; you can delete the `.zip` files afterwards.

## Running Multisensor Visual-Inertial Tracking

```bash
python3 track_multisensor_tartan.py             # multi RGB-D + IMU
python3 track_multisensor_tartan.py --no-imu    # multi RGB-D only (no IMU)
```

`--no-imu` removes the IMU from the rig and skips IMU loading entirely; the
tracker still runs in Multisensor mode with the two RGB-D cameras, which is
useful for A/B-ing the contribution of the IMU.

After running the script, a Rerun visualization window opens with:

- Two RGB camera streams (`lcam_front`, `lcam_back`) on the top row.
- The matching depth streams below them.
- A 3D view with the rig trajectory, current observations, and final
  landmarks.
- IMU acceleration and angular-velocity time-series at the bottom.

## What the example exercises

- **Multisensor odometry mode**
  (`cuvslam.Odometry.OdometryMode.Multisensor`) configured via
  `cuvslam.Odometry.MultisensorSettings(depth_camera_ids=[0, 1], ...)`.
- **Depth in millimetres (uint16)**: cuvslam's Python tracker requires
  `uint16` depth maps. TartanGround ships float32 depth in metres, so
  `load_depth()` multiplies by 1000 and clips to the uint16 range, and the
  tracker is configured with `depth_scale_factor=1000.0` so cuvslam recovers
  metres. The same convention is used by RealSense and ZED depth streams.
- **IMU fusion**: an `ImuCalibration` is attached to the rig, and IMU
  samples are pushed in between `track()` calls with
  `tracker.register_imu_measurement(0, ...)`, identical to the
  `Inertial`-mode pattern in `examples/euroc/track_euroc.py`. Multisensor
  mode automatically enables IMU fusion when the rig contains an IMU.

## Adapting the example

- **Different cameras** — edit `CAMERA_LIST` in `track_multisensor_tartan.py`
  and the cameras block in `tartan_ground.edex` (the multicamera example at
  `../multicamera_edex/tartan_ground.edex` has all 12 TartanGround
  cameras to copy from).
- **Add plain RGB cameras** — Multisensor mode accepts any subset of
  cameras as depth-providers. Drop a camera index from `depth_camera_ids`
  and pass an empty `np.empty(0)` for that camera's slot in `depths`.
- **IMU file names** — if your TartanGround download produced different
  IMU filenames than the ones tried in
  `dataset_utils._load_tartan_imu()`, extend the `candidates` list there.
