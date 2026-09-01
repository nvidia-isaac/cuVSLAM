# Tutorial: ZED recording and offline PyCuVSLAM tracking with Rerun visualisation

See also [StereoLabs: Video Recording](https://www.stereolabs.com/docs/video/recording)

## Setting Up the Environment
Refer to the [Installation ZED SDK](../README.md#zed-sdk-installation) for instructions on installing and configuring
all required dependencies.

For Python replay with the shared example dependencies, use ZED SDK `5.4.0` as recommended in the parent ZED README.
Some older `pyzed` wheels, such as the one shipped with ZED SDK `4.1`, require `numpy<2.0` and are not compatible with
the current `examples/requirements.txt` NumPy pin.

- **record_from_zed** - CLI tool for capturing VGA 100fps stereo images from a ZED 1 camera using GPU-accelerated H.265
  lossy compression. It produces approximately 1GB files every 15 minutes and is suitable for long recordings (>1 hour)
  on a regular laptop. The SVO2 file stores raw RGB-synchronised images and preserves camera calibration. During
  playback, it enables runtime rectification. You can expect 100 fps recording, 350 fps full playback, and 250 fps with
  odometry tracking.
  To build:
  ```
  cmake -S . -B build
  cmake --build build
  ```
  To run recording. Press Ctrl+C to stop recording.
  ```
  ./build/record_from_zed
  ```
- **track_svo** - play the svo2 file, track, and visualise the result.

  The script reads `recording.svo2` from the current working directory. To replay a file produced by
  `record_from_zed`, copy or symlink it so the filename matches:

  ```
  cd track_svo
  ln -sf ../record_from_zed/rec_YYYYMMDD_HH_MM_SS.svo2 recording.svo2
  python3 track_svo.py
  ```

  To use a different filename, edit the `svo_filename` variable at the top of `track_svo.py`.

  Depending on your recording content, you may want to adjust the following parameters:
  * `enable_slam` - Switch between odometry and slam modes.
  * `throttle_vis` - Do not visualize all frames, just one frame per second. Makes sense for long recordings.
  * `rich_vis` - Draw observation and landmarks in addition to the trajectory. May slow down visualization.
  * `border_*` - (top, bottom, left, right) from `cuvslam.Camera`.
    See [Static Masks](../../tum/README.md#masking-regions-to-prevent-feature-selection)
  * `map_cell_size` - from `cuvslam.Slam.Config`. Size of map cell (0 for auto-calculate from camera baseline).
  * `max_map_size` -  from `cuvslam.Slam.Config`. Maximum number of poses in SLAM pose graph (0 for unlimited).

## Tested environment

The performance figures in the recorder description above were measured with this environment. For current Python replay
setup, prefer the ZED SDK version recommended in [ZED SDK Installation](../README.md#zed-sdk-installation).

```
Ubuntu 22.04.5 LTS (python3.10)
ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run
cuda-toolkit-12-2
Driver Version: 570.195.03     CUDA Version: 12.8 (nvidia-driver-570)
ZED 1 camera (SN342990 with firmware zed_fw_v1523_spi.bin)
NVIDIA GeForce RTX 4090
Intel(R) Core(TM) i9-10940X CPU @ 3.30GH
```
