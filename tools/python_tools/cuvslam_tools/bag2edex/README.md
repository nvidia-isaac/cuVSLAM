# Rosbag Extraction Tool


## Install

See [tools/README.md](/tools/README.md) on how to install the `cuvslam_tools` package.


## Configuration

The config is a YAML file with the following fields:
```yaml
# Topic names for CameraInfo messages.
camera_info_topics:
  - /front_stereo_camera/left/camera_info
  - /front_stereo_camera/right/camera_info
  - /back_stereo_camera/left/camera_info
  - /back_stereo_camera/right/camera_info

# Topic names for image messages, in the same order as camera_info_topics.
image_topics:
  - /front_stereo_camera/left/image_compressed
  - /front_stereo_camera/right/image_compressed
  - /back_stereo_camera/left/image_compressed
  - /back_stereo_camera/right/image_compressed

# Name of the rig frame. If unsure, run `rosbag_extract_urdf` and select from list of frames.
rig_frame: base_link

# Name of the IMU frame. (Optional)
imu_frame: front_stereo_camera_imu

# Parameters to resize and reformat images. (Optional)
output_width: 960
output_height: 600
output_format: png

# Maximum threshold for timestamp synchronization, in nanoseconds. Default is 1ms.
sync_threshold_ns: 1000000

# Number of workers for image extraction, must be >=2. Default is number of cores on system.
num_workers: 8

# ROS distribution. Default is "humble".
ros_distribution: humble
```


## Running Extraction Tools

Note: you may see warning messages about `InvalidDataError` during image extraction. This is expected and does not interfere with the execution of the tool.

Extract an EDEX dataset from a rosbag:
```sh
rosbag_extract_edex \
    --config configs/<config>.yaml \
    --rosbag_path path/to/rosbag/directory \
    --output_path path/to/edex/directory
```

Extract images from a rosbag (without camera extrinsics):
```sh
rosbag_extract_images \
    --config configs/<config>.yaml \
    --rosbag_path path/to/rosbag/directory \
    --output_path path/to/edex/directory
```

Extract videos from a rosbag (currently only supports H.264 encoded rosbags):
```sh
rosbag_extract_videos \
    --config configs/<config>.yaml \
    --rosbag_path path/to/rosbag/directory \
    --output_path path/to/edex/directory
```

Extract a URDF file from a rosbag:
```sh
rosbag_extract_urdf \
    --rosbag_path path/to/rosbag/directory \
    --output_path path/to/edex/directory \
    --ros_distribution humble
```


## Generating Missing `metadata.yaml` for MCAP Files

`make_mcap_metadata.py` regenerates the `metadata.yaml` file that `rosbag2` requires next to an `.mcap` file. Use it when you have a standalone `.mcap` file without its companion `metadata.yaml` — for example, after downloading only the `.mcap` from a recording system or after the metadata file was accidentally deleted.

**Prerequisites:** a ROS 2 environment must be sourced with `ros2` on `PATH`, and the MCAP storage plugin must be installed (the script calls `ros2 bag info --storage mcap` internally).

**Usage:**

```sh
./make_mcap_metadata.py path/to/rosbag.mcap
```

The script writes `metadata.yaml` into the same directory as the `.mcap` file. Once generated, the directory can be used with the extraction tools above.

**Optional argument:**
- `--version <int>` — sets the `rosbag2_bagfile_information.version` field in the metadata (default: `9`). Only change this if you need to match a specific rosbag2 version.

**Example:**

```sh
./make_mcap_metadata.py /data/bags/my_recording/my_recording_0.mcap
# Wrote: /data/bags/my_recording/metadata.yaml

rosbag_extract_edex \
    --config configs/my_config.yaml \
    --rosbag_path /data/bags/my_recording \
    --output_path /data/edex/my_recording
```


## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
