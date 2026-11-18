# Tutorial: Running PyCuVSLAM Multicamera Visual Odometry

This tutorial demonstrates how to run PyCuVSLAM in multicamera mode on different datasets

Both scripts can also run in the experimental cuNLS-based `Multisensor` odometry mode, which accepts the same
stereo pairs, including unrectified ones: set `USE_MULTISENSOR_MODE = True` at the top of the script. This requires a
cuNLS-enabled PyCuVSLAM build; official release wheels include cuNLS.

## Set Up the PyCuVSLAM Environment

Refer to the [Installation Guide](../README.md#prerequisites) for detailed environment setup instructions

## Tartan Ground Dataset

cuVSLAM can be run in multicamera mode on setup with hardware-synchronized stereo cameras. This section describes how to run PyCuVSLAM on the [Tartan Ground dataset](https://tartanair.org/tartanground/)

### Download Dataset

Install the [tartanair](https://tartanair.org/installation.html) package and run the download script (see also [TartanGround download docs](https://tartanair.org/examples.html#download-tartanground)):

> **Note**: The `tartanair` package only works on **x86_64**. On aarch64 (e.g. Jetson), it fails at import due to an upstream numba compatibility bug. Download the dataset on an x86_64 machine and transfer it to the target device

```bash
pip install tartanair
python3 download_tartan.py
```

> **Troubleshooting**: If the download fails (for example, with a connection timeout to `airlab-share-02.andrew.cmu.edu`), the PyPI version may be outdated. Install the latest version directly from GitHub (you may also need to fix tartanair API calls in the download script):
> ```bash
> pip install --force-reinstall git+https://github.com/castacks/tartanairpy.git
> ```

> **Note**: Make sure you have sufficient disk space before downloading. The single P2000 sequence with all 12 cameras requires ~8GB to download and ~17GB on disk (including zip files). You can delete the `.zip` files after extraction to reclaim space

### Running 6-Stereo Camera Visual Tracking

```bash
python3 track_multicamera_tartan.py
```

After running the script, you should see a Rerun visualization window displaying the image streams from all the downloaded cameras:

![Tartan Visualization](../assets/tutorial_multicamera_tartan.gif)

## R2B Galileo Dataset

This example is based on the original tutorial provided for ROSbag in the [ISAAC ROS documentation](https://nvidia-isaac-ros.github.io/v/release-3.2/concepts/visual_slam/cuvslam/tutorial_multi_hawk.html).

> **Note**: The linked ISAAC ROS tutorials reference release 3.2, which supports **Jetson Orin** (JetPack 6.1/6.2) and **x86_64** (Ubuntu 22.04+, CUDA 12.6+, Ampere+ GPU) with **ROS 2 Humble** only

### Get Dataset

To obtain the multicamera datasets, you need to download [r2b datasets in ROSbag format](https://registry.ngc.nvidia.com/orgs/nvidia/teams/isaac/resources/r2bdataset2024/files) and convert them using the [edex extractor](https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_common/isaac_ros_rosbag_utils/index.html#edex-extraction).

#### Step 1: Download and Verify ROSbag

1. Follow the original [ISAAC ROS Multi-camera Visual SLAM tutorial](https://nvidia-isaac-ros.github.io/v/release-3.2/concepts/visual_slam/cuvslam/tutorial_multi_hawk.html) for execution on a rosbag
2. Ensure the rosbag file has been downloaded and verify you can play it within the docker container:

```bash
ros2 bag play ${ISAAC_ROS_WS}/isaac_ros_assets/rosbags/r2b_galileo
```

#### Step 2: Extract Data Using EDEX

Set up the [Isaac ROS rosbag utils](https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_common/isaac_ros_rosbag_utils/index.html) package and run within the docker container:

```bash
ros2 run isaac_ros_rosbag_utils extract_edex \
  --config_path src/isaac_ros_common/isaac_ros_rosbag_utils/config/edex_extraction_nova.yaml \
  --rosbag_path ${ISAAC_ROS_WS}/isaac_ros_assets/rosbags/r2b_galileo \
  --edex_path ${ISAAC_ROS_WS}/isaac_ros_assets/r2b_galileo_edex
```

#### Expected Output Structure

The expected structure of the output folder:

```bash
r2b_galileo_edex
├── frame_metadata.jsonl
├── images
│   ├── back_stereo_camera
│   │   ├── left
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   │   └── right
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   ├── front_stereo_camera
│   │   ├── left
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   │   └── right
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   ├── left_stereo_camera
│   │   ├── left
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   │   └── right
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   ├── raw_timestamps.csv
│   ├── right_stereo_camera
│   │   ├── left
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   │   └── right
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   ├── 000002.png
│   │   │   ...
│   └── synced_timestamps.csv
├── robot.urdf
└── stereo.edex
```

3. Exit from docker container and move `r2b_galileo_edex` folder to multicamera example folder:
```bash
cd examples/multicamera_edex
mkdir -p dataset
mv ~/workspaces/isaac/isaac_ros_assets/r2b_galileo_edex dataset/r2b_galileo_edex
```

## Running 4-Stereo Camera Visual Tracking

Go to the `examples/multicamera_edex` folder and run

```bash
python3 track_multicamera_r2b.py
```

After running the script, you should see a Rerun visualization window displaying 8 unrectified image streams, corresponding to the left and right images from each of the 4 stereo cameras:

![R2B Visualization](../assets/tutorial_multicamera_edex.gif)
