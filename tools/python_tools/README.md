# cuvslam-tools

`cuvslam-tools` is the installable Python tools package for cuVSLAM workflows.

It provides command-line tools for:

- Preparing public datasets, including KITTI, EuRoC, TartanGround, TUM RGB-D, and CODa.
- Converting ROS 2 bags to EDEX inputs.
- Running one tracking sequence.
- Running dataset reports.
- Running multi-dataset validation.
- Undistorting EDEX images.

## Install

From the repository root:

```bash
cd tools/python_tools
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -e .
```

With PDF report support:

```bash
pip install -e ".[pdf]"
```

`cuvslam_tracker`, `cuvslam_reporter`, and `cuvslam_validator` require the `cuvslam` Python binding in the same environment. Dataset preparation, ROS bag conversion, and undistortion should stay usable without importing `cuvslam` when their workflows do not need it.

## Install The cuVSLAM Python Binding

If you have a released wheel that matches your platform, Python version, and CUDA version, install it into the same environment:

```bash
pip install /path/to/cuvslam-*.whl
```

When installing from this source tree, build cuVSLAM first, then install the binding from `python`. From `tools/python_tools`:

```bash
cd ../..
mkdir -p ../build
./build_release.sh

cd tools/python_tools
CUVSLAM_BUILD_DIR=/path/to/build/folder pip install ../../python/
```

`CUVSLAM_BUILD_DIR` must be an absolute path to the cuVSLAM build directory and must contain:

```bash
bin/libcuvslam.so
```

After installing the binding, verify it in the same environment:

```bash
python - <<'PY'
import cuvslam
print(cuvslam.get_version())
PY
```

Reinstall the binding after rebuilding `libcuvslam.so`.

## Commands

| Command | Purpose |
|---|---|
| `prepare_kitti` | Download KITTI odometry archives, convert them to cuVSLAM format, and generate KITTI reporter configs. |
| `prepare_euroc` | Download and convert all 11 official EuRoC MAV sequences, or an explicit subset, to portable EDEX and reporter configs. |
| `prepare_tartan` | Download TartanGround data and convert TartanGround stereo pairs or compatible TartanAir-layout sequences to EDEX. |
| `prepare_tum` | Download and convert the 15 evaluated TUM RGB-D freiburg3 sequences, or an explicit subset, to portable EDEX and a reporter config. |
| `prepare_icl_nuim` | Download and convert the eight ICL-NUIM living-room and office trajectories, or an explicit subset, to portable EDEX and a reporter config. |
| `prepare_coda` | Convert manually downloaded CODa sequence archives to portable EDEX and reporter configs. |
| `cuvslam_tracker` | Run one EDEX sequence or supported video input through cuVSLAM. |
| `cuvslam_reporter` | Run one dataset config and generate report outputs. |
| `cuvslam_validator` | Run multiple reporter configs, combine results, and apply validation checks. |
| `rosbag_extract_edex` | Convert a ROS 2 bag to an EDEX sequence directory. |
| `rosbag_extract_images` | Extract images from a ROS 2 bag. |
| `rosbag_extract_urdf` | Inspect/extract TF and URDF data from a ROS 2 bag. |
| `rosbag_extract_videos` | Extract videos from a ROS 2 bag. |
| `undistort_edex_images` | Undistort images from an EDEX sequence. |

Smoke-check installed commands:

```bash
prepare_kitti --help
prepare_euroc --help
prepare_tartan --help
prepare_tum --help
prepare_icl_nuim --help
prepare_coda --help
cuvslam_tracker --help
cuvslam_reporter --help
cuvslam_validator --help
rosbag_extract_edex --help
rosbag_extract_images --help
rosbag_extract_urdf --help
rosbag_extract_videos --help
undistort_edex_images --help
```

## Dataset Preparation

`prepare_kitti` runs `cuvslam_tools.dataset_preparation.kitti.prepare`. It downloads the KITTI odometry archives when needed, converts them to cuVSLAM format, and writes the reporter config files produced by that workflow, including:

- `kitti-vio_gt.cfg`
- `kitti-slam_gt.cfg`
- `kitti-vio_slam.cfg`
- `kitti-vio_slam_gt.cfg`

Example:

```bash
prepare_kitti \
    --raw-dir /path/to/datasets/kitti/raw \
    --output-dir /path/to/datasets/converted/kitti
```

The converted dataset layout is suitable for tracker and reporter workflows. Pass one of the generated KITTI config files to `cuvslam_reporter --test_config`.

Without `--raw-dir` and `--output-dir`, the command uses `./datasets/kitti/raw` and `./datasets/converted`, relative to
the current directory. The same workflow is importable, so scripts can prepare a dataset without going through the CLI:

```python
from cuvslam_tools.dataset_preparation.kitti.prepare import prepare

converted_root = prepare(raw_dir="/data/kitti/raw", output_dir="/data/converted")
```

`prepare_euroc` runs `cuvslam_tools.dataset_preparation.euroc.prepare`. It downloads the official Machine Hall,
Vicon Room 1, and Vicon Room 2 bundles and converts all 11 EuRoC MAV sequences. The output is portable: camera
images are copied under each prepared sequence instead of being linked to raw data outside the prepared root.

```bash
prepare_euroc \
    --raw-dir /path/to/datasets/euroc/raw \
    --output-dir /path/to/datasets/converted
```

Use `--sequences` for a smaller conversion. Only bundles needed by the selected sequences are downloaded:

```bash
prepare_euroc \
    --raw-dir /path/to/datasets/euroc/raw \
    --output-dir /path/to/datasets/converted \
    --sequences MH_01_easy
```

The prepared root is `/path/to/datasets/converted/euroc`. It contains ODOM, SLAM, and combined reporter configs
plus `dataset_metadata.json`. Every sequence contains `stereo.edex`, `frame_metadata.jsonl`, `IMU.jsonl`,
camera-aligned `gt.txt`, and copied `00/` and `01/` media directories.

The reporter layout intentionally uses the recalibrated cam0-relative fisheye parameters checked in under
`examples/euroc/` to reproduce the technical-report and benchmark results. It does not use the original
per-sequence camera calibration from the source archives. The official EuRoC cam0 body-from-sensor transform is
used only to express body-frame ground truth in the cam0 frame.

Run the combined inertial ODOM+SLAM report with:

```bash
cuvslam_reporter \
    --test_config /path/to/datasets/converted/euroc/euroc-vio_slam.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-euroc-reports \
    --odometry_mode inertial \
    --rectified_stereo_camera false \
    --async_sba false \
    --use_segments
```

`prepare_tartan` runs `cuvslam_tools.dataset_preparation.tartan.prepare`. It downloads a TartanGround variant, stages each available `lcam_*`/`rcam_*` stereo pair into the classic TartanAir layout expected by the converter, and converts the staged sequences to EDEX.

```bash
prepare_tartan \
    --variant multicamera \
    --raw-dir /path/to/datasets/tartan/raw \
    --output-dir /path/to/datasets/converted
```

This command needs the [tartanair](https://tartanair.org/installation.html) package for the download step
(`pip install tartanair`). That package is x86_64-only, so on aarch64 download on an x86_64 machine and transfer the
data.

Use `--variant multicamera` for EDEX conversion from the 12-camera TartanGround image variant. Both TartanGround variants also download metadata, including `pose_lcam_*` and `pose_rcam_*` files. The multicamera variant converts each complete stereo orientation, for example `P2000_front`, `P2000_left`, and `P2000_right`. The `multisensor` variant is intended for the RGB-D/IMU example data and can be downloaded with `--download-only`.

`prepare_tum` runs `cuvslam_tools.dataset_preparation.tum.prepare`. It downloads the 15 evaluated TUM RGB-D
freiburg3 sequence archives and converts them to portable EDEX with a reporter config.

```bash
prepare_tum \
    --raw-dir /path/to/datasets/tum/raw \
    --output-dir /path/to/datasets/converted

# One sequence, for a quick check (roughly 1.5 GB of source data)
prepare_tum \
    --raw-dir /path/to/datasets/tum/raw \
    --output-dir /path/to/datasets/converted \
    --sequences rgbd_dataset_freiburg3_long_office_household
```

The prepared root is `/path/to/datasets/converted/tum`. It contains `tum-rgbd_slam.cfg` and
`dataset_metadata.json`. Every sequence contains `stereo.edex`, `frame_metadata.jsonl`, camera-aligned `gt.txt`,
colour PNGs under `00/`, and depth PNGs under `01/`.

Colour and depth frames are associated within 1 ms, which pairs the two views of a single Kinect capture and
rejects pairs stitched across neighbouring captures. Ground truth is interpolated onto the associated frame times
(linear translation, slerp rotation) and written relative to the first frame. Frames outside the ground-truth time
span are dropped so every frame has a pose.

Depth is copied from the source unchanged: 16-bit PNG in TUM units, declared in the EDEX as
`depth_scale_factor: 5000`. All 15 sequences come from the freiburg3 camera series, whose published pinhole
intrinsics carry no distortion, so one calibration covers the selection.

Run the combined RGB-D ODOM+SLAM report with:

```bash
cuvslam_reporter \
    --test_config /path/to/datasets/converted/tum/tum-rgbd_slam.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-tum-reports \
    --odometry_mode rgbd \
    --async_sba false \
    --use_segments
```

`prepare_icl_nuim` runs `cuvslam_tools.dataset_preparation.icl_nuim.prepare`. It downloads the eight ICL-NUIM
TUM-compatible archives and converts them to portable EDEX with a reporter config. Each archive already carries its
`associations.txt` and `.gt.freiburg` poses, so the archive is the only download per sequence.

```bash
prepare_icl_nuim \
    --raw-dir /path/to/datasets/icl_nuim/raw \
    --output-dir /path/to/datasets/converted

# One trajectory, for a quick check
prepare_icl_nuim \
    --raw-dir /path/to/datasets/icl_nuim/raw \
    --output-dir /path/to/datasets/converted \
    --sequences traj2_frei_png
```

The prepared root is `/path/to/datasets/converted/icl_nuim`. It contains `icl_nuim-rgbd_slam.cfg` and
`dataset_metadata.json`. Every sequence contains `stereo.edex`, `frame_metadata.jsonl`, camera-aligned `gt.txt`,
colour PNGs under `00/`, and depth PNGs under `01/`. Depth is copied unchanged as 16-bit PNG with
`depth_scale_factor: 5000`, matching TUM.

ICL-NUIM is rendered rather than recorded, so two things differ from TUM. There are no timestamps: colour, depth,
and pose are matched by frame index and timestamps are synthesized at the published 30 Hz, which keeps the output
reproducible. And the published poses are expressed in the renderer's y-up world while the TUM-compatible PNGs are
stored top-down, so poses are reflected about the XZ plane to reach the camera frame. That reflection is not
cosmetic — skipping it scores 40.39% ATE against 1.51% with it.

The dataset page publishes the camera matrix with a negative `fy` and warns that projections break without it. That
applies to the POVRay-native format, whose image rows run bottom-up; these PNGs use the positive value.

Run the combined RGB-D ODOM+SLAM report with:

```bash
cuvslam_reporter \
    --test_config /path/to/datasets/converted/icl_nuim/icl_nuim-rgbd_slam.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-icl-reports \
    --odometry_mode rgbd \
    --async_sba false \
    --use_segments
```

`prepare_coda` runs `cuvslam_tools.dataset_preparation.coda.prepare`. CODa is license-gated, so nothing is
downloaded: register and accept the dataset license at the
[Texas Dataverse record](https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/BBOQMV), download the
per-sequence archives (`0.zip` … `22.zip`) you want, and place them in the raw directory. The command then converts
every archive it finds there.

```bash
prepare_coda \
    --raw-dir /path/to/datasets/coda/raw \
    --output-dir /path/to/datasets/converted

# One sequence, for a quick check
prepare_coda \
    --raw-dir /path/to/datasets/coda/raw \
    --output-dir /path/to/datasets/converted \
    --sequences 0
```

The prepared root is `/path/to/datasets/converted/coda`. It contains `dataset_metadata.json` and
`coda-vio_slam.cfg`, plus `coda-vio_gt.cfg`, `coda-slam_gt.cfg`, and `coda-vio_slam_gt.cfg` covering the sequences
that shipped poses. Every sequence contains `stereo.edex`, left images under `00/`, right images under `01/`, and
`gt.txt` when the archive carried poses.

Archives are read member by member instead of being extracted, because one sequence unpacks to tens of gigabytes and
only the rectified stereo pair, the calibration, and the poses are used. Images come from `2d_rect/cam0` and
`2d_rect/cam1`, so both cameras share the left camera's intrinsics and differ only by the baseline. The baseline is
derived from cam0's `disparity_matrix`, which agrees with the cam0-to-cam1 extrinsic, rather than cam1's
`projection_matrix`. Ground truth is taken from `poses/dense_global` where it exists and `poses/dense` otherwise
(sequences 8, 14, and 15), moved from the LiDAR frame onto cam0, and written relative to the first frame. Frames past
the end of the pose file are dropped from both the images and `gt.txt` so the two stay one-to-one.

Run the combined stereo ODOM+SLAM report with:

```bash
cuvslam_reporter \
    --test_config /path/to/datasets/converted/coda/coda-vio_slam_gt.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-coda-reports \
    --odometry_mode multicamera \
    --rectified_stereo_camera true \
    --async_sba false \
    --use_segments
```

All dataset preparation commands support `--force-download` and `--download-only`, default to `./datasets/<dataset>/raw`
and `./datasets/converted` relative to the current directory, and are implemented as
`cuvslam_tools.dataset_preparation.<dataset>.prepare`. Each module exposes a `prepare()` function that scripts can call
directly and a `main()` entry point behind the console command. `prepare_coda` accepts `--force-download` only for
parity with the other commands and ignores it: CODa archives are never downloaded or re-downloaded, so the command
always uses whatever is already in the raw directory.

## Tracking

Run one sequence:

```bash
cuvslam_tracker \
    --dataset /path/to/datasets/converted/kitti/00 \
    --config_path /path/to/datasets/converted/kitti/00/stereo.edex \
    --odometry_mode multicamera \
    --output_dir /tmp/cuvslam_tracker-smoke
```

For ROS bag input, first convert the bag to EDEX with `rosbag_extract_edex`.

## Reporting

Run one dataset config:

```bash
cuvslam_reporter \
    --test_config /path/to/datasets/converted/kitti/kitti-vio_gt.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-reports \
    --odometry_mode multicamera \
    --rectified_stereo_camera true \
    --async_sba false \
    --use_segments
```

The reporter requires `--test_config` to point to one config file. Relative paths are resolved from the current working directory; `--datasets_root` is only used to locate dataset folders referenced by that config.

## Validation

Run a multi-dataset validation config:

```bash
cuvslam_validator \
    --validation_config significant-prompt-run.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-validation \
    --odometry_mode multicamera \
    --use_segments
```

The validator runs reporter configs, writes a combined summary CSV, and fails when configured metric checks fail.

## ROS Bag To EDEX

Create a YAML config file. See `cuvslam_tools/bag2edex/configs/` for examples.

```yaml
camera_info_topics:
  - /camera/infra1/camera_info
  - /camera/infra2/camera_info

image_topics:
  - /camera/infra1/image_rect_raw
  - /camera/infra2/image_rect_raw

rig_frame: camera_link
imu_topic: /camera/imu
ros_distribution: humble
```

Extract a full EDEX dataset from a ROS 2 bag:

```bash
rosbag_extract_edex \
    --config cuvslam_tools/bag2edex/configs/realsense.yaml \
    --rosbag_path path/to/bag_folder \
    --output_path path/to/edex_folder
```

List available TF frames:

```bash
rosbag_extract_urdf \
    --rosbag_path path/to/bag_folder \
    --output_path /tmp/urdf_out \
    --ros_distribution humble
```

Extract images only:

```bash
rosbag_extract_images \
    --config cuvslam_tools/bag2edex/configs/realsense.yaml \
    --rosbag_path path/to/bag_folder \
    --output_path path/to/output_folder
```

ROS bag extraction can produce:

| File | Contents |
|---|---|
| `edex` | Camera intrinsics, extrinsics, IMU transform, and sequence frame list. |
| `images/<topic>/NNNNN.png` | Extracted camera frames. |
| `frame_metadata.jsonl` | Per-frame metadata with filenames and timestamps. |
| `imu.jsonl` | IMU samples, if `imu_topic` is set. |
| `robot.urdf` | URDF extracted from `/tf_static`. |

Available sensor configs:

| Config | Sensor |
|---|---|
| `cuvslam_tools/bag2edex/configs/realsense.yaml` | RealSense |
| `cuvslam_tools/bag2edex/configs/realsense_imu.yaml` | RealSense stereo plus IMU |
| `cuvslam_tools/bag2edex/configs/nova_hawk.yaml` | NVIDIA Nova plus HAWK stereo |
| `cuvslam_tools/bag2edex/configs/oak6.yaml` | OAK-6 camera |

RGBD tracking expects an EDEX file with `depth_sequence` and `depth_id` entries. The ROS bag extractor currently writes camera images and optional IMU data; it does not synthesize RGBD depth sequences from bags yet.

## Undistort EDEX Images

Undistort one image using the camera intrinsics in an EDEX file:

```bash
undistort_edex_images \
    /path/to/input.png \
    /path/to/stereo.edex \
    /path/to/output.png
```

Use a specific camera from a multi-camera EDEX file:

```bash
undistort_edex_images \
    /path/to/input.png \
    /path/to/stereo.edex \
    /path/to/output.png \
    --camera 1
```

Use a separate output EDEX camera model instead of the default pinhole output model:

```bash
undistort_edex_images \
    /path/to/input.png \
    /path/to/input.edex \
    /path/to/output.png \
    /path/to/output.edex \
    --camera 0
```

Batch-undistort loose images from a folder:

```bash
undistort_edex_images \
    /path/to/images \
    /path/to/stereo.edex \
    /path/to/undistorted_images \
    --batch \
    --pattern "*.png"
```

If `--pattern` is omitted in batch mode, the tool auto-detects common image formats: `jpg`, `jpeg`, `png`, `tga`, and `bmp`.
