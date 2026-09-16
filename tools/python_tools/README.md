# cuvslam-tools

`cuvslam-tools` is the installable Python tools package for cuVSLAM workflows.

It provides command-line tools for:

- Preparing public datasets, including KITTI, EuRoC, TartanGround, TUM RGB-D, and CODa.
- Converting ROS 2 bags to EDEX inputs.
- Running one tracking sequence.
- Running dataset reports.
- Evaluating visual place recognition across two passes of a route.
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

`cuvslam_tracker`, `cuvslam_reporter`, `cuvslam_vpr_reporter`, and `cuvslam_validator` require the `cuvslam` Python binding in the same environment. Dataset preparation, ROS bag conversion, and undistortion should stay usable without importing `cuvslam` when their workflows do not need it.

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
| `prepare_m3ed_spot` | Convert the 19 M3ED SPOT stereo sequences, or an explicit subset, to portable EDEX and reporter configs, reading the source HDF5 straight from the public bucket. |
| `cuvslam_tracker` | Run one EDEX sequence or supported video input through cuVSLAM. |
| `cuvslam_reporter` | Run one dataset config and generate report outputs. |
| `cuvslam_vpr_reporter` | Build a place recognition map from one sequence, recognize places in another, and report the result. |
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
prepare_m3ed_spot --help
cuvslam_tracker --help
cuvslam_reporter --help
cuvslam_vpr_reporter --help
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
color PNGs under `00/`, and depth PNGs under `01/`.

Colour and depth frames are associated within 1 ms, which pairs the two views of a single Kinect capture and
rejects pairs stitched across neighboring captures. Ground truth is interpolated onto the associated frame times
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
color PNGs under `00/`, and depth PNGs under `01/`. Depth is copied unchanged as 16-bit PNG with
`depth_scale_factor: 5000`, matching TUM.

ICL-NUIM is rendered rather than recorded, so two things differ from TUM. There are no timestamps: color, depth,
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

`prepare_m3ed_spot` runs `cuvslam_tools.dataset_preparation.m3ed_spot.prepare`. It converts the 19 M3ED SPOT stereo
sequences, reading `/ovc/{left,right}/data` out of the published `_data.h5` and the FasterLIO poses out of
`_pose_gt.h5`. The retired reporter config evaluated 16 of them; `hard`, `srt_green_loop` and `stairwell` were
converted but never enabled.

```bash
prepare_m3ed_spot --output-dir /path/to/datasets/converted

# One sequence, or a short prefix of one, for a quick check
prepare_m3ed_spot \
    --output-dir /path/to/datasets/converted \
    --sequences skatepark_2 \
    --frame-limit 120
```

Unlike the other converters this one has no download step. The stereo images exist only inside `_data.h5`, which runs
25-42 GB per sequence because it also carries the event, LiDAR and IMU streams; the compressed image chunks are about
6% of it. The source is therefore read over HTTP range requests, so a sequence transfers roughly 3 GB instead of
downloading 25-42 GB, and nothing is staged on disk. `--force-download` and `--download-only` are rejected for the
same reason. Pass `--raw-dir` to read already-downloaded files instead, laid out as
`<raw-dir>/<published-sequence>/<published-sequence>_{data,pose_gt}.h5`. A `--raw-dir` holding none of those files
means the bucket, since provisioning passes every dataset a directory to download into and this one has nothing to
download; a `--raw-dir` that is not a directory is rejected rather than quietly streaming the whole corpus.

Reading one object takes tens of minutes and M3ED does republish files, so every range read sends the object's ETag as
`If-Range`. A file replaced mid-conversion answers with the whole object instead of the range, which is refused before
the body is read rather than spliced into the output. A source that serves no strong ETag is rejected outright, since
nothing would detect the substitution.

The prepared root is `/path/to/datasets/converted/m3ed_spot`. It contains `dataset_metadata.json` and, as for KITTI
and EuRoC, three reporter configs: `m3ed_spot-vo.cfg`, `m3ed_spot-slam.cfg` and `m3ed_spot-vo_slam.cfg`. Every
sequence contains `stereo.edex`, `frame_metadata.jsonl`, camera-aligned `gt.txt`, and
mono8 PNGs under `00/` (OVC left) and `01/` (OVC right). Each sequence also holds a dot-prefixed
`.conversion_state.json`, which is the converter's own record rather than dataset content: `--skip-existing` reads it
to tell a finished sequence from one truncated by `--frame-limit` or cut short by an interrupted run, and a resumed run
copies its metadata into `dataset_metadata.json` so skipped sequences are described as fully as converted ones.
Calibration is read per sequence from the source, and the
radtan coefficients map onto cuVSLAM's `polynomial` model, whose first four parameters are the same OpenCV values.

Ground truth needs one correction that is easy to miss: the published poses describe the left *event* camera, not the
OVC camera being evaluated, so `ovc/left/calib/T_to_prophesee_left` is applied before the poses are made relative. The
two frames are about 70 mm apart, which does not cancel out under rotation — on `skatepark_2` it changes the measured
path by 4.9 mm over 15.3 m.

Checked against the retired output on `skatepark_2`, the images are byte-identical and the trajectory agrees to 490 µm
rms per frame. Four things differ on purpose: calibration is per sequence rather than one hardcoded set; the EDEX
references `frame_metadata.jsonl` instead of declaring `fps: 30` for a 25 Hz sensor; each frame's pose is the pose at
its own timestamp, where the retired output paired poses with images five frames apart; and images are mono8 rather
than the same pixel replicated across three RGB channels, which makes the output 2.3 times smaller.

Run the combined stereo ODOM+SLAM report for M3ED with:

```bash
cuvslam_reporter \
    --test_config /path/to/datasets/converted/m3ed_spot/m3ed_spot-vo_slam.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-m3ed-reports \
    --odometry_mode multicamera \
    --rectified_stereo_camera false \
    --async_sba false \
    --use_segments
```

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

All dataset preparation commands default to `./datasets/<dataset>/raw` and `./datasets/converted` relative to the
current directory, and are implemented as `cuvslam_tools.dataset_preparation.<dataset>.prepare`. Each module exposes a
`prepare()` function that scripts can call directly and a `main()` entry point behind the console command. Most accept
`--force-download` and `--download-only`; `prepare_coda` ignores `--force-download` because CODa archives are never
downloaded, and `prepare_m3ed_spot` streams its source rather than downloading it.

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

### Ground truth

Every sequence in a reporter config names its reference explicitly, and a sequence that names one it cannot read fails the run:

- `"gt_file_path": "gt.txt"` — KITTI-format poses, absolute or relative to the sequence folder. A missing file is an error.
- `"gt_from_shuttle": true` — the backward pass of a shuttle replay is scored against the forward pass. This measures repeatability, not accuracy: an error that repeats in both directions cancels out. Requires `"repeat_type": "Shuttle"` with `"sequence_num_repeats"` of at least 1, and cannot be combined with `gt_file_path`.
- Neither — the sequence runs without accuracy metrics and its ATE/ARE columns stay blank.

`cuvslam_tracker` takes the same two choices as `--gt_path` and `--gt_from_shuttle`.

## Visual Place Recognition

`cuvslam_vpr_reporter` answers "would this map let a robot recognize where it is?". It takes two EDEX sequences of one
route: a map sequence, and a query sequence driven through the same place later. It tracks the map sequence, offers
every frame to the place recognition map, saves the map, then replays the query sequence against a fresh tracker that
loaded it, and scores every recognition against KITTI-format ground truth. The place recognition map is part of the
SLAM map, so saving it writes the landmark database next to it; those folders are large and are deleted after each
combination unless `--keep_vpr_maps` asks for them.

Two sequences, straight from flags:

```bash
cuvslam_vpr_reporter \
    --map_sequence /path/to/datasets/converted/coda/0 \
    --query_sequence /path/to/datasets/converted/coda/5 \
    --pair_title "CODa-00 to CODa-05" \
    --vpr_modes Simple \
    --output_root /tmp/cuvslam-vpr-reports \
    --pdf
```

Several pairs and backends at once, from a config:

```bash
cuvslam_vpr_reporter \
    --test_config /path/to/configs/coda-vpr.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-vpr-reports \
    --pdf
```

```json
{
  "version": "0.1",
  "success_radius_m": 10.0,
  "vpr_modes": ["Simple", "DBoW2", "AnyLoc"],
  "pairs": [
    {
      "title": "CODa-00 to CODa-05",
      "map_sequence": "coda/0",
      "query_sequence": "coda/5",
      "edex_file": "stereo.edex",
      "map_gt": "gt.txt",
      "query_gt": "gt.txt"
    }
  ]
}
```

`pairs` is the only required key; each pair needs `title`, `map_sequence`, and `query_sequence`, and the other three
keys default to the values above. Sequence paths are resolved against `--datasets_root`, absolute paths are used as
is, and ground-truth paths are resolved against their own sequence folder. `success_radius_m` and `vpr_modes` set the
defaults for the run; `--success_radius` and `--vpr_modes` on the command line override them.

Every combination of a backend and a pair gets one row of the summary table, one section with its plots, and one entry
in `stats/all_vpr_stats.json`. The plot is a bird's eye view: the map sequence's ground truth as a thin grey line,
and the query sequence's ground truth drawn on top of it as a ribbon colored by how each frame was classified, dark
green for recognized, amber for a false positive, red for unrecognized. Under it is a strip of 20 frames of the query
sequence, evenly spaced over the frames that were queried and bordered green where the frame was recognized correctly
and red where the robot did not know where it was, false positives included; each frame carries its index.

Beyond the flags above the command accepts every `cuvslam_tracker` flag; `--frame_limit` cuts both sequences short for
a quick check, `--query_stride` queries every Nth frame, `--vpr_score_threshold` raises the similarity a match must
reach, and `--keep_vpr_maps` leaves the temporary map folders on disk. `--query_tracking false` skips odometry during
the query pass, which times and scores place recognition on its own; the default tracks the query sequence, which is
what a robot localizing against a saved map does.

### Metrics

Ground truth for both sequences is a KITTI-format `gt.txt` of absolute poses. Both are read in the frame they are
written in and compared directly, with no alignment, so the two sequences must already share a world frame. This
holds for the prepared CODa sequences, where every CODa-05 pose is within 6.8 m of the CODa-00 trajectory (median
2.4 m), and trivially for a sequence evaluated against itself.

A recognition names the mapped frame it matched by timestamp, which is looked up in the map sequence's own
timestamp-to-frame index; a timestamp that is not in it falls back to the nearest one and is counted. The distance
between the map frame's ground-truth position and the query frame's ground-truth position is the localization error
of that query, and it decides the classification:

| Column | Meaning |
|---|---|
| Success [%] | query frames whose matched map frame is within `--success_radius` of the query frame |
| False positive [%] | query frames matched to a mapped place further away than that |
| Unrecognized [%] | query frames the backend reported no match for |
| Precision | successes over all map matches |
| Map build [ms/frame] | wall time of the map pass, `track` plus `add_frame_to_vpr_map`, over the mapped frames |
| Add to map [ms/frame] | wall time of one `add_frame_to_vpr_map` call, over the frames of the map pass |
| Search [ms/frame] | wall time of one `recognize_place_by_frame` call, over the queried frames |
| Skipped | queried frames with no ground-truth row, which are scored as nothing |

The localization error over the successes and the similarity the backend reported for its matches are still measured,
as `median_error_m`, `mean_error_m` and `mean_score` in `stats/all_vpr_stats.json`; they are left out of the table
because the classification rates above already say what they are used to judge. The summary row pools each mean by
the frames it was averaged over: the map-pass costs by the mapped frames, the search cost by the queried ones.

A place recognition map loaded from disk is read only, so the query session cannot add its own keyframes to it and
every answer comes from the map. A query frame that matched a keyframe of the query session instead would carry a
query timestamp, which means nothing in the map sequence and cannot be scored; such self matches are therefore
impossible. They are still counted, in `stats/all_vpr_stats.json` as `n_self_matches`, because a non-zero count means
the read-only guarantee broke and every rate of that row is measuring the wrong thing. The reporter then prints a
warning and the report grows a warning row naming the count.

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
