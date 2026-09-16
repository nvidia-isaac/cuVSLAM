# Test util to track, save map, localize using cuvslam API

## Basic Usage

### Odometry Only

To track and save odometry poses:
```bash
./bin/cuvslam_api_launcher -edex=<edex file> -print_odom_poses=<path>
```

### Odometry + SLAM

To enable SLAM and save both odometry and SLAM poses:
```bash
./bin/cuvslam_api_launcher -edex=<edex file> -print_odom_poses=<path> -print_slam_poses=<path> --cfg_enable_slam --cfg_enable_export
```

**Note:** SLAM requires both flags:
- `--cfg_enable_slam` - enables SLAM tracking
- `--cfg_enable_export` - enables observation/landmark export (required for SLAM to get odometry state)

### Save Map

To save a SLAM map:
```bash
./bin/cuvslam_api_launcher -edex=<edex file> -output_map=<map dir> --cfg_enable_slam --cfg_enable_export
```

To localize in map:
```bash
./bin/cuvslam_api_launcher -edex=<edex file> -loc_input_map=<map dir> -loc_input_hints=<hint file> -print_loc_poses=<path>
```

Additional flags with default values:
`-loc_start_frame=0 -loc_retries=0 -loc_hint_noise=0.0 -localize_forever=false -localize_wait=false -loc_random_rot=false -print_nan_on_failure=false`

Hint file rows format: `timestamp x y z [optional quaternion]`
Float timestamps in seconds and int timestamps in ns are supported. Hints must be sorted by timestamps.
To localize, the util will use the latest hint not later than current frame.

## Development Flags

The `--expert_sba_*` gflags are development-only persistent parameters applied once after tracker construction with
`Odometry::ApplyPersistentInternalParameters()`. Normal applications should use library defaults.

## Reporter Integration

`--report_output=<path>` writes JSONL records for every attempted frame, loop closures, final SLAM poses, and an
end-of-run summary. This is a machine interface for `cuvslam_reporter`; human-readable pose outputs remain unchanged.
Reporter mode also uses `--ignore_tracking_errors=true` so losses are recorded and evaluation can continue.

Run this integration through the Python package instead of constructing the machine flags manually:

```bash
cuvslam_reporter \
    --tracker_backend api_launcher \
    --api_launcher_path ./bin/cuvslam_api_launcher \
    ... \
    -- \
    --expert_sba_num_frames=9
```

For RGB-D EDEX input, `--depth_scale_factor` is the divisor used while loading integer depth images. Loaded depths
reach the public API as float metres, so `--cfg_depth_scale_factor=1` avoids applying the scale twice.

# Run tracker on EuRoC MAV Dataset (OBSOLETE)

```shell script
download.sh # Download .bag files

# Install requirements
sudo apt install python-cv-bridge python-opencv python-rosbag

# You should set CUVSLAM_DATASETS environment variable and
# create $CUVSLAM_DATASETS/euroc folder, then run export:
python3 extract_bag.py

# Run tracker
source cuvslam_vars.sh
./bin/cuvslam_api_launcher -edex=$CUVSLAM_DATASETS/euroc/MH_01_easy/stereo.edex

```
