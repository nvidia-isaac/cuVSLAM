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

## Expert SBA Parameters

The `--expert_sba_*` flags are development-only persistent parameters, applied once after tracker construction with
`Odometry::ApplyPersistentInternalParameters()`. Their defaults mirror the library defaults, so leaving them alone
changes nothing. Normal applications should use the default internal parameters.

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
