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
- `--cfg_enable_slam` - constructs the SLAM instance
- `--cfg_enable_export` - enables observation/landmark export (required for SLAM to get odometry state);
  shorthand for `-Podometry.enable_observations_export=true -Podometry.enable_landmarks_export=true`

These two are the only flags left that touch the tracker: everything else in `Odometry::Config` and
`Slam::Config` is a parameter, set through `--params` or `-P` as described below.

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

## Parameters

Configuration and solver parameters come from one flat file, passed with `--params`:

```bash
./bin/cuvslam_api_launcher --params=api_params.txt --edex=<edex file>
```

One `key: value` per line, `#` starts a comment, and every entry is optional. See
`api_params.txt` for a commented example, and run `--list_params` to print every name with its
type, default and description.

Single values can be overridden with a repeatable `-Pkey=value`, which wins over the file:

```bash
./bin/cuvslam_api_launcher --params=api_params.txt -Psba.num_sba_iterations=9 \
    -Pnum_desired_tracks=300 --edex=<edex file>
```

Any unambiguous suffix of a name works, which is why `-Pnum_desired_tracks` reaches
`sof.num_desired_tracks`. An ambiguous abbreviation is rejected with the candidates listed, and an
unknown name is an error rather than a warning, so a typo fails the run instead of silently doing
nothing.

Names fall into two groups, applied at different times:

- `odometry.*` and `slam.*` configure `Odometry::Config` and `Slam::Config`, so they are applied
  before the tracker is built.
- `sof.*`, `kf.*`, `sba.*`, `vo_pnp.*`, `icp.*` and, with an IMU, `sm.*`, `imu_pnp.*` and
  `inertial_stereo_pnp.*` are solver parameters, applied to the tracker once it exists. Which of
  them exist depends on the odometry mode: a tracker with no IMU does not expose the inertial ones
  at all, and naming one is an error.

Solver parameters expose low-level behaviour, are not covered by API stability guarantees, and may
change or disappear in any release. Normal applications should leave them alone. The same values
are reachable from C++ with `Odometry::SetParameter()` and from Python with `set_parameter()` and
`set_parameters()`; `Odometry::GetParameters()` reports every parameter with its current value and
where that value came from. The library itself reads no files: the `--params` format lives in
`libs/utils/param_file.h`, shared by the tools and not shipped, and Python callers hand
`set_parameters()` a dict so they can use a real YAML parser.

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
