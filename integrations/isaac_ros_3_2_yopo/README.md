# Isaac ROS 3.2 external-IMU wrapper patch

This directory carries an out-of-tree patch for the Isaac ROS Visual SLAM
wrapper used by the YOPO integration. It does not modify, build, or link the
cuVSLAM implementation in this repository. In particular, `core17` is not part
of this integration.

"Out-of-tree" describes ownership of the patch: the patch is maintained in
this fork instead of NVIDIA's repository. Applying it creates one reviewed
tracked modification in the NVIDIA checkout. That modification is the approved
exception to the otherwise read-only NVIDIA-source policy; no commit is made in
the NVIDIA repository.

The runtime remains NVIDIA's `libcuvslam.so` distributed with Isaac ROS 3.2.
Only the open-source ROS wrapper at the exact revision listed in
[`COMPATIBILITY.md`](COMPATIBILITY.md) is patched.

## Why this patch is required

Isaac ROS Visual SLAM 3.2 receives IMU and image timestamps in nanoseconds, but
the unpatched wrapper has two independent timestamp defects:

1. It computes each IMU message's timestamp (`imu_ts`) and then passes the
   current image timestamp (`latest_ts`) to
   `CUVSLAM_RegisterImuMeasurement`. When several IMU samples arrive between
   two image frames, they are registered with the same timestamp instead of
   their own strictly increasing timestamps.
2. `MessageStreamSequencer` receives nanosecond timestamps, while its image and
   IMU jitter thresholds are ROS parameters expressed in milliseconds. The
   unpatched constructor passes those thresholds without converting units.

The patch therefore passes `imu_ts` to the SDK and converts both sequencer
thresholds from milliseconds to nanoseconds. Both changes are required to
remove the two wrapper blockers, but they do not by themselves establish a
complete external-IMU data path.

The SDK contract was verified from the Isaac ROS 3.2 container header at:

```text
/opt/ros/humble/share/isaac_ros_nitros/cuvslam/include/cuvslam.h
```

That header requires the timestamp passed to
`CUVSLAM_RegisterImuMeasurement` to be in nanoseconds and to always increase.

## Obtain only this integration in the container

Run these commands inside the Isaac ROS development container:

```bash
cd /workspaces/isaac_ros-dev/src

git clone \
  --depth 1 \
  --branch u5-4/isaac-ros-3.2-yopo-adapter \
  --filter=blob:none \
  --sparse \
  https://github.com/u5-4/cuVSLAM.git \
  cuvslam-yopo-adapter

git -C cuvslam-yopo-adapter sparse-checkout set \
  integrations/isaac_ros_3_2_yopo
```

The three exported paths below are shell-local. Set them again after opening a new
container terminal:

```bash
export VSLAM_SOURCE=/workspaces/isaac_ros-dev/src/isaac_ros_visual_slam
export YOPO_ADAPTER=/workspaces/isaac_ros-dev/src/cuvslam-yopo-adapter/integrations/isaac_ros_3_2_yopo
export VSLAM_PATCH="$YOPO_ADAPTER/patches/isaac_ros_visual_slam_v3_2_15_imu_timestamp.patch"
```

## Verify and apply

The verification script requires a clean NVIDIA checkout. It applies the patch,
checks both corrections, verifies the installed NITROS SDK package/header, and
reverses the patch before it exits:

```bash
"$YOPO_ADAPTER/scripts/verify_visual_slam_patch.sh" "$VSLAM_SOURCE"
```

`--source-only` skips the installed SDK check and exists only for offline patch
development. Do not use it for Jetson acceptance.

Apply the patch for the actual build:

```bash
test -f /workspaces/isaac_ros-dev/install/setup.bash

# Stop the running Visual SLAM launch before changing or rebuilding its library.
git -C "$VSLAM_SOURCE" apply --check \
  "$VSLAM_PATCH"

git -C "$VSLAM_SOURCE" apply \
  "$VSLAM_PATCH"
```

Confirm that the expected patch is the only tracked source change, then rebuild
only the affected ROS package. This package-select build assumes the existing
Isaac ROS workspace overlay has already been built successfully:

```bash
cd /workspaces/isaac_ros-dev
set +u
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash

git -C "$VSLAM_SOURCE" diff --check
test "$(git -C "$VSLAM_SOURCE" status --short --untracked-files=no)" = \
  " M isaac_ros_visual_slam/src/impl/visual_slam_impl.cpp"
git -C "$VSLAM_SOURCE" apply --reverse --check "$VSLAM_PATCH"

colcon build \
  --packages-select isaac_ros_visual_slam \
  --cmake-args -DBUILD_TESTING=OFF

source /workspaces/isaac_ros-dev/install/setup.bash
test "$(ros2 pkg prefix isaac_ros_visual_slam)" = \
  /workspaces/isaac_ros-dev/install/isaac_ros_visual_slam
```

Restart the Visual SLAM launch after the build and verify that its package prefix
resolves to `/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam`.

The clone, patch, build, and `install/` output are in the host bind-mounted
workspace. They persist on the host, but they are not embedded in
`isaac_ros_dev-aarch64:latest`; a production image is a separate later task.

## Update this integration

Never pull a new adapter revision while its current patch is applied. Stop
Visual SLAM, reverse the current patch with the current patch file, pull, run
the verifier, apply the new patch, rebuild, and restart:

```bash
git -C "$VSLAM_SOURCE" apply --reverse --check "$VSLAM_PATCH"
git -C "$VSLAM_SOURCE" apply --reverse "$VSLAM_PATCH"

git -C /workspaces/isaac_ros-dev/src/cuvslam-yopo-adapter pull --ff-only

"$YOPO_ADAPTER/scripts/verify_visual_slam_patch.sh" "$VSLAM_SOURCE"
git -C "$VSLAM_SOURCE" apply --check "$VSLAM_PATCH"
git -C "$VSLAM_SOURCE" apply "$VSLAM_PATCH"

cd /workspaces/isaac_ros-dev
set +u
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
colcon build --packages-select isaac_ros_visual_slam --cmake-args -DBUILD_TESTING=OFF
source /workspaces/isaac_ros-dev/install/setup.bash
```

## Roll back

Stop Visual SLAM first. Reversing the source patch does not replace an already
installed or already loaded library, so rebuild and restart are mandatory:

```bash
git -C "$VSLAM_SOURCE" apply --reverse --check "$VSLAM_PATCH"
git -C "$VSLAM_SOURCE" apply --reverse "$VSLAM_PATCH"

cd /workspaces/isaac_ros-dev
set +u
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
colcon build --packages-select isaac_ros_visual_slam --cmake-args -DBUILD_TESTING=OFF
source /workspaces/isaac_ros-dev/install/setup.bash
```

Restart Visual SLAM only after this build succeeds.

## Validation boundary

A successful build proves only that the wrapper compiles and links. Before
external FCU IMU fusion is accepted, validate at runtime that:

- the FCU IMU topic uses the intended clock and strictly increasing timestamps;
- duplicate and non-monotonic FCU timestamps are rejected and counted by the
  future adapter health gate instead of being forwarded to this wrapper;
- the Visual SLAM node has exactly the intended FCU `sensor_msgs/msg/Imu`
  subscription/remap and is not still consuming `/camera/imu` from the D435i;
- the calibrated constant camera/FCU clock offset is applied with its documented
  sign convention before SDK registration;
- all IMU samples use the expected axes and the calibrated transform direction
  is converted into the documented ROS TF chain; for the selected rectified
  stereo input, the Kalibr result is treated as `T_Crect0_I`, not as a raw
  optical-camera transform;
- noise density and random-walk values come from the selected FCU IMU, with the
  units expected by the Isaac ROS parameters, rather than D435i/default values;
- Visual SLAM logs do not report failed IMU registrations;
- tracking and timestamp checks pass on a new recording made after this patch.

The earlier 10-minute visual tracking test used the D435i onboard IMU and
predates this patch. It does not prove that FCU IMU measurements are routed,
calibrated, or registered with correct timestamps.

The included verifier is a source and SDK-contract check, not a behavioral
fusion test. Production acceptance still requires the rebuilt Jetson package
and a new FCU-IMU rosbag/runtime validation.
