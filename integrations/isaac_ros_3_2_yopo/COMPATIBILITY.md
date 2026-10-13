# Compatibility contract

This adapter is intentionally pinned. Do not apply it to another Isaac ROS
Visual SLAM revision without reviewing the upstream wrapper again.

| Component | Required value |
| --- | --- |
| NVIDIA repository | `NVIDIA-ISAAC-ROS/isaac_ros_visual_slam` |
| NVIDIA tag | `v3.2-15` |
| NVIDIA commit | `e31f4cc1d41a329a01946e5fe63669f8b15da677` |
| ROS distribution | ROS 2 Humble |
| Patched package | `isaac_ros_visual_slam` |
| NITROS package | `ros-humble-isaac-ros-nitros=3.2.5-0jammy` |
| SDK interface checked | `CUVSLAM_RegisterImuMeasurement` from the Isaac ROS 3.2 NITROS cuVSLAM header |
| Patched binary marker | `ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1` |
| Runtime calibration | `d435i_243622070369_factory_rectified_px4_imu_20260720` |

NVIDIA tags `v3.2-14` and `v3.2-15` may both describe the same commit in a
shallow checkout. This integration uses the `v3.2-15` release label and treats
the full commit ID above as the authoritative compatibility check.

## Ownership boundary

- The patch changes only
  `isaac_ros_visual_slam/src/impl/visual_slam_impl.cpp` in NVIDIA's wrapper.
- NVIDIA's packaged `libcuvslam.so` remains the runtime implementation.
- No library under this fork's `libs/` tree is built or linked.
- `core17` is not built, linked, copied, or redistributed by this adapter.
- The NVIDIA SDK binary and header are not vendored here.
- The verifier checks the installed Debian package version, its file-integrity
  metadata, and the required timestamp contract in `cuvslam.h` before applying
  the source patch. The patch also embeds the marker above in
  `libvisual_slam_node.so`; runtime bringup refuses an installed overlay that
  lacks it. This proves the current binary was rebuilt from the marked wrapper,
  but does not claim behavioral IMU-fusion coverage.

## Revalidation triggers

Repeat the source audit and runtime validation before using the adapter with:

- a different Isaac ROS Visual SLAM commit;
- a different NITROS/cuVSLAM SDK package;
- a different timestamp clock or FCU bridge;
- a different camera/FCU-IMU frame convention;
- a new camera-to-FCU-IMU calibration or time-offset estimate.
