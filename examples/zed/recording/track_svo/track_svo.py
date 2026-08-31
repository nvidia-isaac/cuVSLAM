# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import cuvslam
import pyzed.sl as sl
import rerun as rr
import rerun.blueprint as rrb

svo_filename = "recording.svo2"
rich_vis = True  # Draw observation and landmarks in addition to the trajectory. May slow down visualization.
throttle_vis = True  # Do not visualize all frames, just one frame per second. Makes sense for long recordings.
enable_slam = True


def _zed_sdk_version_at_least(major: int, minor: int) -> bool:
    try:
        version = sl.Camera.get_sdk_version()
        version_parts = version.split(".")
        sdk_version = int(version_parts[0]), int(version_parts[1])
    except (AttributeError, IndexError, ValueError):
        return False

    return sdk_version >= (major, minor)


def configure_zed_timestamp_clock() -> None:
    if (
        _zed_sdk_version_at_least(5, 3)
        and hasattr(sl.Camera, "set_timestamp_clock")
        and hasattr(sl, "TIMESTAMP_CLOCK")
    ):
        sl.Camera.set_timestamp_clock(sl.TIMESTAMP_CLOCK.MONOTONIC_CLOCK)


def init_zed():
    configure_zed_timestamp_clock()

    zed = sl.Camera()

    # disable depth to speedup processing to 1.5x
    init = sl.InitParameters(coordinate_units=sl.UNIT.METER, depth_mode=sl.DEPTH_MODE.NONE)
    init.set_from_svo_file(svo_filename)

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file '{svo_filename}': {err}")
    return zed


def create_cuvslam_camera_from_zed_params(zed_params: sl.CameraParameters):
    cu_camera = cuvslam.Camera()
    zed_resolution = zed_params.image_size
    cu_camera.size = [zed_resolution.width, zed_resolution.height]
    cu_camera.principal = [zed_params.cx, zed_params.cy]
    cu_camera.focal = [zed_params.fx, zed_params.fy]
    return cu_camera


def init_cuvslam_from_zed(zed_calibration: sl.CalibrationParameters):
    cu_cameras = [create_cuvslam_camera_from_zed_params(zed_calibration.left_cam),
                  create_cuvslam_camera_from_zed_params(zed_calibration.right_cam)]
    cu_cameras[1].rig_from_camera.translation[0] = zed_calibration.get_camera_baseline()
    cu_odom_cfg = cuvslam.Odometry.Config(async_sba=False, enable_final_landmarks_export=rich_vis,
                                                 rectified_stereo_camera=True,
                                                 multicam_mode=cuvslam.Odometry.MulticameraMode.Performance)
    if enable_slam:
        cu_slam_cfg = cuvslam.Slam.Config(enable_reading_internals=True, map_cell_size=2,
                                                 sync_mode=True, max_map_size=10000)
    else:
        cu_slam_cfg = None
    return cuvslam.Tracker(cuvslam.Rig(cu_cameras), cu_odom_cfg, cu_slam_cfg)


# Generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier):
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


def init_rerun():
    default_blueprint = rrb.Blueprint(rrb.TimePanel(state="collapsed"),
                                      rrb.Vertical(row_shares=[0.6, 0.4],
                                                   contents=[rrb.Spatial3DView(),
                                                             rrb.Spatial2DView(origin='humanoid/cam0')]))
    rr.init('forest', strict=True, spawn=True, default_blueprint=default_blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


def rerun_visualize(tracker, frame, odom_pose, trajectory, image):
    if throttle_vis and frame % 100 != 0:
        return
    rr.set_time('frame', sequence=frame)
    rr.log('trajectory', rr.LineStrips3D(trajectory))  # add static=True for speedup vis

    if not rich_vis:
        return
    observations = tracker.odometry.get_last_observations(0)  # get observation from left camera
    landmarks = tracker.odometry.get_last_landmarks()
    final_landmarks = tracker.odometry.get_final_landmarks()

    observations_uv = [[o.u, o.v] for o in observations]
    observations_colors = [color_from_id(o.id) for o in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]

    rr.log('final_odometry_landmarks', rr.Points3D(list(final_landmarks.values()), radii=0.03))
    rr.log('humanoid', rr.Transform3D(translation=odom_pose.translation, quaternion=odom_pose.rotation))
    rr.log('humanoid/head', rr.Ellipsoids3D(centers=[0, 0, 0], half_sizes=[0.09, 0.09, 0.09]), static=True)
    rr.log('humanoid/landmarks_center', rr.Points3D(landmark_xyz, radii=0.1, colors=landmarks_colors))
    rr.log('humanoid/landmarks_lines', rr.Arrows3D(vectors=landmark_xyz, radii=0.005, colors=landmarks_colors))
    rr.log('humanoid/cam0', rr.Pinhole(image_plane_distance=1, focal_length=100,
                                       width=image.get_width(), height=image.get_height()))
    rr.log('humanoid/cam0/image', rr.Image(image).compress(jpeg_quality=80))
    rr.log('humanoid/cam0/observations', rr.Points2D(observations_uv, radii=3, colors=observations_colors))


def main():
    zed = init_zed()
    view_left = sl.VIEW.LEFT_GRAY
    view_right = sl.VIEW.RIGHT_GRAY

    tracker = init_cuvslam_from_zed(zed.get_camera_information().camera_configuration.calibration_parameters)
    init_rerun()

    image_left = sl.Mat()
    image_right = sl.Mat()
    print("Starting SVO playback...")

    runtime = sl.RuntimeParameters()
    trajectory = []
    frame = 0
    while zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_left, view_left)
        zed.retrieve_image(image_right, view_right)
        if frame % 100 == 0:
            print('fps=', zed.get_current_fps(), 'dropped=', zed.get_frame_dropped_count())

        frame_left = image_left.get_data()  # to numpy array
        frame_right = image_right.get_data()

        images = [frame_left, frame_right]

        timestamp_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        odom_pose_estimate, slam_pose_estimate = tracker.track(timestamp_ns, images)
        frame += 1

        if odom_pose_estimate.world_from_rig is None:
            print(f"Warning: Failed to track frame {frame}")
            continue

        pose = slam_pose_estimate if enable_slam else odom_pose_estimate.world_from_rig.pose

        if not throttle_vis or frame % 100 == 0:
            trajectory.append(pose.translation)
        rerun_visualize(tracker, frame, pose, trajectory, image_left)

    print("End of SVO file reached.")
    zed.close()


if __name__ == "__main__":
    main()
