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

import argparse
import os

import cuvslam as vslam
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dataset_utils import (
    DEPTH_SCALE_MM,
    build_imu_calibration,
    load_depth,
    load_rgb,
    prepare_tartan_frames,
    read_stereo_edex,
)


def color_from_id(identifier):
    """Generate pseudo-random color from integer identifier for visualization."""
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


parser = argparse.ArgumentParser(
    description='Multisensor (multi RGB-D + IMU) tracking on TartanGround.')
parser.add_argument('--no-imu', action='store_true',
                    help='Disable the IMU: run Multisensor mode with only the two RGB-D cameras.')
args = parser.parse_args()
use_imu = not args.no_imu

CAMERA_LIST = ['lcam_front', 'lcam_back']
data_path = 'dataset/tartan_ground/OldTownFall/Data_anymal/P2000'


# setup rerun visualizer
rr.init('tartan_ground_multisensor', strict=True, spawn=True)
rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Horizontal(
        column_shares=[0.5, 0.5],
        contents=[
            rrb.Vertical(contents=[
                rrb.Horizontal(contents=[
                    rrb.Spatial2DView(origin='car/cam0/image', name='lcam_front'),
                    rrb.Spatial2DView(origin='car/cam1/image', name='lcam_back'),
                ]),
                rrb.Horizontal(contents=[
                    rrb.Spatial2DView(origin='car/cam0/depth', name='lcam_front depth'),
                    rrb.Spatial2DView(origin='car/cam1/depth', name='lcam_back depth'),
                ]),
                rrb.Horizontal(contents=[
                    rrb.TimeSeriesView(
                        name="IMU Acceleration",
                        origin="world/imu/accel",
                        overrides={
                            "world/imu/accel/x": rr.SeriesLines(colors=[255, 0, 0]),
                            "world/imu/accel/y": rr.SeriesLines(colors=[0, 255, 0]),
                            "world/imu/accel/z": rr.SeriesLines(colors=[0, 0, 255]),
                        },
                    ),
                    rrb.TimeSeriesView(
                        name="IMU Angular Velocity",
                        origin="world/imu/gyro",
                        overrides={
                            "world/imu/gyro/x": rr.SeriesLines(colors=[255, 0, 0]),
                            "world/imu/gyro/y": rr.SeriesLines(colors=[0, 255, 0]),
                            "world/imu/gyro/z": rr.SeriesLines(colors=[0, 0, 255]),
                        },
                    ),
                ]),
            ]),
            rrb.Spatial3DView(
                name="3D",
                defaults=[rr.Pinhole.from_fields(image_plane_distance=0.5)]
            ),
        ],
    ),
), make_active=True)

# cuvslam uses right-hand system with X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Build rig: 2 RGB-D cameras (+ 1 IMU unless --no-imu was passed)
cameras = read_stereo_edex('tartan_ground.edex')
rig = vslam.Rig()
rig.cameras = cameras
if use_imu:
    rig.imus = [build_imu_calibration()]

# Configure Multisensor odometry: both cameras provide depth, depth in meters
multisensor_settings = vslam.Odometry.MultisensorSettings(
    depth_camera_ids=[0, 1],
    depth_scale_factor=DEPTH_SCALE_MM,
    enable_depth_stereo_tracking=True,
)
odom_cfg = vslam.Odometry.Config(
    odometry_mode=vslam.Odometry.OdometryMode.Multisensor,
    multisensor_settings=multisensor_settings,
    enable_final_landmarks_export=True,
    rectified_stereo_camera=False,
    async_sba=False,
)

tracker = vslam.Tracker(rig, odom_cfg)
print(f"cuVSLAM Tracker initialized with odometry mode: {cfg.odometry_mode}"
      f"{' (IMU disabled)' if not use_imu else ''}")

frames_metadata = prepare_tartan_frames(data_path, CAMERA_LIST, include_imu=use_imu)
print(f"Loaded {sum(1 for r in frames_metadata if r['type'] == 'rgbd')} RGB-D frames and "
      f"{sum(1 for r in frames_metadata if r['type'] == 'imu')} IMU samples")

trajectory = []
frame_id = 0
last_camera_timestamp = None
imu_count_since_last_camera = 0
last_accel = None
last_gyro = None

for record in frames_metadata:
    timestamp = record['timestamp']

    if record['type'] == 'imu':
        last_accel = record['accel']
        last_gyro = record['gyro']
        imu_measurement = vslam.ImuMeasurement()
        imu_measurement.timestamp_ns = int(timestamp)
        imu_measurement.linear_accelerations = np.asarray(last_accel)
        imu_measurement.angular_velocities = np.asarray(last_gyro)
        tracker.register_imu_measurement(0, imu_measurement)
        imu_count_since_last_camera += 1
        continue

    if use_imu and last_camera_timestamp is not None and imu_count_since_last_camera == 0:
        print(f"Warning: No IMU measurements between timestamps "
              f"{last_camera_timestamp} and {timestamp}")
    last_camera_timestamp = timestamp
    imu_count_since_last_camera = 0

    try:
        images = [load_rgb(p) for p in record['image_paths']]
        depths = [load_depth(p) for p in record['depth_paths']]
    except FileNotFoundError as e:
        print(f"Error: missing data for frame {frame_id}: {e}")
        frame_id += 1
        continue

    odom_pose_estimate, _ = tracker.track(timestamp, images, depths=depths)
    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: failed to track frame {frame_id}")
        frame_id += 1
        continue

    odom_pose = odom_pose_estimate.world_from_rig.pose
    observations = [tracker.odometry.get_last_observations(i) for i in range(len(CAMERA_LIST))]
    landmarks = tracker.odometry.get_last_landmarks()
    final_landmarks = tracker.odometry.get_final_landmarks()
    gravity = tracker.odometry.get_last_gravity() if use_imu else None

    observations_uv = [[[o.u, o.v] for o in obs] for obs in observations]
    observations_colors = [[color_from_id(o.id) for o in obs] for obs in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]
    trajectory.append(odom_pose.translation)

    rr.set_time('frame', sequence=frame_id)
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('final_landmarks', rr.Points3D(list(final_landmarks.values()), radii=0.01))
    rr.log('car', rr.Transform3D(translation=odom_pose.translation, quaternion=odom_pose.rotation))
    rr.log('car/body', rr.Boxes3D(centers=[0, 0.3 / 2, 0], sizes=[[0.35, 0.3, 0.66]]))
    rr.log('car/landmarks_center', rr.Points3D(landmark_xyz, radii=0.02, colors=landmarks_colors))

    for i in range(len(cameras)):
        rr.log(f'car/cam{i}/image', rr.Image(images[i]).compress(jpeg_quality=80))
        rr.log(f'car/cam{i}/image/observations',
               rr.Points2D(observations_uv[i], radii=5, colors=observations_colors[i]))
        rr.log(f'car/cam{i}/depth', rr.DepthImage(depths[i], meter=DEPTH_SCALE_MM))
        rr.log(f'car/cam{i}',
               rr.Transform3D(translation=cameras[i].rig_from_camera.translation,
                              rotation=rr.Quaternion(xyzw=cameras[i].rig_from_camera.rotation),
                              relation=rr.TransformRelation.ParentFromChild))
        rr.log(f'car/cam{i}',
               rr.Pinhole(image_plane_distance=1.,
                          image_from_camera=np.array([[cameras[i].focal[0], 0, cameras[i].principal[0]],
                                                      [0, cameras[i].focal[1], cameras[i].principal[1]],
                                                      [0, 0, 1]]),
                          width=cameras[i].size[0], height=cameras[i].size[1]))

    if last_accel is not None:
        rr.log("world/imu/accel/x", rr.Scalars(last_accel[0]))
        rr.log("world/imu/accel/y", rr.Scalars(last_accel[1]))
        rr.log("world/imu/accel/z", rr.Scalars(last_accel[2]))
        rr.log("world/imu/gyro/x", rr.Scalars(last_gyro[0]))
        rr.log("world/imu/gyro/y", rr.Scalars(last_gyro[1]))
        rr.log("world/imu/gyro/z", rr.Scalars(last_gyro[2]))

    if gravity is not None:
        rr.log('car/gravity',
               rr.Arrows3D(vectors=[gravity], colors=[[255, 0, 0]], radii=0.015))

    frame_id += 1
