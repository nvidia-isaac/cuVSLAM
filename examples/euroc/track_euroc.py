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

import os

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import cuvslam
from dataset_utils import prepare_frame_metadata_euroc, get_rig, load_frame

# Set up dataset path
sequence_path = os.path.join(os.path.dirname(__file__), "dataset/mav0")

def color_from_id(identifier):
    """Generate pseudo-random color from integer identifier for visualization."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256
    ]


# Setup rerun visualizer
rr.init("cuVSLAM Visualizer", spawn=True)

# Setup coordinate basis for root, cuvslam uses right-hand system with
# X-right, Y-down, Z-forward
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Setup rerun views
blueprint = rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Horizontal(
        column_shares=[0.5, 0.5],
        contents=[
            rrb.Vertical(contents=[
                rrb.Horizontal(contents=[
                    rrb.Spatial2DView(origin='world/camera_0'),
                    rrb.Spatial2DView(origin='world/camera_1')
                ]),
                rrb.Vertical(contents=[
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
                )
                ])
            ]),
            rrb.Spatial3DView(origin='world')
        ]
    )
)
rr.send_blueprint(blueprint)
blueprint_path = os.path.join(os.path.dirname(__file__), "cpp", "euroc_blueprint")
# Uncomment the next line to regenerate the blueprint file for the C++ example
# blueprint.save(blueprint_path)

# Available tracking modes:
# 0: Multicamera - Visual tracking using stereo camera (can be extended to multiple stereo cameras)
# 1: Inertial - Visual-inertial tracking using stereo camera + IMU
# 2: RGBD - Visual tracking using monocular camera + depth (supports grayscale input)
# 3: Mono - Visual tracking using monocular camera (without scale, accurate rotation only)

euroc_tracking_mode = cuvslam.Odometry.OdometryMode.Inertial

# Configure tracker
odom_cfg = cuvslam.Odometry.Config(
    async_sba=False,
    enable_observations_export=True,
    enable_final_landmarks_export=True,
    rectified_stereo_camera=False,
    odometry_mode=euroc_tracking_mode
)

# Get camera rig
rig = get_rig(sequence_path)

# Initialize tracker
tracker = cuvslam.Tracker(rig, odom_cfg)
print(f"cuVSLAM Tracker initilized with odometry mode: {cfg.odometry_mode}")

# Track frames
last_camera_timestamp = None
imu_count_since_last_camera = 0
frame_id = 0
trajectory = []
frames_metadata = prepare_frame_metadata_euroc(
    sequence_path, euroc_tracking_mode
)

odom_trajectory = []

for frame_metadata in frames_metadata:
    timestamp = frame_metadata['timestamp']

    if frame_metadata['type'] == 'imu':
        accel_data = frame_metadata['accel']
        gyro_data = frame_metadata['gyro']
        imu_measurement = cuvslam.ImuMeasurement()
        imu_measurement.timestamp_ns = int(timestamp)
        imu_measurement.linear_accelerations = np.asarray(accel_data)
        imu_measurement.angular_velocities = np.asarray(gyro_data)
        tracker.register_imu_measurement(0, imu_measurement)
        imu_count_since_last_camera += 1
        continue

    images = [load_frame(image_path) for image_path in frame_metadata['images_paths']]

    # Check IMU measurements before tracking
    if (cfg.odometry_mode == cuvslam.Odometry.OdometryMode.Inertial
            and last_camera_timestamp is not None):
        if imu_count_since_last_camera == 0:
            print(
                f"Warning: No IMU measurements between timestamps "
                f"{last_camera_timestamp} and {timestamp}"
            )

    # Reset counters
    last_camera_timestamp = timestamp
    imu_count_since_last_camera = 0

    # Track frame
    odom_pose_estimate, _ = tracker.track(timestamp, images)

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame_id}")
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odom_pose_estimate.world_from_rig.pose
    current_observations_main_cam = tracker.odometry.get_last_observations(0)
    trajectory.append(odom_pose.translation)
    odom_trajectory.append([timestamp] + list(odom_pose.translation) + list(odom_pose.rotation))

    gravity = None
    if cfg.odometry_mode == cuvslam.Odometry.OdometryMode.Inertial:
        # Gravity estimation requires collecting sufficient number of keyframes with motion diversity
        gravity = tracker.odometry.get_last_gravity()

    # Visualize
    rr.set_time("frame", sequence=frame_id)
    rr.log("world/trajectory", rr.LineStrips3D(trajectory), static=True)
    rr.log(
        "world/camera_0",
        rr.Transform3D(
            translation=odom_pose.translation,
            quaternion=odom_pose.rotation
        ),
        rr.Arrows3D(
            vectors=np.eye(3) * 0.2,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ axes
        )
    )

    points = np.array([[obs.u, obs.v] for obs in current_observations_main_cam])
    colors = np.array([color_from_id(obs.id) for obs in current_observations_main_cam])
    rr.log(
        "world/camera_0/observations",
        rr.Points2D(positions=points, colors=colors, radii=5.0),
        rr.Image(images[0]).compress(jpeg_quality=80)
    )

    rr.log(
        "world/camera_1/observations",
        rr.Points2D(positions=points, colors=colors, radii=5.0),
        rr.Image(images[1]).compress(jpeg_quality=80)
    )

    if gravity is not None:
        rr.log(
            "world/camera_0/gravity",
            rr.Arrows3D(vectors=gravity, colors=[[255, 0, 0]], radii=0.015)
        )

    if cfg.odometry_mode == cuvslam.Odometry.OdometryMode.Inertial:
        rr.log("world/imu/accel/x", rr.Scalars(accel_data[0]), static=False)
        rr.log("world/imu/accel/y", rr.Scalars(accel_data[1]), static=False)
        rr.log("world/imu/accel/z", rr.Scalars(accel_data[2]), static=False)

        rr.log("world/imu/gyro/x", rr.Scalars(gyro_data[0]), static=False)
        rr.log("world/imu/gyro/y", rr.Scalars(gyro_data[1]), static=False)
        rr.log("world/imu/gyro/z", rr.Scalars(gyro_data[2]), static=False)

    frame_id += 1
