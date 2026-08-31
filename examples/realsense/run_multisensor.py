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

"""Multisensor RGB-D + IMU tracking on one RealSense camera."""

import queue
import threading
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyrealsense2 as rs

import cuvslam as vslam
from camera_utils import get_rs_camera, get_rs_imu
from visualizer import RerunVisualizer

# Constants
RESOLUTION = (640, 360)
FPS = 30
IMU_FREQUENCY_ACCEL = 200
IMU_FREQUENCY_GYRO = 200
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_NS = 35 * 1e6  # 35ms in nanoseconds
IMU_JITTER_THRESHOLD_NS = 6 * 1e6  # 6ms in nanoseconds
IMU_QUEUE_MAX_SIZE = IMU_FREQUENCY_ACCEL * 5
NUM_VIZ_CAMERAS = 2
SHOW_GRAVITY = False


@dataclass
class ImuSample:
    """Raw IMU sample buffered before registration with cuVSLAM."""

    timestamp_ns: int
    linear_accelerations: tuple[float, float, float]
    angular_velocities: tuple[float, float, float]


def configure_video_streams(config: rs.config) -> None:
    """Enable RGB-D streams."""
    config.enable_stream(
        rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS
    )
    config.enable_stream(
        rs.stream.depth, RESOLUTION[0], RESOLUTION[1], rs.format.z16, FPS
    )


def configure_motion_streams(config: rs.config) -> None:
    """Enable RealSense IMU streams."""
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, IMU_FREQUENCY_ACCEL)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, IMU_FREQUENCY_GYRO)


def setup_camera_parameters() -> tuple[dict[str, dict[str, object]], float]:
    """Read RealSense intrinsics, extrinsics, and depth scale."""
    config = rs.config()
    configure_video_streams(config)
    configure_motion_streams(config)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        accel_profile = profile.get_stream(rs.stream.accel)

        camera_params: dict[str, dict[str, object]] = {
            'color': {
                'intrinsics': color_profile.intrinsics
            },
            'imu': {
                'cam_from_imu': accel_profile.get_extrinsics_to(color_profile)
            }
        }
    finally:
        pipeline.stop()

    return camera_params, depth_scale


def get_rs_multisensor_rig(
    camera_params: dict[str, dict[str, object]]
) -> vslam.Rig:
    """Create a Multisensor rig for RGB-D + IMU on a D455."""
    rig = vslam.Rig()
    rig.cameras = [
        get_rs_camera(camera_params['color']['intrinsics'])
    ]
    rig.imus = [get_rs_imu(camera_params['imu']['cam_from_imu'])]
    return rig


def configure_emitter(video_pipe: rs.pipeline, video_config: rs.config) -> None:
    """Enable the IR emitter for better depth quality."""
    pipeline_wrapper = rs.pipeline_wrapper(video_pipe)
    pipeline_profile = video_config.resolve(pipeline_wrapper)
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)


def imu_thread(
    imu_queue: queue.Queue[ImuSample],
    motion_pipe: rs.pipeline,
    stop_event: threading.Event
) -> None:
    """Capture IMU samples into a queue without touching the cuVSLAM tracker."""
    prev_timestamp: Optional[int] = None
    drop_count = 0
    queue_drop_count = 0

    try:
        while not stop_event.is_set():
            imu_frames = motion_pipe.wait_for_frames()
            accel_frame = imu_frames.first_or_default(rs.stream.accel)
            gyro_frame = imu_frames.first_or_default(rs.stream.gyro)
            if not accel_frame or not gyro_frame:
                continue
            current_timestamp = int(accel_frame.timestamp * 1e6)

            if prev_timestamp is not None:
                timestamp_diff = current_timestamp - prev_timestamp
                if timestamp_diff < 0:
                    continue
                if timestamp_diff > IMU_JITTER_THRESHOLD_NS:
                    drop_count += 1
                    if drop_count % 100 == 1:
                        print(f"Warning: IMU drops detected ({drop_count} total, last gap: {timestamp_diff/1e6:.2f} ms)")

            prev_timestamp = current_timestamp

            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            sample = ImuSample(
                timestamp_ns=current_timestamp,
                linear_accelerations=(accel_data.x, accel_data.y, accel_data.z),
                angular_velocities=(gyro_data.x, gyro_data.y, gyro_data.z)
            )

            try:
                imu_queue.put_nowait(sample)
            except queue.Full:
                queue_drop_count += 1
                with suppress(queue.Empty):
                    imu_queue.get_nowait()
                imu_queue.put_nowait(sample)
                if queue_drop_count % 100 == 1:
                    print(f"Warning: IMU queue overflow ({queue_drop_count} dropped samples)")
    except RuntimeError as e:
        if not stop_event.is_set():
            print(f"IMU thread error: {e}")


def register_imu_until(
    tracker: vslam.Tracker,
    imu_queue: queue.Queue[ImuSample],
    pending_imu: deque[ImuSample],
    timestamp_ns: int,
    last_tracker_timestamp_ns: Optional[int]
) -> Optional[int]:
    """Register queued IMU samples up to the image timestamp."""
    while True:
        try:
            pending_imu.append(imu_queue.get_nowait())
        except queue.Empty:
            break

    while pending_imu and pending_imu[0].timestamp_ns <= timestamp_ns:
        sample = pending_imu.popleft()
        if last_tracker_timestamp_ns is not None and sample.timestamp_ns < last_tracker_timestamp_ns:
            continue
        imu_measurement = vslam.ImuMeasurement()
        imu_measurement.timestamp_ns = sample.timestamp_ns
        imu_measurement.linear_accelerations = sample.linear_accelerations
        imu_measurement.angular_velocities = sample.angular_velocities
        tracker.register_imu_measurement(0, imu_measurement)
        last_tracker_timestamp_ns = sample.timestamp_ns
    return last_tracker_timestamp_ns


def main() -> None:
    """Run RealSense Multisensor tracking with RGB-D and IMU."""
    camera_params, depth_scale = setup_camera_parameters()
    rig = get_rs_multisensor_rig(camera_params)

    multisensor_settings = vslam.Odometry.MultisensorSettings(
        depth_camera_ids=[0],
        depth_scale_factor=1 / depth_scale,
        enable_depth_stereo_tracking=True
    )
    odom_cfg = vslam.Odometry.Config(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        odometry_mode=vslam.Odometry.OdometryMode.Multisensor,
        multisensor_settings=multisensor_settings,
        rectified_stereo_camera=False
    )
    tracker = vslam.Tracker(rig, odom_cfg)

    video_pipe = rs.pipeline()
    video_config = rs.config()
    configure_video_streams(video_config)
    configure_emitter(video_pipe, video_config)
    align = rs.align(rs.stream.color)

    motion_pipe = rs.pipeline()
    motion_config = rs.config()
    configure_motion_streams(motion_config)

    imu_queue = queue.Queue(maxsize=IMU_QUEUE_MAX_SIZE)
    pending_imu: deque[ImuSample] = deque()
    stop_event = threading.Event()
    visualizer = RerunVisualizer(
        num_viz_cameras=NUM_VIZ_CAMERAS,
        image_size=RESOLUTION,
        show_gravity=SHOW_GRAVITY
    )

    motion_pipe.start(motion_config)
    video_pipe.start(video_config)
    imu_thread_obj = threading.Thread(
        target=imu_thread,
        args=(imu_queue, motion_pipe, stop_event),
        daemon=True
    )
    imu_thread_obj.start()

    frame_id = 0
    prev_timestamp: Optional[int] = None
    last_tracker_timestamp: Optional[int] = None
    trajectory: list[np.ndarray] = []

    try:
        while True:
            frames = video_pipe.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                print("Warning: missing color or depth frame")
                continue

            timestamp = int(color_frame.timestamp * 1e6)
            if prev_timestamp is not None:
                timestamp_diff = timestamp - prev_timestamp
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                    print(
                        f"Warning: Camera stream message drop: timestamp gap "
                        f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms"
                    )

            frame_id += 1
            if frame_id <= WARMUP_FRAMES:
                prev_timestamp = timestamp
                continue

            images = [
                np.asanyarray(color_frame.get_data())
            ]
            depths = [np.asanyarray(depth_frame.get_data())]

            last_tracker_timestamp = register_imu_until(
                tracker, imu_queue, pending_imu, timestamp, last_tracker_timestamp
            )
            odom_pose_estimate, _ = tracker.track(timestamp, images=images, depths=depths)
            last_tracker_timestamp = timestamp
            prev_timestamp = timestamp

            odom_pose_with_cov = odom_pose_estimate.world_from_rig
            if odom_pose_with_cov is None:
                print(f"Tracking failed at frame {frame_id}")
                continue

            odom_pose = odom_pose_with_cov.pose
            trajectory.append(odom_pose.translation)
            observations = tracker.odometry.get_last_observations(0)
            gravity = tracker.odometry.get_last_gravity() if SHOW_GRAVITY else None
            if gravity is not None:
                gravity = np.asarray(gravity, dtype=np.float32)

            visualizer.visualize_frame(
                frame_id=frame_id,
                images=[images[0], depths[0]],
                pose=odom_pose,
                observations_main_cam=[observations, observations],
                trajectory=trajectory,
                timestamp=timestamp,
                gravity=gravity
            )

    except KeyboardInterrupt:
        print("Stopping Multisensor tracking...")
    finally:
        stop_event.set()
        motion_pipe.stop()
        video_pipe.stop()


if __name__ == "__main__":
    main()
