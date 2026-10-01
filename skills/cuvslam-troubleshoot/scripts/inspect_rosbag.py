#!/usr/bin/env python3
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

# Inspect a ROS 2 bag for cuVSLAM troubleshooting.
#
# Prints a summary of:
#   - ros_distro, duration, topic list with message counts and Hz
#   - Camera intrinsics and computed stereo baseline (from camera_info P matrix)
#   - All static TF transforms (for IMU-to-camera extrinsic verification)
#   - Sensor model guess (D435i vs D455) based on baseline and resolution
#   - Data-completeness warnings for required cuVSLAM topics
#
# Usage:
#   source /opt/ros/<distro>/setup.bash
#   python3 scripts/inspect_rosbag.py <path/to/bag_folder> [--vio]
#
# --vio  Also check for IMU topic (required when tracking_mode=1)

import argparse
import sys
from pathlib import Path

import yaml
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage


# ROS 2 distros in release order — used for version comparison (index-based).
ROS_DISTROS_ORDERED = ['foxy', 'galactic', 'humble', 'iron', 'jazzy', 'kilted']
_JAZZY_IDX = ROS_DISTROS_ORDERED.index('jazzy')


def _ros_distro_index(distro: str) -> int | None:
    """Return the position of *distro* in ROS_DISTROS_ORDERED, or None if unknown."""
    try:
        return ROS_DISTROS_ORDERED.index(distro.lower().strip())
    except ValueError:
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _open_reader(bag_folder: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_folder, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader


def _fmt_tf(t) -> str:
    tr = t.transform.translation
    ro = t.transform.rotation
    return (
        f't=({tr.x:.6f}, {tr.y:.6f}, {tr.z:.6f})  '
        f'q=({ro.x:.4f}, {ro.y:.4f}, {ro.z:.4f}, {ro.w:.4f})'
    )


# ── inspection logic ──────────────────────────────────────────────────────────

def inspect(bag_folder: str, check_imu: bool) -> None:
    bag_path = Path(bag_folder)
    meta_path = bag_path / 'metadata.yaml'

    # ── metadata.yaml ──────────────────────────────────────────────────────
    print('=' * 60)
    print('BAG METADATA')
    print('=' * 60)

    if not meta_path.exists():
        print(f'  WARNING: metadata.yaml not found at {meta_path}')
        ros_distro = 'unknown'
        duration_s = None
    else:
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        info = meta.get('rosbag2_bagfile_information', {})
        ros_distro = info.get('ros_distro', 'unknown')
        duration_ns = info.get('duration', {}).get('nanoseconds', 0)
        duration_s = duration_ns / 1e9
        msg_count = info.get('message_count', 0)

        print(f'  ros_distro : {ros_distro}')
        print(f'  duration   : {duration_s:.1f} s')
        print(f'  messages   : {msg_count}')
        print()
        print('  Topics (name | count | Hz):')

        topics = info.get('topics_with_message_count', [])
        for t in topics:
            tm = t['topic_metadata']
            name = tm['name']
            count = t['message_count']
            hz = count / duration_s if duration_s else 0
            print(f'    {name}')
            print(f'      {count} msgs  {hz:.1f} Hz')

    # ── required topic checks ──────────────────────────────────────────────
    print()
    print('=' * 60)
    print('REQUIRED TOPIC CHECK')
    print('=' * 60)

    if meta_path.exists():
        topic_names = [
            t['topic_metadata']['name']
            for t in info.get('topics_with_message_count', [])
        ]
        topic_counts = {
            t['topic_metadata']['name']: t['message_count']
            for t in info.get('topics_with_message_count', [])
        }

        image_topics = [n for n in topic_names if 'image_rect_raw' in n or 'image_raw' in n]
        cam_info_topics = [n for n in topic_names if 'camera_info' in n]
        imu_topics = [n for n in topic_names if '/imu' in n and 'metadata' not in n]

        ok = True

        if len(image_topics) >= 2:
            print(f'  [OK] Stereo image topics found: {image_topics}')
            counts = [topic_counts[t] for t in image_topics]
            if max(counts) - min(counts) > 10:
                print(f'  [WARN] Left/right image counts differ by >{max(counts)-min(counts)} — possible sync issue')
        else:
            print(f'  [FAIL] Expected >=2 image topics, found: {image_topics}')
            ok = False

        if len(cam_info_topics) >= 2:
            print(f'  [OK] Camera info topics found: {cam_info_topics}')
        else:
            print(f'  [FAIL] Expected >=2 camera_info topics, found: {cam_info_topics}')
            ok = False

        if check_imu:
            if imu_topics:
                print(f'  [OK] IMU topic found: {imu_topics}')
                imu_hz = topic_counts[imu_topics[0]] / duration_s if duration_s else 0
                if imu_hz < 50:
                    print(f'  [WARN] IMU rate {imu_hz:.0f} Hz seems low (expected ~200 Hz for D435i/D455)')
            else:
                print('  [FAIL] VIO mode requires an IMU topic — none found')
                ok = False

        if not ok:
            print()
            print('  One or more required topics are missing. Report this to the user before proceeding.')

    # ── camera intrinsics and baseline ─────────────────────────────────────
    print()
    print('=' * 60)
    print('CAMERA INTRINSICS  (from first camera_info messages)')
    print('=' * 60)

    try:
        reader = _open_reader(bag_folder)
        reader.set_filter(rosbag2_py.StorageFilter(topics=cam_info_topics[:2]))
        intrinsics = {}
        while reader.has_next() and len(intrinsics) < 2:
            topic, raw, _ = reader.read_next()
            if topic in intrinsics:
                continue
            msg = deserialize_message(raw, CameraInfo)
            P = msg.p
            intrinsics[topic] = {
                'width': msg.width, 'height': msg.height,
                'fx': P[0], 'fy': P[5], 'cx': P[2], 'cy': P[6],
                'P3': P[3],
                'distortion_model': msg.distortion_model,
                'D': list(msg.d),
            }

        for topic, ci in intrinsics.items():
            label = 'Left ' if 'infra1' in topic or topic == cam_info_topics[0] else 'Right'
            print(f'  {label}  ({topic})')
            print(f'    resolution : {ci["width"]}x{ci["height"]}')
            print(f'    fx={ci["fx"]:.2f}  fy={ci["fy"]:.2f}  cx={ci["cx"]:.2f}  cy={ci["cy"]:.2f}')
            print(f'    distortion_model={ci["distortion_model"]}  D={ci["D"]}')
            if ci['D'] == [0.0, 0.0, 0.0, 0.0, 0.0]:
                print('    [OK] Zero distortion — images are rectified')
            else:
                print('    [INFO] Non-zero distortion — images are raw (unrectified)')

        # Baseline from right camera P[3] = -fx * baseline
        right_topics = [t for t in intrinsics if 'infra2' in t or t == cam_info_topics[1]]
        if right_topics:
            ri = intrinsics[right_topics[0]]
            if ri['fx'] != 0 and ri['P3'] != 0:
                baseline = -ri['P3'] / ri['fx']
                print()
                print(f'  Stereo baseline (from P[3]): {baseline:.6f} m  ({baseline*1000:.1f} mm)')
                width = ri['width']
                if abs(baseline - 0.050) < 0.005 and width == 640 and ri['height'] == 480:
                    print('  [INFO] Matches RealSense D435i (baseline ~50 mm, 640x480)')
                elif abs(baseline - 0.095) < 0.010:
                    print('  [INFO] Matches RealSense D455 (baseline ~95 mm)')
                    if ri['height'] == 360:
                        print('  [INFO] Resolution 640x360 also consistent with D455')
                else:
                    print('  [INFO] Unknown sensor model for this baseline')
                print()
                print('  NOTE: Verify bag2edex hardcoded baseline matches this value.')
                print('  bag_to_edex.py defaults to 0.05 m (D435i).')
                print('  bag_to_edex_ros2.py defaults to 0.0950183 m (D455).')
                print('  Edit the script if this sensor has a different baseline.')
            else:
                print('  [WARN] Cannot compute baseline — P[3]=0 or fx=0 in right camera_info')

    except Exception as e:
        print(f'  Could not read camera_info: {e}')

    # ── static TF transforms ───────────────────────────────────────────────
    print()
    print('=' * 60)
    print('STATIC TF TRANSFORMS  (/tf_static)')
    print('=' * 60)

    try:
        reader = _open_reader(bag_folder)
        reader.set_filter(rosbag2_py.StorageFilter(topics=['/tf_static']))
        static_tfs = {}
        while reader.has_next():
            topic, raw, _ = reader.read_next()
            msg = deserialize_message(raw, TFMessage)
            for t in msg.transforms:
                key = f'{t.header.frame_id} → {t.child_frame_id}'
                static_tfs[key] = t

        if static_tfs:
            for key, t in static_tfs.items():
                print(f'  {key}')
                print(f'    {_fmt_tf(t)}')
        else:
            print('  No /tf_static messages found in bag.')

        # Highlight the IMU-to-camera transforms
        imu_keys = [k for k in static_tfs if 'gyro' in k.lower() or 'accel' in k.lower() or 'imu' in k.lower()]
        if imu_keys:
            print()
            print('  IMU-related transforms (relevant for VIO calibration):')
            for k in imu_keys:
                t = static_tfs[k]
                tr = t.transform.translation
                print(f'    {k}')
                print(f'      translation = ({tr.x:.6f}, {tr.y:.6f}, {tr.z:.6f})')

    except Exception as e:
        print(f'  Could not read /tf_static: {e}')

    print()
    print('=' * 60)
    print('INSPECTION COMPLETE')
    print('=' * 60)
    print(f'  ros_distro: {ros_distro}')
    distro_idx = _ros_distro_index(ros_distro) if ros_distro else None
    if distro_idx is None:
        if ros_distro:
            print(f'  [WARN] Unknown ROS distro "{ros_distro}" — falling back to older script.')
        print('  Use bag_to_edex.py for conversion (Humble or earlier).')
    elif distro_idx >= _JAZZY_IDX:
        print('  Use bag_to_edex_ros2.py for conversion (Jazzy or later).')
    else:
        print('  Use bag_to_edex.py for conversion (Humble or earlier).')


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Inspect a ROS 2 bag for cuVSLAM troubleshooting.')
    parser.add_argument('bag_folder', help='Path to the bag directory (contains .db3 + metadata.yaml)')
    parser.add_argument('--vio', action='store_true',
                        help='Also check for IMU topic (required for tracking_mode=1)')
    args = parser.parse_args()

    inspect(args.bag_folder, check_imu=args.vio)


if __name__ == '__main__':
    main()
