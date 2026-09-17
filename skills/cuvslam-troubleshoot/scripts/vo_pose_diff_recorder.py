#!/usr/bin/env python3
# REFERENCE SCRIPT - This file is for reference only.
#
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

"""
ROS2 node that subscribes to /visual_slam/tracking/vo_pose and writes
per-frame translation diffs to a CSV file.

Columns: frame_id, timestamp_ns, translation_diff_m

Run standalone:
    python3 vo_pose_diff_recorder.py --output /path/to/vo_pose_diff.txt

Or launched from a launch file via ExecuteProcess.
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class VoPoseDiffRecorder(Node):

    def __init__(self, output_path: str) -> None:
        super().__init__('vo_pose_diff_recorder')

        self._output_path = output_path
        self._prev_pos = None
        self._prev_ts_ns = None
        self._frame_id = 0
        self._bag_start_ns = None

        self._file = open(output_path, 'w')
        self._file.write('frame_id,timestamp_ns,translation_diff_m\n')
        self._file.flush()

        self._sub = self.create_subscription(
            PoseStamped,
            '/visual_slam/tracking/vo_pose',
            self._callback,
            10,
        )
        self.get_logger().info(f'Recording vo_pose diffs to {output_path}')

    def _callback(self, msg: PoseStamped) -> None:
        ts_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

        if self._bag_start_ns is None:
            self._bag_start_ns = ts_ns

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        if self._prev_pos is not None:
            dx = x - self._prev_pos[0]
            dy = y - self._prev_pos[1]
            dz = z - self._prev_pos[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            self._file.write(f'{self._frame_id},{ts_ns},{dist:.8f}\n')
            self._file.flush()

        self._prev_pos = (x, y, z)
        self._prev_ts_ns = ts_ns
        self._frame_id += 1

    def destroy_node(self) -> None:
        self._file.close()
        self.get_logger().info(
            f'Saved {self._frame_id} pose diff entries to {self._output_path}'
        )
        super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        default='/tmp/vo_pose_diff.txt',
        help='Output CSV file path',
    )
    # Strip ROS2 args before argparse sees them
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = VoPoseDiffRecorder(args.output)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
