#!/usr/bin/env python3
# REFERENCE SCRIPT — Isaac ROS release 4.3
# This file is for reference only.
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

"""Throttle a stereo image pair to a lower publish rate for Isaac ROS Visual SLAM.

This is a relay node: it subscribes to the original camera topics and
republishes on `*_throttled` variants at a capped rate.  The SLAM node's
remappings are then pointed at the throttled topics.

VIO frame-rate interaction
   Throttling image rate affects VIO and VO differently — try it as part of
   diagnosis and observe whether tracking improves or degrades.
   At 30 Hz the IMU integration interval triples (11 ms → 33 ms), receiving
   ~6–7 IMU messages per frame instead of ~2–3 at 90 Hz, and increasing
   per-interval drift by √3.  On a RealSense D435i dataset this turned 368
   jumps (max 4.1 m) into 995 jumps (max 16.5 m).  If throttling degrades
   VIO, switch to VO mode (tracking_mode: 0) instead.

ROS 2 parameters
----------------
target_hz   float  Target publish rate (default 30.0).
                   Recommended: ≥ 60.0 for VIO; any value for VO-only.
input_left  str    Input left-image topic  (default /camera/infra1/image_rect_raw)
input_right str    Input right-image topic (default /camera/infra2/image_rect_raw)
output_left  str   Output left-image topic  (default <input_left>_throttled)
output_right str   Output right-image topic (default <input_right>_throttled)

image_jitter_threshold_ms guideline
------------------------------------
Set the SLAM node's image_jitter_threshold_ms to at least 3 × frame_period_ms
to absorb timer scheduling jitter from this relay:

    target_hz  frame_period  recommended threshold
      90 Hz       11 ms            33 ms
      60 Hz       17 ms            51 ms
      30 Hz       33 ms            99 ms

isaac_ros_visual_slam_realsense_bag_vio_throttled.launch.py computes this
automatically as ceil(3000 / target_hz).

Usage — add to your launch file
---------------------------------
    from launch.actions import ExecuteProcess
    throttle = ExecuteProcess(
        cmd=['python3', '/path/to/stereo_image_throttle.py',
             '--ros-args',
             '-p', 'target_hz:=60.0',
             '-p', 'input_left:=/camera/infra1/image_rect_raw',
             '-p', 'input_right:=/camera/infra2/image_rect_raw'],
        output='screen',
    )

Then remap the SLAM node's image subscriptions:
    ('visual_slam/image_0', '/camera/infra1/image_rect_raw_throttled'),
    ('visual_slam/image_1', '/camera/infra2/image_rect_raw_throttled'),
"""
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image


class StereoThrottle(Node):
    def __init__(self) -> None:
        super().__init__(
            'stereo_throttle',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True),
            ],
        )
        self.declare_parameter('target_hz',    30.0)
        self.declare_parameter('input_left',   '/camera/infra1/image_rect_raw')
        self.declare_parameter('input_right',  '/camera/infra2/image_rect_raw')
        self.declare_parameter('output_left',  '')
        self.declare_parameter('output_right', '')

        target_hz    = self.get_parameter('target_hz').value
        input_left   = self.get_parameter('input_left').value
        input_right  = self.get_parameter('input_right').value
        output_left  = self.get_parameter('output_left').value  or f'{input_left}_throttled'
        output_right = self.get_parameter('output_right').value or f'{input_right}_throttled'

        if target_hz < 60.0:
            self.get_logger().warn(
                f'target_hz={target_hz:.0f} Hz: IMU integration interval grows to '
                f'{1000.0/target_hz:.0f} ms. Monitor for increased VIO jumps — '
                'if tracking degrades, consider VO mode (tracking_mode: 0) instead.'
            )

        self._latest_left:  Image | None = None
        self._latest_right: Image | None = None

        self.create_subscription(Image, input_left,  self._cb_left,  10)
        self.create_subscription(Image, input_right, self._cb_right, 10)
        self._pub_left  = self.create_publisher(Image, output_left,  10)
        self._pub_right = self.create_publisher(Image, output_right, 10)

        self.create_timer(1.0 / target_hz, self._timer_cb)
        self.get_logger().info(
            f'stereo_throttle: {input_left} + {input_right} → '
            f'{output_left} + {output_right} @ {target_hz:.0f} Hz'
        )

    def _cb_left(self, msg: Image) -> None:
        self._latest_left = msg

    def _cb_right(self, msg: Image) -> None:
        self._latest_right = msg

    def _timer_cb(self) -> None:
        if self._latest_left is not None and self._latest_right is not None:
            self._pub_left.publish(self._latest_left)
            self._pub_right.publish(self._latest_right)
            self._latest_left  = None
            self._latest_right = None


def main() -> None:
    rclpy.init()
    node = StereoThrottle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
