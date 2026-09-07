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

from PIL import Image as PIL_Image
from rosbags.rosbag2 import Writer
from rosbags.serde import serialize_cdr
import numpy as np
from scipy.spatial.transform import Rotation

from rosbags.typesys.types import \
    sensor_msgs__msg__Image as Image, \
    sensor_msgs__msg__Imu as Imu, \
    sensor_msgs__msg__CameraInfo as CameraInfo, \
    sensor_msgs__msg__RegionOfInterest as RegionOfInterest, \
    geometry_msgs__msg__Transform as Transform, \
    geometry_msgs__msg__TransformStamped as TransformStamped, \
    geometry_msgs__msg__Vector3 as Vector3, \
    geometry_msgs__msg__Quaternion as Quaternion, \
    tf2_msgs__msg__TFMessage as TFMessage, \
    std_msgs__msg__Header as Header, \
    builtin_interfaces__msg__Time as Time


DISTORTION_MODEL_EDEX2ROS = {
    'pinhole': 'plumb_bob',
    'fisheye': 'equidistant',
    'brown5k': 'plumb_bob',
    'polynomial': 'rational_polynomial',
}

# edex brown5k is [k1, k2, k3, p1, p2], ROS plumb_bob is [k1, k2, p1, p2, k3].
_BROWN5K_TO_PLUMB_BOB = [0, 1, 3, 4, 2]


def edex_distortion_to_ros(intrinsics):
    """Translate edex distortion naming and coefficient order to the ROS CameraInfo convention."""
    model = intrinsics['distortion_model']
    params = list(intrinsics['distortion_params'])
    if model == 'brown5k':
        # fisheye and polynomial already match the ROS order, brown5k does not.
        assert len(params) == 5, 'brown5k needs 5 coefficients, got %d' % len(params)
        params = [params[i] for i in _BROWN5K_TO_PLUMB_BOB]
    return DISTORTION_MODEL_EDEX2ROS[model], params


def make_matrix_square(m3x4):
    m4x4 = m3x4.copy()
    m4x4.append([0.0, 0.0, 0.0, 1.0])
    return m4x4


def change_basis(a_to_b, transform_in_a):
    return np.matmul(np.matmul(a_to_b, transform_in_a), np.linalg.inv(a_to_b))


def cuvslam_transform_to_ros(transform):
    ros_to_cuvslam = np.array([
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    return change_basis(np.linalg.inv(ros_to_cuvslam), make_matrix_square(transform))


def time_from_timestamp(timestamp):
    return Time(sec=int(timestamp / 1000000000), nanosec=timestamp % 1000000000)


def quaternion_from_rot_matrix(rotation_matrix_3x3):
    return Quaternion(*tuple(Rotation.from_matrix(rotation_matrix_3x3).as_quat()))


def make_tf2_transform(cuvslam_transform):
    ros_transform = cuvslam_transform_to_ros(cuvslam_transform)
    translation = Vector3(ros_transform[0][3], ros_transform[1][3], ros_transform[2][3])
    rotation = quaternion_from_rot_matrix(ros_transform[:3, :3])
    return Transform(translation=translation, rotation=rotation)


class BagWriter:
    def __init__(self, filename, intrinsics, transforms, imu_data):
        w = Writer(filename)
        w.open()
        self.writer = w
        self.con_image_l = w.add_connection('/stereo_camera/left/image', Image.__msgtype__)
        self.con_image_r = w.add_connection('/stereo_camera/right/image', Image.__msgtype__)
        self.con_info_l = w.add_connection('/stereo_camera/left/camera_info', CameraInfo.__msgtype__)
        self.con_info_r = w.add_connection('/stereo_camera/right/camera_info', CameraInfo.__msgtype__)
        self.con_tf = w.add_connection('/tf', TFMessage.__msgtype__)
        self.con_imu = w.add_connection('/stereo_camera/imu', Imu.__msgtype__) if imu_data else None

        self.intrinsics = intrinsics
        self.tf_l = make_tf2_transform(transforms[0])
        self.tf_r = make_tf2_transform(transforms[1])
        self.tf_imu = make_tf2_transform(imu_data['transform']) if imu_data else None
        self.timestamp = 0
        self.time = Time(sec=0, nanosec=0)
        self.identity = Transform(translation=Vector3(0, 0, 0), rotation=Quaternion(0, 0, 0, 1))

    def close(self):
        self.writer.close()

    def write_imu_measurement(self, m):
        timestamp = m['timestamp']
        header = Header(stamp=time_from_timestamp(timestamp), frame_id='imu_frame')
        lin_acc = Vector3(m['LinearAccelerationX'], m['LinearAccelerationY'], m['LinearAccelerationZ'])
        ang_vel = Vector3(m['AngularVelocityX'], m['AngularVelocityY'], m['AngularVelocityZ'])
        default_covariance = np.zeros((9,), dtype=np.float64)
        message = Imu(header=header,
                      orientation=Quaternion(0, 0, 0, 1),
                      orientation_covariance=default_covariance,
                      angular_velocity=ang_vel,
                      angular_velocity_covariance=default_covariance,
                      linear_acceleration=lin_acc,
                      linear_acceleration_covariance=default_covariance)
        self.writer.write(self.con_imu, timestamp, serialize_cdr(message, Imu.__msgtype__).tobytes())

    def write_frame(self, timestamp, image_filenames):
        self.timestamp = timestamp
        self.time = time_from_timestamp(timestamp)
        self._publish_tf2_optical()
        self._publish_tf2_camera_frames()
        if self.tf_imu:
            self._publish_imu_frame()
        self._publish_camera_info(self.con_info_l, self.intrinsics[0])
        self._publish_camera_info(self.con_info_r, self.intrinsics[1])
        self._publish_image(self.con_image_l, 'left_optical_frame', image_filenames[0])
        self._publish_image(self.con_image_r, 'right_optical_frame', image_filenames[1])

    def _publish_tf2_optical(self):
        header = Header(stamp=self.time, frame_id='map')
        ts1 = TransformStamped(header=header, child_frame_id="left_optical_frame", transform=self.identity)
        ts2 = TransformStamped(header=header, child_frame_id="right_optical_frame", transform=self.identity)
        message = TFMessage(transforms=[ts1, ts2])
        self.writer.write(self.con_tf, self.timestamp, serialize_cdr(message, TFMessage.__msgtype__).tobytes())

    def _publish_tf2_camera_frames(self):
        header = Header(stamp=self.time, frame_id='base_link')
        ts_l = TransformStamped(header=header, child_frame_id="left_camera_frame", transform=self.tf_l)
        ts_r = TransformStamped(header=header, child_frame_id="right_camera_frame", transform=self.tf_r)
        message = TFMessage(transforms=[ts_l, ts_r])
        self.writer.write(self.con_tf, self.timestamp, serialize_cdr(message, TFMessage.__msgtype__).tobytes())

    def _publish_imu_frame(self):
        header = Header(stamp=self.time, frame_id='left_camera_frame')
        ts_imu = TransformStamped(header=header, child_frame_id="imu_frame", transform=self.tf_imu)
        message = TFMessage(transforms=[ts_imu])
        self.writer.write(self.con_tf, self.timestamp, serialize_cdr(message, TFMessage.__msgtype__).tobytes())

    def _publish_image(self, connection, frame, filename):
        header = Header(stamp=self.time, frame_id=frame)
        im_frame = PIL_Image.open(filename)
        w, h = im_frame.size
        data = np.array(im_frame.getdata(), dtype=np.uint8)
        message = Image(header=header, height=h, width=w, encoding='mono8', is_bigendian=0, step=w, data=data)
        self.writer.write(connection, self.timestamp, serialize_cdr(message, Image.__msgtype__).tobytes())

    def _publish_camera_info(self, connection, intrinsics):
        header = Header(stamp=self.time, frame_id='')
        f = intrinsics['focal'][0]
        principal = intrinsics['principal']
        cx = principal[0]
        cy = principal[1]
        size = intrinsics['size']
        distortion_model, distortion_params = edex_distortion_to_ros(intrinsics)
        message = CameraInfo(header=header, height=size[1], width=size[0],
                             distortion_model=distortion_model,
                             d=np.array(distortion_params, dtype=np.float64),
                             k=np.array([f, 0, cx,
                                         0, f, cy,
                                         0, 0, 1], dtype=np.float64),
                             p=np.array([f, 0, cx, 0,
                                         0, f, cy, 0,
                                         0, 0, 1, 0], dtype=np.float64),
                             r=np.zeros((9,), dtype=np.float64),
                             binning_x=0, binning_y=0,
                             roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
        self.writer.write(connection, self.timestamp, serialize_cdr(message, CameraInfo.__msgtype__).tobytes())
