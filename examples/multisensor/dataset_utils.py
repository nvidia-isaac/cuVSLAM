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

import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import cuvslam as vslam


def to_distortion_model(distortion: str) -> vslam.Distortion.Model:
    """Convert string distortion model name to vslam.Distortion.Model enum."""
    distortion_models = {
        'pinhole': vslam.Distortion.Model.Pinhole,
        'fisheye': vslam.Distortion.Model.Fisheye,
        'brown': vslam.Distortion.Model.Brown,
        'polynomial': vslam.Distortion.Model.Polynomial
    }
    if distortion not in distortion_models:
        raise ValueError(f"Unknown distortion model: {distortion}")
    return distortion_models[distortion]


def opengl_to_opencv_transform(rotation: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert from edex coordinate system to OpenCV coordinate system."""
    K = np.array([
        [1, 0,  0],
        [0, -1, 0],
        [0, 0, -1]])
    return (K @ rotation @ K.T), K @ translation


def transform_to_pose(transform_16: List[float]) -> vslam.Pose:
    """Convert a 4x4 transformation matrix to a vslam.Pose object."""
    transform = np.array(transform_16).reshape([-1, 4])
    rotation_opencv, translation_opencv = opengl_to_opencv_transform(transform[:3, :3], transform[:3, 3])
    rotation_quat = Rotation.from_matrix(rotation_opencv).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_opencv)


def read_stereo_edex(file_path: str) -> List[vslam.Camera]:
    """Load cameras from a tartan_ground.edex file (cuvslam EDEX schema)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EDEX file not found: {file_path}")

    with open(file_path, 'r') as file:
        data = json.load(file)

    cameras = []
    for cam_data in data[0]['cameras']:
        config = {
            'camera_model': cam_data['intrinsics']['distortion_model'],
            'distortion_coefficients': cam_data['intrinsics']['distortion_params'],
            'intrinsics': cam_data['intrinsics']['focal'] + cam_data['intrinsics']['principal'],
            'resolution': cam_data['intrinsics']['size'],
            'extrinsics': cam_data['transform']
        }

        cam = vslam.Camera()
        cam.distortion = vslam.Distortion(
            to_distortion_model(config['camera_model']),
            config['distortion_coefficients']
        )
        cam.focal = config['intrinsics'][0:2]
        cam.principal = config['intrinsics'][2:4]
        cam.size = config['resolution']
        cam.rig_from_camera = transform_to_pose(config['extrinsics'])
        cameras.append(cam)

    return cameras


def load_rgb(image_path: str) -> np.ndarray:
    """Load an RGB image (TartanGround stores PNGs)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return np.asarray(Image.open(image_path))


DEPTH_SCALE_MM = 1000.0  # cuvslam depth_scale_factor: uint16 value / 1000 = meters
_DEPTH_MAX_UINT16 = np.iinfo(np.uint16).max


def load_depth(depth_path: str) -> np.ndarray:
    """Load a TartanGround depth frame and convert to uint16 millimetres.

    TartanGround ships depth as float32 numpy arrays in meters; some downloads
    also produce the lossless 4-channel PNG packed-float32 format documented at
    https://tartanair.org/modalities.html. The cuvslam Python binding requires
    `(H, W) uint16`, so we convert to millimetres and clip to the uint16 range
    (sky pixels would otherwise wrap around). cuvslam undoes the scale via
    `Odometry.MultisensorSettings.depth_scale_factor=DEPTH_SCALE_MM`.

    PNG path uses cv2.imread + view('<f4') exactly as the official TartanAir tools loader
    (castacks/tartanairpy reader.depth_rgba_float32) — cv2 returns the file's 4 channel bytes
    in BGRA order, which is the byte order the float32 value was packed in.  PIL.Image.open
    returns RGBA and would silently produce wrong floats (R↔B byte swap garbles every value).
    """
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    if depth_path.endswith('.npy'):
        depth_m = np.load(depth_path).astype(np.float32, copy=False)
    elif depth_path.endswith('.png'):
        import cv2  # local import: only this branch needs it
        rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise OSError(f"cv2 failed to read depth PNG: {depth_path}")
        if rgba.ndim == 3 and rgba.shape[2] == 4:
            depth_m = rgba.view('<f4').reshape(rgba.shape[:2]).copy()
        else:
            depth_m = rgba.astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth file extension: {depth_path}")

    depth_mm = depth_m * DEPTH_SCALE_MM
    np.clip(depth_mm, 0, _DEPTH_MAX_UINT16, out=depth_mm)
    return depth_mm.astype(np.uint16)


def build_imu_calibration() -> vslam.ImuCalibration:
    """Build an IMU calibration matching the TartanGround anymal IMU.

    TartanGround publishes acc/gyro in the robot **body** frame using the FRD
    convention (X-forward, Y-right, Z-down). Verification on the published
    data: `acc - acc_nograv_body` (gravity-induced specific force, body frame)
    is `[0, 0, -9.80]` at every sample, so the accelerometer reads `-g` along
    `+Z_body`, which means `Z_body` points along gravity, i.e. down. Combined
    with `vel_body`'s +X = forward, the right-hand frame closes with Y = right.

    cuvslam's rig frame is OpenCV (X-right, Y-down, Z-forward), so we install
    the fixed rotation mapping FRD body vectors into the OpenCV rig:

        OpenCV X-right    = +Y_FRD
        OpenCV Y-down     = +Z_FRD
        OpenCV Z-forward  = +X_FRD

    The IMU is co-located with the rig origin (translation 0). Noise densities
    are typical consumer-grade values; they only weight the IMU factors.
    """
    R_opencv_from_frd = np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [1, 0, 0]], dtype=np.float64)
    q_xyzw = Rotation.from_matrix(R_opencv_from_frd).as_quat()

    imu = vslam.ImuCalibration()
    imu.rig_from_imu = vslam.Pose(rotation=q_xyzw.tolist(), translation=[0, 0, 0])
    imu.gyroscope_noise_density = 0.00016
    imu.gyroscope_random_walk = 0.000022
    imu.accelerometer_noise_density = 0.0028
    imu.accelerometer_random_walk = 0.00086
    imu.frequency = 100.0
    return imu


def _load_tartan_imu(imu_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load TartanGround IMU arrays. Returns (acc, gyro, time) where:
      - acc is (N, 3) float32, body-frame linear acceleration in m/s^2
      - gyro is (N, 3) float32, body-frame angular velocity in rad/s
      - time is (N,) float64, timestamps in seconds (monotonic, near-constant rate)

    TartanAir v2 publishes IMU as separate numpy arrays under the sequence's
    `imu/` directory. File names vary slightly between releases; we try the
    layouts we've seen in the wild and surface a clear error if none match.
    """
    candidates = [
        ('acc.npy', 'gyro.npy', 'imu_time.npy'),
        ('acc.npy', 'gyro.npy', 'time.npy'),
        ('acc_xyz_body.npy', 'gyro_xyz_body.npy', 'imu_time.npy'),
        ('imu_acc.npy', 'imu_gyro.npy', 'imu_time.npy'),
    ]
    for acc_name, gyro_name, time_name in candidates:
        acc_p = os.path.join(imu_dir, acc_name)
        gyro_p = os.path.join(imu_dir, gyro_name)
        time_p = os.path.join(imu_dir, time_name)
        if all(os.path.exists(p) for p in (acc_p, gyro_p, time_p)):
            acc = np.asarray(np.load(acc_p), dtype=np.float32)
            gyro = np.asarray(np.load(gyro_p), dtype=np.float32)
            time = np.asarray(np.load(time_p), dtype=np.float64)
            if acc.shape != gyro.shape or acc.shape[0] != time.shape[0]:
                raise ValueError(
                    f"IMU array shape mismatch in {imu_dir}: "
                    f"acc {acc.shape}, gyro {gyro.shape}, time {time.shape}")
            return acc, gyro, time

    tried = '\n  '.join(', '.join(c) for c in candidates)
    raise FileNotFoundError(
        f"No TartanGround IMU arrays found under {imu_dir}. Tried:\n  {tried}\n"
        f"Re-download with modality=['imu'] (see download_tartan.py), or "
        f"adjust the candidate list in dataset_utils._load_tartan_imu().")


def prepare_tartan_frames(data_path: str, camera_list: List[str],
                          include_imu: bool = True) -> List[dict]:
    """Build a timestamp-sorted list of frame (and optionally IMU) records.

    Each camera frame has the same logical timestamp (TartanGround is
    rig-synchronous); we use the frame index times the nominal frame period
    so that IMU samples interleave correctly with images. Returned records:

      {'type': 'rgbd', 'timestamp': int, 'image_paths': [...], 'depth_paths': [...]}
      {'type': 'imu',  'timestamp': int, 'accel': [x,y,z], 'gyro': [x,y,z]}

    When `include_imu` is False, IMU samples are skipped (no IMU files are
    read) and only `rgbd` records are returned.
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Tartan sequence path not found: {data_path}")

    cam0_dir = os.path.join(data_path, f'image_{camera_list[0]}')
    if not os.path.isdir(cam0_dir):
        raise FileNotFoundError(f"Image directory not found: {cam0_dir}")
    num_frames = len(os.listdir(cam0_dir))

    cam_time_path = os.path.join(data_path, 'imu', 'cam_time.npy')
    if os.path.exists(cam_time_path):
        cam_times_s = np.load(cam_time_path).astype(np.float64)
        num_frames = min(num_frames, len(cam_times_s))
    else:
        cam_times_s = np.arange(num_frames, dtype=np.float64) * 0.1

    def find_depth_path(cam: str, frame_id: int) -> str:
        npy = os.path.join(data_path, f'depth_{cam}', f'{frame_id:06d}_{cam}_depth.npy')
        if os.path.exists(npy):
            return npy
        png = os.path.join(data_path, f'depth_{cam}', f'{frame_id:06d}_{cam}_depth.png')
        if os.path.exists(png):
            return png
        raise FileNotFoundError(
            f"Depth frame not found for camera {cam} at index {frame_id}: tried "
            f"{npy} and {png}")

    rgbd_records = []
    for frame_id in range(num_frames):
        image_paths = [os.path.join(data_path, f'image_{cam}', f'{frame_id:06d}_{cam}.png')
                       for cam in camera_list]
        depth_paths = [find_depth_path(cam, frame_id) for cam in camera_list]
        rgbd_records.append({
            'type': 'rgbd',
            'timestamp': int(float(cam_times_s[frame_id]) * 1e9),
            'image_paths': image_paths,
            'depth_paths': depth_paths,
        })

    if not include_imu:
        return rgbd_records

    acc, gyro, imu_time_s = _load_tartan_imu(os.path.join(data_path, 'imu'))
    # TartanGround publishes cam_time.npy and imu_time.npy in the same clock
    # (both start at 0). No anchoring needed; just convert to ns.
    imu_records = [{
        'type': 'imu',
        'timestamp': int(float(t) * 1e9),
        'accel': [float(a) for a in acc[i]],
        'gyro':  [float(g) for g in gyro[i]],
    } for i, t in enumerate(imu_time_s)]

    records = rgbd_records + imu_records
    records.sort(key=lambda r: r['timestamp'])
    return records
