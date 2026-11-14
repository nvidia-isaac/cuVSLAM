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

"""KITTI stereo calibrations and the maps that move a rectified pixel from one calibration to another."""

import zipfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# Recording date of every odometry sequence; each date has its own calibration.
SEQUENCE_DATES = {f"{s:02d}": "2011_10_03" for s in (0, 1, 2, *range(13, 22))}
SEQUENCE_DATES["03"] = "2011_09_26"
SEQUENCE_DATES.update({f"{s:02d}": "2011_09_30" for s in range(4, 13)})


class Camera(NamedTuple):
    """Raw intrinsics K, OpenCV distortion D (k1 k2 p1 p2 k3), rectifying rotation R, rectified projection P."""

    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    P: np.ndarray


class Stereo(NamedTuple):
    """Left and right rectified cameras and the rectified image size (w, h)."""

    cameras: tuple
    size: tuple


def read_kitti_calibration(calib_zip: Path, date: str) -> tuple:
    """KITTI's rectification of cameras 00 and 01 and the board square size (m) from a raw-data calib zip."""
    with zipfile.ZipFile(calib_zip) as archive:
        text = archive.read(f"{date}/calib_cam_to_cam.txt").decode()
    values = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key and key != "calib_time":
            values[key] = np.array(value.split(), dtype=float)
    cameras = tuple(Camera(values[f"K_{c}"].reshape(3, 3), values[f"D_{c}"], values[f"R_rect_{c}"].reshape(3, 3),
                           values[f"P_rect_{c}"].reshape(3, 4)) for c in ("00", "01"))
    return Stereo(cameras, tuple(int(v) for v in values["S_rect_00"])), float(values["corner_dist"][0])


def read_stereo_calibration(path: Path) -> Stereo:
    """Calibration in the Cvišić et al. format: CAM1 (left) and CAM2 (right) K, D, R, P."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"cannot read {path}")
    cameras = tuple(Camera(*(storage.getNode(f"CAM{i}_{k}").mat() for k in ("K", "D", "R", "P"))) for i in (1, 2))
    size = (int(storage.getNode("image_width").real()), int(storage.getNode("image_height").real()))
    storage.release()
    return Stereo(tuple(c._replace(D=c.D.ravel()) for c in cameras), size)


def distort(x, y, D):
    """OpenCV radial-tangential distortion of normalized coordinates."""
    k1, k2, p1, p2, k3 = D
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    return x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def undistort(xd, yd, D):
    """Inverse of distort (Newton iterations)."""
    x, y = xd.copy(), yd.copy()
    eps = 1e-7
    for _ in range(50):
        fx, fy = distort(x, y, D)
        ax, ay = distort(x + eps, y, D)
        bx, by = distort(x, y + eps, D)
        j11, j21, j12, j22 = (ax - fx) / eps, (ay - fy) / eps, (bx - fx) / eps, (by - fy) / eps
        det = j11 * j22 - j12 * j21
        rx, ry = fx - xd, fy - yd
        x, y = x - (j22 * rx - j12 * ry) / det, y - (-j21 * rx + j11 * ry) / det
    return x, y


def rectified_to_raw(camera: Camera, u, v):
    """Raw pixel (x, y) of rectified pixels (u, v)."""
    rays = camera.R.T @ np.linalg.solve(camera.P[:, :3], np.stack([u.ravel(), v.ravel(), np.ones(u.size)]))
    xd, yd = distort(rays[0] / rays[2], rays[1] / rays[2], camera.D)
    K = camera.K
    return (K[0, 0] * xd + K[0, 1] * yd + K[0, 2]).reshape(u.shape), (K[1, 1] * yd + K[1, 2]).reshape(u.shape)


def _rays(camera: Camera, x, y):
    """Rectified-frame rays (3 x N, z = 1 before rotation) of raw pixels (x, y)."""
    K = camera.K
    yd = (y - K[1, 2]) / K[1, 1]
    xn, yn = undistort((x - K[0, 2] - K[0, 1] * yd) / K[0, 0], yd, camera.D)
    return camera.R @ np.stack([xn.ravel(), yn.ravel(), np.ones(xn.size)])


def raw_to_rectified(camera: Camera, x, y):
    """Rectified pixel (u, v) of raw pixels (x, y)."""
    p = camera.P[:, :3] @ _rays(camera, x, y)
    return (p[0] / p[2]).reshape(x.shape), (p[1] / p[2]).reshape(x.shape)


def rerectification_maps(kitti: Stereo, new: Stereo) -> list:
    """Per camera, KITTI's rectified pixel (x, y maps) of every pixel rectified with the new calibration."""
    w, h = new.size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return [raw_to_rectified(old, *rectified_to_raw(cam, u, v)) for cam, old in zip(new.cameras, kitti.cameras)]


def common_crop(maps: list, size: tuple) -> tuple:
    """Rectangle (top, bottom, left, right; inclusive) where all maps stay inside an image of size (w, h)."""
    valid = np.ones(maps[0][0].shape, dtype=bool)
    for mx, my in maps:
        valid &= (mx >= 0) & (mx <= size[0] - 1) & (my >= 0) & (my <= size[1] - 1)
    top, bottom, left, right = 0, valid.shape[0] - 1, 0, valid.shape[1] - 1
    while not valid[top:bottom + 1, left:right + 1].all():
        # Trim the side with the largest invalid share.
        sub = valid[top:bottom + 1, left:right + 1]
        bad = {"top": (~sub[0]).mean(), "bottom": (~sub[-1]).mean(), "left": (~sub[:, 0]).mean(),
               "right": (~sub[:, -1]).mean()}
        side = max(bad, key=bad.get)
        top, bottom = top + (side == "top"), bottom - (side == "bottom")
        left, right = left + (side == "left"), right - (side == "right")
    return top, bottom, left, right


def frame_rotation(kitti: Stereo, new: Stereo) -> np.ndarray:
    """Rotation from KITTI's left rectified camera frame to the new one (Kabsch over rays of the same raw pixels)."""
    w, h = kitti.size
    u, v = np.meshgrid(np.arange(0, w, 8, dtype=np.float64), np.arange(0, h, 8, dtype=np.float64))
    x, y = rectified_to_raw(kitti.cameras[0], u, v)
    old_rays, new_rays = (r / np.linalg.norm(r, axis=0) for r in (_rays(kitti.cameras[0], x, y),
                                                                   _rays(new.cameras[0], x, y)))
    U, _, Vt = np.linalg.svd(new_rays @ old_rays.T)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
