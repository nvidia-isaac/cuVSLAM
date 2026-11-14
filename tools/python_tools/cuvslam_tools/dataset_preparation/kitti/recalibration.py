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

"""Re-rectify converted KITTI odometry sequences with the Cvišić et al. (ECMR 2021) stereo calibration."""

import json
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np

from cuvslam_tools.dataset_preparation.common import PreparationError, dataset_file

# Recording date of every odometry sequence; each date has its own calibration.
SEQUENCE_DATES = {f"{s:02d}": "2011_10_03" for s in (0, 1, 2, *range(13, 22))}
SEQUENCE_DATES["03"] = "2011_09_26"
SEQUENCE_DATES.update({f"{s:02d}": "2011_09_30" for s in range(4, 13)})
# The authors' calibrations (downloaded); they published none for 2011_09_26, which uses ours (bundled).
AUTHORS_CALIBRATIONS = {"2011_10_03": "lamor_00_02.yaml", "2011_09_30": "lamor_04_12.yaml"}
OWN_CALIBRATION = "calib_2011_09_26.json"
GT_ROTATIONS = "gt_rotations.json"

_MAPS = {}


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
        raise PreparationError(f"cannot read {path}")
    cameras = tuple(Camera(*(storage.getNode(f"CAM{i}_{k}").mat() for k in ("K", "D", "R", "P"))) for i in (1, 2))
    size = (int(storage.getNode("image_width").real()), int(storage.getNode("image_height").real()))
    storage.release()
    return Stereo(tuple(c._replace(D=c.D.ravel()) for c in cameras), size)


def new_calibration(raw_dir: Path, date: str) -> Stereo:
    """The recalibration of one recording date."""
    name = AUTHORS_CALIBRATIONS.get(date)
    return read_stereo_calibration(raw_dir / name if name else dataset_file(__file__, OWN_CALIBRATION))


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


def _set_maps(maps):
    _MAPS["maps"] = maps


def _remap(job):
    camera, path = job
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise PreparationError(f"cannot read {path}")
    map_x, map_y = _MAPS["maps"][camera]
    if not cv2.imwrite(path, cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)):
        raise PreparationError(f"cannot write {path}")
    Path(path + ".tga").unlink(missing_ok=True)  # EdexReader's cache of the old image


def recalibrate_date(dataset_dir: Path, sequences: List[str], kitti: Stereo, new: Stereo,
                     gt_rotation: np.ndarray) -> None:
    """Rewrite sequences of one date in place: images, stereo.edex with KITTI's baseline, gt.txt as A^T G A."""
    maps = rerectification_maps(kitti, new)
    top, bottom, left, right = common_crop(maps, kitti.size)
    maps = [tuple(m[top:bottom + 1, left:right + 1].astype(np.float32) for m in xy) for xy in maps]
    size = [right - left + 1, bottom - top + 1]
    A = np.eye(4)
    A[:3, :3] = gt_rotation
    with ProcessPoolExecutor(initializer=_set_maps, initargs=(maps,)) as pool:
        for seq in sequences:
            seq_dir = dataset_dir / seq
            edex = json.loads((seq_dir / "stereo.edex").read_text())
            cameras = edex[0]["cameras"]
            for camera, new_camera in zip(cameras, new.cameras):
                camera["intrinsics"]["focal"] = [float(new_camera.P[0, 0]), float(new_camera.P[1, 1])]
                camera["intrinsics"]["principal"] = [float(new_camera.P[0, 2] - left), float(new_camera.P[1, 2] - top)]
                camera["intrinsics"]["size"] = size
            cameras[1]["transform"][0][3] = float(-kitti.cameras[1].P[0, 3] / kitti.cameras[1].P[0, 0])
            (seq_dir / "stereo.edex").write_text(json.dumps(edex, indent=4))

            if (seq_dir / "gt.txt").is_file():
                lines = []
                for g in np.loadtxt(seq_dir / "gt.txt").reshape(-1, 3, 4):
                    pose = A.T @ np.vstack([g, [0, 0, 0, 1]]) @ A
                    lines.append(" ".join(f"{v:.9e}" for v in pose[:3].ravel()))
                (seq_dir / "gt.txt").write_text("\n".join(lines) + "\n")

            jobs = [(camera, str(p)) for camera, sub in enumerate(("00", "01"))
                    for p in sorted((seq_dir / sub).glob("*.png"))]
            list(pool.map(_remap, jobs, chunksize=64))
            print(f"  recalibrated {seq}: {len(jobs)} images, {size[0]}x{size[1]}", flush=True)


def recalibrate(dataset_dir: Path, raw_dir: Path) -> None:
    """Re-rectify every sequence of a converted KITTI dataset in place with the calibrations in raw_dir."""
    by_date = defaultdict(list)
    for edex in sorted(Path(dataset_dir).glob("*/stereo.edex")):
        by_date[SEQUENCE_DATES[edex.parent.name]].append(edex.parent.name)
    rotations = json.loads(dataset_file(__file__, GT_ROTATIONS).read_text())
    for date, sequences in sorted(by_date.items()):
        kitti, _ = read_kitti_calibration(Path(raw_dir) / f"{date}_calib.zip", date)
        recalibrate_date(Path(dataset_dir), sequences, kitti, new_calibration(Path(raw_dir), date),
                         np.array(rotations[date]))
