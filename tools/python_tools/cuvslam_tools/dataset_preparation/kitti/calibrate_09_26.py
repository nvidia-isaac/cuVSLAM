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

"""Reproduce calib_2011_09_26.json from KITTI's 2011_09_26 checkerboard shot and sequence 03.

Cvišić et al. (ECMR 2021, Sec. III-B) recipe: all boards of one stereo shot, left fx/cu/cv and Tx fixed.
Not in the paper: right fy and stereo pitch/roll fitted to the row alignment of sequence 03 stereo matches, and stereo
yaw set to the authors' disparity at infinity relative to KITTI's rectification.
"""

import argparse
import zipfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.spatial.transform import Rotation

from cuvslam_tools.dataset_preparation.common import dataset_file, run_download_script

from . import recalibration as cal
from .prepare import DOWNLOAD_SCRIPT

DATE = "2011_09_26"
DRIVE_ZIP = "2011_09_26_drive_0119_extract.zip"
FRAME = "2011_09_26/2011_09_26_drive_0119_extract/image_0{}/data/0000000000.png"
ODOMETRY_ZIP = "data_odometry_gray.zip"
SEQUENCE_03 = "dataset/sequences/03/image_{}/"
# The authors' fixed 975 / 700 / 247 / -0.539 with fx and cv moved by KITTI's 09_26 vs 09_30, 10_03 left camera.
FIXED = {"l_fx": 979.9, "l_cu": 700.0, "l_cv": 231.0, "tx": -0.539}
INITIAL_FOCAL = 980.0
VERGENCE_PX = 0.055  # the authors' disparity at infinity vs KITTI's rectification on 09_30 and 10_03
MATCH_FRAME_STEP = 20
INTRINSICS = ("fx", "fy", "cu", "cv", "k1", "k2", "p1", "p2", "k3")
NAMES = tuple(f"{s}_{n}" for s in "lr" for n in INTRINSICS) + ("rx", "ry", "rz", "tx", "ty", "tz")
SB_FLAGS = cv2.CALIB_CB_LARGER | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-3))


def canonical(grid):
    """Corner grid (rows, cols, 2) reordered so that +col points right and +row points down in the image."""
    col_dir, row_dir = (grid[:, -1] - grid[:, 0]).mean(0), (grid[-1, :] - grid[0, :]).mean(0)
    if abs(col_dir[0]) < abs(row_dir[0]):
        grid = grid.transpose(1, 0, 2)
    if (grid[:, -1] - grid[:, 0]).mean(0)[0] < 0:
        grid = grid[:, ::-1]
    if (grid[-1, :] - grid[0, :]).mean(0)[1] < 0:
        grid = grid[::-1, :]
    return np.ascontiguousarray(grid)


def _sb(image, scale):
    """Full checkerboard grid in an image upscaled by scale, in the input's pixels, or None."""
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, corners, meta = cv2.findChessboardCornersSBWithMeta(image, (4, 4), flags=SB_FLAGS)
    if not ok:
        return None
    return (corners.reshape(*meta.shape[:2], 2).astype(np.float64) + 0.5) / scale - 0.5


def _outline(grid):
    """Polygon one square beyond the outer corners."""
    rows, cols = grid.shape[:2]
    ext = np.empty((rows + 2, cols + 2, 2))
    ext[1:-1, 1:-1] = grid
    ext[0, 1:-1], ext[-1, 1:-1] = 2 * grid[0] - grid[1], 2 * grid[-1] - grid[-2]
    ext[:, 0], ext[:, -1] = 2 * ext[:, 1] - ext[:, 2], 2 * ext[:, -2] - ext[:, -3]
    return cv2.convexHull(ext.reshape(-1, 2).astype(np.float32)).astype(np.int32)


def detect_boards(image, max_boards=30):
    """Every checkerboard: found one at a time and masked, then searched again in 1x and 2x windows."""
    work, fill, boards = image.copy(), int(np.median(image)), []

    def add(grid):
        boards.append(canonical(grid))
        cv2.fillConvexPoly(work, _outline(grid), fill)

    for _ in range(max_boards):
        grid = _sb(work, 1)
        if grid is None:
            break
        add(grid)
    h, w = image.shape
    for scale in (1, 2):
        for wh, ww in ((256, 464), (192, 348)):
            for y0 in sorted(set(range(0, h - wh + 1, wh // 2)) | {h - wh}):
                for x0 in sorted(set(range(0, w - ww + 1, ww // 2)) | {w - ww}):
                    grid = _sb(work[y0:y0 + wh, x0:x0 + ww], scale)
                    if grid is None:
                        continue
                    grid = grid + (x0, y0)
                    # A grid at the window edge may be cut: retry on its padded bounding box.
                    pad = 3 * np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2))
                    x_lo, y_lo = np.maximum(np.floor(grid.reshape(-1, 2).min(0) - pad), 0).astype(int)
                    x_hi, y_hi = np.minimum(np.ceil(grid.reshape(-1, 2).max(0) + pad), (w - 1, h - 1)).astype(int)
                    full = _sb(work[y_lo:y_hi + 1, x_lo:x_hi + 1], scale)
                    if full is not None and full.size >= grid.size:
                        grid = full + (x_lo, y_lo)
                    add(grid)
    return boards


def match_boards(left, right):
    """(left, right) pairs of equal-size boards 20-200 px apart in x and within 25 px in y, ordered by x."""
    candidates = []
    for i, gl in enumerate(left):
        cl = gl.reshape(-1, 2).mean(0)
        for j, gr in enumerate(right):
            cr = gr.reshape(-1, 2).mean(0)
            if gr.shape == gl.shape and 20 < cl[0] - cr[0] < 200 and abs(cl[1] - cr[1]) < 25:
                candidates.append((abs(cl[1] - cr[1]), i, j))
    used_left, used_right, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            pairs.append((left[i], right[j]))
    return sorted(pairs, key=lambda p: p[0][..., 0].mean())


def project(X, intrinsics):
    """Pixels of camera-frame points X (N x 3) through fx fy cu cv k1 k2 p1 p2 k3."""
    fx, fy, cu, cv, *D = intrinsics
    xd, yd = cal.distort(X[:, 0] / X[:, 2], X[:, 1] / X[:, 2], D)
    return np.column_stack([fx * xd + cu, fy * yd + cv])


def _board_points(grid, square):
    rows, cols = grid.shape[:2]
    g = np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), -1).reshape(-1, 2) * square
    return np.column_stack([g, np.zeros(len(g))])


def _board_pose(points, pixels, intrinsics):
    K = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
    ok, r, t = cv2.solvePnP(points, pixels, K, np.array(intrinsics[4:]), flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    return np.r_[r.ravel(), t.ravel()]


def _transform(points, pose):
    return points @ Rotation.from_rotvec(pose[:3]).as_matrix().T + pose[3:]


def _refine_camera(boards, square, side, params, fixed):
    """Calibrate one camera alone (better conditioned than the stereo problem), keeping its fixed values."""
    free = [n for n in INTRINSICS if f"{side}_{n}" not in fixed]
    values = {n: params[f"{side}_{n}"] for n in INTRINSICS}
    data = [(_board_points(g, square), g.reshape(-1, 2)) for g in boards]
    poses = [_board_pose(p, o, [values[n] for n in INTRINSICS]) for p, o in data]

    def residuals(x):
        values.update(zip(free, x[:len(free)]))
        intrinsics = [values[n] for n in INTRINSICS]
        return np.concatenate([(project(_transform(p, pose), intrinsics) - o).ravel()
                               for (p, o), pose in zip(data, x[len(free):].reshape(-1, 6))])

    x = least_squares(residuals, np.r_[[values[n] for n in free], np.ravel(poses)], method="lm", x_scale="jac",
                      max_nfev=50000).x
    return dict(zip((f"{side}_{n}" for n in free), x[:len(free)]))


def solve(pairs, square, fixed, size):
    """Stereo calibration (NAMES -> value) and the reprojection errors (px) of all corners."""
    params = {f"{s}_{n}": 0.0 for s in "lr" for n in INTRINSICS[4:]}
    for s in "lr":
        params.update({f"{s}_fx": INITIAL_FOCAL, f"{s}_fy": INITIAL_FOCAL, f"{s}_cu": size[0] / 2,
                       f"{s}_cv": size[1] / 2})
    params.update(fixed)
    for i, side in enumerate("lr"):
        params.update(_refine_camera([p[i] for p in pairs], square, side, params, fixed))

    data, poses, relative = [], [], []
    for gl, gr in pairs:
        points = _board_points(gl, square)
        data.append((points, gl.reshape(-1, 2), gr.reshape(-1, 2)))
        pl, pr = (_board_pose(points, g.reshape(-1, 2), [params[f"{s}_{n}"] for n in INTRINSICS])
                  for g, s in ((gl, "l"), (gr, "r")))
        Rl, Rr = (Rotation.from_rotvec(p[:3]).as_matrix() for p in (pl, pr))
        poses.append(np.r_[Rotation.from_matrix(Rl).as_rotvec(), pl[3:]])  # this start gives the bundled file
        relative.append((Rr @ Rl.T, pr[3:] - Rr @ Rl.T @ pl[3:]))
    params.update(zip(("rx", "ry", "rz"), Rotation.from_matrix([r for r, _ in relative]).mean().as_rotvec()))
    params.update(zip(("tx", "ty", "tz"), np.median([t for _, t in relative], axis=0)))
    params.update(fixed)
    free = [n for n in NAMES if n not in fixed]

    def residuals(x):
        v = dict(params, **dict(zip(free, x[:len(free)])))
        R, T = Rotation.from_rotvec([v["rx"], v["ry"], v["rz"]]).as_matrix(), np.array([v["tx"], v["ty"], v["tz"]])
        out = []
        for (points, ol, orr), pose in zip(data, x[len(free):].reshape(-1, 6)):
            X1 = _transform(points, pose)
            out += [(project(X1, [v[f"l_{n}"] for n in INTRINSICS]) - ol).ravel(),
                    (project(X1 @ R.T + T, [v[f"r_{n}"] for n in INTRINSICS]) - orr).ravel()]
        return np.concatenate(out)

    sol = least_squares(residuals, np.r_[[params[n] for n in free], np.ravel(poses)], method="lm", x_scale="jac",
                        max_nfev=50000)
    params.update(zip(free, sol.x[:len(free)]))
    return params, np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)


def rectify(params, raw_size, rect_size) -> cal.Stereo:
    """OpenCV rectification (zero disparity at infinity, no black border) of a stereo calibration."""
    K = [np.array([[params[f"{s}_fx"], 0, params[f"{s}_cu"]], [0, params[f"{s}_fy"], params[f"{s}_cv"]], [0, 0, 1]])
         for s in "lr"]
    D = [np.array([params[f"{s}_{n}"] for n in INTRINSICS[4:]]) for s in "lr"]
    R = Rotation.from_rotvec([params["rx"], params["ry"], params["rz"]]).as_matrix()
    T = np.array([[params["tx"]], [params["ty"]], [params["tz"]]])
    R1, R2, P1, P2, *_ = cv2.stereoRectify(K[0], D[0], K[1], D[1], tuple(raw_size), R, T,
                                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=tuple(rect_size))
    return cal.Stereo((cal.Camera(K[0], D[0], R1, P1), cal.Camera(K[1], D[1], R2, P2)), tuple(rect_size))


def stereo_matches(left, right):
    """Left points (N x 2) and their right matches (N x 2): forward-backward consistent LK on corners."""
    pts = cv2.goodFeaturesToTrack(left, maxCorners=3000, qualityLevel=0.005, minDistance=5, blockSize=7)
    if pts is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(left, right, pts, None, **LK)
    p0, st0, _ = cv2.calcOpticalFlowPyrLK(right, left, p1, None, **LK)
    pts, p1 = pts.reshape(-1, 2), p1.reshape(-1, 2)
    d = pts[:, 0] - p1[:, 0]
    ok = (st1.ravel() == 1) & (st0.ravel() == 1) & (np.linalg.norm(p0.reshape(-1, 2) - pts, axis=1) < 0.05)
    ok &= (d > 1) & (d < 150)
    return pts[ok], p1[ok]


def _read_image(archive: zipfile.ZipFile, name: str):
    return cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_GRAYSCALE)


def refine_rows(params, kitti: cal.Stereo, gray_zip: Path, raw_size):
    """Right fy and stereo pitch/roll (rx, rz) that row-align stereo matches of KITTI-rectified sequence 03."""
    raw_left, raw_right = [], []
    with zipfile.ZipFile(gray_zip) as archive:
        left_dir, right_dir = SEQUENCE_03.format(0), SEQUENCE_03.format(1)
        names = sorted(n for n in archive.namelist() if n.startswith(left_dir) and n.endswith(".png"))
        for name in names[::MATCH_FRAME_STEP]:
            pl, pr = stereo_matches(_read_image(archive, name), _read_image(archive, name.replace(left_dir, right_dir)))
            raw_left.append(np.column_stack(cal.rectified_to_raw(kitti.cameras[0], pl[:, 0], pl[:, 1])))
            raw_right.append(np.column_stack(cal.rectified_to_raw(kitti.cameras[1], pr[:, 0], pr[:, 1])))
    raw_left, raw_right = np.concatenate(raw_left), np.concatenate(raw_right)

    def dy(x):
        stereo = rectify(dict(params, r_fy=x[0], rx=x[1], rz=x[2]), raw_size, kitti.size)
        return (cal.raw_to_rectified(stereo.cameras[1], raw_right[:, 0], raw_right[:, 1])[1] -
                cal.raw_to_rectified(stereo.cameras[0], raw_left[:, 0], raw_left[:, 1])[1])

    x0 = np.array([params["r_fy"], params["rx"], params["rz"]])
    x = least_squares(dy, x0, loss="soft_l1", f_scale=0.3, x_scale=np.array([1.0, 1e-4, 1e-4])).x
    print(f"{len(raw_left)} matches: row error median {np.median(dy(x0)):+.3f} -> {np.median(dy(x)):+.3f} px")
    return dict(params, r_fy=x[0], rx=x[1], rz=x[2])


def disparity_at_infinity(params, kitti: cal.Stereo, raw_size):
    """Median x difference of left and right new-rectified pixels for KITTI-rectified points at infinity."""
    stereo = rectify(params, raw_size, kitti.size)
    w, h = kitti.size
    u, v = np.meshgrid(np.arange(20, w - 20, 16, dtype=np.float64), np.arange(20, h - 20, 16, dtype=np.float64))
    xs = [cal.raw_to_rectified(new, *cal.rectified_to_raw(old, u, v))[0]
          for new, old in zip(stereo.cameras, kitti.cameras)]
    return float(np.median(xs[0] - xs[1]))


def match_vergence(params, kitti: cal.Stereo, raw_size):
    """Stereo yaw (ry) that gives VERGENCE_PX disparity at infinity relative to KITTI's rectification."""
    def offset(d):
        return disparity_at_infinity(dict(params, ry=params["ry"] + d), kitti, raw_size) - VERGENCE_PX

    d = brentq(offset, np.radians(-0.3), np.radians(0.3), xtol=1e-9)
    print(f"disparity at infinity {offset(0.0) + VERGENCE_PX:+.3f} -> {offset(d) + VERGENCE_PX:+.3f} px")
    return dict(params, ry=params["ry"] + d)


def write_stereo_calibration(path: Path, stereo: cal.Stereo, params, raw_size) -> None:
    """Write a calibration in the Cvišić et al. format."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    for key, value in (("image_width_raw", raw_size[0]), ("image_height_raw", raw_size[1]),
                       ("image_width", stereo.size[0]), ("image_height", stereo.size[1]), ("cameras", 2)):
        storage.write(key, int(value))
    storage.write("model", "radtan")
    for i, camera in enumerate(stereo.cameras, start=1):
        for name, value in zip("KDRP", (camera.K, camera.D.reshape(-1, 1), camera.R, camera.P)):
            storage.write(f"CAM{i}_{name}", np.asarray(value, float))
    storage.write("CAM2_TO_CAM1_R", Rotation.from_rotvec([params["rx"], params["ry"], params["rz"]]).as_matrix())
    storage.write("CAM2_TO_CAM1_T", np.array([[params["tx"]], [params["ty"]], [params["tz"]]]))
    storage.write("KITTI_TO_CAM1_R", np.eye(3))
    storage.release()


def calibrate(raw_dir: Path, out: Path) -> None:
    """Download what is missing into raw_dir, calibrate, and write the calibration to out."""
    run_download_script(dataset_file(__file__, DOWNLOAD_SCRIPT), [str(raw_dir), "--recalibrated", "--checkerboards"])
    kitti, square = cal.read_kitti_calibration(raw_dir / f"{DATE}_calib.zip", DATE)
    with zipfile.ZipFile(raw_dir / DRIVE_ZIP) as archive:
        left, right = (_read_image(archive, FRAME.format(c)) for c in (0, 1))
    pairs = match_boards(detect_boards(left), detect_boards(right))
    raw_size = left.shape[::-1]
    params, errors = solve(pairs, square, FIXED, raw_size)
    print(f"{len(pairs)} boards: reprojection error rms {np.sqrt(np.mean(errors ** 2)):.4f} px")
    params = refine_rows(params, kitti, raw_dir / ODOMETRY_ZIP, raw_size)
    params = match_vergence(params, kitti, raw_size)
    write_stereo_calibration(out, rectify(params, raw_size, kitti.size), params, raw_size)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and write the 2011_09_26 calibration."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="KITTI archives directory (prepare_kitti --raw-dir).")
    parser.add_argument("--out", type=Path, required=True, help="Calibration file to write (.json or .yaml).")
    args = parser.parse_args(argv)
    calibrate(args.raw_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
