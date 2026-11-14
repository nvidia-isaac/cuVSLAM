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

"""prepare_kitti --recalibrated and its calibration tools on synthetic calibrations, images and trajectories."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cuvslam_tools.dataset_preparation.common import dataset_file
from cuvslam_tools.dataset_preparation.kitti import calibrate_09_26 as calibrate
from cuvslam_tools.dataset_preparation.kitti import fit_gt_rotation as fit
from cuvslam_tools.dataset_preparation.kitti import prepare
from cuvslam_tools.dataset_preparation.kitti import recalibration as cal

W, H = 120, 80


def _stereo(cx_shift=0.0, cy_shift=0.0, rotation=np.eye(3)):
    """Distorted raw cameras rectified with a 0.5 m baseline, principal point shifted by (cx_shift, cy_shift)."""
    K = np.array([[100.0, 0, 60], [0, 100.0, 40], [0, 0, 1]])
    D = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])
    cameras = tuple(cal.Camera(K, D, rotation, np.array([[90.0, 0, 60 + cx_shift, -90.0 * baseline],
                                                        [0, 90.0, 40 + cy_shift, 0], [0, 0, 1, 0]]))
                    for baseline in (0.0, 0.5))
    return cal.Stereo(cameras, (W, H))


class TestRecalibrationGeometry(unittest.TestCase):
    def test_same_calibration_maps_every_pixel_to_itself(self):
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        for mx, my in cal.rerectification_maps(_stereo(), _stereo()):
            np.testing.assert_allclose(mx, u, atol=1e-6)
            np.testing.assert_allclose(my, v, atol=1e-6)

    def test_crop_keeps_only_pixels_both_maps_sample_inside(self):
        maps = cal.rerectification_maps(_stereo(), _stereo(2.5, 1.5))

        self.assertEqual(cal.common_crop(maps, (W, H)), (2, H - 1, 3, W - 1))

    def test_frame_rotation_is_the_rectifying_rotation_difference(self):
        Q = Rotation.from_rotvec([0.01, -0.02, 0.005]).as_matrix()

        np.testing.assert_allclose(cal.frame_rotation(_stereo(), _stereo(rotation=Q)), Q, atol=1e-9)


class TestRecalibrate(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        root = Path(self._temporary.name)
        self.raw_dir, self.dataset_dir = root / "raw", root / "kitti"
        self.raw_dir.mkdir()
        lines = ["calib_time: 09-Jan-2012 14:00:15", "corner_dist: 9.950000e-02", f"S_rect_00: {W} {H}"]
        for c, camera in zip(("00", "01"), _stereo().cameras):
            lines += [f"{key}_{c}: " + " ".join(map(str, value.ravel()))
                      for key, value in zip(("K", "D", "R_rect", "P_rect"), camera)]
        with zipfile.ZipFile(self.raw_dir / "2011_10_03_calib.zip", "w") as archive:
            archive.writestr("2011_10_03/calib_cam_to_cam.txt", "\n".join(lines) + "\n")
        storage = cv2.FileStorage(str(self.raw_dir / "lamor_00_02.yaml"), cv2.FILE_STORAGE_WRITE)
        storage.write("image_width", W)
        storage.write("image_height", H)
        for i, camera in enumerate(_stereo(2.5, 1.5).cameras, start=1):
            for key, value in zip("KDRP", camera):
                storage.write(f"CAM{i}_{key}", value)
        storage.release()
        self.gt = np.array([np.eye(4)[:3].ravel(), np.hstack([Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix(),
                                                              [[1.0], [2.0], [3.0]]]).ravel()])
        seq = self.dataset_dir / "00"
        for camera in ("00", "01"):
            (seq / camera).mkdir(parents=True)
            cv2.imwrite(str(seq / camera / f"00.{int(camera)}.0001.png"), np.full((H, W), 77, np.uint8))
        (seq / "00" / "00.0.0001.png.tga").write_bytes(b"cache of the old image")
        intrinsics = {"focal": [90.0, 90.0], "principal": [60.0, 40.0], "size": [W, H]}
        cameras = [{"intrinsics": dict(intrinsics), "transform": [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0],
                                                                  [0.0, 0.0, 1.0, 0.0]]} for x in (0.0, 0.537)]
        (seq / "stereo.edex").write_text(json.dumps([{"cameras": cameras}, {"fps": 10}]))
        np.savetxt(seq / "gt.txt", self.gt)

    def tearDown(self):
        self._temporary.cleanup()

    def test_sequence_is_rerectified_cropped_and_rotated_in_place(self):
        cal.recalibrate(self.dataset_dir, self.raw_dir)

        cameras = json.loads((self.dataset_dir / "00" / "stereo.edex").read_text())[0]["cameras"]
        for camera in cameras:
            self.assertEqual(camera["intrinsics"], {"focal": [90.0, 90.0], "principal": [59.5, 39.5],
                                                    "size": [W - 3, H - 2]})
        self.assertEqual(cameras[1]["transform"][0][3], 0.5, "KITTI's baseline")
        for camera in ("00", "01"):
            image = cv2.imread(str(self.dataset_dir / "00" / camera / f"00.{int(camera)}.0001.png"),
                               cv2.IMREAD_UNCHANGED)
            self.assertEqual(image.shape, (H - 2, W - 3))
            self.assertTrue((image == 77).all())
        self.assertFalse((self.dataset_dir / "00" / "00" / "00.0.0001.png.tga").exists(), "stale reader cache")
        A = np.eye(4)
        A[:3, :3] = json.loads(dataset_file(cal.__file__, cal.GT_ROTATIONS).read_text())["2011_10_03"]
        expected = [(A.T @ np.vstack([g.reshape(3, 4), [0, 0, 0, 1]]) @ A)[:3].ravel() for g in self.gt]
        np.testing.assert_allclose(np.loadtxt(self.dataset_dir / "00" / "gt.txt"), expected, atol=1e-8)


class TestPrepareRecalibrated(unittest.TestCase):
    def _prepare(self, recalibrated):
        with mock.patch.object(prepare, "run_download_script") as download, \
                mock.patch.object(prepare.convert_kitti, "convert") as convert, \
                mock.patch.object(prepare.recalibration, "recalibrate") as recalibrate:
            prepare.prepare(raw_dir=Path("raw"), output_dir=Path("out"), recalibrated=recalibrated)
        convert.assert_called_once_with(Path("raw"), Path("out"))
        return download.call_args.args[1], recalibrate

    def test_flag_downloads_the_calibrations_and_recalibrates_the_conversion(self):
        arguments, recalibrate = self._prepare(recalibrated=True)

        self.assertIn("--recalibrated", arguments)
        recalibrate.assert_called_once_with(Path("out"), Path("raw"))

    def test_without_the_flag_the_conversion_stays_kitti_s(self):
        arguments, recalibrate = self._prepare(recalibrated=False)

        self.assertNotIn("--recalibrated", arguments)
        recalibrate.assert_not_called()


TRUE = dict(l_fx=979.9, l_fy=976.0, l_cu=700.0, l_cv=231.0, l_k1=-0.35, l_k2=0.17, l_p1=0.002, l_p2=0.0007, l_k3=-0.05,
            r_fx=983.0, r_fy=983.5, r_cu=690.0, r_cv=241.0, r_k1=-0.36, r_k2=0.18, r_p1=0.001, r_p2=0.0005, r_k3=-0.06,
            rx=0.007, ry=-0.004, rz=-0.019, tx=-0.539, ty=0.007, tz=-0.0006)
RAW_SIZE = (1392, 512)


class TestCalibrate0926(unittest.TestCase):
    def test_canonical_grid_runs_right_and_down(self):
        grid = np.stack(np.meshgrid(np.arange(4) * 10.0, np.arange(3) * 10.0), -1)

        np.testing.assert_array_equal(calibrate.canonical(grid.transpose(1, 0, 2)), grid)
        np.testing.assert_array_equal(calibrate.canonical(grid[::-1, ::-1]), grid)

    def test_boards_pair_by_size_and_disparity(self):
        grid = np.stack(np.meshgrid(np.arange(4) * 10.0, np.arange(3) * 10.0), -1)
        left = [grid + (600, 200), grid + (300, 200)]
        right = [np.zeros((4, 4, 2)) + (260, 200), grid + (550, 205), grid + (250, 195)]

        pairs = calibrate.match_boards(left, right)

        self.assertEqual([(gl[0, 0, 0], gr[0, 0, 0]) for gl, gr in pairs], [(300, 250), (600, 550)])

    def test_solve_recovers_a_stereo_calibration(self):
        R = Rotation.from_rotvec([TRUE["rx"], TRUE["ry"], TRUE["rz"]]).as_matrix()
        T = np.array([TRUE["tx"], TRUE["ty"], TRUE["tz"]])
        rows, cols, square = 6, 9, 0.1
        points = np.column_stack([np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), -1).reshape(-1, 2) * square,
                                  np.zeros(rows * cols)])
        pairs = []
        for i, (x, y, z) in enumerate([(-1.6, -0.4, 4), (0, -0.3, 3), (1.4, -0.4, 4.5), (-1, 0.2, 5), (0.6, 0.1, 3.5),
                                       (-0.4, -0.5, 6), (1.2, 0.3, 5.5), (-1.8, 0.1, 3.5)]):
            board = Rotation.from_rotvec([0.15 * np.sin(i), 0.4 * np.cos(i), 0.1 * np.sin(2 * i)]).as_matrix()
            X = points @ board.T + (x, y, z)
            pairs.append(tuple(calibrate.project(P, [TRUE[f"{s}_{n}"] for n in calibrate.INTRINSICS])
                               .reshape(rows, cols, 2) for P, s in ((X, "l"), (X @ R.T + T, "r"))))
        fixed = {n: TRUE[n] for n in ("l_fx", "l_cu", "l_cv", "tx")}

        params, errors = calibrate.solve(pairs, square, fixed, RAW_SIZE)

        self.assertLess(errors.max(), 1e-5)
        for name in calibrate.NAMES:
            delta = 1e-4 if name.endswith(("fx", "fy", "cu", "cv")) else 1e-6  # px or unitless/rad/m
            self.assertAlmostEqual(params[name], TRUE[name], delta=delta, msg=name)

    def test_vergence_matches_the_target_disparity(self):
        kitti = calibrate.rectify(dict(TRUE, ry=TRUE["ry"] + 0.001), RAW_SIZE, (1242, 375))

        params = calibrate.match_vergence(TRUE, kitti, RAW_SIZE)

        self.assertAlmostEqual(calibrate.disparity_at_infinity(params, kitti, RAW_SIZE), calibrate.VERGENCE_PX,
                               delta=1e-6)


class TestFitGtRotation(unittest.TestCase):
    def test_fit_recovers_the_frame_rotation(self):
        poses = [np.eye(4)]
        for i in range(1, 1200):
            step = np.eye(4)
            step[:3, :3] = Rotation.from_rotvec([0.002 * np.sin(i / 50), 0.03 * np.sin(i / 90),
                                                 0.002 * np.cos(i / 70)]).as_matrix()
            step[:3, 3] = (0.0, 0.0, 1.0)
            poses.append(poses[-1] @ step)
        gt = np.array(poses)
        A = np.eye(4)
        A[:3, :3] = Rotation.from_rotvec(np.radians([0.3, -0.2, 0.25])).as_matrix()

        fitted = fit.fit([fit.Segments(gt, A.T @ gt @ A)], np.eye(3))

        self.assertLess(np.degrees(Rotation.from_matrix(fitted @ A[:3, :3].T).magnitude()), 2e-3)

    def test_rotations_file_round_trips(self):
        rotations = {"2011_10_03": Rotation.from_rotvec([0.01, 0.02, 0.03]).as_matrix(), "2011_09_26": np.eye(3)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotations.json"
            fit.write_rotations(path, rotations)
            loaded = json.loads(path.read_text())

        self.assertEqual(loaded, {d: m.tolist() for d, m in rotations.items()})


if __name__ == "__main__":
    unittest.main()
