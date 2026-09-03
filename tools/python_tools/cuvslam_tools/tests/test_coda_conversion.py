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

"""CODa conversion: archive reading, calibration, ground truth, and reporter configs."""

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from cuvslam_tools.dataset_preparation.coda import convert_coda
from cuvslam_tools.dataset_preparation.coda import prepare as coda_prepare
from cuvslam_tools.dataset_preparation.common import PreparationError

_FOCAL = (700.0, 710.0)
_PRINCIPAL = (610.0, 180.0)
_SIZE = (1224, 1024)
_BASELINE = 0.5

# Camera 0 sits one metre above the LiDAR, with no rotation between them.
_CAMERA_FROM_LIDAR_Z = 1.0

_SQRT_HALF = 0.7071067811865476


def _intrinsics_yaml(disparity_q14=None):
    """Return a cam0 intrinsics file in CODa's OpenCV YAML shape."""
    q14 = -1.0 / _BASELINE if disparity_q14 is None else disparity_q14
    projection = [_FOCAL[0], 0.0, _PRINCIPAL[0], 0.0, 0.0, _FOCAL[1], _PRINCIPAL[1], 0.0, 0.0, 0.0, 1.0, 0.0]
    disparity = [
        1.0, 0.0, 0.0, -_PRINCIPAL[0],
        0.0, 1.0, 0.0, -_PRINCIPAL[1],
        0.0, 0.0, 0.0, _FOCAL[0],
        0.0, 0.0, q14, 0.0,
    ]
    return (
        f"image_width: {_SIZE[0]}\n"
        f"image_height: {_SIZE[1]}\n"
        "camera_name: cam0\n"
        "projection_matrix:\n"
        "  rows: 3\n"
        "  cols: 4\n"
        f"  data: [{', '.join(repr(value) for value in projection)}]\n"
        "disparity_matrix:\n"
        "  rows: 4\n"
        "  cols: 4\n"
        f"  data: [{', '.join(repr(value) for value in disparity)}]\n"
    )


def _extrinsics_yaml():
    """Return calib_os1_to_cam0.yaml: T_cam0_from_os1, a pure translation in z."""
    extrinsic = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, _CAMERA_FROM_LIDAR_Z,
        0.0, 0.0, 0.0, 1.0,
    ]
    return (
        "extrinsic_matrix:\n"
        "  rows: 4\n"
        "  cols: 4\n"
        f"  data: [{', '.join(repr(value) for value in extrinsic)}]\n"
    )


def _default_poses():
    """Frame 0 at the origin, frame 1 rotated 90 degrees about z at x = 1."""
    return (
        "0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0\n"
        f"0.1 1.0 0.0 0.0 {_SQRT_HALF} 0.0 0.0 {_SQRT_HALF}\n"
        "0.2 2.0 0.0 0.0 1.0 0.0 0.0 0.0\n"
    )


def _archive_bytes(
    sequence="0",
    left_frames=(0, 1, 2),
    right_frames=(0, 1, 2),
    poses=None,
    pose_member=None,
    intrinsics=None,
    omit_extrinsics=False,
):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            f"calibrations/{sequence}/calib_cam0_intrinsics.yaml",
            _intrinsics_yaml() if intrinsics is None else intrinsics,
        )
        if not omit_extrinsics:
            archive.writestr(f"calibrations/{sequence}/calib_os1_to_cam0.yaml", _extrinsics_yaml())
        for camera, frames in ((0, left_frames), (1, right_frames)):
            for frame in frames:
                archive.writestr(
                    f"2d_rect/cam{camera}/{sequence}/2d_rect_cam{camera}_{sequence}_{frame}.png",
                    f"cam{camera}-frame{frame}".encode(),
                )
        if poses is not None:
            member = pose_member or f"{convert_coda._GT_PREFERRED}/{sequence}.txt"
            archive.writestr(member, poses)
    return buffer.getvalue()


def _write_archive(raw_dir, sequence="0", **kwargs):
    archive = raw_dir / convert_coda.archive_name(sequence)
    archive.write_bytes(_archive_bytes(sequence=sequence, **kwargs))
    return archive


def _convert(raw_dir, output_dir, sequences=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return convert_coda.convert(raw_dir, output_dir, sequences)


class CodaConversionTestCase(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        root = Path(self._temporary.name)
        self.raw_dir = root / "raw"
        self.output_dir = root / "converted" / "coda"
        self.raw_dir.mkdir(parents=True)

    def tearDown(self):
        self._temporary.cleanup()


class TestCodaLayout(CodaConversionTestCase):
    def test_converted_sequence_has_the_reporter_layout(self):
        _write_archive(self.raw_dir, poses=_default_poses())

        metadata = _convert(self.raw_dir, self.output_dir)

        sequence_dir = self.output_dir / "0"
        self.assertEqual(
            sorted(path.name for path in (sequence_dir / "00").iterdir()),
            ["0.0.00001.png", "0.0.00002.png", "0.0.00003.png"],
        )
        self.assertEqual(
            sorted(path.name for path in (sequence_dir / "01").iterdir()),
            ["0.1.00001.png", "0.1.00002.png", "0.1.00003.png"],
        )
        # Output index 1 is source frame 0, so the payloads must line up.
        self.assertEqual((sequence_dir / "00" / "0.0.00001.png").read_bytes(), b"cam0-frame0")
        self.assertEqual((sequence_dir / "01" / "0.1.00003.png").read_bytes(), b"cam1-frame2")
        self.assertEqual(metadata["sequences"][0]["converted_counts"]["frames"], 3)

    def test_edex_carries_the_rectified_stereo_rig(self):
        _write_archive(self.raw_dir, poses=_default_poses())

        _convert(self.raw_dir, self.output_dir)

        edex = json.loads((self.output_dir / "0" / "stereo.edex").read_text())
        header, body = edex
        self.assertEqual(header["frame_start"], 1)
        self.assertEqual(header["frame_end"], 3)
        left, right = header["cameras"]
        self.assertEqual(left["intrinsics"]["focal"], list(_FOCAL))
        self.assertEqual(left["intrinsics"]["principal"], list(_PRINCIPAL))
        self.assertEqual(left["intrinsics"]["size"], list(_SIZE))
        # Rectified pair: same intrinsics, offset by the baseline along x.
        self.assertEqual(right["intrinsics"], left["intrinsics"])
        self.assertEqual(left["transform"][0][3], 0.0)
        self.assertAlmostEqual(right["transform"][0][3], _BASELINE)
        self.assertEqual(body["sequence"], [["00/0.0.00001.png"], ["01/0.1.00001.png"]])

    def test_baseline_comes_from_the_disparity_matrix(self):
        _write_archive(self.raw_dir, poses=_default_poses(), intrinsics=_intrinsics_yaml(disparity_q14=-4.0))

        _convert(self.raw_dir, self.output_dir)

        edex = json.loads((self.output_dir / "0" / "stereo.edex").read_text())
        self.assertAlmostEqual(edex[0]["cameras"][1]["transform"][0][3], 0.25)

    def test_zero_disparity_entry_is_rejected(self):
        _write_archive(self.raw_dir, poses=_default_poses(), intrinsics=_intrinsics_yaml(disparity_q14=0.0))

        with self.assertRaisesRegex(convert_coda.ConversionError, "cannot derive baseline"):
            _convert(self.raw_dir, self.output_dir)

    def test_unterminated_data_list_is_reported(self):
        # A truncated calibration file: the data list never reaches its "]".
        truncated = _intrinsics_yaml().split("disparity_matrix")[0].rstrip("\n")[:-1]
        _write_archive(self.raw_dir, poses=_default_poses(), intrinsics=truncated)

        with self.assertRaisesRegex(
            convert_coda.ConversionError, "projection_matrix data list is never closed"
        ):
            _convert(self.raw_dir, self.output_dir)

    def test_non_numeric_calibration_entry_is_reported(self):
        corrupted = _intrinsics_yaml().replace(repr(_PRINCIPAL[0]), "nul", 1)
        _write_archive(self.raw_dir, poses=_default_poses(), intrinsics=corrupted)

        with self.assertRaisesRegex(
            convert_coda.ConversionError, "projection_matrix holds a non-numeric entry"
        ):
            _convert(self.raw_dir, self.output_dir)

    def test_non_finite_calibration_entry_is_reported(self):
        # float() parses these, so they need a guard of their own.
        for literal in ("nan", "inf"):
            with self.subTest(literal=literal):
                corrupted = _intrinsics_yaml().replace(repr(_PRINCIPAL[0]), literal, 1)
                _write_archive(self.raw_dir, poses=_default_poses(), intrinsics=corrupted)

                with self.assertRaisesRegex(
                    convert_coda.ConversionError, "projection_matrix holds a non-finite entry"
                ):
                    _convert(self.raw_dir, self.output_dir)

    def test_missing_calibration_member_is_reported(self):
        _write_archive(self.raw_dir, poses=_default_poses(), omit_extrinsics=True)

        with self.assertRaisesRegex(convert_coda.ConversionError, "calib_os1_to_cam0.yaml not found"):
            _convert(self.raw_dir, self.output_dir)


class TestCodaGroundTruth(CodaConversionTestCase):
    def _gt_rows(self):
        return [
            [float(value) for value in line.split()]
            for line in (self.output_dir / "0" / "gt.txt").read_text().strip().splitlines()
        ]

    def test_poses_are_relative_to_the_first_camera_frame(self):
        _write_archive(self.raw_dir, poses=_default_poses())

        _convert(self.raw_dir, self.output_dir)

        rows = self._gt_rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        # Frame 1 is the LiDAR pose moved onto cam0: a 90 degree yaw at x = 1.
        # The camera offset cancels because it is constant and rotation-free here.
        expected = [0, -1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
        for actual, wanted in zip(rows[1], expected):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_dense_poses_are_used_when_dense_global_is_absent(self):
        _write_archive(self.raw_dir, poses=_default_poses(), pose_member=f"{convert_coda._GT_FALLBACK}/0.txt")

        metadata = _convert(self.raw_dir, self.output_dir)

        self.assertEqual(metadata["sequences"][0]["ground_truth_source"], "poses/dense/0.txt")
        self.assertEqual(len(self._gt_rows()), 3)

    def test_frames_past_the_pose_file_are_dropped_from_images_and_poses(self):
        # Four stereo pairs but only three poses: the last pair has no ground truth.
        _write_archive(
            self.raw_dir, left_frames=(0, 1, 2, 3), right_frames=(0, 1, 2, 3), poses=_default_poses()
        )

        metadata = _convert(self.raw_dir, self.output_dir)

        self.assertEqual(len(list((self.output_dir / "0" / "00").iterdir())), 3)
        self.assertEqual(len(list((self.output_dir / "0" / "01").iterdir())), 3)
        self.assertEqual(len(self._gt_rows()), 3)
        self.assertEqual(metadata["sequences"][0]["dropped_outside_ground_truth"], 1)

    def test_unpaired_frames_are_skipped(self):
        _write_archive(self.raw_dir, left_frames=(0, 1, 2), right_frames=(0, 2), poses=_default_poses())

        metadata = _convert(self.raw_dir, self.output_dir)

        self.assertEqual(metadata["sequences"][0]["converted_counts"]["frames"], 2)
        self.assertEqual((self.output_dir / "0" / "00" / "0.0.00002.png").read_bytes(), b"cam0-frame2")

    def test_sequence_without_poses_produces_no_gt_file(self):
        _write_archive(self.raw_dir, poses=None)

        metadata = _convert(self.raw_dir, self.output_dir)

        self.assertFalse((self.output_dir / "0" / "gt.txt").exists())
        self.assertIsNone(metadata["sequences"][0]["ground_truth_source"])


class TestCodaReporterConfigs(CodaConversionTestCase):
    def test_configs_cover_both_modes_and_name_the_dataset_folder(self):
        _write_archive(self.raw_dir, poses=_default_poses())

        metadata = _convert(self.raw_dir, self.output_dir)

        self.assertEqual(
            metadata["reporter_configs"],
            ["coda-slam_gt.cfg", "coda-vio_gt.cfg", "coda-vio_slam.cfg", "coda-vio_slam_gt.cfg"],
        )
        config = json.loads((self.output_dir / "coda-vio_slam.cfg").read_text())
        self.assertEqual(config["dataset_folder"], "coda/")
        entries = config["sequence_cfgs"]
        self.assertEqual([entry["sequence_title"] for entry in entries], ["CODA-00-ODOM", "CODA-00-SLAM"])
        self.assertNotIn("use_slam", entries[0])
        self.assertTrue(entries[1]["use_slam"])
        self.assertTrue(all(entry["gt_file_path"] == "gt.txt" for entry in entries))

    def test_configs_omit_ground_truth_when_the_archive_has_none(self):
        _write_archive(self.raw_dir, poses=None)

        metadata = _convert(self.raw_dir, self.output_dir)

        # A gt_file_path pointing at a file that was never written fails the
        # sequence, and the _gt configs would have no sequences at all.
        self.assertEqual(metadata["reporter_configs"], ["coda-vio_slam.cfg"])
        config = json.loads((self.output_dir / "coda-vio_slam.cfg").read_text())
        self.assertTrue(all("gt_file_path" not in entry for entry in config["sequence_cfgs"]))

    def test_dataset_metadata_records_the_source_archives(self):
        archive = _write_archive(self.raw_dir, poses=_default_poses())

        metadata = _convert(self.raw_dir, self.output_dir)

        written = json.loads((self.output_dir / "dataset_metadata.json").read_text())
        self.assertEqual(written, metadata)
        self.assertEqual(
            written["source"]["archives"], [{"name": "0.zip", "size_bytes": archive.stat().st_size}]
        )


class TestCodaSequenceSelection(CodaConversionTestCase):
    def test_every_present_archive_is_converted_by_default(self):
        _write_archive(self.raw_dir, sequence="0", poses=_default_poses())
        _write_archive(self.raw_dir, sequence="10", poses=_default_poses())

        metadata = _convert(self.raw_dir, self.output_dir)

        self.assertEqual([entry["sequence"] for entry in metadata["sequences"]], ["0", "10"])

    def test_explicit_selection_restricts_the_conversion(self):
        _write_archive(self.raw_dir, sequence="0", poses=_default_poses())
        _write_archive(self.raw_dir, sequence="10", poses=_default_poses())

        metadata = _convert(self.raw_dir, self.output_dir, ["10"])

        self.assertEqual([entry["sequence"] for entry in metadata["sequences"]], ["10"])
        self.assertFalse((self.output_dir / "0").exists())

    def test_missing_archive_for_a_selected_sequence_is_reported(self):
        _write_archive(self.raw_dir, sequence="0", poses=_default_poses())

        with self.assertRaisesRegex(convert_coda.ConversionError, r"missing archive\(s\).*5\.zip"):
            _convert(self.raw_dir, self.output_dir, ["5"])

    def test_empty_raw_directory_is_reported(self):
        with self.assertRaisesRegex(convert_coda.ConversionError, "no CODa sequence archives found"):
            _convert(self.raw_dir, self.output_dir)

    def test_unknown_and_duplicate_selections_are_rejected(self):
        with self.assertRaisesRegex(convert_coda.ConversionError, "unknown sequence"):
            convert_coda.required_archives(["99"])
        with self.assertRaisesRegex(convert_coda.ConversionError, "duplicate sequence"):
            convert_coda.required_archives(["1", "1"])
        with self.assertRaisesRegex(convert_coda.ConversionError, "no sequences selected"):
            convert_coda.required_archives([])

    def test_no_archives_are_required_without_a_selection(self):
        self.assertEqual(convert_coda.required_archives(), [])
        self.assertEqual(convert_coda.required_archives(["2", "0"]), ["0.zip", "2.zip"])


class TestCodaPreparation(CodaConversionTestCase):
    def test_prepare_writes_under_the_dataset_id_and_returns_that_root(self):
        _write_archive(self.raw_dir, poses=_default_poses())
        converted_root = self.output_dir.parent

        with mock.patch.object(coda_prepare, "run_download_script") as download:
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = coda_prepare.prepare(raw_dir=self.raw_dir, output_dir=converted_root)

        self.assertEqual(prepared, converted_root / "coda")
        self.assertTrue((prepared / "coda-vio_slam.cfg").is_file())
        self.assertTrue((prepared / "0" / "stereo.edex").is_file())
        # The archive check runs against the raw directory, with no selection to forward.
        arguments = download.call_args.args[1]
        self.assertEqual(arguments, [str(self.raw_dir)])

    def test_prepare_forwards_the_selected_archives_and_force_flag(self):
        _write_archive(self.raw_dir, sequence="3", poses=_default_poses())

        with mock.patch.object(coda_prepare, "run_download_script") as download:
            with contextlib.redirect_stdout(io.StringIO()):
                coda_prepare.prepare(
                    raw_dir=self.raw_dir,
                    output_dir=self.output_dir.parent,
                    sequences=["3"],
                    force_download=True,
                )

        self.assertEqual(
            download.call_args.args[1], [str(self.raw_dir), "--force", "--archive", "3.zip"]
        )

    def test_download_only_stops_before_conversion(self):
        with mock.patch.object(coda_prepare, "run_download_script"):
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = coda_prepare.prepare(
                    raw_dir=self.raw_dir, output_dir=self.output_dir.parent, download_only=True
                )

        self.assertEqual(prepared, self.raw_dir)
        self.assertFalse(self.output_dir.exists())

    def test_conversion_failure_becomes_a_preparation_error(self):
        with mock.patch.object(coda_prepare, "run_download_script"):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(PreparationError, "no CODa sequence archives found"):
                    coda_prepare.prepare(raw_dir=self.raw_dir, output_dir=self.output_dir.parent)


if __name__ == "__main__":
    unittest.main()
