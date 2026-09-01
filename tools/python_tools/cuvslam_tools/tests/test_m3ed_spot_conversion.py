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

"""M3ED SPOT conversion: source reading, rig geometry, ground truth, and layout.

The sequences are 25-42 GB each, so these build small synthetic HDF5 files with
the same structure as the published ones.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from cuvslam_tools.dataset_preparation import rgbd
from cuvslam_tools.dataset_preparation.m3ed_spot import convert_m3ed_spot

h5py = None
try:
    import h5py as _h5py

    h5py = _h5py
except ImportError:  # pragma: no cover - h5py is a declared dependency
    pass

WIDTH = 8
HEIGHT = 4
FRAME_INTERVAL_US = 40_000
POSE_INTERVAL_US = 100_000

LEFT_INTRINSICS = (1058.5, 1058.9, 673.4, 336.6)
RIGHT_INTRINSICS = (1052.4, 1053.2, 670.1, 329.4)
LEFT_DISTORTION = (-0.3937, 0.158, -0.00013, 0.00058)
RIGHT_DISTORTION = (-0.3918, 0.152, -0.00034, 0.00119)

# The left OVC camera sits 70 mm from the left event camera, the right a further
# 120 mm along x, matching the published rig to the millimetre.
LEFT_TO_PROPHESEE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.070],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
RIGHT_TO_PROPHESEE = np.array(
    [
        [1.0, 0.0, 0.0, 0.120],
        [0.0, 1.0, 0.0, -0.070],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _write_calibration(group, intrinsics, distortion, transform):
    group.create_dataset("camera_model", data=b"pinhole")
    group.create_dataset("distortion_model", data=b"radtan")
    group.create_dataset("intrinsics", data=np.array(intrinsics, dtype=np.float64))
    group.create_dataset("distortion_coeffs", data=np.array(distortion, dtype=np.float64))
    group.create_dataset("resolution", data=np.array([WIDTH, HEIGHT], dtype=np.int64))
    group.create_dataset("T_to_prophesee_left", data=transform)


def write_data_file(path, frames=6, first_us=0, interval_us=FRAME_INTERVAL_US, **overrides):
    """Write a synthetic ``_data.h5`` with the published group layout."""
    with h5py.File(path, "w") as handle:
        ovc = handle.create_group("ovc")
        stamps = np.array(
            [first_us + index * interval_us for index in range(frames)], dtype=np.int64
        )
        ovc.create_dataset("ts", data=overrides.get("ts", stamps))
        for side, intrinsics, distortion, transform in (
            ("left", LEFT_INTRINSICS, LEFT_DISTORTION, LEFT_TO_PROPHESEE),
            ("right", RIGHT_INTRINSICS, RIGHT_DISTORTION, RIGHT_TO_PROPHESEE),
        ):
            group = ovc.create_group(side)
            _write_calibration(
                group.create_group("calib"),
                intrinsics,
                distortion,
                overrides.get(f"{side}_transform", transform),
            )
            count = overrides.get(f"{side}_frames", frames)
            # Distinct pixel values per side and frame, so a swapped camera or a
            # misindexed frame is visible in the output bytes.
            images = np.zeros((count, HEIGHT, WIDTH, 1), dtype=np.uint8)
            for index in range(count):
                images[index, :, :, 0] = index + (0 if side == "left" else 100)
            group.create_dataset("data", data=images, chunks=(1, HEIGHT, WIDTH, 1))
    return path


def write_pose_file(path, poses=None, stamps=None):
    """Write a synthetic ``_pose_gt.h5``.

    ``Cn_T_C0`` maps a point from the first camera frame into frame n, so a
    camera moving forward in +x has a pose matrix with -x translation.
    """
    if poses is None:
        count = 8
        poses = np.stack([np.eye(4) for _ in range(count)])
        for index in range(count):
            poses[index, 0, 3] = -0.1 * index
    if stamps is None:
        stamps = np.array(
            [index * POSE_INTERVAL_US for index in range(poses.shape[0])], dtype=np.int64
        )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Cn_T_C0", data=poses)
        handle.create_dataset("ts", data=np.asarray(stamps, dtype=np.int64))
    return path


@unittest.skipIf(h5py is None, "h5py is required to build the synthetic sources")
class TestSourceReading(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def _data(self, **kwargs):
        return write_data_file(self.root / "data.h5", **kwargs)

    def test_calibration_maps_radtan_onto_the_polynomial_model(self):
        with h5py.File(self._data(), "r") as handle:
            left = convert_m3ed_spot.read_camera_calibration(handle, "left")
        self.assertEqual(left.focal, (1058.5, 1058.9))
        self.assertEqual(left.principal, (673.4, 336.6))
        self.assertEqual(left.size, (WIDTH, HEIGHT))
        # cuVSLAM's Polynomial model takes the first eight OpenCV coefficients,
        # so the four published radtan values are k1, k2, p1, p2 and the
        # rational terms are zero.
        self.assertEqual(left.distortion, LEFT_DISTORTION + (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(len(left.distortion), 8)

    def test_unsupported_distortion_model_is_rejected(self):
        path = self._data()
        with h5py.File(path, "r+") as handle:
            del handle["ovc/left/calib/distortion_model"]
            handle["ovc/left/calib"].create_dataset("distortion_model", data=b"equidistant")
        with h5py.File(path, "r") as handle:
            with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "equidistant"):
                convert_m3ed_spot.read_camera_calibration(handle, "left")

    def test_frame_timestamps_are_converted_to_nanoseconds(self):
        with h5py.File(self._data(frames=3), "r") as handle:
            self.assertEqual(
                convert_m3ed_spot.read_frame_timestamps(handle), [0, 40_000_000, 80_000_000]
            )

    def test_non_monotonic_frame_timestamps_are_rejected(self):
        path = self._data(frames=3, ts=np.array([0, 40_000, 40_000], dtype=np.int64))
        with h5py.File(path, "r") as handle:
            with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "strictly increasing"):
                convert_m3ed_spot.read_frame_timestamps(handle)

    def test_trajectory_inverts_the_published_transform(self):
        with h5py.File(write_pose_file(self.root / "pose.h5"), "r") as handle:
            trajectory = convert_m3ed_spot.read_trajectory(handle)
        self.assertEqual(len(trajectory), 8)
        self.assertEqual(trajectory[0][0], 0)
        self.assertEqual(trajectory[1][0], 100_000_000)
        # Cn_T_C0 carries -0.1 m per sample, so the pose of the camera is +0.1 m.
        self.assertAlmostEqual(trajectory[1][1][0], 0.1)
        self.assertAlmostEqual(trajectory[3][1][0], 0.3)

    def test_corrupt_pose_rotation_is_rejected(self):
        poses = np.stack([np.eye(4) for _ in range(3)])
        poses[1, :3, :3] = np.zeros((3, 3))
        path = write_pose_file(self.root / "pose.h5", poses=poses)
        with h5py.File(path, "r") as handle:
            with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "orthonormal"):
                convert_m3ed_spot.read_trajectory(handle)

    def test_mirrored_pose_rotation_is_rejected(self):
        # Orthonormal but left-handed, so it passes the transpose check and only
        # the determinant catches it.
        poses = np.stack([np.eye(4) for _ in range(3)])
        poses[1, :3, :3] = np.diag([1.0, 1.0, -1.0])
        path = write_pose_file(self.root / "pose.h5", poses=poses)
        with h5py.File(path, "r") as handle:
            with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "right-handed"):
                convert_m3ed_spot.read_trajectory(handle)

    def test_non_finite_pose_is_rejected(self):
        poses = np.stack([np.eye(4) for _ in range(3)])
        poses[2, 0, 3] = np.nan
        path = write_pose_file(self.root / "pose.h5", poses=poses)
        with h5py.File(path, "r") as handle:
            with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "non-finite"):
                convert_m3ed_spot.read_trajectory(handle)

    def test_mismatched_pose_arrays_are_rejected(self):
        poses = np.stack([np.eye(4) for _ in range(4)])
        path = write_pose_file(self.root / "pose.h5", poses=poses, stamps=np.arange(3))
        with h5py.File(path, "r") as handle:
            with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "entries but"):
                convert_m3ed_spot.read_trajectory(handle)

    def test_baseline_comes_from_the_two_extrinsics(self):
        with h5py.File(self._data(), "r") as handle:
            left = convert_m3ed_spot.read_camera_calibration(handle, "left")
            right = convert_m3ed_spot.read_camera_calibration(handle, "right")
        _, translation = convert_m3ed_spot.left_from_right(left, right)
        self.assertAlmostEqual(translation[0], 0.120)
        self.assertAlmostEqual(translation[1], 0.0)
        self.assertAlmostEqual(translation[2], 0.0)


@unittest.skipIf(h5py is None, "h5py is required to build the synthetic sources")
class TestConvertSequence(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.sequence = convert_m3ed_spot.ALL_SEQS[0]
        self.output = self.root / "out"

    def tearDown(self):
        self._temporary.cleanup()

    def _convert(self, frame_limit=None, **data_kwargs):
        data = write_data_file(self.root / "data.h5", **data_kwargs)
        pose = write_pose_file(self.root / "pose.h5")
        with h5py.File(data, "r") as data_handle, h5py.File(pose, "r") as pose_handle:
            return convert_m3ed_spot.convert_sequence(
                data_handle, pose_handle, self.sequence, self.output, frame_limit=frame_limit
            )

    def test_emits_the_expected_layout(self):
        metadata = self._convert()
        sequence_dir = self.output / self.sequence
        self.assertEqual(
            sorted(entry.name for entry in sequence_dir.iterdir()),
            ["00", "01", "frame_metadata.jsonl", "gt.txt", "stereo.edex"],
        )
        self.assertEqual(metadata["converted_counts"]["frames"], 6)
        self.assertEqual(
            sorted(entry.name for entry in (sequence_dir / "00").iterdir()),
            [f"{index:06d}.png" for index in range(6)],
        )

    def test_left_and_right_land_in_their_own_directories(self):
        self._convert()
        sequence_dir = self.output / self.sequence
        left = np.array(Image.open(sequence_dir / "00" / "000002.png"))
        right = np.array(Image.open(sequence_dir / "01" / "000002.png"))
        self.assertEqual(left.shape, (HEIGHT, WIDTH))
        self.assertEqual(left.dtype, np.uint8)
        # The fixture marks right-camera frames with +100, so a swap is visible.
        self.assertEqual(int(left[0, 0]), 2)
        self.assertEqual(int(right[0, 0]), 102)

    def test_frame_metadata_lists_both_cameras_at_one_timestamp(self):
        self._convert()
        lines = (self.output / self.sequence / "frame_metadata.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 6)
        record = json.loads(lines[1])
        self.assertEqual(record["frame_id"], 1)
        self.assertEqual(
            record["cams"],
            [
                {"filename": "00/000001.png", "id": 0, "timestamp": 40_000_000},
                {"filename": "01/000001.png", "id": 1, "timestamp": 40_000_000},
            ],
        )

    def test_edex_describes_the_stereo_pair_with_the_baseline(self):
        self._convert()
        document = json.loads((self.output / self.sequence / "stereo.edex").read_text())
        rig, metadata = document
        self.assertEqual(rig["frame_start"], 0)
        self.assertEqual(rig["frame_end"], 5)
        self.assertEqual(len(rig["cameras"]), 2)
        # Camera 0 is the rig origin; camera 1 carries the baseline.
        self.assertEqual(
            rig["cameras"][0]["transform"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        )
        self.assertAlmostEqual(rig["cameras"][1]["transform"][0][3], 0.120)
        self.assertEqual(rig["cameras"][0]["intrinsics"]["distortion_model"], "polynomial")
        self.assertEqual(rig["cameras"][0]["intrinsics"]["focal"], [1058.5, 1058.9])
        self.assertEqual(rig["cameras"][1]["intrinsics"]["focal"], [1052.4, 1053.2])
        # Replay must use the real frame times, not a synthesized frame rate.
        self.assertEqual(metadata["frame_metadata"], "frame_metadata.jsonl")
        self.assertNotIn("fps", metadata)

    def test_ground_truth_applies_the_event_camera_extrinsic(self):
        self._convert()
        rows = (self.output / self.sequence / "gt.txt").read_text().splitlines()
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(len(row.split()), 12)
        # The fixture translates along x only, and the extrinsic is a pure
        # translation, so it cancels here and the camera advances 0.1 m per
        # 100 ms, meaning 0.04 m per 40 ms frame.
        second = [float(value) for value in rows[1].split()]
        self.assertAlmostEqual(second[3], 0.04, places=6)

    def test_extrinsic_does_not_cancel_under_rotation(self):
        # A pose that rotates 90 degrees about z: the 70 mm offset between the
        # event camera and the OVC left camera then moves the camera centre, so
        # dropping the extrinsic changes the trajectory.
        poses = np.stack([np.eye(4) for _ in range(4)])
        for index in range(4):
            angle = math.radians(30.0 * index)
            poses[index, :3, :3] = np.array(
                [
                    [math.cos(angle), math.sin(angle), 0.0],
                    [-math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        pose_path = write_pose_file(self.root / "rot.h5", poses=poses)
        with h5py.File(pose_path, "r") as handle:
            trajectory = convert_m3ed_spot.read_trajectory(handle)
        timestamps = [0, 100_000_000, 200_000_000]
        extrinsic = rgbd.invert_transform(
            tuple(tuple(row[:3]) for row in LEFT_TO_PROPHESEE[:3]),
            tuple(row[3] for row in LEFT_TO_PROPHESEE[:3]),
        )
        with_extrinsic = rgbd.relative_ground_truth_lines(
            trajectory, timestamps, body_from_camera=extrinsic
        )
        without = rgbd.relative_ground_truth_lines(trajectory, timestamps)
        self.assertNotEqual(with_extrinsic[1], without[1])

    def test_frames_outside_the_pose_span_are_dropped(self):
        # Poses cover 0..700 ms; frames every 40 ms starting at 800 ms would all
        # fall outside, so start inside and run past the end.
        metadata = self._convert(frames=30)
        self.assertEqual(metadata["source_counts"]["frames"], 30)
        self.assertLess(metadata["converted_counts"]["frames"], 30)
        self.assertEqual(
            metadata["dropped_outside_ground_truth"],
            30 - metadata["converted_counts"]["frames"],
        )

    def test_no_frame_inside_the_pose_span_is_rejected(self):
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "ground-truth time span"):
            self._convert(frames=3, first_us=10_000_000)

    def test_frame_limit_truncates_without_changing_the_source_count(self):
        metadata = self._convert(frame_limit=2)
        self.assertEqual(metadata["converted_counts"]["frames"], 2)
        self.assertEqual(metadata["source_counts"]["frames"], 6)
        self.assertEqual(metadata["frame_limit"], 2)
        rows = (self.output / self.sequence / "gt.txt").read_text().splitlines()
        self.assertEqual(len(rows), 2)

    def test_camera_frame_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "left has 6 frames"):
            self._convert(frames=6, right_frames=5)

    def test_timestamp_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "ovc/ts has"):
            self._convert(frames=6, ts=np.arange(5, dtype=np.int64) * FRAME_INTERVAL_US)

    def test_rerunning_replaces_the_previous_output(self):
        self._convert()
        stale = self.output / self.sequence / "00" / "999999.png"
        stale.write_bytes(b"stale")
        self._convert()
        self.assertFalse(stale.exists())


class TestSequenceSelection(unittest.TestCase):
    def test_sixteen_sequences_map_to_published_names(self):
        self.assertEqual(len(convert_m3ed_spot.ALL_SEQS), 16)
        self.assertEqual(len(set(convert_m3ed_spot.ALL_SEQS)), 16)
        published = [name for _, name in convert_m3ed_spot.SEQUENCES]
        self.assertEqual(len(set(published)), 16)
        for name in published:
            self.assertTrue(name.startswith("spot_"), name)

    def test_source_name_rejects_unknown_sequences(self):
        self.assertEqual(
            convert_m3ed_spot.source_name("skatepark_2"), "spot_outdoor_day_skatepark_2"
        )
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "unknown sequence"):
            convert_m3ed_spot.source_name("skatepark_9")

    def test_selection_is_validated_and_ordered(self):
        self.assertEqual(
            convert_m3ed_spot._selected_sequences(["stairs", "easy_1"]), ["easy_1", "stairs"]
        )
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "no sequences selected"):
            convert_m3ed_spot._selected_sequences([])
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "duplicate sequence"):
            convert_m3ed_spot._selected_sequences(["stairs", "stairs"])
        with self.assertRaisesRegex(convert_m3ed_spot.ConversionError, "unknown sequence"):
            convert_m3ed_spot._selected_sequences(["nope"])


class TestReporterConfig(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.output = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def _config(self, sequences=None):
        selected = list(sequences or convert_m3ed_spot.ALL_SEQS)
        names = convert_m3ed_spot._write_configs(self.output, selected)
        return names, json.loads((self.output / names[0]).read_text())

    def test_one_config_covers_every_sequence_in_both_modes(self):
        names, config = self._config()
        self.assertEqual(names, ["m3ed_spot-vo_slam.cfg"])
        self.assertEqual(len(config["sequence_cfgs"]), 32)

    def test_dataset_folder_matches_the_registry_id(self):
        _, config = self._config()
        self.assertEqual(config["dataset_folder"], "m3ed_spot/")

    def test_config_name_derives_the_intended_kpi_prefix(self):
        names, _ = self._config()
        self.assertEqual(Path(names[0]).stem.split("-")[0].upper(), "M3ED_SPOT")

    def test_every_entry_names_its_edex_and_ground_truth(self):
        _, config = self._config()
        for entry in config["sequence_cfgs"]:
            self.assertEqual(entry["edex_file"], "stereo.edex")
            self.assertEqual(entry["gt_file_path"], "gt.txt")
            self.assertTrue(entry["enable"])

    def test_titles_pair_one_odom_and_one_slam_per_sequence(self):
        _, config = self._config(["art_plaza_loop"])
        self.assertEqual(
            [entry["sequence_title"] for entry in config["sequence_cfgs"]],
            ["M3ED-art-plaza-loop-ODOM", "M3ED-art-plaza-loop-SLAM"],
        )
        self.assertNotIn("use_slam", config["sequence_cfgs"][0])
        self.assertTrue(config["sequence_cfgs"][1]["use_slam"])

    def test_segment_lengths_match_the_outdoor_scale(self):
        _, config = self._config()
        self.assertEqual(config["segment_lengths"], [5, 10, 15, 20, 25, 50, 100])


if __name__ == "__main__":
    unittest.main()
