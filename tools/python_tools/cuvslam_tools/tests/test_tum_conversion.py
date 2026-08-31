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

"""TUM RGB-D conversion: association, ground-truth handling, and emitted layout.

The converter copies image bytes without decoding them, so the synthetic
sequences below use placeholder file contents.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from cuvslam_tools.dataset_preparation import rgbd
from cuvslam_tools.dataset_preparation.tum import convert_tum

MILLISECOND_NS = 1_000_000


def _index_text(entries):
    lines = ["# color images", "# file: 'synthetic'", "# timestamp filename"]
    lines += [f"{timestamp} {path}" for timestamp, path in entries]
    return "\n".join(lines) + "\n"


def _trajectory_text(rows):
    lines = ["# ground truth trajectory", "# timestamp tx ty tz qx qy qz qw"]
    for timestamp, translation, quaternion in rows:
        values = " ".join(f"{value}" for value in list(translation) + list(quaternion))
        lines.append(f"{timestamp} {values}")
    return "\n".join(lines) + "\n"


class TestTimestampIndex(unittest.TestCase):
    def test_comments_and_blank_lines_are_skipped(self):
        text = "# header\n\n1.000000 rgb/a.png\n1.033333 rgb/b.png\n"
        entries = rgbd.read_timestamp_index(text, "rgb.txt")
        self.assertEqual(entries, [(1_000_000_000, "rgb/a.png"), (1_033_333_000, "rgb/b.png")])

    def test_timestamp_scales_exactly_to_nanoseconds(self):
        # Parsed as Decimal, so the value does not depend on binary float rounding.
        entries = rgbd.read_timestamp_index("1341847980.722988 rgb/a.png\n", "rgb.txt")
        self.assertEqual(entries[0][0], 1_341_847_980_722_988_000)

    def test_wrong_column_count_is_rejected(self):
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "expected 2 columns"):
            rgbd.read_timestamp_index("1.0 rgb/a.png extra\n", "rgb.txt")

    def test_non_increasing_timestamps_are_rejected(self):
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "strictly increasing"):
            rgbd.read_timestamp_index("2.0 rgb/a.png\n1.0 rgb/b.png\n", "rgb.txt")

    def test_non_finite_timestamps_are_rejected(self):
        # Decimal accepts these, and converting them to an integer raises
        # outside the module's error contract unless they are caught first.
        for text in ("inf", "-inf", "Infinity", "nan"):
            with self.subTest(timestamp=text):
                with self.assertRaises(rgbd.RgbdConversionError):
                    rgbd.read_timestamp_index(f"{text} rgb/a.png\n", "rgb.txt")

    def test_empty_index_is_rejected(self):
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "no entries"):
            rgbd.read_timestamp_index("# only a comment\n", "rgb.txt")


class TestTrajectory(unittest.TestCase):
    def test_quaternion_is_normalized(self):
        text = "1.0 0 0 0 0 0 0 2\n"
        rows = rgbd.read_tum_trajectory(text, "gt")
        self.assertEqual(rows[0][2], [0.0, 0.0, 0.0, 1.0])

    def test_wrong_column_count_is_rejected(self):
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "expected 8 columns"):
            rgbd.read_tum_trajectory("1.0 0 0 0 0 0 0\n", "gt")

    def test_zero_length_quaternion_is_rejected(self):
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "zero-length quaternion"):
            rgbd.read_tum_trajectory("1.0 0 0 0 0 0 0 0\n", "gt")

    def test_extreme_quaternion_magnitudes_still_normalize(self):
        # Squaring these overflows to infinity and underflows to zero
        # respectively. Either one silently yields an all-zero quaternion, which
        # quaternion_to_matrix then reads as an identity rotation.
        for magnitude in ("1e308", "1e-200"):
            with self.subTest(magnitude=magnitude):
                rows = rgbd.read_tum_trajectory(f"1.0 0 0 0 {magnitude} 0 0 0\n", "gt")
                self.assertEqual(rows[0][2], [1.0, 0.0, 0.0, 0.0])
                self.assertEqual(
                    rgbd.quaternion_to_matrix(rows[0][2]),
                    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                )

    def test_non_finite_pose_values_are_rejected(self):
        # float() accepts these, and they would otherwise reach every
        # interpolated pose as silently wrong ground truth.
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "non-finite pose value"):
            rgbd.read_tum_trajectory("1.0 inf 0 0 0 0 0 1\n", "gt")
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "non-finite pose value"):
            rgbd.read_tum_trajectory("1.0 0 0 0 nan 0 0 1\n", "gt")


class TestAssociation(unittest.TestCase):
    def test_closest_pair_wins_over_earlier_candidate(self):
        # Depth at 5 ms and 11 ms; colour at 10 ms. Accepting the first candidate
        # in stream order would take 5 ms, but 11 ms is closer.
        color = [(10 * MILLISECOND_NS, "rgb/a.png")]
        depth = [(5 * MILLISECOND_NS, "depth/a.png"), (11 * MILLISECOND_NS, "depth/b.png")]
        matched = rgbd.associate(color, depth, 20 * MILLISECOND_NS)
        self.assertEqual([pair[3] for pair in matched], ["depth/b.png"])

    def test_each_frame_is_used_at_most_once(self):
        color = [(10 * MILLISECOND_NS, "rgb/a.png"), (11 * MILLISECOND_NS, "rgb/b.png")]
        depth = [(10 * MILLISECOND_NS, "depth/a.png")]
        matched = rgbd.associate(color, depth, 20 * MILLISECOND_NS)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][1], "rgb/a.png")

    def test_tolerance_boundary_is_exclusive(self):
        color = [(0, "rgb/a.png")]
        depth = [(MILLISECOND_NS, "depth/a.png")]
        with self.assertRaises(rgbd.RgbdConversionError):
            rgbd.associate(color, depth, MILLISECOND_NS)
        self.assertEqual(len(rgbd.associate(color, depth, MILLISECOND_NS + 1)), 1)

    def test_result_is_sorted_by_colour_timestamp(self):
        # Offsets shrink along the sequence, so closest-first consumes the last
        # colour frame first and the output order comes from the final sort
        # rather than the scan order. Both indices are sorted, as
        # read_timestamp_index guarantees for real input.
        color = [(0, "rgb/a.png"), (100_000, "rgb/b.png"), (200_000, "rgb/c.png")]
        depth = [(50_000, "depth/a.png"), (130_000, "depth/b.png"), (200_000, "depth/c.png")]
        matched = rgbd.associate(color, depth, MILLISECOND_NS)
        self.assertEqual([pair[0] for pair in matched], [0, 100_000, 200_000])
        self.assertEqual(
            [pair[3] for pair in matched],
            ["depth/a.png", "depth/b.png", "depth/c.png"],
        )

    def test_window_pairs_every_frame_when_the_tolerance_is_tight(self):
        # A tolerance on the scale of the data, rather than one that spans the
        # whole index, is what exercises the sliding window.
        color = [(0, "rgb/a.png"), (100_000, "rgb/b.png"), (200_000, "rgb/c.png")]
        depth = [(1_000, "depth/a.png"), (101_000, "depth/b.png"), (201_000, "depth/c.png")]
        matched = rgbd.associate(color, depth, 2_000)
        self.assertEqual(len(matched), 3)
        self.assertEqual([pair[1] for pair in matched], ["rgb/a.png", "rgb/b.png", "rgb/c.png"])

    def test_non_positive_tolerance_is_rejected(self):
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "tolerance must be positive"):
            rgbd.associate([(0, "a")], [(0, "b")], 0)


class TestGroundTruthSampling(unittest.TestCase):
    def _trajectory(self):
        # Rotate 90 degrees about z between the two samples, translating in x.
        half = math.sin(math.radians(45))
        return [
            (0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            (1_000_000_000, [2.0, 0.0, 0.0], [0.0, 0.0, half, half]),
        ]

    def test_exact_sample_is_returned_unchanged(self):
        trajectory = self._trajectory()
        rotation, translation = rgbd.interpolate_pose(trajectory, [row[0] for row in trajectory], 0)
        self.assertEqual(translation, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(rotation[0][0], 1.0)

    def test_midpoint_interpolates_translation_and_rotation(self):
        trajectory = self._trajectory()
        rotation, translation = rgbd.interpolate_pose(
            trajectory, [row[0] for row in trajectory], 500_000_000
        )
        self.assertAlmostEqual(translation[0], 1.0)
        # Half of a 90 degree turn about z.
        self.assertAlmostEqual(math.degrees(math.atan2(rotation[1][0], rotation[0][0])), 45.0)

    def test_outside_range_clamps_to_the_end_samples(self):
        trajectory = self._trajectory()
        timestamps = [row[0] for row in trajectory]
        _, before = rgbd.interpolate_pose(trajectory, timestamps, -1_000)
        _, after = rgbd.interpolate_pose(trajectory, timestamps, 2_000_000_000)
        self.assertEqual(before, [0.0, 0.0, 0.0])
        self.assertEqual(after, [2.0, 0.0, 0.0])

    def test_first_row_is_exactly_the_identity(self):
        trajectory = self._trajectory()
        pairs = [(0, "a", 0, "a"), (1_000_000_000, "b", 1_000_000_000, "b")]
        lines = rgbd.relative_ground_truth_lines(trajectory, pairs)
        self.assertEqual(len(lines), 2)
        self.assertEqual([float(value) for value in lines[0].split()][0], 1.0)
        self.assertEqual([float(value) for value in lines[0].split()][3], 0.0)

    def test_poses_are_relative_to_the_first_frame(self):
        # Frame 0 sits at the midpoint, so frame 1 must be 1 m ahead of it, not 2.
        trajectory = self._trajectory()
        pairs = [(500_000_000, "a", 500_000_000, "a"), (1_000_000_000, "b", 1_000_000_000, "b")]
        lines = rgbd.relative_ground_truth_lines(trajectory, pairs)
        values = [float(value) for value in lines[1].split()]
        translation = [values[3], values[7], values[11]]
        self.assertAlmostEqual(math.hypot(*translation[:2]), 1.0, places=6)

    def test_frames_outside_the_trajectory_span_are_dropped(self):
        trajectory = self._trajectory()
        pairs = [
            (-5, "early", -5, "early"),
            (10, "inside", 10, "inside"),
            (2_000_000_000, "late", 2_000_000_000, "late"),
        ]
        kept = rgbd.restrict_to_trajectory(pairs, trajectory)
        self.assertEqual([pair[1] for pair in kept], ["inside"])

    def test_no_frame_inside_the_span_is_rejected(self):
        trajectory = self._trajectory()
        with self.assertRaisesRegex(rgbd.RgbdConversionError, "inside the ground-truth"):
            rgbd.restrict_to_trajectory([(2_000_000_000, "late", 2_000_000_000, "late")], trajectory)


class TestSequenceSelection(unittest.TestCase):
    def test_default_is_the_fifteen_evaluated_sequences(self):
        self.assertEqual(len(convert_tum.ALL_SEQS), 15)
        self.assertEqual(len(set(convert_tum.ALL_SEQS)), 15)
        self.assertEqual(len(convert_tum.required_archives()), 15)

    def test_every_sequence_is_from_the_freiburg3_series(self):
        # One intrinsics set covers the selection only because all of it is fr3.
        for sequence in convert_tum.ALL_SEQS:
            self.assertTrue(sequence.startswith("rgbd_dataset_freiburg3_"), sequence)

    def test_unknown_sequence_is_rejected(self):
        with self.assertRaisesRegex(convert_tum.ConversionError, "unknown sequence"):
            convert_tum.required_archives(["rgbd_dataset_freiburg1_desk"])

    def test_duplicate_sequence_is_rejected(self):
        name = convert_tum.ALL_SEQS[0]
        with self.assertRaisesRegex(convert_tum.ConversionError, "duplicate sequence"):
            convert_tum.required_archives([name, name])

    def test_empty_selection_is_rejected(self):
        with self.assertRaisesRegex(convert_tum.ConversionError, "no sequences selected"):
            convert_tum.required_archives([])

    def test_selection_order_follows_the_report_order(self):
        requested = [convert_tum.ALL_SEQS[3], convert_tum.ALL_SEQS[1]]
        self.assertEqual(
            convert_tum.required_archives(requested),
            [convert_tum.archive_name(convert_tum.ALL_SEQS[1]),
             convert_tum.archive_name(convert_tum.ALL_SEQS[3])],
        )

    def test_downloader_lists_exactly_the_selected_archives(self):
        script = (
            Path(convert_tum.__file__).resolve().with_name("download_tum.sh")
        ).read_text(encoding="utf-8")
        listed = {
            line.strip().strip('"')
            for line in script.splitlines()
            if line.strip().startswith('"rgbd_dataset_freiburg3_')
        }
        self.assertEqual(listed, set(convert_tum.required_archives()))


class TestConvertSequence(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.sequence = convert_tum.ALL_SEQS[0]

    def tearDown(self):
        self._temporary.cleanup()

    def _write_source(self, frame_count=4, step_ns=33_000_000):
        source = self.root / "source"
        (source / "rgb").mkdir(parents=True)
        (source / "depth").mkdir(parents=True)
        color_entries = []
        depth_entries = []
        for index in range(frame_count):
            timestamp = 1_000_000_000 + index * step_ns
            color_name = f"rgb/{timestamp}.png"
            depth_name = f"depth/{timestamp}.png"
            (source / color_name).write_bytes(b"color-" + str(index).encode())
            (source / depth_name).write_bytes(b"depth-" + str(index).encode())
            color_entries.append((_seconds(timestamp), color_name))
            # Depth lags colour by 20 us, as the real streams do.
            depth_entries.append((_seconds(timestamp + 20_000), depth_name))
        (source / "rgb.txt").write_text(_index_text(color_entries), encoding="utf-8")
        (source / "depth.txt").write_text(_index_text(depth_entries), encoding="utf-8")

        span = step_ns * (frame_count - 1)
        rows = [
            (_seconds(1_000_000_000 - step_ns), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            (_seconds(1_000_000_000 + span + step_ns), [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ]
        (source / "groundtruth.txt").write_text(_trajectory_text(rows), encoding="utf-8")
        return source

    def test_emits_the_expected_layout(self):
        source = self._write_source()
        output = self.root / "out"
        metadata = convert_tum.convert_sequence(source, self.sequence, output)

        sequence_dir = output / self.sequence
        self.assertEqual(metadata["converted_counts"]["frames"], 4)
        self.assertEqual(
            sorted(entry.name for entry in sequence_dir.iterdir()),
            ["00", "01", "frame_metadata.jsonl", "gt.txt", "stereo.edex"],
        )
        self.assertEqual(
            sorted(entry.name for entry in (sequence_dir / "00").iterdir()),
            [f"{index:06d}.png" for index in range(4)],
        )
        # Colour lands in 00 and depth in 01, and the bytes are copied verbatim.
        self.assertEqual((sequence_dir / "00" / "000002.png").read_bytes(), b"color-2")
        self.assertEqual((sequence_dir / "01" / "000002.png").read_bytes(), b"depth-2")

    def test_frame_metadata_pairs_colour_with_depth(self):
        source = self._write_source()
        output = self.root / "out"
        convert_tum.convert_sequence(source, self.sequence, output)

        lines = (output / self.sequence / "frame_metadata.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 4)
        first = json.loads(lines[0])
        self.assertEqual(first["frame_id"], 0)
        self.assertEqual(first["cams"], [{"filename": "00/000000.png", "id": 0, "timestamp": 1_000_000_000}])
        self.assertEqual(
            first["depth"], [{"filename": "01/000000.png", "id": 0, "timestamp": 1_000_020_000}]
        )

    def test_edex_describes_one_rgbd_camera(self):
        source = self._write_source()
        output = self.root / "out"
        convert_tum.convert_sequence(source, self.sequence, output)

        document = json.loads((output / self.sequence / "stereo.edex").read_text())
        rig, metadata = document
        self.assertEqual(rig["frame_start"], 0)
        self.assertEqual(rig["frame_end"], 3)
        camera = rig["cameras"][0]
        self.assertEqual(camera["intrinsics"]["focal"], [535.4, 539.2])
        self.assertEqual(camera["intrinsics"]["principal"], [320.1, 247.6])
        self.assertEqual(camera["intrinsics"]["size"], [640, 480])
        self.assertEqual(camera["intrinsics"]["distortion_model"], "pinhole")
        self.assertEqual(camera["depth_id"], 0)
        # 16-bit PNG depth in TUM units. Without this the reader would divide by
        # 1, and depth would be read as thousands of metres.
        self.assertEqual(camera["depth_scale_factor"], 5000.0)
        self.assertEqual(metadata["frame_metadata"], "frame_metadata.jsonl")
        self.assertEqual(metadata["depth_sequence"], [["01/000000.png"]])

    def test_ground_truth_has_one_row_per_frame(self):
        source = self._write_source()
        output = self.root / "out"
        convert_tum.convert_sequence(source, self.sequence, output)

        rows = (output / self.sequence / "gt.txt").read_text().splitlines()
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(len(row.split()), 12)

    def test_missing_index_file_is_reported(self):
        source = self._write_source()
        (source / "groundtruth.txt").unlink()
        with self.assertRaisesRegex(convert_tum.ConversionError, "missing groundtruth.txt"):
            convert_tum.convert_sequence(source, self.sequence, self.root / "out")

    def test_missing_image_is_reported(self):
        source = self._write_source()
        (source / "depth").joinpath(sorted(p.name for p in (source / "depth").iterdir())[0]).unlink()
        with self.assertRaisesRegex(convert_tum.ConversionError, "missing image"):
            convert_tum.convert_sequence(source, self.sequence, self.root / "out")

    def test_rerunning_replaces_the_previous_output(self):
        source = self._write_source()
        output = self.root / "out"
        convert_tum.convert_sequence(source, self.sequence, output)
        stale = output / self.sequence / "00" / "999999.png"
        stale.write_bytes(b"stale")
        convert_tum.convert_sequence(source, self.sequence, output)
        self.assertFalse(stale.exists())


class TestReporterConfig(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.output = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def _config(self, sequences=None):
        selected = list(sequences or convert_tum.ALL_SEQS)
        names = convert_tum._write_configs(self.output, selected)
        return names, json.loads((self.output / names[0]).read_text())

    def test_one_config_covers_every_sequence_in_both_modes(self):
        names, config = self._config()
        self.assertEqual(names, ["tum-rgbd_slam.cfg"])
        self.assertEqual(len(config["sequence_cfgs"]), 30)

    def test_dataset_folder_matches_the_registry_id(self):
        _, config = self._config()
        self.assertEqual(config["dataset_folder"], "tum/")

    def test_config_name_derives_the_intended_kpi_prefix(self):
        names, _ = self._config()
        # The KPI collector takes everything before the first hyphen, uppercased.
        self.assertEqual(Path(names[0]).stem.split("-")[0].upper(), "TUM")

    def test_every_entry_names_its_edex_and_ground_truth(self):
        # The retired config omitted gt_file_path, which makes the reporter run
        # the sequence with no ground truth instead of failing.
        _, config = self._config()
        for entry in config["sequence_cfgs"]:
            self.assertEqual(entry["edex_file"], "stereo.edex")
            self.assertEqual(entry["gt_file_path"], "gt.txt")
            self.assertTrue(entry["enable"])

    def test_titles_pair_one_odom_and_one_slam_per_sequence(self):
        _, config = self._config([convert_tum.ALL_SEQS[7]])
        titles = [entry["sequence_title"] for entry in config["sequence_cfgs"]]
        self.assertEqual(titles, ["freiburg3-sitting-xyz-ODOM", "freiburg3-sitting-xyz-SLAM"])
        self.assertNotIn("use_slam", config["sequence_cfgs"][0])
        self.assertTrue(config["sequence_cfgs"][1]["use_slam"])

    def test_segment_lengths_match_the_other_converters(self):
        _, config = self._config()
        self.assertEqual(config["segment_lengths"], [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 35, 45])


def _seconds(nanoseconds):
    """Render nanoseconds as a TUM-style fixed-point seconds string."""
    return f"{nanoseconds // 1_000_000_000}.{nanoseconds % 1_000_000_000:09d}"


if __name__ == "__main__":
    unittest.main()
