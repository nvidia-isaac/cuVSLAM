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

"""ICL-NUIM conversion: frame indexing, the world reflection, and emitted layout.

The converter copies image bytes without decoding them, so the synthetic
sequences below use placeholder file contents.
"""

import inspect
import json
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from cuvslam_tools.dataset_preparation import rgbd
from cuvslam_tools.dataset_preparation.icl_nuim import convert_icl_nuim
from cuvslam_tools.dataset_preparation.icl_nuim import prepare as icl_prepare


def _associations_text(indices):
    return "\n".join(f"{i} depth/{i}.png {i} rgb/{i}.png" for i in indices) + "\n"


def _pose_text(rows):
    lines = []
    for index, translation, quaternion in rows:
        values = " ".join(str(value) for value in list(translation) + list(quaternion))
        lines.append(f"{index} {values}")
    return "\n".join(lines) + "\n"


class TestSequenceSelection(unittest.TestCase):
    def test_default_is_the_eight_published_trajectories(self):
        self.assertEqual(len(convert_icl_nuim.ALL_SEQS), 8)
        self.assertEqual(len(set(convert_icl_nuim.SEQUENCE_NAMES)), 8)
        self.assertEqual(len(convert_icl_nuim.required_archives()), 8)

    def test_four_living_room_and_four_office_trajectories(self):
        titles = [spec.title for spec in convert_icl_nuim.ALL_SEQS]
        self.assertEqual(len([t for t in titles if t.startswith("living-room-")]), 4)
        self.assertEqual(len([t for t in titles if t.startswith("office-")]), 4)

    def test_office_titles_disambiguate_the_bare_archive_names(self):
        # "traj0_frei_png" is the office scene; the archive stem alone would not
        # say which room a report row came from.
        spec = convert_icl_nuim._spec("traj0_frei_png")
        self.assertEqual(spec.title, "office-traj0")
        self.assertEqual(spec.ground_truth, "traj0.gt.freiburg")

    def test_unknown_sequence_is_rejected(self):
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "unknown sequence"):
            convert_icl_nuim.required_archives(["living_room_traj9_frei_png"])

    def test_duplicate_sequence_is_rejected(self):
        name = convert_icl_nuim.SEQUENCE_NAMES[0]
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "duplicate sequence"):
            convert_icl_nuim.required_archives([name, name])

    def test_empty_selection_is_rejected(self):
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "no sequences selected"):
            convert_icl_nuim.required_archives([])

    def test_selection_order_follows_the_published_order(self):
        requested = ["traj0_frei_png", "living_room_traj1_frei_png"]
        self.assertEqual(
            convert_icl_nuim.required_archives(requested),
            ["living_room_traj1_frei_png.tar.gz", "traj0_frei_png.tar.gz"],
        )

    def test_downloader_lists_exactly_the_selected_archives(self):
        script = (
            Path(convert_icl_nuim.__file__).resolve().with_name("download_icl_nuim.sh")
        ).read_text(encoding="utf-8")
        listed = {
            line.strip().strip('"')
            for line in script.splitlines()
            if line.strip().startswith('"') and line.strip().endswith('.tar.gz"')
        }
        self.assertEqual(listed, set(convert_icl_nuim.required_archives()))


class TestWorldReflection(unittest.TestCase):
    """The published poses live in a y-up world; the images do not."""

    def test_translation_y_is_negated(self):
        translation, _ = convert_icl_nuim.flip_world_y([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(translation, [1.0, -2.0, 3.0])

    def test_rotation_is_conjugated_by_the_reflection(self):
        # The quaternion form must equal S R S with S = diag(1, -1, 1); anything
        # else silently mirrors the ground truth against the imagery.
        quaternion = [0.2, 0.3, -0.5, 0.7]
        norm = sum(component * component for component in quaternion) ** 0.5
        quaternion = [component / norm for component in quaternion]
        rotation = rgbd.quaternion_to_matrix(quaternion)
        expected = [
            [rotation[row][column] * (-1 if (row == 1) != (column == 1) else 1) for column in range(3)]
            for row in range(3)
        ]

        _, flipped = convert_icl_nuim.flip_world_y([0.0, 0.0, 0.0], quaternion)
        for row, (got, want) in enumerate(zip(rgbd.quaternion_to_matrix(flipped), expected)):
            for column, (a, b) in enumerate(zip(got, want)):
                self.assertAlmostEqual(a, b, places=12, msg=f"element ({row}, {column})")

    def test_applied_while_reading_the_pose_file(self):
        text = _pose_text([(1, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))])
        rows = convert_icl_nuim.read_indexed_trajectory(text, "gt")
        self.assertEqual(rows[0][1], [0.0, -1.0, 0.0])


class TestIndexedTrajectory(unittest.TestCase):
    def test_frame_index_maps_to_the_synthesized_frame_time(self):
        text = _pose_text([(0, (0, 0, 0), (0, 0, 0, 1)), (30, (0, 0, 0), (0, 0, 0, 1))])
        rows = convert_icl_nuim.read_indexed_trajectory(text, "gt")
        self.assertEqual(rows[0][0], 0)
        # Index 30 at 30 Hz is one second in.
        self.assertEqual(rows[1][0], 30 * convert_icl_nuim.NANOSECONDS_PER_FRAME)

    def test_fractional_frame_index_is_rejected(self):
        # The first column is a frame number, so a fractional value means the
        # file is not the pose format this converter expects.
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "must be an integer"):
            convert_icl_nuim.read_indexed_trajectory("0.5 0 0 0 0 0 0 1\n", "gt")


class TestAssociations(unittest.TestCase):
    def test_parses_frame_index_to_media_paths(self):
        associations = convert_icl_nuim.read_associations(_associations_text([1, 2]), "assoc")
        self.assertEqual(associations[1], ("rgb/1.png", "depth/1.png"))
        self.assertEqual(associations[2], ("rgb/2.png", "depth/2.png"))

    def test_wrong_column_count_is_rejected(self):
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "expected 4 columns"):
            convert_icl_nuim.read_associations("1 depth/1.png 1\n", "assoc")

    def test_mismatched_keys_are_rejected(self):
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "does not match"):
            convert_icl_nuim.read_associations("1 depth/1.png 2 rgb/2.png\n", "assoc")

    def test_duplicate_frame_is_rejected(self):
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "duplicate frame"):
            convert_icl_nuim.read_associations(_associations_text([1, 1]), "assoc")

    def test_unexpected_media_path_is_rejected(self):
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "unexpected media path"):
            convert_icl_nuim.read_associations("1 depth/1.jpg 1 rgb/1.png\n", "assoc")

    def test_traversing_media_path_is_rejected(self):
        # convert_sequence joins these paths onto the extracted sequence root, so
        # the pattern is what keeps a repackaged archive from reaching outside it.
        for path in ("../../etc/passwd", "rgb/../../1.png", "/abs/1.png"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    convert_icl_nuim.ConversionError, "unexpected media path"
                ):
                    convert_icl_nuim.read_associations(f"1 {path} 1 rgb/1.png\n", "assoc")

    def test_empty_file_is_rejected(self):
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "no associations"):
            convert_icl_nuim.read_associations("# nothing\n", "assoc")


class TestConvertSequence(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.spec = convert_icl_nuim._spec("traj0_frei_png")

    def tearDown(self):
        self._temporary.cleanup()

    def _write_source(self, image_indices=(1, 2, 3, 4), pose_indices=(2, 3, 4)):
        source = self.root / "source"
        (source / "rgb").mkdir(parents=True)
        (source / "depth").mkdir(parents=True)
        for index in image_indices:
            (source / "rgb" / f"{index}.png").write_bytes(f"color-{index}".encode())
            (source / "depth" / f"{index}.png").write_bytes(f"depth-{index}".encode())
        (source / "associations.txt").write_text(
            _associations_text(image_indices), encoding="utf-8"
        )
        (source / self.spec.ground_truth).write_text(
            _pose_text(
                [(index, (index * 0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)) for index in pose_indices]
            ),
            encoding="utf-8",
        )
        return source

    def test_frames_without_a_pose_are_dropped(self):
        # The office scenes render from index 1 but their poses start at 2.
        source = self._write_source()
        output = self.root / "out"
        metadata = convert_icl_nuim.convert_sequence(source, self.spec, output)

        self.assertEqual(metadata["converted_counts"]["frames"], 3)
        self.assertEqual(metadata["first_source_frame_index"], 2)
        self.assertEqual((output / self.spec.name / "00" / "000000.png").read_bytes(), b"color-2")

    def test_emits_the_expected_layout(self):
        source = self._write_source()
        output = self.root / "out"
        convert_icl_nuim.convert_sequence(source, self.spec, output)

        sequence_dir = output / self.spec.name
        self.assertEqual(
            sorted(entry.name for entry in sequence_dir.iterdir()),
            ["00", "01", "frame_metadata.jsonl", "gt.txt", "stereo.edex"],
        )
        # Colour lands in 00 and depth in 01, copied verbatim.
        self.assertEqual((sequence_dir / "01" / "000001.png").read_bytes(), b"depth-3")

    def test_frame_metadata_uses_synthesized_thirty_hertz_timestamps(self):
        source = self._write_source()
        output = self.root / "out"
        convert_icl_nuim.convert_sequence(source, self.spec, output)

        lines = (output / self.spec.name / "frame_metadata.jsonl").read_text().splitlines()
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        step = second["cams"][0]["timestamp"] - first["cams"][0]["timestamp"]
        self.assertEqual(step, convert_icl_nuim.NANOSECONDS_PER_FRAME)
        # Colour and depth are the same rendered frame, so they share a timestamp.
        self.assertEqual(first["cams"][0]["timestamp"], first["depth"][0]["timestamp"])

    def test_timestamps_are_reproducible(self):
        # The retired converter stamped frames with the wall clock at conversion
        # time, so its output changed on every run.
        source = self._write_source()
        first = self.root / "first"
        second = self.root / "second"
        convert_icl_nuim.convert_sequence(source, self.spec, first)
        convert_icl_nuim.convert_sequence(source, self.spec, second)
        self.assertEqual(
            (first / self.spec.name / "frame_metadata.jsonl").read_bytes(),
            (second / self.spec.name / "frame_metadata.jsonl").read_bytes(),
        )

    def test_edex_declares_the_icl_camera(self):
        source = self._write_source()
        output = self.root / "out"
        convert_icl_nuim.convert_sequence(source, self.spec, output)

        rig, metadata = json.loads((output / self.spec.name / "stereo.edex").read_text())
        camera = rig["cameras"][0]
        self.assertEqual(camera["intrinsics"]["focal"], [481.20, 480.00])
        self.assertEqual(camera["intrinsics"]["principal"], [319.50, 239.50])
        self.assertEqual(camera["intrinsics"]["size"], [640, 480])
        # Positive fy: the published K uses -480 for the POVRay-native rendering,
        # whose rows run bottom-up, not for these top-down PNGs.
        self.assertGreater(camera["intrinsics"]["focal"][1], 0)
        self.assertEqual(camera["depth_scale_factor"], 5000.0)
        self.assertEqual(rig["frame_end"], 2)
        self.assertEqual(metadata["depth_sequence"], [["01/000000.png"]])

    def test_ground_truth_has_one_exact_row_per_frame(self):
        source = self._write_source()
        output = self.root / "out"
        convert_icl_nuim.convert_sequence(source, self.spec, output)

        rows = (output / self.spec.name / "gt.txt").read_text().splitlines()
        self.assertEqual(len(rows), 3)
        # Poses are exact per frame, so frame 1 sits exactly 0.1 m along x.
        second = [float(value) for value in rows[1].split()]
        self.assertAlmostEqual(second[3], 0.1, places=9)

    def test_posed_frame_with_no_association_is_reported(self):
        source = self._write_source(image_indices=(1, 2), pose_indices=(2, 3))
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "no entry for 1 posed"):
            convert_icl_nuim.convert_sequence(source, self.spec, self.root / "out")

    def test_missing_media_file_is_reported(self):
        source = self._write_source()
        (source / "rgb" / "3.png").unlink()
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "missing 1 media file"):
            convert_icl_nuim.convert_sequence(source, self.spec, self.root / "out")

    def test_missing_pose_file_is_reported(self):
        source = self._write_source()
        (source / self.spec.ground_truth).unlink()
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "missing traj0.gt.freiburg"):
            convert_icl_nuim.convert_sequence(source, self.spec, self.root / "out")

    def test_missing_associations_file_is_reported(self):
        source = self._write_source()
        (source / "associations.txt").unlink()
        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "missing associations.txt"):
            convert_icl_nuim.convert_sequence(source, self.spec, self.root / "out")

    def test_symlinked_output_directory_is_refused(self):
        # Converting would otherwise delete whatever the link points at.
        source = self._write_source()
        output = self.root / "out"
        output.mkdir()
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "keep.txt").write_text("keep", encoding="utf-8")
        (output / self.spec.name).symlink_to(elsewhere, target_is_directory=True)

        with self.assertRaisesRegex(convert_icl_nuim.ConversionError, "symlinked output"):
            convert_icl_nuim.convert_sequence(source, self.spec, output)
        self.assertTrue((elsewhere / "keep.txt").is_file())

    def test_reconversion_removes_stale_frames(self):
        # A shorter second run must not leave frames from the longer first one,
        # which would be listed by neither frame_metadata.jsonl nor gt.txt.
        output = self.root / "out"
        long_source = self._write_source(image_indices=(1, 2, 3, 4), pose_indices=(2, 3, 4))
        convert_icl_nuim.convert_sequence(long_source, self.spec, output)
        self.assertEqual(len(list((output / self.spec.name / "00").iterdir())), 3)

        shutil.rmtree(long_source)
        short_source = self._write_source(image_indices=(1, 2, 3), pose_indices=(2, 3))
        convert_icl_nuim.convert_sequence(short_source, self.spec, output)

        frames = sorted(path.name for path in (output / self.spec.name / "00").iterdir())
        self.assertEqual(frames, ["000000.png", "000001.png"])
        self.assertEqual(
            len((output / self.spec.name / "gt.txt").read_text().splitlines()), len(frames)
        )


class TestReporterConfig(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.output = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def _config(self):
        names = convert_icl_nuim._write_configs(self.output, list(convert_icl_nuim.ALL_SEQS))
        return names, json.loads((self.output / names[0]).read_text())

    def test_one_config_covers_every_trajectory_in_both_modes(self):
        names, config = self._config()
        self.assertEqual(names, ["icl_nuim-rgbd_slam.cfg"])
        self.assertEqual(len(config["sequence_cfgs"]), 16)

    def test_dataset_folder_matches_the_registry_id(self):
        _, config = self._config()
        self.assertEqual(config["dataset_folder"], "icl_nuim/")

    def test_config_name_derives_the_intended_kpi_prefix(self):
        # The collector takes everything before the first hyphen, so the retired
        # "icl-nuim.cfg" produced "ICL" while this produces "ICL_NUIM".
        names, _ = self._config()
        self.assertEqual(Path(names[0]).stem.split("-")[0].upper(), "ICL_NUIM")

    def test_every_entry_names_its_edex_and_ground_truth(self):
        _, config = self._config()
        for entry in config["sequence_cfgs"]:
            self.assertEqual(entry["edex_file"], "stereo.edex")
            self.assertEqual(entry["gt_file_path"], "gt.txt")
            self.assertTrue(entry["enable"])

    def test_titles_match_the_retired_report_rows(self):
        _, config = self._config()
        titles = [entry["sequence_title"] for entry in config["sequence_cfgs"]]
        self.assertEqual(titles[:2], ["living-room-traj0-ODOM", "living-room-traj0-SLAM"])
        self.assertEqual(titles[-2:], ["office-traj3-ODOM", "office-traj3-SLAM"])


class TestPreparationContract(unittest.TestCase):
    """The shared contract test only covers registered datasets.

    ICL-NUIM is not in the registry yet — registering provision targets is a
    later change — so the same contract is asserted here rather than widened
    there, where it would claim the dataset is provisionable.
    """

    def test_prepare_and_main_match_the_shared_contract(self):
        self.assertTrue(callable(icl_prepare.prepare))
        self.assertTrue(callable(icl_prepare.main))
        parameters = inspect.signature(icl_prepare.prepare).parameters
        self.assertLessEqual(
            {"raw_dir", "output_dir", "force_download", "download_only"}, set(parameters)
        )
        for name, parameter in parameters.items():
            self.assertIsNot(parameter.default, inspect.Parameter.empty, name)

    def test_provisioning_arguments_are_accepted(self):
        arguments = ["--raw-dir", "/raw", "--output-dir", "/converted", "--force-download"]
        with unittest.mock.patch.object(icl_prepare, "prepare") as prepare:
            self.assertEqual(icl_prepare.main(arguments), 0)
        self.assertEqual(prepare.call_args.kwargs["raw_dir"], Path("/raw"))
        self.assertEqual(prepare.call_args.kwargs["output_dir"], Path("/converted"))
        self.assertTrue(prepare.call_args.kwargs["force_download"])


class TestSequenceDiscovery(unittest.TestCase):
    """ICL archives hold rgb/ and depth/ at the root, unlike TUM's nested layout."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self):
        self._temporary.cleanup()

    def _extracted(self, prefix=""):
        destination = self.root / "out"
        base = destination / prefix if prefix else destination
        (base / "rgb").mkdir(parents=True)
        (base / "depth").mkdir(parents=True)
        return destination

    def test_root_layout_is_accepted(self):
        destination = self._extracted()
        with unittest.mock.patch.object(icl_prepare, "extract_tar_archive"):
            found = icl_prepare.extract_sequence(Path("archive.tar.gz"), destination)
        self.assertEqual(found, destination)

    def test_nested_layout_is_accepted(self):
        destination = self._extracted(prefix="traj0_frei_png")
        with unittest.mock.patch.object(icl_prepare, "extract_tar_archive"):
            found = icl_prepare.extract_sequence(Path("archive.tar.gz"), destination)
        self.assertEqual(found.name, "traj0_frei_png")

    def test_archive_without_both_streams_is_reported(self):
        destination = self.root / "out"
        (destination / "rgb").mkdir(parents=True)
        with unittest.mock.patch.object(icl_prepare, "extract_tar_archive"):
            with self.assertRaisesRegex(icl_prepare.PreparationError, "rgb/ and depth/"):
                icl_prepare.extract_sequence(Path("archive.tar.gz"), destination)


if __name__ == "__main__":
    unittest.main()
