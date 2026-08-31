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

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from cuvslam_tools.dataset_preparation.euroc import convert_euroc
from cuvslam_tools.dataset_preparation.euroc import prepare as euroc_prepare


_CAMERA_X_IN_BODY = [
    convert_euroc._R_BC0[0][0],
    convert_euroc._R_BC0[1][0],
    convert_euroc._R_BC0[2][0],
]


def _camera_csv(entries):
    lines = ["#timestamp [ns],filename"]
    lines.extend(f"{timestamp},{filename}" for timestamp, filename in entries)
    return "\n".join(lines) + "\n"


def _default_ground_truth():
    end_position = [2.0 * value for value in _CAMERA_X_IN_BODY]
    return (
        "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n"
        "100,0,0,0,1,0,0,0\n"
        f"300,{end_position[0]},{end_position[1]},{end_position[2]},1,0,0,0\n"
    )


def _default_imu():
    return (
        "#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n"
        "90,1,2,3,4,5,6\n"
        "100,2,3,4,5,6,7\n"
        "150,3,4,5,6,7,8\n"
        "250,4,5,6,7,8,9\n"
        "300,5,6,7,8,9,10\n"
        "310,6,7,8,9,10,11\n"
    )


def _default_camera_entries():
    cam0 = [
        (50, "cam0-50.png"),
        (100, "cam0-100.png"),
        (150, "cam0-only.png"),
        (200, "cam0-200.png"),
        (300, "cam0-300.png"),
        (350, "cam0-350.png"),
    ]
    cam1 = [
        (50, "cam1-50.png"),
        (100, "cam1-100.png"),
        (200, "cam1-200.png"),
        (250, "cam1-only.png"),
        (300, "cam1-300.png"),
        (350, "cam1-350.png"),
    ]
    return cam0, cam1


def _inner_archive_bytes(
    cam0_entries=None,
    cam1_entries=None,
    cam0_csv=None,
    imu_csv=None,
    omit_member=None,
    omit_image=None,
):
    default_cam0, default_cam1 = _default_camera_entries()
    cam0_entries = default_cam0 if cam0_entries is None else cam0_entries
    cam1_entries = default_cam1 if cam1_entries is None else cam1_entries
    members = {
        "mav0/cam0/data.csv": _camera_csv(cam0_entries) if cam0_csv is None else cam0_csv,
        "mav0/cam1/data.csv": _camera_csv(cam1_entries),
        "mav0/imu0/data.csv": _default_imu() if imu_csv is None else imu_csv,
        "mav0/state_groundtruth_estimate0/data.csv": _default_ground_truth(),
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, contents in members.items():
            if name != omit_member:
                archive.writestr(name, contents)
        for camera, entries in (("cam0", cam0_entries), ("cam1", cam1_entries)):
            for timestamp, filename in entries:
                if filename == omit_image:
                    continue
                try:
                    convert_euroc._validate_safe_archive_path(filename, "test image")
                except convert_euroc.ConversionError:
                    continue
                archive.writestr(
                    f"mav0/{camera}/data/{filename}",
                    f"{camera}-image-{timestamp}".encode(),
                )
    return stream.getvalue()


def _write_outer_archive(raw_dir, archive_name, sequence_archives, extra_members=None):
    raw_dir.mkdir(parents=True, exist_ok=True)
    group = archive_name[:-4]
    with zipfile.ZipFile(raw_dir / archive_name, "w", zipfile.ZIP_DEFLATED) as archive:
        for sequence, inner_contents in sequence_archives.items():
            archive.writestr(f"{group}/{sequence}/{sequence}.zip", inner_contents)
        for name, contents in extra_members or []:
            archive.writestr(name, contents)


def _read_json_lines(path):
    if not path.read_text(encoding="utf-8"):
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestEurocConvertedSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.raw = cls.root / "raw"
        cls.output = cls.root / "converted" / "euroc"
        _write_outer_archive(
            cls.raw,
            "machine_hall.zip",
            {"MH_01_easy": _inner_archive_bytes()},
        )
        cls.metadata = convert_euroc.convert(
            cls.raw,
            cls.output,
            ["MH_01_easy"],
        )
        cls.sequence_dir = cls.output / "MH_01_easy"

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_exact_frame_intersection_and_portable_loose_media(self):
        metadata = _read_json_lines(self.sequence_dir / "frame_metadata.jsonl")
        self.assertEqual([frame["cams"][0]["timestamp"] for frame in metadata], [100, 200, 300])
        self.assertEqual([frame["cams"][1]["timestamp"] for frame in metadata], [100, 200, 300])
        self.assertEqual(
            sorted(path.name for path in (self.sequence_dir / "00").iterdir()),
            ["l.000000.png", "l.000001.png", "l.000002.png"],
        )
        self.assertEqual(
            sorted(path.name for path in (self.sequence_dir / "01").iterdir()),
            ["r.000000.png", "r.000001.png", "r.000002.png"],
        )
        self.assertEqual(
            (self.sequence_dir / "00" / "l.000001.png").read_bytes(),
            b"cam0-image-200",
        )
        self.assertEqual(
            (self.sequence_dir / "01" / "r.000001.png").read_bytes(),
            b"cam1-image-200",
        )
        self.assertFalse(any(path.is_symlink() for path in self.output.rglob("*")))

    def test_recalibrated_cam0_relative_fisheye_edex(self):
        edex = json.loads((self.sequence_dir / "stereo.edex").read_text(encoding="utf-8"))
        header = edex[0]
        self.assertEqual(header["frame_start"], 0)
        self.assertEqual(header["frame_end"], 2)
        self.assertEqual(header["cameras"][0]["intrinsics"]["distortion_model"], "fisheye")
        self.assertEqual(
            header["cameras"][0]["intrinsics"]["distortion_params"],
            [
                -0.0062748193357009315,
                0.029005519692414498,
                -0.03438856012105873,
                0.014830434499283266,
            ],
        )
        self.assertEqual(
            header["cameras"][0]["transform"],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        )
        self.assertEqual(
            header["cameras"][1]["transform"],
            [
                [0.9999967, 0.0021889, -0.0013548, 0.1099839],
                [-0.0022078, 0.9998979, -0.0141205, 0.0005322],
                [0.0013237, 0.0141234, 0.9998994, -0.0004407],
            ],
        )
        self.assertEqual(edex[1]["sequence"], [["00/l.000000.png"], ["01/r.000000.png"]])

    def test_imu_values_and_timestamp_range(self):
        imu = _read_json_lines(self.sequence_dir / "IMU.jsonl")
        self.assertEqual([sample["timestamp"] for sample in imu], [150, 250])
        self.assertTrue(all(isinstance(sample["timestamp"], int) for sample in imu))
        self.assertEqual(imu[0]["AngularVelocityX"], 3.0)
        self.assertEqual(imu[0]["LinearAccelerationZ"], 8.0)
        self.assertTrue(all(sample["type"] == "imu_data" for sample in imu))

    def test_ground_truth_is_camera_aligned_and_interpolated(self):
        rows = [
            [float(value) for value in line.split()]
            for line in (self.sequence_dir / "gt.txt").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            rows[0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.assertAlmostEqual(rows[1][3], 1.0, places=6)
        self.assertAlmostEqual(rows[1][7], 0.0, places=6)
        self.assertAlmostEqual(rows[1][11], 0.0, places=6)
        self.assertAlmostEqual(rows[2][3], 2.0, places=6)

    def test_subset_archive_selection_configs_and_metadata(self):
        self.assertEqual(
            self.metadata["source"]["archives"][0]["name"],
            "machine_hall.zip",
        )
        self.assertEqual(len(self.metadata["source"]["archives"]), 1)
        self.assertEqual(self.metadata["sequences"][0]["name"], "MH_01_easy")
        self.assertEqual(
            self.metadata["sequences"][0]["source_counts"],
            {
                "cam0_frames": 6,
                "cam1_frames": 6,
                "imu_samples": 6,
                "ground_truth_poses": 2,
            },
        )
        self.assertEqual(self.metadata["sequences"][0]["converted_counts"]["frames"], 3)
        combined_entries = json.loads((self.output / "euroc-vio_slam.cfg").read_text())["sequence_cfgs"]
        self.assertEqual({entry["sequence_folder"] for entry in combined_entries}, {"MH_01_easy"})
        self.assertEqual(
            self.metadata["generated_configs"],
            ["euroc-slam.cfg", "euroc-vio.cfg", "euroc-vio_slam.cfg"],
        )

    def test_metadata_is_deterministic_and_contains_only_relative_layout(self):
        second_output = self.root / "second" / "euroc"
        convert_euroc.convert(self.raw, second_output, ["MH_01_easy"])
        first_metadata = (self.output / "dataset_metadata.json").read_bytes()
        second_metadata = (second_output / "dataset_metadata.json").read_bytes()
        self.assertEqual(first_metadata, second_metadata)
        metadata = json.loads(first_metadata)
        self.assertEqual(metadata["schema_version"], "1.0")
        self.assertEqual(metadata["converter_version"], "1.0")
        self.assertEqual(metadata["source"]["doi"], "10.3929/ethz-b-000690084")
        self.assertEqual(metadata["source"]["url"], "https://doi.org/10.3929/ethz-b-000690084")
        self.assertEqual(metadata["media_layout"]["storage"], "loose_files")
        self.assertEqual(len(metadata["source"]["archives"][0]["sha256"]), 64)
        self.assertNotIn(str(self.root), first_metadata.decode())
        self.assertNotIn("timestamp", metadata)
        self.assertEqual(metadata["generated_configs"], sorted(metadata["generated_configs"]))


class TestEurocConfigGeneration(unittest.TestCase):
    def test_prepare_converts_explicit_subset_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            output = root / "converted"
            _write_outer_archive(
                raw,
                "machine_hall.zip",
                {"MH_01_easy": _inner_archive_bytes()},
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            md5sum = fake_bin / "md5sum"
            md5sum.write_text(
                "#!/usr/bin/env bash\n"
                "printf '363f5c2502b469cdd97ef85997714806  %s\\n' \"$2\"\n"
            )
            md5sum.chmod(0o755)

            # The cached archive lets the real download script verify and reuse it.
            with mock.patch.dict(os.environ, {"PATH": f"{fake_bin}:{os.environ['PATH']}"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    prepared = euroc_prepare.prepare(
                        raw_dir=raw,
                        output_dir=output,
                        sequences=["MH_01_easy"],
                    )

            self.assertEqual(prepared, output / "euroc")
            self.assertTrue((output / "euroc" / "dataset_metadata.json").is_file())
            self.assertTrue((output / "euroc" / "MH_01_easy" / "gt.txt").is_file())

    def test_default_conversion_selects_all_three_archives_and_eleven_sequences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            output = root / "converted" / "euroc"
            inner_contents = _inner_archive_bytes()
            for archive_name, sequences in convert_euroc._OUTER_ZIPS:
                _write_outer_archive(
                    raw,
                    archive_name,
                    {sequence: inner_contents for sequence in sequences},
                )

            metadata = convert_euroc.convert(raw, output)

            self.assertEqual(
                [archive["name"] for archive in metadata["source"]["archives"]],
                ["machine_hall.zip", "vicon_room1.zip", "vicon_room2.zip"],
            )
            self.assertEqual(
                [sequence["name"] for sequence in metadata["sequences"]],
                convert_euroc.ALL_SEQS,
            )
            self.assertTrue(all((output / sequence / "stereo.edex").is_file() for sequence in convert_euroc.ALL_SEQS))

    def test_all_eleven_configs_resolve_and_name_edex_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "custom-euroc-output"
            output.mkdir()
            for sequence in convert_euroc.ALL_SEQS:
                sequence_dir = output / sequence
                sequence_dir.mkdir()
                (sequence_dir / "stereo.edex").write_text("{}", encoding="utf-8")
                (sequence_dir / "gt.txt").write_text("", encoding="utf-8")

            names = convert_euroc._write_configs(output, convert_euroc.ALL_SEQS)

            self.assertEqual(len(convert_euroc.ALL_SEQS), 11)
            self.assertEqual(
                names,
                ["euroc-slam.cfg", "euroc-vio.cfg", "euroc-vio_slam.cfg"],
            )
            combined = json.loads((output / "euroc-vio_slam.cfg").read_text(encoding="utf-8"))
            self.assertEqual(combined["dataset_folder"], "custom-euroc-output/")
            self.assertEqual(
                (output.parent / combined["dataset_folder"]).resolve(),
                output.resolve(),
            )
            self.assertEqual(len(combined["sequence_cfgs"]), 22)
            self.assertEqual(
                {entry["sequence_folder"] for entry in combined["sequence_cfgs"]},
                set(convert_euroc.ALL_SEQS),
            )
            for config_name in names:
                config = json.loads((output / config_name).read_text(encoding="utf-8"))
                for entry in config["sequence_cfgs"]:
                    self.assertEqual(entry["edex_file"], "stereo.edex")
                    self.assertEqual(entry["gt_file_path"], "gt.txt")
                    sequence_dir = output / entry["sequence_folder"]
                    self.assertTrue((sequence_dir / entry["edex_file"]).is_file())
                    self.assertTrue((sequence_dir / entry["gt_file_path"]).is_file())

    def test_defaults_and_positional_cli_paths(self):
        expected_root = Path(__file__).resolve().parents[4]
        self.assertEqual(convert_euroc._repo_root(), expected_root)
        with mock.patch.object(convert_euroc, "convert") as convert:
            self.assertEqual(convert_euroc.main([]), 0)
        convert.assert_called_once_with(
            expected_root / "datasets" / "euroc" / "raw",
            expected_root / "datasets" / "converted" / "euroc",
            None,
        )
        with mock.patch.object(convert_euroc, "convert") as convert:
            self.assertEqual(
                convert_euroc.main(["/raw", "/out", "--sequences", "MH_01_easy"]),
                0,
            )
        convert.assert_called_once_with(Path("/raw"), Path("/out"), ["MH_01_easy"])


class TestEurocConversionFailures(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.raw = self.root / "raw"
        self.output = self.root / "output"

    def tearDown(self):
        self._temporary.cleanup()

    def test_unknown_sequence_fails_clearly(self):
        with self.assertRaisesRegex(convert_euroc.ConversionError, "unknown sequence"):
            convert_euroc.convert(self.raw, self.output, ["not_official"])

    def test_missing_archive_fails_clearly(self):
        with self.assertRaisesRegex(convert_euroc.ConversionError, "missing archive.*machine_hall.zip"):
            convert_euroc.convert(self.raw, self.output, ["MH_01_easy"])

    def test_required_member_missing_fails_clearly(self):
        _write_outer_archive(
            self.raw,
            "machine_hall.zip",
            {
                "MH_01_easy": _inner_archive_bytes(
                    omit_member="mav0/imu0/data.csv",
                )
            },
        )
        with self.assertRaisesRegex(convert_euroc.ConversionError, "required archive member missing.*imu0"):
            convert_euroc.convert(self.raw, self.output, ["MH_01_easy"])

    def test_required_associated_image_missing_fails_clearly(self):
        _write_outer_archive(
            self.raw,
            "machine_hall.zip",
            {
                "MH_01_easy": _inner_archive_bytes(
                    omit_image="cam1-200.png",
                )
            },
        )
        with self.assertRaisesRegex(convert_euroc.ConversionError, "required archive member missing.*cam1-200"):
            convert_euroc.convert(self.raw, self.output, ["MH_01_easy"])

    def test_malformed_csv_fails_clearly(self):
        _write_outer_archive(
            self.raw,
            "machine_hall.zip",
            {
                "MH_01_easy": _inner_archive_bytes(
                    cam0_csv="100,filename.png,unexpected\n",
                )
            },
        )
        with self.assertRaisesRegex(convert_euroc.ConversionError, "malformed CSV.*expected 2 columns"):
            convert_euroc.convert(self.raw, self.output, ["MH_01_easy"])

    def test_no_associated_frames_fails_clearly(self):
        cam1_entries = [(125, "cam1-125.png"), (275, "cam1-275.png")]
        _write_outer_archive(
            self.raw,
            "machine_hall.zip",
            {
                "MH_01_easy": _inner_archive_bytes(
                    cam1_entries=cam1_entries,
                )
            },
        )
        with self.assertRaisesRegex(convert_euroc.ConversionError, "no associated cam0/cam1 frames"):
            convert_euroc.convert(self.raw, self.output, ["MH_01_easy"])

    def test_no_imu_inside_ground_truth_range_fails_before_writing_sequence(self):
        boundary_only_imu = (
            "#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n"
            "100,1,2,3,4,5,6\n"
            "300,2,3,4,5,6,7\n"
        )
        _write_outer_archive(
            self.raw,
            "machine_hall.zip",
            {
                "MH_01_easy": _inner_archive_bytes(
                    imu_csv=boundary_only_imu,
                )
            },
        )
        with self.assertRaisesRegex(convert_euroc.ConversionError, "no IMU samples inside"):
            convert_euroc.convert(self.raw, self.output, ["MH_01_easy"])
        self.assertFalse((self.output / "MH_01_easy").exists())

    def test_unsafe_outer_archive_member_paths_are_rejected(self):
        unsafe_paths = [
            "/absolute.zip",
            "\\absolute.zip",
            "C:\\escape.zip",
            "C:escape.zip",
            "../escape.zip",
            "..\\escape.zip",
        ]
        for index, unsafe_path in enumerate(unsafe_paths):
            with self.subTest(path=unsafe_path):
                raw = self.root / f"outer-{index}"
                _write_outer_archive(
                    raw,
                    "machine_hall.zip",
                    {},
                    [(unsafe_path, b"not a nested archive")],
                )
                with self.assertRaisesRegex(convert_euroc.ConversionError, "unsafe.*outer archive"):
                    convert_euroc.convert(raw, self.output / str(index), ["MH_01_easy"])

    def test_unsafe_inner_csv_image_paths_are_rejected(self):
        unsafe_paths = [
            "/absolute.png",
            "\\absolute.png",
            "C:\\escape.png",
            "C:escape.png",
            "../escape.png",
            "..\\escape.png",
        ]
        for index, unsafe_path in enumerate(unsafe_paths):
            with self.subTest(path=unsafe_path):
                raw = self.root / f"inner-{index}"
                cam0_entries, _ = _default_camera_entries()
                cam0_entries[1] = (100, unsafe_path)
                _write_outer_archive(
                    raw,
                    "machine_hall.zip",
                    {
                        "MH_01_easy": _inner_archive_bytes(
                            cam0_entries=cam0_entries,
                        )
                    },
                )
                with self.assertRaisesRegex(convert_euroc.ConversionError, "unsafe image filename"):
                    convert_euroc.convert(raw, self.output / str(index), ["MH_01_easy"])


class TestEurocBundleSelection(unittest.TestCase):
    """Selecting the wrong bundle downloads 6-12 GB that is not needed.

    The default and unknown-sequence cases are covered by the conversion tests
    above; these pin the per-group mapping and the resulting download arguments.
    """

    def test_each_sequence_group_maps_to_its_own_bundle(self):
        cases = {
            "MH_03_medium": ["machine_hall.zip"],
            "V1_02_medium": ["vicon_room1.zip"],
            "V2_03_difficult": ["vicon_room2.zip"],
        }
        for sequence, expected in cases.items():
            with self.subTest(sequence=sequence):
                self.assertEqual(convert_euroc.required_archives([sequence]), expected)

    def test_mixed_subset_keeps_bundle_order(self):
        self.assertEqual(
            convert_euroc.required_archives(["V2_01_easy", "MH_01_easy"]),
            ["machine_hall.zip", "vicon_room2.zip"],
        )

    def test_explicit_subset_downloads_only_its_bundles(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary) / "raw"
            with mock.patch.object(euroc_prepare, "run_download_script") as download:
                with contextlib.redirect_stdout(io.StringIO()):
                    euroc_prepare.prepare(
                        raw_dir=raw_dir,
                        output_dir=Path(temporary) / "converted",
                        sequences=["MH_01_easy", "V1_01_easy"],
                        download_only=True,
                    )

        self.assertEqual(
            download.call_args.args[1],
            [str(raw_dir), "--archive", "machine_hall.zip", "--archive", "vicon_room1.zip"],
        )


class TestEurocDownloadScript(unittest.TestCase):
    def test_corrupt_cached_archive_is_rejected(self):
        """A truncated multi-GB download must not be handed to the converter."""
        script = Path(convert_euroc.__file__).with_name("download_euroc.sh")
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            (raw_dir / "machine_hall.zip").write_bytes(b"corrupt")

            completed = subprocess.run(
                ["bash", str(script), "--archive", "machine_hall.zip", str(raw_dir)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("md5 mismatch for existing", completed.stderr)
        self.assertIn("re-run with --force", completed.stderr)


if __name__ == "__main__":
    unittest.main()
