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

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from cuvslam_tools.dataset_preparation.tartanair_v1 import convert_tartanair_v1 as convert

# seasidetown is small and has both stable and flaky trajectories.
ENV = "seasidetown"
FRAMES = 3
STEP_M = 0.5
SKY_M = 13000.0


def _png(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def _depth():
    depth = np.full(convert.IMAGE_SIZE[::-1], 2.5, dtype=np.float32)
    depth[:10, :10] = SKY_M
    depth[10:20, :10] = np.nan
    buffer = io.BytesIO()
    np.save(buffer, depth)
    return buffer.getvalue()


def _pose_lines(offset_y=0.0):
    # NED: forward along x, the right camera BASELINE_M along +y.
    return "".join(f"{frame * STEP_M:.6f} {offset_y:.6f} 0.000000 0 0 0 1\n" for frame in range(FRAMES))


def write_zips(raw_dir: Path, env=ENV, trajectories=None, right_offset=convert.BASELINE_M, drop=None):
    """Write the three per-environment zips in the public release's member layout."""
    trajectories = trajectories or convert.selected([env])[env]
    raw_dir.mkdir(parents=True, exist_ok=True)
    image = _png(*convert.IMAGE_SIZE)
    depth = _depth()
    with zipfile.ZipFile(raw_dir / convert.zip_filename(env, "image_left"), "w") as left, \
         zipfile.ZipFile(raw_dir / convert.zip_filename(env, "image_right"), "w") as right, \
         zipfile.ZipFile(raw_dir / convert.zip_filename(env, "depth_left"), "w") as depth_zip:
        for trajectory in trajectories:
            base = f"{env}/Hard/{trajectory}"
            for archive in (left, right, depth_zip):
                archive.writestr(f"{base}/pose_left.txt", _pose_lines())
                archive.writestr(f"{base}/pose_right.txt", _pose_lines(right_offset))
            for frame in range(FRAMES):
                members = (
                    (left, f"{base}/image_left/{frame:06d}_left.png", image),
                    (right, f"{base}/image_right/{frame:06d}_right.png", image),
                    (depth_zip, f"{base}/depth_left/{frame:06d}_left_depth.npy", depth),
                )
                for archive, name, data in members:
                    if name != drop:
                        archive.writestr(name, data)


class _Converted(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name) / "raw"
        self.output = Path(self._tmp.name) / convert.DATASET_ID
        write_zips(self.raw)
        convert.convert(self.raw, self.output, [ENV], workers=1)
        self.sequence = self.output / ENV / "P000"

    def tearDown(self):
        self._tmp.cleanup()


class TestSequence(_Converted):
    def test_images_are_copied_byte_for_byte(self):
        with zipfile.ZipFile(self.raw / convert.zip_filename(ENV, "image_right")) as archive:
            source = archive.read(f"{ENV}/Hard/P000/image_right/000002_right.png")

        self.assertEqual((self.sequence / "01" / "000002.png").read_bytes(), source)

    def test_ground_truth_is_the_left_camera_relative_to_frame_0(self):
        poses = np.loadtxt(self.sequence / convert.GROUND_TRUTH_FILE)

        self.assertEqual(len(poses), FRAMES)
        np.testing.assert_allclose(poses[0].reshape(3, 4), np.eye(4)[:3], atol=1e-9)
        # Forward in NED is +z in the OpenCV frame the tracker reports in.
        np.testing.assert_allclose(poses[-1].reshape(3, 4)[:, 3], [0, 0, (FRAMES - 1) * STEP_M], atol=1e-9)

    def test_ground_truth_keeps_the_osmo_number_format(self):
        first = (self.sequence / convert.GROUND_TRUTH_FILE).read_text().splitlines()[0].split()

        self.assertEqual(first[0], "1.000000e+00")
        self.assertEqual(len(first), 12)

    def test_depth_is_left_only_in_millimetres_with_no_return_as_zero(self):
        depth = np.array(Image.open(self.sequence / "depth_00" / "000001.png"))

        self.assertFalse((self.sequence / "depth_01").exists())
        self.assertEqual(depth.dtype, np.uint16)
        self.assertEqual(int(depth[100, 100]), 2500)
        self.assertTrue((depth[:10, :10] == 0).all(), "sky must read as no return")
        self.assertTrue((depth[10:20, :10] == 0).all(), "non-finite depth must read as no return")

    def test_the_rig_is_the_osmo_stereo_pair_with_depth_on_the_left(self):
        edex = json.loads((self.sequence / convert.EDEX_FILE).read_text())
        left, right = edex[0]["cameras"]

        self.assertEqual(right["transform"][0][3], convert.BASELINE_M)
        self.assertEqual((left["depth_id"], left["depth_scale_factor"]), (0, convert.DEPTH_SCALE_FACTOR))
        self.assertNotIn("depth_id", right)
        self.assertEqual(left["intrinsics"]["principal"], [320.0, 240.0])
        self.assertEqual(edex[1]["fps"], 30)
        self.assertEqual(edex[0]["frame_end"], FRAMES - 1)


class TestConfigs(_Converted):
    def _config(self, name):
        return json.loads((self.output / name).read_text())

    def test_each_trajectory_is_in_exactly_one_split_config_with_both_modes(self):
        stable = self._config("tartan_stable-vo_slam.cfg")["sequence_cfgs"]
        flaky = self._config("tartan_flaky-vo_slam.cfg")["sequence_cfgs"]
        folders = lambda entries: sorted({entry["sequence_folder"] for entry in entries})

        self.assertEqual(folders(flaky), [f"{ENV}/P003"])
        self.assertEqual(folders(stable), [f"{ENV}/{t}" for t in ("P000", "P001", "P002", "P004")])
        self.assertEqual(len(stable), 2 * len(folders(stable)))

    def test_the_full_config_is_the_stable_and_flaky_entries_together(self):
        key = lambda entry: entry["sequence_title"]
        split = (self._config("tartan_stable-vo_slam.cfg")["sequence_cfgs"]
                 + self._config("tartan_flaky-vo_slam.cfg")["sequence_cfgs"])
        full = self._config("tartan-vo_slam.cfg")["sequence_cfgs"]

        self.assertEqual(sorted(full, key=key), sorted(split, key=key))

    def test_entries_keep_the_osmo_titles_and_link_ground_truth(self):
        entries = self._config("tartan-vo_slam.cfg")["sequence_cfgs"]

        self.assertEqual([e["sequence_title"] for e in entries[:2]],
                         [f"{ENV}-Hard-P000-ODOM", f"{ENV}-Hard-P000-SLAM"])
        self.assertEqual({e["gt_file_path"] for e in entries}, {"gt.txt"})
        self.assertEqual([e.get("use_slam", False) for e in entries[:2]], [False, True])

    def test_configs_have_distinct_kpi_prefixes_and_the_osmo_segments(self):
        for name in convert.CONFIGS:
            with self.subTest(config=name):
                config = self._config(name)
                self.assertEqual(config["dataset_folder"], f"{convert.DATASET_ID}/")
                self.assertEqual(config["segment_lengths"], list(convert.SEGMENT_LENGTHS))
        self.assertEqual({name.split("-")[0].upper() for name in convert.CONFIGS},
                         {"TARTAN", "TARTAN_STABLE", "TARTAN_FLAKY"})


class TestSelection(unittest.TestCase):
    def test_every_trajectory_is_stable_or_flaky_never_both(self):
        for env in convert.ENVIRONMENTS:
            with self.subTest(env=env):
                self.assertFalse(set(convert.STABLE.get(env, ())) & set(convert.FLAKY.get(env, ())))

    def test_the_selection_is_the_osmo_one_with_its_two_unassigned_trajectories_placed(self):
        self.assertEqual(len(convert.ENVIRONMENTS), 16)
        self.assertEqual(sum(map(len, convert.STABLE.values())), 130)
        self.assertEqual(sum(map(len, convert.FLAKY.values())), 9)
        self.assertIn("P003", convert.STABLE["neighborhood"])
        self.assertIn("P001", convert.FLAKY["westerndesert"])
        self.assertTrue({"abandonedfactory_night", "gascola"}.isdisjoint(convert.ENVIRONMENTS))

    def test_the_download_manifest_pins_exactly_the_zips_the_converter_reads(self):
        rows = [line.split() for line in convert.MANIFEST.read_text().splitlines() if line.strip()]
        keys = {key for key, _, _ in rows}

        self.assertEqual(keys, {f"{env}/Hard/{kind}.zip" for env in convert.ENVIRONMENTS for kind in convert.ZIP_KINDS})
        for key, md5, size in rows:
            with self.subTest(zip=key):
                self.assertRegex(md5, r"^[0-9a-f]{32}$")
                self.assertGreater(int(size), 0)

    def test_a_partial_run_writes_only_configs_it_can_fill(self):
        chosen = convert.selected(["carwelding"])

        self.assertEqual(list(convert.config_subsets(chosen)), ["tartan-vo_slam.cfg", "tartan_stable-vo_slam.cfg"])

    def test_an_unknown_environment_is_rejected(self):
        with self.assertRaisesRegex(convert.ConversionError, "unknown environments"):
            convert.selected(["gascola"])


class TestValidation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name) / "raw"
        self.output = Path(self._tmp.name) / convert.DATASET_ID

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_zip_holding_other_trajectories_than_pinned_is_rejected(self):
        write_zips(self.raw, trajectories=("P000", "P001", "P002", "P003", "P004", "P005"))

        with self.assertRaisesRegex(convert.ConversionError, "the selection pins"):
            convert.convert(self.raw, self.output, [ENV], workers=1)

    def test_a_right_camera_off_the_baseline_is_rejected(self):
        write_zips(self.raw, right_offset=0.3)

        with self.assertRaisesRegex(convert.ConversionError, "baseline"):
            convert.convert(self.raw, self.output, [ENV], workers=1)

    def test_a_missing_frame_is_rejected(self):
        write_zips(self.raw, drop=f"{ENV}/Hard/P002/depth_left/000001_left_depth.npy")

        with self.assertRaisesRegex(convert.ConversionError, "missing"):
            convert.convert(self.raw, self.output, [ENV], workers=1)

    def test_a_missing_zip_names_the_download_step(self):
        with self.assertRaisesRegex(convert.ConversionError, "download step"):
            convert.convert(self.raw, self.output, [ENV], workers=1)


if __name__ == "__main__":
    unittest.main()
