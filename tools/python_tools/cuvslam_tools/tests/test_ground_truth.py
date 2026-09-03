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

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cuvslam_tools.tracker.ground_truth import load_gt_transforms, resolve_gt_file

IDENTITY_POSE = "1 0 0 0 0 1 0 0 0 0 1 0"


class TestResolveGtFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dataset = Path(self._tmp.name)

    def test_relative_path_resolves_against_the_dataset_directory(self):
        (self.dataset / "gt_rect.txt").write_text(IDENTITY_POSE, encoding="utf-8")

        resolved = resolve_gt_file(str(self.dataset), "gt_rect.txt", False, "none", 0)

        self.assertEqual(resolved, os.path.join(str(self.dataset), "gt_rect.txt"))

    def test_relative_path_resolves_beside_a_video_dataset(self):
        # A video input names the file, not a directory, so its poses sit next to it.
        video = self.dataset / "run.mp4"
        video.write_bytes(b"")
        (self.dataset / "gt_rect.txt").write_text(IDENTITY_POSE, encoding="utf-8")

        resolved = resolve_gt_file(
            str(video), "gt_rect.txt", gt_from_shuttle=False, repeat_type="none", num_loops=0
        )

        self.assertEqual(resolved, os.path.join(str(self.dataset), "gt_rect.txt"))

    def test_absolute_path_is_used_as_is(self):
        gt_file = self.dataset / "poses.txt"
        gt_file.write_text(IDENTITY_POSE, encoding="utf-8")

        resolved = resolve_gt_file("/somewhere/else", str(gt_file), False, "none", 0)

        self.assertEqual(resolved, str(gt_file))

    def test_missing_ground_truth_file_stops_the_run(self):
        with self.assertRaises(FileNotFoundError) as cm:
            resolve_gt_file(str(self.dataset), "gt_rect.txt", False, "shuttle", 1)

        self.assertIn("gt_rect.txt", str(cm.exception))

    def test_no_requested_ground_truth_reads_no_file(self):
        # A gt.txt sitting in the dataset directory must not be picked up implicitly.
        (self.dataset / "gt.txt").write_text(IDENTITY_POSE, encoding="utf-8")

        self.assertIsNone(resolve_gt_file(str(self.dataset), None, False, "none", 0))
        self.assertIsNone(resolve_gt_file(str(self.dataset), "", False, "shuttle", 1))

    def test_shuttle_ground_truth_reads_no_file(self):
        self.assertIsNone(resolve_gt_file(str(self.dataset), None, True, "shuttle", 1))

    def test_shuttle_ground_truth_rejects_a_pose_file(self):
        (self.dataset / "gt.txt").write_text(IDENTITY_POSE, encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Conflicting ground truth"):
            resolve_gt_file(str(self.dataset), "gt.txt", True, "shuttle", 1)

    def test_shuttle_ground_truth_requires_a_shuttle_replay(self):
        with self.assertRaisesRegex(ValueError, "repeat_type='shuttle'"):
            resolve_gt_file(str(self.dataset), None, True, "repeat", 1)

        with self.assertRaisesRegex(ValueError, "num_loops > 0"):
            resolve_gt_file(str(self.dataset), None, True, "shuttle", 0)


class TestLoadGtTransforms(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dataset = Path(self._tmp.name)

    def test_each_line_becomes_a_homogeneous_transform(self):
        gt_file = self.dataset / "gt.txt"
        gt_file.write_text(f"{IDENTITY_POSE}\n1 0 0 2 0 1 0 3 0 0 1 4\n", encoding="utf-8")

        transforms = load_gt_transforms(str(gt_file))

        self.assertEqual(len(transforms), 2)
        np.testing.assert_allclose(transforms[0], np.eye(4))
        np.testing.assert_allclose(transforms[1][:3, 3], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(transforms[1][3], [0.0, 0.0, 0.0, 1.0])

    def test_a_line_with_a_non_finite_value_names_the_line(self):
        # float() parses these, so they need a guard of their own.
        for literal in ("nan", "inf"):
            with self.subTest(literal=literal):
                gt_file = self.dataset / "gt.txt"
                gt_file.write_text(f"{IDENTITY_POSE}\n1 0 0 {literal} 0 1 0 0 0 0 1 0\n", encoding="utf-8")

                with self.assertRaisesRegex(
                    ValueError, rf"gt\.txt:2 holds a non-finite value \({literal}\)"
                ):
                    load_gt_transforms(str(gt_file))

    def test_a_line_without_twelve_values_names_the_line(self):
        gt_file = self.dataset / "gt.txt"
        gt_file.write_text(f"{IDENTITY_POSE}\n1 0 0 2\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, r"gt\.txt:2"):
            load_gt_transforms(str(gt_file))


if __name__ == "__main__":
    unittest.main()
