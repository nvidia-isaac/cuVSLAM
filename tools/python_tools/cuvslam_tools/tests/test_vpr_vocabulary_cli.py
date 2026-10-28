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

import builtins
import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from cuvslam_tools.vpr_vocabulary import cli


class _FakeFitter:
    """Stand-in for Slam.VprVocabularyFitter that records its calls."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.calls = []

    def add_image(self, image):
        self.calls.append(("add_image", image))

    def save(self, path):
        self.calls.append(("save", path))


class _FakeReader:
    """Stand-in for EdexReader whose images name the sequence, frame, and camera they come from."""

    def __init__(self, edex_dir: str, frame_ids, n_cameras: int = 2):
        self.edex_dir = edex_dir
        self.rig = types.SimpleNamespace(cameras=[None] * n_cameras)
        self.frames = {frame_id: {} for frame_id in frame_ids}

    def load_camera_image(self, frame_id, camera_id):
        return (self.edex_dir, frame_id, camera_id)


def _without_cuvslam():
    """Fail the test if cuvslam, or the EDEX reader that imports it, is imported."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ("cuvslam", "cuvslam_tools.tracker.edex_reader"):
            raise AssertionError(f"{name} was imported, so the arguments were not rejected first")
        return original_import(name, *args, **kwargs)

    return mock.patch.object(builtins, "__import__", side_effect=guarded_import)


class TestSelectFrames(unittest.TestCase):
    def test_picks_spread_evenly_over_all_sequences(self):
        # A quarter of the frames get a quarter of the picks.
        self.assertEqual(cli.select_frames([10, 30], 8),
                         [(0, 2), (0, 7), (1, 2), (1, 7), (1, 12), (1, 17), (1, 22), (1, 27)])

    def test_picks_are_sorted_and_unique(self):
        picks = cli.select_frames([1101, 1, 4541], 189)

        self.assertEqual(len(picks), 189)
        self.assertEqual(picks, sorted(set(picks)))

    def test_below_capacity_every_frame_is_picked(self):
        self.assertEqual(cli.select_frames([3, 2], 189), [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])


class TestFitVocabulary(unittest.TestCase):
    def test_images_of_the_camera_are_spread_over_the_sequences_and_saved_once(self):
        fitter = _FakeFitter(capacity=8)
        # Frames are taken in frame id order, not in the order the reader holds them.
        readers = [_FakeReader("a", range(109, 99, -1)), _FakeReader("b", range(30))]

        counts = cli.fit_vocabulary(fitter, readers, 1, "kitti.vocabulary")

        self.assertEqual(counts, [2, 6])
        self.assertEqual(fitter.calls,
                         [("add_image", ("a", frame_id, 1)) for frame_id in (102, 107)]
                         + [("add_image", ("b", frame_id, 1)) for frame_id in (2, 7, 12, 17, 22, 27)]
                         + [("save", "kitti.vocabulary")])

    def test_camera_outside_a_rig_fails_before_any_image(self):
        fitter = _FakeFitter(capacity=8)
        # Only the second rig lacks the camera.
        readers = [_FakeReader("a", range(10), n_cameras=2), _FakeReader("b", range(30), n_cameras=1)]

        with self.assertRaisesRegex(ValueError, "--camera 1 is not a camera of b"):
            cli.fit_vocabulary(fitter, readers, 1, "kitti.vocabulary")

        self.assertEqual(fitter.calls, [])


class TestVprVocabularyCli(unittest.TestCase):
    def setUp(self):
        # Hide the caller's CUVSLAM_ANYLOC_ENGINE, the --vpr_model_path default.
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("CUVSLAM_ANYLOC_ENGINE", None)

        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = root.name
        self.engine = os.path.join(self.root, "anyloc.engine")
        self.sequence = os.path.join(self.root, "11")
        self.output = os.path.join(self.root, "kitti.vocabulary")
        open(self.engine, "w").close()
        os.mkdir(self.sequence)
        open(os.path.join(self.sequence, "stereo.edex"), "w").close()

    def _usage_error(self, argv: list) -> str:
        """Run cli.main, expect a usage error before cuvslam is imported, and return stderr."""
        stderr = io.StringIO()
        with _without_cuvslam(), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli.main(argv)

        self.assertEqual(cm.exception.code, 2)
        return stderr.getvalue()

    def _stub_cuvslam(self, fitter):
        """Stand in for the binding and the EDEX reader; returns the patch and the fitter and reader class mocks."""
        fitter_class = mock.Mock(return_value=fitter)
        vslam = types.ModuleType("cuvslam")
        vslam.Slam = types.SimpleNamespace(VprVocabularyFitter=fitter_class)
        reader_class = mock.Mock(side_effect=lambda sequence, stereo_edex: _FakeReader(sequence, range(10)))
        edex_reader = types.ModuleType("cuvslam_tools.tracker.edex_reader")
        edex_reader.EdexReader = reader_class
        modules = {"cuvslam": vslam, "cuvslam_tools.tracker.edex_reader": edex_reader}
        return mock.patch.dict(sys.modules, modules), fitter_class, reader_class

    def test_help_does_not_import_cuvslam(self):
        stdout = io.StringIO()
        with _without_cuvslam(), contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--help"])

        self.assertEqual(cm.exception.code, 0)
        for flag in ("--output", "--vpr_model_path", "--edex_filename", "--camera"):
            self.assertIn(flag, stdout.getvalue())

    def test_usage_errors_exit_before_cuvslam_is_imported(self):
        missing = os.path.join(self.root, "missing")
        engine = ["--vpr_model_path", self.engine]
        cases = (
            (["--output", self.output, self.sequence], "--vpr_model_path"),
            (["--output", self.output, "--vpr_model_path", missing, self.sequence], "--vpr_model_path"),
            (["--output", self.output, *engine, self.sequence, missing], f"{missing} has no stereo.edex"),
            (["--output", self.output, *engine, "--edex_filename", "mono.edex", self.sequence], "has no mono.edex"),
            (["--output", self.output, *engine, "--camera", "-1", self.sequence], "--camera"),
            (["--output", self.root, *engine, self.sequence], "is a folder"),
            (["--output", os.path.join(missing, "kitti.vocabulary"), *engine, self.sequence], "does not exist"),
        )
        for argv, message in cases:
            with self.subTest(argv=argv):
                self.assertIn(message, self._usage_error(argv))

    def test_output_is_not_abbreviated(self):
        error = self._usage_error(["--output", self.output, "--vpr_model_path", self.engine, self.sequence,
                                   "--out", self.output])

        self.assertIn("unrecognized arguments: --out", error)

    def test_fits_every_sequence_with_the_engine(self):
        fitter = _FakeFitter(capacity=8)
        modules, fitter_class, reader_class = self._stub_cuvslam(fitter)
        stdout = io.StringIO()
        with modules, contextlib.redirect_stdout(stdout):
            exit_code = cli.main(["--output", self.output, "--vpr_model_path", self.engine, self.sequence])

        self.assertEqual(exit_code, 0)
        reader_class.assert_called_once_with(self.sequence, stereo_edex=os.path.join(self.sequence, "stereo.edex"))
        # Only the engine is passed; the rest of the config keeps the binding's defaults.
        fitter_class.Config.assert_called_once_with(vpr_model_path=self.engine)
        fitter_class.assert_called_once_with(fitter_class.Config.return_value)
        self.assertEqual(fitter.calls[-1], ("save", self.output))
        self.assertIn(f"{self.sequence}: 8 images", stdout.getvalue())

    def test_runtime_error_from_cuvslam_is_a_usage_error(self):
        # The engine, TensorRT, and the file write fail with RuntimeError.
        fitter = _FakeFitter(capacity=8)
        fitter.save = mock.Mock(side_effect=RuntimeError("Failed to write the vocabulary"))
        modules, _, _ = self._stub_cuvslam(fitter)
        stderr = io.StringIO()
        with modules, contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--output", self.output, "--vpr_model_path", self.engine, self.sequence])

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("Failed to write the vocabulary", stderr.getvalue())

    def test_missing_binding_is_a_usage_error(self):
        stderr = io.StringIO()
        with mock.patch.dict(sys.modules, {"cuvslam": None}), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--output", self.output, "--vpr_model_path", self.engine, self.sequence])

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("requires the cuVSLAM Python binding", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
