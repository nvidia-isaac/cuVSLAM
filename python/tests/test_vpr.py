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
import shutil
import tempfile
import unittest

import numpy as np

# Unit tests must not require the external Rerun Viewer, even in USE_RERUN builds.
os.environ.setdefault("RERUN", "0")

import cuvslam as vslam
import data_gen as data

W, H = 640, 480         # Image size of tracked images
BASELINE = 0.25         # Baseline distance between cameras (m)
STEPS = 30              # Number of tracked frames
FRAME_PERIOD_NS = 1_000_000
NO_MATCH = "No mapped place matched."  # error of a place recognition query that matched nothing

# Real loop-closing imagery for the KITTI-07 tests, which skip without it.
KITTI_07_DIR = os.environ.get("CUVSLAM_KITTI_07_DIR") or os.path.join(
    os.environ.get("CUVSLAM_DATASETS", ""), "kitti", "07"
)

# AnyLoc TensorRT engine of a USE_TENSORRT build; cuvslam_vars.sh exports its path.
ANYLOC_ENGINE = os.environ.get("CUVSLAM_ANYLOC_ENGINE", "")
needs_anyloc = unittest.skipUnless(
    os.path.isfile(ANYLOC_ENGINE), "needs the AnyLoc engine of a USE_TENSORRT build in CUVSLAM_ANYLOC_ENGINE")

def make_rig():
    cameras = data.generate_stereo_camera(W, H, BASELINE)
    return vslam.Rig(cameras, [vslam.ImuCalibration()]), cameras

def make_configs(use_vpr):
    odom_cfg = vslam.Odometry.Config()
    odom_cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
    odom_cfg.rectified_stereo_camera = True
    odom_cfg.async_sba = False

    slam_cfg = vslam.Slam.Config()
    slam_cfg.sync_mode = True
    # Lets the tests read back the pose graph node a match names.
    slam_cfg.enable_reading_internals = True
    slam_cfg.use_vpr = use_vpr
    if use_vpr:
        slam_cfg.vpr_model_path = ANYLOC_ENGINE

    return odom_cfg, slam_cfg

def noise_images(seed):
    """Stereo pair of untrackable noise, used where only the pixels matter."""
    rng = np.random.RandomState(seed)
    return [np.ascontiguousarray(rng.randint(0, 256, (H, W), dtype=np.uint8)) for _ in range(2)]

def recognize_place(slam, images, timestamp_ns):
    """Return the (result, error_message) of a place recognition query; needs sync_mode."""
    answers = []
    slam.recognize_place_by_frame(images, timestamp_ns, lambda result, error: answers.append((result, error)))
    if len(answers) != 1:
        raise AssertionError(f"sync_mode must call the callback exactly once, it ran {len(answers)} times")
    return answers[0]

def recognize_match(slam, images, timestamp_ns):
    """Return the matched place, or None when no mapped place matched; fail on any other error."""
    place, error = recognize_place(slam, images, timestamp_ns)
    if place is None and error != NO_MATCH:
        raise AssertionError(f"place recognition failed: {error}")
    return place

def localization_settings(**kwargs):
    """Localization settings with the small probe grid the synthetic map needs."""
    return vslam.Slam.LocalizationSettings(
        horizontal_search_radius=0.25, vertical_search_radius=0.25,
        horizontal_step=0.0625, vertical_step=0.0625, angular_step_rads=0.03125, **kwargs)

def localize_in_map(slam, folder, images, timestamp_ns, settings, guess_pose=None):
    """Localize and return the (pose, error_message) pair the callback got."""
    answers = []
    slam.localize_in_map(folder, timestamp_ns, guess_pose, images, settings,
                         lambda: None, lambda pose, error: answers.append((pose, error)))
    if len(answers) != 1:
        raise AssertionError(f"sync_mode must call the callback exactly once, it ran {len(answers)} times")
    return answers[0]

def save_map(slam, folder):
    """Save the map and return the flag the callback reported."""
    answers = []
    slam.save_map(folder, answers.append)
    if len(answers) != 1:
        raise AssertionError(f"sync_mode must call the callback exactly once, it ran {len(answers)} times")
    return answers[0]

def pose_arrays(pose):
    """Translation and rotation of a Pose as numpy arrays."""
    return np.array(pose.translation, dtype=np.float64), np.array(pose.rotation, dtype=np.float64)


class TestVprBindings(unittest.TestCase):
    """The place recognition API surface, without a tracker."""

    def test_vpr_defaults(self):
        cfg = vslam.Slam.Config()
        self.assertFalse(cfg.use_vpr)
        self.assertAlmostEqual(cfg.vpr_score_threshold, 0.68, places=5)
        self.assertEqual(cfg.vpr_vocabulary_path, "")
        self.assertEqual(cfg.vpr_vocabulary_min_keyframes, 8)
        self.assertEqual(cfg.vpr_vocabulary_max_descriptors, 100000)
        self.assertEqual(localization_settings().vpr_candidates, 5)

    def test_config_vpr_keyword_arguments(self):
        # Distinct values, so an argument bound to the wrong field shows.
        cfg = vslam.Slam.Config(
            use_vpr=True,
            vpr_model_path="/tmp/anyloc.engine",
            vpr_score_threshold=0.75,
            vpr_vocabulary_path="/tmp/a.vocabulary",
            vpr_vocabulary_min_keyframes=3,
            vpr_vocabulary_max_descriptors=7,
            loop_closure_mode=vslam.Slam.LoopClosureMode.VPR)

        self.assertTrue(cfg.use_vpr)
        self.assertEqual(cfg.vpr_model_path, "/tmp/anyloc.engine")
        self.assertAlmostEqual(cfg.vpr_score_threshold, 0.75, places=5)
        self.assertEqual(cfg.vpr_vocabulary_path, "/tmp/a.vocabulary")
        self.assertEqual(cfg.vpr_vocabulary_min_keyframes, 3)
        self.assertEqual(cfg.vpr_vocabulary_max_descriptors, 7)
        self.assertEqual(cfg.loop_closure_mode, vslam.Slam.LoopClosureMode.VPR)

    def test_config_vpr_modifiers(self):
        cfg = vslam.Slam.Config()
        cfg.use_vpr = True
        cfg.vpr_model_path = "/tmp/other.engine"
        cfg.vpr_score_threshold = 0.25
        cfg.vpr_vocabulary_path = "/tmp/b.vocabulary"
        cfg.vpr_vocabulary_min_keyframes = 4
        cfg.vpr_vocabulary_max_descriptors = 9

        self.assertTrue(cfg.use_vpr)
        self.assertEqual(cfg.vpr_model_path, "/tmp/other.engine")
        self.assertAlmostEqual(cfg.vpr_score_threshold, 0.25, places=5)
        self.assertEqual(cfg.vpr_vocabulary_path, "/tmp/b.vocabulary")
        self.assertEqual(cfg.vpr_vocabulary_min_keyframes, 4)
        self.assertEqual(cfg.vpr_vocabulary_max_descriptors, 9)

    def test_config_repr_mentions_vpr(self):
        text = repr(vslam.Slam.Config(use_vpr=True, vpr_score_threshold=0.5, vpr_vocabulary_path="/tmp/a.vocabulary",
                                      vpr_vocabulary_min_keyframes=3, vpr_vocabulary_max_descriptors=7))
        for field in ('use_vpr', 'vpr_model_path', 'vpr_score_threshold'):
            self.assertIn(field, text)
        self.assertIn('use_vpr=True', text)
        self.assertIn('vpr_vocabulary_path=/tmp/a.vocabulary, vpr_vocabulary_min_keyframes=3, '
                      'vpr_vocabulary_max_descriptors=7', text)

    def test_config_vocabulary_settings_are_unsigned(self):
        with self.assertRaises(TypeError):
            vslam.Slam.Config(vpr_vocabulary_min_keyframes=-1)
        cfg = vslam.Slam.Config()
        with self.assertRaises(TypeError):
            cfg.vpr_vocabulary_min_keyframes = -1

    def test_loop_closure_mode_enum(self):
        self.assertEqual(set(vslam.Slam.LoopClosureMode.__members__), {'LSI', 'VPR'})
        self.assertEqual(vslam.Slam.Config().loop_closure_mode, vslam.Slam.LoopClosureMode.LSI)

    def test_config_loop_closure_mode(self):
        cfg = vslam.Slam.Config(loop_closure_mode=vslam.Slam.LoopClosureMode.VPR)
        self.assertEqual(cfg.loop_closure_mode, vslam.Slam.LoopClosureMode.VPR)
        self.assertIn('loop_closure_mode=LoopClosureMode.VPR', repr(cfg))

        cfg.loop_closure_mode = vslam.Slam.LoopClosureMode.LSI
        self.assertEqual(cfg.loop_closure_mode, vslam.Slam.LoopClosureMode.LSI)

    def test_place_recognition_fields(self):
        place = vslam.Slam.PlaceRecognition()
        place.node_id = 7
        place.pose = vslam.Pose(translation=[1.0, 2.0, 3.0], rotation=[0.0, 0.0, 0.0, 1.0])
        place.score = 0.875
        place.timestamp_ns = 123456789

        self.assertEqual(place.node_id, 7)
        self.assertEqual(list(place.pose.translation), [1.0, 2.0, 3.0])
        self.assertAlmostEqual(place.score, 0.875, places=5)
        self.assertEqual(place.timestamp_ns, 123456789)
        self.assertIn('score', repr(place))

    def test_localization_settings_round_trip_vpr_candidates(self):
        settings = vslam.Slam.LocalizationSettings(
            horizontal_search_radius=1.0, vertical_search_radius=0.5,
            horizontal_step=0.25, vertical_step=0.25, angular_step_rads=0.1,
            vpr_candidates=7)
        self.assertEqual(settings.vpr_candidates, 7)
        self.assertIn('vpr_candidates=7', repr(settings))

        settings.vpr_candidates = 3
        self.assertEqual(settings.vpr_candidates, 3)

    def test_vocabulary_fitter_config(self):
        cfg = vslam.Slam.VprVocabularyFitter.Config()
        self.assertEqual(cfg.vpr_model_path, "")
        self.assertEqual(cfg.max_descriptors, 100000)

        cfg = vslam.Slam.VprVocabularyFitter.Config(vpr_model_path="/tmp/anyloc.engine", max_descriptors=7)
        self.assertEqual(cfg.vpr_model_path, "/tmp/anyloc.engine")
        self.assertEqual(cfg.max_descriptors, 7)
        self.assertEqual(
            repr(cfg), "cuvslam.Slam.VprVocabularyFitter.Config(vpr_model_path=/tmp/anyloc.engine, max_descriptors=7)")

        # unnamed parameters are not supported (kw_only)
        with self.assertRaises(TypeError):
            vslam.Slam.VprVocabularyFitter.Config("/tmp/anyloc.engine", 7)  # type: ignore

    def test_vocabulary_fitter_min_images(self):
        # As many as a session fits its own vocabulary to by default.
        self.assertEqual(vslam.Slam.VprVocabularyFitter.min_images, 8)
        self.assertEqual(vslam.Slam.VprVocabularyFitter.min_images, vslam.Slam.Config().vpr_vocabulary_min_keyframes)


class TestVprOff(unittest.TestCase):
    """use_vpr=False has to leave every place recognition call inert rather than fail."""

    def setUp(self):
        self.rig, _ = make_rig()
        self.images = noise_images(1)
        odom_cfg, slam_cfg = make_configs(False)
        self.tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)

    def test_recognize_reports_an_error_instead_of_a_result(self):
        result, error = recognize_place(self.tracker.slam, self.images, FRAME_PERIOD_NS)

        # A no-match error would hide the misconfiguration.
        self.assertIsNone(result)
        self.assertIn("use_vpr", error)

    def test_localize_in_map_without_a_guess_pose_is_rejected(self):
        # A usage error: raises before reading the map, so the folder need not exist.
        with self.assertRaises(ValueError) as caught:
            localize_in_map(self.tracker.slam, "/nonexistent/map", self.images, FRAME_PERIOD_NS,
                            localization_settings())
        self.assertIn("use_vpr", str(caught.exception))


@needs_anyloc
class TestVprFrameSelection(unittest.TestCase):
    """A frame without a primary camera image is refused, and one matching nothing gets the no-match error."""

    def setUp(self):
        self.rig, _ = make_rig()
        self.images = noise_images(2)
        _, self.slam_cfg = make_configs(True)

    def test_a_frame_with_no_images_reports_an_error(self):
        slam = vslam.Slam(self.rig, [0, 1], self.slam_cfg)

        result, error = recognize_place(slam, [], FRAME_PERIOD_NS)

        self.assertIsNone(result)
        self.assertIn("primary camera", error)

    def test_a_frame_from_no_primary_camera_reports_an_error(self):
        # Only primary cameras are mapped; others would yield a misleading no-match error.
        slam = vslam.Slam(self.rig, [1], self.slam_cfg)

        result, error = recognize_place(slam, self.images[:1], FRAME_PERIOD_NS)

        self.assertIsNone(result)
        self.assertIn("primary camera", error)

    def test_an_empty_map_reports_no_match(self):
        slam = vslam.Slam(self.rig, [0, 1], self.slam_cfg)

        result, error = recognize_place(slam, self.images, FRAME_PERIOD_NS)

        self.assertIsNone(result)
        self.assertEqual(error, NO_MATCH)


class TestVprLoopClosureModeConfig(unittest.TestCase):
    """LoopClosureMode.VPR on a synthetic sequence, and the configurations it rejects."""

    def setUp(self):
        self.rig, self.cameras = make_rig()

    def make_tracker(self, slam_cfg):
        odom_cfg, _ = make_configs(False)
        return vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)

    def vpr_loop_closure_config(self, use_vpr=True):
        _, slam_cfg = make_configs(use_vpr)
        slam_cfg.loop_closure_mode = vslam.Slam.LoopClosureMode.VPR
        return slam_cfg

    def assert_slam_rejects(self, slam_cfg, field):
        # Slam, not Tracker, which rejects some of these configurations first.
        with self.assertRaises(ValueError) as caught:
            vslam.Slam(self.rig, [0, 1], slam_cfg)
        self.assertIn("loop_closure_mode", str(caught.exception))
        self.assertIn(field, str(caught.exception))

    def test_needs_a_place_recognition_backend(self):
        self.assert_slam_rejects(self.vpr_loop_closure_config(use_vpr=False), "use_vpr")

    def test_rejects_gt_align_mode(self):
        slam_cfg = self.vpr_loop_closure_config()
        slam_cfg.gt_align_mode = True
        self.assert_slam_rejects(slam_cfg, "gt_align_mode")

    @needs_anyloc
    def test_tracks_a_synthetic_sequence(self):
        tracker = self.make_tracker(self.vpr_loop_closure_config())
        generator = data.ImageGenerator(self.cameras, STEPS)

        tracked = 0
        for i in range(STEPS):
            images, _ = generator.generate_zoomed_images(i)
            odom_pose, _ = tracker.track(i * FRAME_PERIOD_NS, images)
            tracked += odom_pose.world_from_rig is not None

        self.assertGreater(tracked, 0, "the synthetic sequence did not track")
        # Every frame is within the recent past that loop closure ignores.
        self.assertEqual(len(tracker.slam.get_loop_closure_poses()), 0)


class TestVprAnyLocConfig(unittest.TestCase):
    """The configurations and settings the AnyLoc backend rejects."""

    def test_needs_the_gpu(self):
        # Checked before loading the engine, so it needs neither the file nor USE_TENSORRT.
        _, slam_cfg = make_configs(True)
        slam_cfg.use_gpu = False
        slam_cfg.vpr_model_path = "/nonexistent/anyloc.engine"
        with self.assertRaises(ValueError) as caught:
            vslam.Slam(make_rig()[0], [0, 1], slam_cfg)
        self.assertIn("use_gpu", str(caught.exception))

    def test_needs_a_score_threshold_in_0_1(self):
        for threshold in (-0.1, 1.5, float("nan")):
            _, slam_cfg = make_configs(True)
            slam_cfg.vpr_score_threshold = threshold
            slam_cfg.vpr_model_path = "/nonexistent/anyloc.engine"
            with self.assertRaises(ValueError, msg=threshold) as caught:
                vslam.Slam(make_rig()[0], [0, 1], slam_cfg)
            self.assertIn("vpr_score_threshold", str(caught.exception))

    def test_vocabulary_settings_need_use_vpr(self):
        for name, value in (("vpr_vocabulary_path", "/nonexistent/test.vocabulary"),
                            ("vpr_vocabulary_min_keyframes", 3), ("vpr_vocabulary_max_descriptors", 7)):
            _, slam_cfg = make_configs(False)
            setattr(slam_cfg, name, value)
            with self.assertRaises(ValueError, msg=name) as caught:
                vslam.Slam(make_rig()[0], [0, 1], slam_cfg)
            self.assertIn("use_vpr", str(caught.exception))

    def test_a_vocabulary_file_replaces_the_online_fit_settings(self):
        for name, value in (("vpr_vocabulary_min_keyframes", 3), ("vpr_vocabulary_max_descriptors", 7)):
            _, slam_cfg = make_configs(True)
            slam_cfg.vpr_model_path = "/nonexistent/anyloc.engine"
            slam_cfg.vpr_vocabulary_path = "/nonexistent/test.vocabulary"
            setattr(slam_cfg, name, value)
            with self.assertRaises(ValueError, msg=name) as caught:
                vslam.Slam(make_rig()[0], [0, 1], slam_cfg)
            self.assertIn("vpr_vocabulary_path", str(caught.exception))

    def test_online_fit_settings_must_be_positive(self):
        for name in ("vpr_vocabulary_min_keyframes", "vpr_vocabulary_max_descriptors"):
            _, slam_cfg = make_configs(True)
            slam_cfg.vpr_model_path = "/nonexistent/anyloc.engine"
            setattr(slam_cfg, name, 0)
            with self.assertRaises(ValueError, msg=name) as caught:
                vslam.Slam(make_rig()[0], [0, 1], slam_cfg)
            self.assertIn(name, str(caught.exception))

    def test_vocabulary_fitter_rejects_an_empty_training_set(self):
        # Checked before loading the engine.
        cfg = vslam.Slam.VprVocabularyFitter.Config(vpr_model_path="/nonexistent/anyloc.engine", max_descriptors=0)
        with self.assertRaises(ValueError) as caught:
            vslam.Slam.VprVocabularyFitter(cfg)
        self.assertIn("max_descriptors", str(caught.exception))

    def test_vocabulary_fitter_needs_an_engine(self):
        cfg = vslam.Slam.VprVocabularyFitter.Config(vpr_model_path="/nonexistent/anyloc.engine")
        with self.assertRaises(RuntimeError):
            vslam.Slam.VprVocabularyFitter(cfg)

    @needs_anyloc
    def test_localization_without_a_guess_pose_needs_a_candidate(self):
        slam = vslam.Slam(make_rig()[0], [0, 1], make_configs(True)[1])
        # Raises before reading the map, so the folder need not exist.
        with self.assertRaises(ValueError) as caught:
            localize_in_map(slam, "/nonexistent/map", noise_images(3), FRAME_PERIOD_NS,
                            localization_settings(vpr_candidates=0))
        self.assertIn("vpr_candidates", str(caught.exception))


@needs_anyloc
class TestVprVocabularyFitter(unittest.TestCase):
    """A vocabulary file fitted to noise images, and Slam loading it."""

    PATCHES = 529  # patch descriptors the engine makes of an image
    FILE_BYTES = 24 + 32 * 384 * 4 + 8  # header, 32 centers of 384 floats, no places

    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.path = os.path.join(self.folder, "noise.vocabulary")

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def make_fitter(self, max_descriptors=PATCHES * 10):
        return vslam.Slam.VprVocabularyFitter(
            vslam.Slam.VprVocabularyFitter.Config(vpr_model_path=ANYLOC_ENGINE, max_descriptors=max_descriptors))

    @staticmethod
    def add_images(fitter, count):
        for seed in range(count):
            fitter.add_image(noise_images(seed)[0])

    def vocabulary_config(self):
        _, slam_cfg = make_configs(True)
        slam_cfg.vpr_vocabulary_path = self.path
        return slam_cfg

    def test_capacity_counts_whole_images(self):
        fitter = self.make_fitter()
        self.assertEqual(fitter.capacity, 10)
        self.assertEqual(fitter.image_count, 0)

        self.add_images(fitter, 10)
        self.assertEqual(fitter.image_count, 10)
        with self.assertRaises(ValueError):
            fitter.add_image(noise_images(10)[0])
        self.assertEqual(fitter.image_count, 10)

    def test_slam_loads_a_saved_vocabulary(self):
        fitter = self.make_fitter()
        self.add_images(fitter, vslam.Slam.VprVocabularyFitter.min_images)
        fitter.save(self.path)
        self.assertEqual(os.path.getsize(self.path), self.FILE_BYTES)
        # Below capacity, so only the save refuses it.
        with self.assertRaises(ValueError):
            fitter.add_image(noise_images(10)[0])

        vslam.Slam(make_rig()[0], [0, 1], self.vocabulary_config())

    def test_save_needs_min_images(self):
        fitter = self.make_fitter()
        self.add_images(fitter, vslam.Slam.VprVocabularyFitter.min_images - 1)
        with self.assertRaises(ValueError):
            fitter.save(self.path)
        self.assertEqual(os.listdir(self.folder), [])

    def test_save_reports_a_path_it_cannot_write(self):
        fitter = self.make_fitter()
        self.add_images(fitter, vslam.Slam.VprVocabularyFitter.min_images)
        path = os.path.join(self.folder, "missing", "noise.vocabulary")
        with self.assertRaises(RuntimeError) as caught:
            fitter.save(path)
        self.assertIn(path, str(caught.exception))

    def test_training_set_must_hold_min_images(self):
        too_small = self.PATCHES * vslam.Slam.VprVocabularyFitter.min_images - 1
        with self.assertRaises(ValueError):
            self.make_fitter(too_small)

        _, slam_cfg = make_configs(True)
        slam_cfg.vpr_vocabulary_max_descriptors = too_small
        with self.assertRaises(ValueError):
            vslam.Slam(make_rig()[0], [0, 1], slam_cfg)

        # Holds 8 images: refused only if the session's minimum reaches AnyLoc.
        slam_cfg.vpr_vocabulary_max_descriptors = self.PATCHES * 8
        slam_cfg.vpr_vocabulary_min_keyframes = 9
        with self.assertRaises(ValueError):
            vslam.Slam(make_rig()[0], [0, 1], slam_cfg)
        slam_cfg.vpr_vocabulary_min_keyframes = 8
        vslam.Slam(make_rig()[0], [0, 1], slam_cfg)

    def test_localize_in_map_reports_a_vocabulary_gone_since_construction(self):
        fitter = self.make_fitter()
        self.add_images(fitter, vslam.Slam.VprVocabularyFitter.min_images)
        fitter.save(self.path)
        slam = vslam.Slam(make_rig()[0], [0, 1], self.vocabulary_config())
        os.remove(self.path)

        # Localizing reads the file again; its error reaches the callback instead of escaping.
        pose, error = localize_in_map(slam, self.folder, noise_images(0), 0, localization_settings())
        self.assertIsNone(pose)
        self.assertIn(self.path, error)

    def test_slam_rejects_a_file_that_is_no_vocabulary(self):
        with open(self.path, "wb") as file:
            file.write(b"not a vocabulary")
        with self.assertRaises(RuntimeError) as caught:
            vslam.Slam(make_rig()[0], [0, 1], self.vocabulary_config())
        self.assertIn(self.path, str(caught.exception))


@unittest.skipUnless(os.path.isdir(KITTI_07_DIR), f"needs real imagery at {KITTI_07_DIR}")
@needs_anyloc
class TestVprAnyLoc(unittest.TestCase):
    """End to end place recognition on KITTI-07; AnyLoc needs 8+ mapped places to answer."""

    # Before the loop: dozens of keyframes, no revisits.
    N_FRAMES = 300
    QUERY_STRIDE = 10
    # No revisits, so a match must be a node mapped about the same time.
    MAX_TIME_TO_NODE_NS = 3_000_000_000

    @classmethod
    def setUpClass(cls):
        try:
            from cuvslam_tools.tracker.edex_reader import EdexReader
        except ImportError as exc:
            raise unittest.SkipTest(f"needs cuvslam_tools to read the sequence: {exc}")
        cls.make_reader = staticmethod(
            lambda: EdexReader(KITTI_07_DIR, stereo_edex=os.path.join(KITTI_07_DIR, "stereo.edex")))
        cls.rig = cls.make_reader().rig
        cls.map_dir = tempfile.mkdtemp()
        cls.vocabulary_dir = tempfile.mkdtemp()
        cls.tracker = cls.make_tracker(cls.anyloc_config())
        cls.tracker.slam.get_pose_graph()  # enables publishing the pose graph
        cls.frames = cls.track_prefix(cls.tracker)
        if not save_map(cls.tracker.slam, cls.map_dir):
            raise RuntimeError(f"could not save the KITTI-07 map to {cls.map_dir}")
        cls.vocabulary_path = os.path.join(cls.vocabulary_dir, "kitti_07.vocabulary")
        cls.fit_vocabulary(cls.vocabulary_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.map_dir, ignore_errors=True)
        shutil.rmtree(cls.vocabulary_dir, ignore_errors=True)

    @staticmethod
    def anyloc_config():
        _, slam_cfg = make_configs(True)
        slam_cfg.max_map_size = 0  # never merge nodes out of the VPR map
        return slam_cfg

    @classmethod
    def fit_vocabulary(cls, path):
        """Fit a vocabulary file to images spread over the whole sequence, up to the fitter's capacity."""
        fitter = vslam.Slam.VprVocabularyFitter(vslam.Slam.VprVocabularyFitter.Config(vpr_model_path=ANYLOC_ENGINE))
        reader = cls.make_reader()
        reader.replay(_VocabularyReplay(fitter, reader.total_frames // fitter.capacity + 1))
        fitter.save(path)

    @classmethod
    def make_tracker(cls, slam_cfg):
        odom_cfg, _ = make_configs(False)
        return vslam.Tracker(cls.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)

    @classmethod
    def track_prefix(cls, tracker, n_frames=None):
        """Track the sequence start into the map and return the frames kept to query."""
        replay = _PrefixReplay(tracker, n_frames or cls.N_FRAMES, cls.QUERY_STRIDE)
        try:
            cls.make_reader().replay(replay)
        except _StopReplay:
            pass
        return replay.kept

    def recognize_all(self, slam, frames):
        """Query every kept frame and return the confidently recognized ones."""
        recognized = []
        for timestamp, images in frames:
            place = recognize_match(slam, images, timestamp)
            if place is not None and place.score > 0.9:
                recognized.append((timestamp, place))
        return recognized

    def test_mapped_frames_recognize_themselves(self):
        recognized = self.recognize_all(self.tracker.slam, self.frames)
        self.assertGreater(len(recognized), 0, "the sequence mapped no recognizable place")

        # A match names a node of this session and reports its live pose.
        nodes = {node.id: node.node_pose for node in self.tracker.slam.get_pose_graph().nodes}
        for timestamp, place in recognized:
            self.assertIn(place.node_id, nodes)
            np.testing.assert_allclose(
                np.array(place.pose.translation), np.array(nodes[place.node_id].translation), atol=1e-5)
            self.assertLessEqual(place.score, 1.0 + 1e-5)
            self.assertLessEqual(abs(place.timestamp_ns - timestamp), self.MAX_TIME_TO_NODE_NS)

    def test_the_reader_loads_the_images_its_replay_delivers(self):
        # cuvslam_vpr_vocabulary reads the images it fits to this way.
        replay = _FirstFrames(3)
        reader = self.make_reader()
        try:
            reader.replay(replay)
        except _StopReplay:
            pass
        self.assertEqual(len(replay.frames), 3)
        for frame_id, images in replay.frames:
            for camera, image in enumerate(images):
                np.testing.assert_array_equal(reader.load_camera_image(frame_id, camera), image)
        frame_id, images = replay.frames[0]
        with self.assertRaises(ValueError):
            reader.load_camera_image(frame_id, len(images))

    def test_unrelated_image_is_not_recognized(self):
        timestamp, images = self.frames[-1]
        rng = np.random.RandomState(99)
        noise = [np.ascontiguousarray(rng.randint(0, 256, image.shape).astype(image.dtype)) for image in images]

        place = recognize_match(self.tracker.slam, noise, timestamp + 1)
        if place is not None:
            self.assertLess(place.score, 0.9, "an unrelated image was recognized")

    def test_a_new_session_gets_the_places_by_localizing_in_the_saved_map(self):
        # One LMDB database holds both maps, so localizing in it brings the places along.
        self.assertEqual(os.listdir(self.map_dir), ["data.mdb"])
        mapped = self.recognize_all(self.tracker.slam, self.frames)
        self.assertTrue(mapped)

        session = self.make_tracker(self.anyloc_config())
        localized_at = None
        for timestamp, images in self.frames[::3]:
            session.track(timestamp, images)
            pose, _ = localize_in_map(session.slam, self.map_dir, images, timestamp, localization_settings())
            if pose is not None:
                localized_at = timestamp
                break
        self.assertIsNotNone(localized_at, "no frame relocalized in the saved map")

        # Same node ids and timestamps: the VPR map came with the pose graph its ids refer to.
        images_by_timestamp = dict(self.frames)
        for timestamp, place in mapped:
            if timestamp == localized_at:
                continue  # its own frame was just added to the map again
            reloaded = recognize_match(session.slam, images_by_timestamp[timestamp], timestamp)
            self.assertIsNotNone(reloaded)
            self.assertGreater(reloaded.score, 0.9)
            self.assertEqual(reloaded.node_id, place.node_id)
            self.assertEqual(reloaded.timestamp_ns, place.timestamp_ns)

    def test_an_attached_map_keeps_its_places_when_the_session_ends(self):
        # With map_path the map lives in LMDB; ending the session writes the VPR map there too.
        map_dir = tempfile.mkdtemp()
        try:
            slam_cfg = self.anyloc_config()
            slam_cfg.map_path = map_dir
            tracker = self.make_tracker(slam_cfg)
            frames = self.track_prefix(tracker, n_frames=200)
            del tracker

            results = [self.relocalize(timestamp, images, localization_settings(), map_dir)
                       for timestamp, images in frames[::2]]
            self.assertTrue(any(pose is not None for pose, _ in results),
                            f"nothing relocalized in the attached map; errors were {sorted({e for _, e in results})}")
        finally:
            shutil.rmtree(map_dir, ignore_errors=True)

    def relocalize(self, timestamp, images, settings, folder, slam_cfg=None):
        """Localize one frame of a fresh session in the saved map, without a guess pose."""
        tracker = self.make_tracker(slam_cfg or self.anyloc_config())
        tracker.track(timestamp, images)
        return localize_in_map(tracker.slam, folder, images, timestamp, settings)

    def test_localization_without_a_guess_pose_is_seeded_by_the_saved_index(self):
        # Kidnapped robot: the saved index proposes candidates, each verified geometrically.
        settings = localization_settings(vpr_candidates=3)
        results = [self.relocalize(timestamp, images, settings, self.map_dir)
                   for timestamp, images in self.frames[::3]]
        localized = [pose for pose, _ in results if pose is not None]
        self.assertTrue(localized, "no frame of the sequence relocalized without a guess pose; "
                                   f"errors were {sorted({error for _, error in results})}")
        # Failures come from places VPR did not recognize or candidates that did not verify.
        for pose, error in results:
            if pose is None:
                self.assertTrue("did not recognize this place" in error or "Can't localize" in error, error)

    def test_localization_without_a_guess_pose_needs_a_map_that_carries_an_index(self):
        # Saved without place recognition, so it must fail rather than grid-search the whole map.
        _, slam_cfg = make_configs(False)
        tracker = self.make_tracker(slam_cfg)
        frames = self.track_prefix(tracker, n_frames=60)
        folder = tempfile.mkdtemp()
        try:
            self.assertTrue(save_map(tracker.slam, folder))
            del tracker

            timestamp, images = frames[0]
            pose, error = self.relocalize(timestamp, images, localization_settings(vpr_candidates=3), folder)

            self.assertIsNone(pose)
            self.assertIn("did not recognize this place", error)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_a_session_of_another_vocabulary_leaves_the_saved_index_unused(self):
        # The saved map fitted its own vocabulary, so only a guess pose localizes in it.
        slam_cfg = self.anyloc_config()
        slam_cfg.vpr_vocabulary_path = self.vocabulary_path
        poses = {stamped.timestamp_ns: stamped.pose for stamped in self.tracker.slam.get_all_slam_poses()}

        localized = 0
        for timestamp, images in self.frames[::3]:
            session = self.make_tracker(slam_cfg)
            session.track(timestamp, images)
            pose, error = localize_in_map(session.slam, self.map_dir, images, timestamp,
                                          localization_settings(vpr_candidates=3))
            self.assertIsNone(pose)
            self.assertIn("did not recognize this place", error)

            pose, _ = localize_in_map(session.slam, self.map_dir, images, timestamp, localization_settings(),
                                      poses[timestamp])
            localized += pose is not None
        self.assertGreater(localized, 0, "no frame localized in the saved map even with a guess pose")

    def test_sessions_that_share_a_vocabulary_file_localize_in_each_others_maps(self):
        slam_cfg = self.anyloc_config()
        slam_cfg.vpr_vocabulary_path = self.vocabulary_path
        tracker = self.make_tracker(slam_cfg)
        frames = self.track_prefix(tracker, n_frames=200)
        folder = tempfile.mkdtemp()
        try:
            self.assertTrue(save_map(tracker.slam, folder))
            del tracker

            results = [self.relocalize(timestamp, images, localization_settings(vpr_candidates=3), folder, slam_cfg)
                       for timestamp, images in frames[::3]]
            self.assertTrue(any(pose is not None for pose, _ in results),
                            f"no frame relocalized without a guess pose; errors were {sorted({e for _, e in results})}")
        finally:
            shutil.rmtree(folder, ignore_errors=True)


@unittest.skipUnless(os.path.isdir(KITTI_07_DIR), f"needs a loop-closing sequence at {KITTI_07_DIR}")
@needs_anyloc
class TestVprFollowsPoseGraph(unittest.TestCase):
    """A recognized place reports its node's current pose, not the one it was mapped with."""

    # Queried mid-run and again at the end; early, so the late loop closures can move them.
    EARLY_FRAMES = (60, 100, 140, 180)
    # Stride of the frames checked against the final pose graph.
    QUERY_STRIDE = 50
    # Minimum early-node motion for the test to prove anything.
    MIN_DISPLACEMENT_M = 0.01
    # Querying a map with its own sequence is the easiest case, so the bar is high.
    MIN_RECALL = 0.9
    MAX_MATCH_DISTANCE_M = 10.0

    def setUp(self):
        try:
            from cuvslam_tools.tracker.edex_reader import EdexReader
        except ImportError as exc:
            self.skipTest(f"needs cuvslam_tools to read the sequence: {exc}")

        self.reader = EdexReader(KITTI_07_DIR, stereo_edex=os.path.join(KITTI_07_DIR, "stereo.edex"))
        self.assertTrue(self.reader.validate_rig(), f"invalid rig in {KITTI_07_DIR}")

        odom_cfg = vslam.Odometry.Config()
        odom_cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        odom_cfg.rectified_stereo_camera = True
        odom_cfg.async_sba = False
        odom_cfg.enable_observations_export = True
        odom_cfg.enable_landmarks_export = True

        slam_cfg = vslam.Slam.Config()
        slam_cfg.sync_mode = True
        slam_cfg.enable_reading_internals = True
        # Unbounded, so early nodes are not merged away before the loop closes.
        slam_cfg.max_map_size = 0
        slam_cfg.use_vpr = True
        slam_cfg.vpr_model_path = ANYLOC_ENGINE

        self.tracker = vslam.Tracker(
            self.reader.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        self.tracker.slam.get_pose_graph()  # enables publishing the pose graph

    def test_recognized_pose_follows_the_optimization(self):
        replay = _VprReplay(self, self.tracker)
        self.reader.replay(replay)
        self.assertGreater(replay.n_frames, 0, f"no frames replayed from {KITTI_07_DIR}")

        loop_closures = self.tracker.slam.get_loop_closure_poses()
        self.assertGreater(len(loop_closures), 0,
                           f"{KITTI_07_DIR} closed no loop, so no optimization moved any node")

        nodes = {node.id: node.node_pose for node in self.tracker.slam.get_pose_graph().nodes}
        self.assertGreater(len(nodes), 0, "pose graph came back empty")

        # (a) Answers carry the named node's live pose, near where the frame was taken.
        n_checked = 0
        far = []
        for frame_id, (timestamp, images, tracked_pose) in sorted(replay.kept_frames.items()):
            place = recognize_match(self.tracker.slam, images, timestamp)
            if place is None:
                continue
            self.assertIn(place.node_id, nodes, "a match named a node the pose graph does not have")
            place_t, place_q = pose_arrays(place.pose)
            node_t, node_q = pose_arrays(nodes[place.node_id])
            np.testing.assert_allclose(place_t, node_t, atol=1e-5)
            self.assertAlmostEqual(abs(float(np.dot(place_q, node_q))), 1.0, places=5)
            distance = float(np.linalg.norm(place_t - pose_arrays(tracked_pose)[0]))
            if distance > self.MAX_MATCH_DISTANCE_M:
                far.append((frame_id, distance))
            n_checked += 1
        self.assertGreater(n_checked, 0, "no frame was recognized, so nothing was checked")

        recall = n_checked / len(replay.kept_frames)
        print(f"place recognition recall over {len(replay.kept_frames)} sampled frames of "
              f"{replay.n_frames}: {recall:.2f}, worst match distance "
              f"{max((d for _, d in far), default=0.0):.2f} m")
        self.assertGreaterEqual(
            recall, self.MIN_RECALL,
            f"only {recall:.2f} of the sampled frames were recognized in a map built from the same "
            f"sequence, below the {self.MIN_RECALL:.2f} this has held at")
        self.assertFalse(far, f"matches further than {self.MAX_MATCH_DISTANCE_M} m from where the "
                              f"frame was taken: {far}")

        # (b) The optimization did move those nodes.
        displacements = {}
        for frame_id, (timestamp, images, first) in sorted(replay.early_results.items()):
            again = recognize_match(self.tracker.slam, images, timestamp)
            if again is None or again.node_id != first.node_id:
                # Matched another node, so the pose difference is meaningless.
                continue
            displacements[frame_id] = float(
                np.linalg.norm(pose_arrays(again.pose)[0] - pose_arrays(first.pose)[0]))

        print(f"\nplace recognition pose displacement over {replay.n_frames} frames and "
              f"{len(loop_closures)} loop closures, per early frame (m): "
              + ", ".join(f"{frame_id}: {value:.4f}" for frame_id, value in displacements.items()))

        self.assertTrue(displacements,
                        "no early frame kept matching the same node, so this run compared nothing; "
                        "the check above is vacuous - pick other early frames or another sequence")
        worst = max(displacements.values())
        self.assertGreater(
            worst, self.MIN_DISPLACEMENT_M,
            f"the optimization moved no early node by more than {worst:.4f} m, so this run does NOT "
            f"show that place recognition follows the pose graph - it would pass unchanged against a "
            f"backend that reported the pose stored when the frame was mapped. Pick an early frame "
            f"or a sequence where the optimization actually moves a node.")


@unittest.skipUnless(os.path.isdir(KITTI_07_DIR), f"needs a loop-closing sequence at {KITTI_07_DIR}")
@needs_anyloc
class TestVprLoopClosureMode(unittest.TestCase):
    """LoopClosureMode.VPR closes real loops from place recognition candidates."""

    # Loop closures must revisit, in ground truth, a place this close and this old.
    REVISIT_RADIUS_M = 15.0
    REVISIT_MIN_AGE_NS = 20_000_000_000

    def setUp(self):
        try:
            from cuvslam_tools.tracker.edex_reader import EdexReader
        except ImportError as exc:
            self.skipTest(f"needs cuvslam_tools to read the sequence: {exc}")

        self.reader = EdexReader(KITTI_07_DIR, stereo_edex=os.path.join(KITTI_07_DIR, "stereo.edex"))
        self.assertTrue(self.reader.validate_rig(), f"invalid rig in {KITTI_07_DIR}")

        odom_cfg = vslam.Odometry.Config()
        odom_cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        odom_cfg.rectified_stereo_camera = True
        odom_cfg.async_sba = False
        odom_cfg.enable_observations_export = True
        odom_cfg.enable_landmarks_export = True

        slam_cfg = vslam.Slam.Config()
        slam_cfg.sync_mode = True
        slam_cfg.enable_reading_internals = True
        slam_cfg.max_map_size = 0
        slam_cfg.use_vpr = True
        slam_cfg.vpr_model_path = ANYLOC_ENGINE
        slam_cfg.loop_closure_mode = vslam.Slam.LoopClosureMode.VPR

        self.tracker = vslam.Tracker(
            self.reader.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        self.tracker.slam.get_pose_graph()  # enables publishing the pose graph

    def test_vpr_driven_loop_closure_closes_a_loop(self):
        replay = _VprReplay(self, self.tracker)
        self.reader.replay(replay)
        self.assertGreater(replay.n_frames, 0, f"no frames replayed from {KITTI_07_DIR}")

        loop_closures = self.tracker.slam.get_loop_closure_poses()
        self.assertGreater(
            len(loop_closures), 0,
            f"loop_closure_mode=VPR closed no loop on {KITTI_07_DIR}, which LSI does close")

        # KITTI ground truth has one row per frame, in replay order.
        gt_positions = np.loadtxt(os.path.join(KITTI_07_DIR, "gt.txt")).reshape(-1, 3, 4)[:, :, 3]
        self.assertEqual(len(gt_positions), len(replay.timestamps))
        frame_of = {timestamp: frame for frame, timestamp in enumerate(replay.timestamps)}
        for loop_closure in loop_closures:
            frame = frame_of[loop_closure.timestamp_ns]
            earlier = [f for f, timestamp in enumerate(replay.timestamps)
                       if loop_closure.timestamp_ns - timestamp >= self.REVISIT_MIN_AGE_NS]
            self.assertTrue(earlier, f"loop closure at frame {frame} has no earlier past to be back at")
            distance = np.min(np.linalg.norm(gt_positions[earlier] - gt_positions[frame], axis=1))
            self.assertLess(distance, self.REVISIT_RADIUS_M,
                            f"loop closure at frame {frame} is {distance:.1f} m from anywhere the car had been")


class _VprReplay:
    """EdexReader processor that tracks and maps every frame and keeps a few to query."""

    def __init__(self, test, tracker):
        self.test = test
        self.tracker = tracker
        self.kept_frames = {}     # frame_id -> (timestamp, images, pose), queried after tracking
        self.early_results = {}   # frame_id -> (timestamp, images, place found mid-run)
        self.timestamps = []      # of every frame, in replay order
        self.n_frames = 0

    def process_images(self, frame_id, timestamps, images, masks, depths=None):
        timestamp = max(timestamps)
        self.tracker.track(timestamp, images, masks, depths)
        self.timestamps.append(timestamp)
        self.n_frames += 1

        # The reader reuses its images list, so copy kept frames.
        if frame_id % TestVprFollowsPoseGraph.QUERY_STRIDE == 0:
            self.kept_frames[frame_id] = (timestamp,
                                          [np.array(image, copy=True) for image in images],
                                          self.tracker.slam.get_pose())
        if frame_id in TestVprFollowsPoseGraph.EARLY_FRAMES:
            place = recognize_match(self.tracker.slam, images, timestamp)
            if place is not None:
                self.early_results[frame_id] = (
                    timestamp, [np.array(image, copy=True) for image in images], place)

    def process_imu(self, timestamp, linear_accelerations, angular_velocities):
        """Ignore IMU samples: place recognition is answered from images alone."""

    def get_camera_pose(self, frame_id):
        """Poses come from the pose graph instead."""
        return None

    def set_frame_metadata(self, frame_id, metadata):
        """Ignore replay metadata: the sequence is replayed once, forward."""


class _StopReplay(Exception):
    """Ends EdexReader.replay() early."""


class _PrefixReplay:
    """EdexReader processor that maps the first n_frames and keeps every stride-th."""

    def __init__(self, tracker, n_frames, stride):
        self.tracker = tracker
        self.n_frames = n_frames
        self.stride = stride
        self.kept = []  # (timestamp, images) to query with
        self.n_tracked = 0

    def process_images(self, frame_id, timestamps, images, masks, depths=None):
        if self.n_tracked == self.n_frames:
            raise _StopReplay()
        timestamp = max(timestamps)
        self.tracker.track(timestamp, images, masks, depths)
        # The reader reuses its images list, so copy kept frames.
        if self.n_tracked % self.stride == 0:
            self.kept.append((timestamp, [np.array(image, copy=True) for image in images]))
        self.n_tracked += 1

    def process_imu(self, timestamp, linear_accelerations, angular_velocities):
        """Ignore IMU samples: place recognition is answered from images alone."""

    def get_camera_pose(self, frame_id):
        """Poses come from the pose graph instead."""
        return None

    def set_frame_metadata(self, frame_id, metadata):
        """Ignore replay metadata: the sequence is replayed once, forward."""


class _FirstFrames:
    """EdexReader processor that keeps the ids and images of the first n_frames."""

    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.frames = []  # (frame_id, images)

    def process_images(self, frame_id, timestamps, images, masks, depths=None):
        if len(self.frames) == self.n_frames:
            raise _StopReplay()
        # The reader reuses its images list, so copy them.
        self.frames.append((frame_id, [np.array(image, copy=True) for image in images]))

    def process_imu(self, timestamp, linear_accelerations, angular_velocities):
        """Ignore IMU samples: only the images are compared."""

    def get_camera_pose(self, frame_id):
        """Nothing is tracked."""
        return None

    def set_frame_metadata(self, frame_id, metadata):
        """Ignore replay metadata: the sequence is replayed once, forward."""


class _VocabularyReplay:
    """EdexReader processor that adds the first camera's image of every stride-th frame to a vocabulary fitter."""

    def __init__(self, fitter, stride):
        self.fitter = fitter
        self.stride = stride
        self.n_frames = 0

    def process_images(self, frame_id, timestamps, images, masks, depths=None):
        if self.n_frames % self.stride == 0:
            self.fitter.add_image(images[0])
        self.n_frames += 1

    def process_imu(self, timestamp, linear_accelerations, angular_velocities):
        """Ignore IMU samples: the vocabulary is fitted to images alone."""

    def get_camera_pose(self, frame_id):
        """Nothing is tracked."""
        return None

    def set_frame_metadata(self, frame_id, metadata):
        """Ignore replay metadata: the sequence is replayed once, forward."""


if __name__ == "__main__":
    unittest.main()
