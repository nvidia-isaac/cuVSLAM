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

VPR_MAP_FILE = "vpr_map.bin"   # what Slam.save_map writes for the place recognition part of a map

# A sequence that loop-closes, needed by TestVprFollowsPoseGraph: the optimization is only observable
# where there is a loop to close. Point CUVSLAM_KITTI_07_DIR at one directly, or let it fall out of
# the CUVSLAM_DATASETS root the rest of the repo's tooling uses. Nothing else in this file needs a
# dataset, so that one test skips itself when neither resolves.
KITTI_07_DIR = os.environ.get("CUVSLAM_KITTI_07_DIR") or os.path.join(
    os.environ.get("CUVSLAM_DATASETS", ""), "kitti", "07"
)

def make_rig():
    cameras = data.generate_stereo_camera(W, H, BASELINE)
    return vslam.Rig(cameras, [vslam.ImuCalibration()]), cameras

def make_configs(vpr_mode, vpr_map_path=""):
    odom_cfg = vslam.Odometry.Config()
    odom_cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
    odom_cfg.rectified_stereo_camera = True
    odom_cfg.async_sba = False

    slam_cfg = vslam.Slam.Config()
    slam_cfg.sync_mode = True
    # A match names a pose graph node, so the tests need to read the graph back to check it.
    slam_cfg.enable_reading_internals = True
    slam_cfg.vpr_mode = vpr_mode
    slam_cfg.vpr_map_path = vpr_map_path

    return odom_cfg, slam_cfg

def noise_images(seed):
    """Stereo pair of untrackable noise, used where only the pixels matter."""
    rng = np.random.RandomState(seed)
    return [np.ascontiguousarray(rng.randint(0, 256, (H, W), dtype=np.uint8)) for _ in range(2)]

def recognize_place(slam, images, timestamp_ns):
    """Query place recognition and return the (result, error_message) pair the callback got.

    `recognize_place_by_frame` is asynchronous, but `Slam.Config.sync_mode` makes the callback run
    before the call returns, so collecting it into a local and reading it afterwards is enough.
    """
    answers = []
    slam.recognize_place_by_frame(images, timestamp_ns, lambda result, error: answers.append((result, error)))
    if len(answers) != 1:
        raise AssertionError(f"sync_mode must call the callback exactly once, it ran {len(answers)} times")
    return answers[0]

def localization_settings(vpr_candidates=0):
    """Localization settings with the small probe grid the synthetic map needs."""
    return vslam.Slam.LocalizationSettings(
        horizontal_search_radius=0.25, vertical_search_radius=0.25,
        horizontal_step=0.0625, vertical_step=0.0625, angular_step_rads=0.03125,
        vpr_candidates=vpr_candidates)

def localize_in_map(slam, folder, images, timestamp_ns, settings, guess_pose=None):
    """Localize and return the (pose, error_message) pair the callback got. Synchronous for the same
    reason `recognize_place` is."""
    answers = []
    slam.localize_in_map(folder, timestamp_ns, guess_pose, images, settings,
                         lambda: None, lambda pose, error: answers.append((pose, error)))
    if len(answers) != 1:
        raise AssertionError(f"sync_mode must call the callback exactly once, it ran {len(answers)} times")
    return answers[0]

def save_map(slam, folder):
    """Save the map and return the flag the callback reported. Synchronous for the same reason."""
    answers = []
    slam.save_map(folder, answers.append)
    if len(answers) != 1:
        raise AssertionError(f"sync_mode must call the callback exactly once, it ran {len(answers)} times")
    return answers[0]

def pose_arrays(pose):
    """Translation and rotation of a Pose as numpy arrays, for comparing two poses."""
    return np.array(pose.translation, dtype=np.float64), np.array(pose.rotation, dtype=np.float64)


class TestVprBindings(unittest.TestCase):
    """The place recognition API surface, without a tracker."""

    def test_vpr_mode_enum(self):
        self.assertEqual(set(vslam.Slam.VprMode.__members__), {'Off', 'Simple', 'DBoW2', 'AnyLoc', 'Bow'})
        self.assertEqual(vslam.Slam.Config().vpr_mode, vslam.Slam.VprMode.Off)

    def test_config_vpr_keyword_arguments(self):
        cfg = vslam.Slam.Config(
            vpr_mode=vslam.Slam.VprMode.Simple,
            vpr_map_path="/tmp/vpr_map",
            vpr_model_path="/tmp/dinov2.onnx",
            vpr_score_threshold=0.75)

        self.assertEqual(cfg.vpr_mode, vslam.Slam.VprMode.Simple)
        self.assertEqual(cfg.vpr_map_path, "/tmp/vpr_map")
        self.assertEqual(cfg.vpr_model_path, "/tmp/dinov2.onnx")
        self.assertAlmostEqual(cfg.vpr_score_threshold, 0.75, places=5)

    def test_config_vpr_modifiers(self):
        cfg = vslam.Slam.Config()
        cfg.vpr_mode = vslam.Slam.VprMode.AnyLoc
        cfg.vpr_map_path = "/tmp/other_map"
        cfg.vpr_model_path = "/tmp/other.onnx"
        cfg.vpr_score_threshold = 0.25

        self.assertEqual(cfg.vpr_mode, vslam.Slam.VprMode.AnyLoc)
        self.assertEqual(cfg.vpr_map_path, "/tmp/other_map")
        self.assertEqual(cfg.vpr_model_path, "/tmp/other.onnx")
        self.assertAlmostEqual(cfg.vpr_score_threshold, 0.25, places=5)

    def test_config_repr_mentions_vpr(self):
        text = repr(vslam.Slam.Config(vpr_mode=vslam.Slam.VprMode.Simple, vpr_score_threshold=0.5))
        for field in ('vpr_mode', 'vpr_map_path', 'vpr_model_path', 'vpr_score_threshold'):
            self.assertIn(field, text)
        self.assertIn('Simple', text)

    def test_place_recognition_fields(self):
        place = vslam.Slam.PlaceRecognition()
        self.assertFalse(place.found)

        place.found = True
        place.node_id = 7
        place.pose = vslam.Pose(translation=[1.0, 2.0, 3.0], rotation=[0.0, 0.0, 0.0, 1.0])
        place.score = 0.875
        place.timestamp_ns = 123456789
        place.imported = True

        self.assertTrue(place.found)
        self.assertEqual(place.node_id, 7)
        self.assertEqual(list(place.pose.translation), [1.0, 2.0, 3.0])
        self.assertAlmostEqual(place.score, 0.875, places=5)
        self.assertEqual(place.timestamp_ns, 123456789)
        self.assertTrue(place.imported)
        self.assertIn('score', repr(place))
        self.assertIn('imported', repr(place))

    def test_localization_settings_round_trip_vpr_candidates(self):
        settings = vslam.Slam.LocalizationSettings(
            horizontal_search_radius=1.0, vertical_search_radius=0.5,
            horizontal_step=0.25, vertical_step=0.25, angular_step_rads=0.1,
            vpr_candidates=7)
        self.assertEqual(settings.vpr_candidates, 7)
        self.assertIn('vpr_candidates=7', repr(settings))

        settings.vpr_candidates = 3
        self.assertEqual(settings.vpr_candidates, 3)
        # 0 is not "no candidates", it is "use the backend default", so it has to stay settable.
        settings.vpr_candidates = 0
        self.assertEqual(settings.vpr_candidates, 0)


class TestVprOff(unittest.TestCase):
    """VprMode.Off has to leave every place recognition call inert rather than fail."""

    def setUp(self):
        self.rig, _ = make_rig()
        self.images = noise_images(1)
        odom_cfg, slam_cfg = make_configs(vslam.Slam.VprMode.Off)
        self.tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)

    def test_add_frame_is_a_no_op(self):
        # Offering a frame must neither raise nor quietly switch place recognition on, so the query
        # that follows still says the mode is Off rather than answering from a map built behind it.
        self.tracker.slam.add_frame_to_vpr_map(self.images, FRAME_PERIOD_NS)

        _, error = recognize_place(self.tracker.slam, self.images, FRAME_PERIOD_NS)
        self.assertIn("vpr_mode", error)

    def test_recognize_reports_an_error_instead_of_a_result(self):
        result, error = recognize_place(self.tracker.slam, self.images, FRAME_PERIOD_NS)

        # An error and no result, rather than a raised exception or a PlaceRecognition that reads as
        # a genuine "I have never been here", which a caller cannot tell from a misconfigured build.
        self.assertIsNone(result)
        self.assertIn("vpr_mode", error)

    def test_localize_in_map_without_a_guess_pose_is_rejected(self):
        # Without a guess pose the map is searched by appearance, and with place recognition off
        # there is nothing to search it with. That is a programming error rather than a failed
        # localization, so it raises instead of reporting through the callback - and it raises
        # before the map folder is touched, which is why this one does not have to exist.
        with self.assertRaises(ValueError) as caught:
            localize_in_map(self.tracker.slam, "/nonexistent/map", self.images, FRAME_PERIOD_NS,
                            localization_settings())
        self.assertIn("vpr_mode", str(caught.exception))


class TestVprFrameSelection(unittest.TestCase):
    """Only a primary camera's pixels are comparable to the map; a frame without any is refused."""

    def setUp(self):
        self.rig, _ = make_rig()
        self.images = noise_images(2)
        _, self.slam_cfg = make_configs(vslam.Slam.VprMode.Simple)

    def test_a_frame_with_no_images_reports_an_error(self):
        slam = vslam.Slam(self.rig, [0, 1], self.slam_cfg)

        result, error = recognize_place(slam, [], FRAME_PERIOD_NS)

        self.assertIsNone(result)
        self.assertIn("primary camera", error)

    def test_a_frame_from_no_primary_camera_reports_an_error(self):
        # SLAM maps primary cameras only, so a frame that carries none has nothing comparable to the
        # map. Querying with another camera's pixels would come back as "this place is not in the
        # map", which is indistinguishable from the truthful answer.
        slam = vslam.Slam(self.rig, [1], self.slam_cfg)

        result, error = recognize_place(slam, self.images[:1], FRAME_PERIOD_NS)

        self.assertIsNone(result)
        self.assertIn("primary camera", error)


class TestVprSimple(unittest.TestCase):
    """End to end place recognition over a synthetic sequence; needs no dataset."""

    def setUp(self):
        self.rig, cameras = make_rig()
        self.generator = data.ImageGenerator(cameras, STEPS)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # save_map writes the LMDB landmark database next to the place recognition map, so this is
        # megabytes rather than the kilobytes the recognition map alone costs.
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def track_sequence(self, tracker):
        """Track the synthetic sequence, offering every frame to the map. Returns the frames."""
        frames = []
        for i in range(STEPS):
            images, _ = self.generator.generate_zoomed_images(i)
            timestamp = i * FRAME_PERIOD_NS
            tracker.track(timestamp, images)
            tracker.slam.add_frame_to_vpr_map(images, timestamp)
            frames.append((timestamp, images))
        return frames

    def build_map(self):
        """Track the sequence into a fresh session's place recognition map."""
        odom_cfg, slam_cfg = make_configs(vslam.Slam.VprMode.Simple)
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        # Reading the pose graph enables the layer that publishes it, so it is populated by the end.
        tracker.slam.get_pose_graph()

        frames = self.track_sequence(tracker)
        return tracker, frames

    def recognize_all(self, slam, frames):
        """Query every tracked frame, keeping the ones the map recognizes as itself."""
        recognized = []
        for timestamp, images in frames:
            place, error = recognize_place(slam, images, timestamp)
            self.assertEqual(error, "")
            if place.found and place.score > 0.9:
                recognized.append((timestamp, place))
        return recognized

    def test_mapped_frames_recognize_themselves(self):
        tracker, frames = self.build_map()

        recognized = self.recognize_all(tracker.slam, frames)
        self.assertGreater(len(recognized), 0, "the synthetic sequence mapped no recognizable place")

        # A match names a pose graph node of this session, and reports that node's live pose.
        nodes = {node.id: node.node_pose for node in tracker.slam.get_pose_graph().nodes}
        for timestamp, place in recognized:
            self.assertFalse(place.imported, "a place mapped by this session is not an imported one")
            self.assertIn(place.node_id, nodes)
            np.testing.assert_allclose(
                np.array(place.pose.translation), np.array(nodes[place.node_id].translation), atol=1e-5)
            self.assertLessEqual(place.score, 1.0 + 1e-5)
            self.assertLessEqual(abs(place.timestamp_ns - timestamp), 2 * FRAME_PERIOD_NS)

    def test_unrelated_image_is_not_recognized(self):
        tracker, _ = self.build_map()

        place, error = recognize_place(tracker.slam, noise_images(99), STEPS * FRAME_PERIOD_NS)
        self.assertEqual(error, "")
        self.assertTrue(not place.found or place.score < 0.9, f"unrelated image scored {place.score}")

    def test_saved_map_recognizes_places_in_a_new_session(self):
        tracker, frames = self.build_map()
        mapped = self.recognize_all(tracker.slam, frames)
        self.assertTrue(save_map(tracker.slam, self.temp_dir))
        self.assertIn(VPR_MAP_FILE, os.listdir(self.temp_dir))
        del tracker

        odom_cfg, slam_cfg = make_configs(vslam.Slam.VprMode.Simple, vpr_map_path=self.temp_dir)
        loaded = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)

        images_by_timestamp = {timestamp: images for timestamp, images in frames}
        for timestamp, place in mapped:
            reloaded, error = recognize_place(loaded.slam, images_by_timestamp[timestamp], timestamp)
            self.assertEqual(error, "")
            self.assertTrue(reloaded.found)
            self.assertGreater(reloaded.score, 0.9)
            self.assertEqual(reloaded.timestamp_ns, place.timestamp_ns)
            # An imported entry carries the pose the session that mapped it had, and is renumbered so
            # its id cannot collide with the ids of the pose graph that loaded it.
            self.assertTrue(reloaded.imported)
            self.assertNotEqual(reloaded.node_id, place.node_id)
            np.testing.assert_array_almost_equal(
                np.array(reloaded.pose.translation), np.array(place.pose.translation), decimal=5)

    def test_loaded_map_is_read_only(self):
        tracker, frames = self.build_map()
        self.assertTrue(save_map(tracker.slam, self.temp_dir))
        del tracker

        odom_cfg, slam_cfg = make_configs(vslam.Slam.VprMode.Simple, vpr_map_path=self.temp_dir)
        loaded = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        before = [recognize_place(loaded.slam, images, timestamp)[0] for timestamp, images in frames]

        # The loading session tracks the very same route and offers every frame, which without the
        # read-only rule would put its own keyframes in the map and win most of these queries.
        self.track_sequence(loaded)

        after = [recognize_place(loaded.slam, images, timestamp)[0] for timestamp, images in frames]
        self.assertTrue(any(place.found for place in before))
        for old, new in zip(before, after):
            self.assertEqual((old.found, old.node_id, old.timestamp_ns), (new.found, new.node_id, new.timestamp_ns))
            self.assertAlmostEqual(old.score, new.score, places=6)
            np.testing.assert_array_equal(np.array(old.pose.translation), np.array(new.pose.translation))

        # Saving reports the SLAM map, which was written; the recognition map declines, which is not
        # a failure of the save. That decline is visible as the missing file: VprMap::Save returns
        # before writing anything when the map it holds was loaded from disk.
        second_dir = tempfile.mkdtemp()
        try:
            self.assertTrue(save_map(loaded.slam, second_dir))
            self.assertNotIn(VPR_MAP_FILE, os.listdir(second_dir))
        finally:
            shutil.rmtree(second_dir, ignore_errors=True)

    def relocalize(self, step, settings, folder):
        """Track one frame of the sequence in a fresh session and localize it with no guess pose."""
        odom_cfg, slam_cfg = make_configs(vslam.Slam.VprMode.Simple)
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        images, _ = self.generator.generate_zoomed_images(step)
        timestamp = step * FRAME_PERIOD_NS
        tracker.track(timestamp, images)
        return localize_in_map(tracker.slam, folder, images, timestamp, settings)

    def test_localization_without_a_guess_pose_is_seeded_by_the_saved_index(self):
        tracker, _ = self.build_map()
        self.assertTrue(save_map(tracker.slam, self.temp_dir))
        del tracker

        # This is the kidnapped robot: it has just been switched on inside a map it has never
        # tracked, so it has no pose to search around. The map's own recognition index proposes a
        # few places that look like the frame and each is verified geometrically until one holds.
        settings = localization_settings(vpr_candidates=3)
        results = [self.relocalize(step, settings, self.temp_dir) for step in range(0, 12)]
        localized = [pose for pose, _ in results if pose is not None]
        self.assertTrue(localized, "no frame of the sequence relocalized without a guess pose; "
                                   f"errors were {sorted({error for _, error in results})}")
        # Every frame that did not localize says so because no candidate verified, not because the
        # index was missing or the settings were rejected.
        for pose, error in results:
            if pose is None:
                self.assertIn("did not recognize this place", error)

    def test_localization_without_a_guess_pose_needs_a_map_that_carries_an_index(self):
        # The same map saved by a session with place recognition off has no vpr_map.bin in it, so
        # there is nothing to propose candidates from and the call fails instead of falling back to
        # the blind grid, which would be a search of the whole map.
        odom_cfg, slam_cfg = make_configs(vslam.Slam.VprMode.Off)
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        self.track_sequence(tracker)
        self.assertTrue(save_map(tracker.slam, self.temp_dir))
        self.assertNotIn(VPR_MAP_FILE, os.listdir(self.temp_dir))
        del tracker

        pose, error = self.relocalize(0, localization_settings(vpr_candidates=3), self.temp_dir)

        self.assertIsNone(pose)
        self.assertIn("did not recognize this place", error)


@unittest.skipUnless(os.path.isdir(KITTI_07_DIR), f"needs a loop-closing sequence at {KITTI_07_DIR}")
class TestVprFollowsPoseGraph(unittest.TestCase):
    """A recognized place reports the node's pose as it is now, not as it was when it was mapped.

    The whole design rests on it: place recognition resolves a match to a pose graph node and reads
    that node's pose out of the graph at query time. So after a loop closure and a pose graph
    optimization the same image has to come back with the corrected pose. A backend that answered
    from the pose stored beside the descriptor would pass every other test in this file and fail
    this one.
    """

    # Frames whose place is queried while the sequence is still being tracked, then again at the
    # end. Early enough that the optimization triggered by the loop closures near the end of the
    # sequence has had a whole lap to move them.
    EARLY_FRAMES = (60, 100, 140, 180)
    # Every fiftieth frame is kept to check the reported poses against the final pose graph.
    QUERY_STRIDE = 50
    # The optimization has to move an early node by more than this for the test to prove anything.
    MIN_DISPLACEMENT_M = 0.01
    # Replaying a sequence against a map built from that same sequence is the easiest case there is,
    # so the bar is high: a backend that recognizes fewer than this many of its own frames, or that
    # answers with a node this far from where the frame was actually taken, has regressed.
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
        # The recognition map holds one image per pose graph node, so a bounded pose graph would
        # merge the early nodes away long before the loop closes on them.
        slam_cfg.max_map_size = 0
        slam_cfg.vpr_mode = vslam.Slam.VprMode.Simple

        self.tracker = vslam.Tracker(
            self.reader.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        self.tracker.slam.get_pose_graph()  # enables the layer, so the graph is published from now on

    def test_recognized_pose_follows_the_optimization(self):
        replay = _VprReplay(self, self.tracker)
        self.reader.replay(replay)
        self.assertGreater(replay.n_frames, 0, f"no frames replayed from {KITTI_07_DIR}")

        loop_closures = self.tracker.slam.get_loop_closure_poses()
        self.assertGreater(len(loop_closures), 0,
                           f"{KITTI_07_DIR} closed no loop, so no optimization moved any node")

        nodes = {node.id: node.node_pose for node in self.tracker.slam.get_pose_graph().nodes}
        self.assertGreater(len(nodes), 0, "pose graph came back empty")

        # (a) every answer is the live pose of the node it names, not a copy made when it was mapped,
        #     and it names a node that is actually near where the frame was taken
        n_checked = 0
        far = []
        for frame_id, (timestamp, images, tracked_pose) in sorted(replay.kept_frames.items()):
            place, error = recognize_place(self.tracker.slam, images, timestamp)
            self.assertEqual(error, "")
            if not place.found:
                continue
            self.assertFalse(place.imported)
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

        # (b) and the optimization did move those nodes, so following them is worth something
        displacements = {}
        for frame_id, (timestamp, images, first) in sorted(replay.early_results.items()):
            again, error = recognize_place(self.tracker.slam, images, timestamp)
            self.assertEqual(error, "")
            if not again.found or again.node_id != first.node_id:
                # The frame now matches a different node, so the difference between the two poses is
                # not this node's motion and says nothing about the optimization.
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


class _VprReplay:
    """EdexReader processor: tracks the sequence, maps every frame, and keeps a few for querying."""

    def __init__(self, test, tracker):
        self.test = test
        self.tracker = tracker
        self.kept_frames = {}     # frame_id -> (timestamp, images, pose), queried after tracking ends
        self.early_results = {}   # frame_id -> (timestamp, images, the place found mid-sequence)
        self.n_frames = 0

    def process_images(self, frame_id, timestamps, images, masks, depths=None):
        timestamp = max(timestamps)
        self.tracker.track(timestamp, images, masks, depths)
        self.tracker.slam.add_frame_to_vpr_map(images, timestamp)
        self.n_frames += 1

        # The reader reuses its image buffers between frames, so a kept frame has to be copied.
        if frame_id % TestVprFollowsPoseGraph.QUERY_STRIDE == 0:
            self.kept_frames[frame_id] = (timestamp,
                                          [np.array(image, copy=True) for image in images],
                                          self.tracker.slam.get_pose())
        if frame_id in TestVprFollowsPoseGraph.EARLY_FRAMES:
            place, error = recognize_place(self.tracker.slam, images, timestamp)
            self.test.assertEqual(error, "")
            if place.found:
                self.early_results[frame_id] = (
                    timestamp, [np.array(image, copy=True) for image in images], place)

    def process_imu(self, timestamp, linear_accelerations, angular_velocities):
        """Ignore IMU samples: place recognition is answered from images alone."""

    def get_camera_pose(self, frame_id):
        """Poses are read from the pose graph, not collected here, as the reader protocol allows."""
        return None

    def set_frame_metadata(self, frame_id, metadata):
        """Ignore replay metadata: the sequence is replayed once, forward."""


if __name__ == "__main__":
    unittest.main()
