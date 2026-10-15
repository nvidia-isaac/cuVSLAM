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

import multiprocessing
import os
import tempfile
import shutil
import unittest
from unittest import mock
import numpy as np
import threading
from numpy.testing import assert_array_almost_equal

# Unit tests must not require the external Rerun Viewer, even in USE_RERUN builds.
os.environ.setdefault("RERUN", "0")

import cuvslam as vslam
import data_gen as data

W, H = 640, 480  # Image size of tracked images
BASELINE = 0.25  # Baseline distance between cameras (m)
STEPS = 30       # Number of steps

def end_with_pending_callbacks(use_tracker, queue):
    """Drops an async session while the SLAM thread is inside a callback and has more callbacks queued."""
    rig = vslam.Rig(data.generate_stereo_camera(W, H, BASELINE), [vslam.ImuCalibration()])
    slam_cfg = vslam.Slam.Config(sync_mode=False)
    if use_tracker:
        odom_cfg = vslam.Odometry.Config(odometry_mode=vslam.Odometry.OdometryMode.Multicamera,
                                         rectified_stereo_camera=True, async_sba=True)
        session = [vslam.Tracker(rig, vslam.Tracker.Mode.OdometryWithSlamRealtime, odom_cfg, slam_cfg)]
        session.append(session[0].slam)
    else:
        session = [vslam.Slam(rig, [0], slam_cfg)]
    images = [np.zeros((H, W), dtype=np.uint8)] * 2
    settings = vslam.Slam.LocalizationSettings(horizontal_search_radius=0.25, vertical_search_radius=0.25,
                                               horizontal_step=0.0625, vertical_step=0.0625, angular_step_rads=0.03125)
    callback_threads = []
    started = threading.Event()
    release = threading.Event()

    def first():
        callback_threads.append(threading.get_ident())
        started.set()
        release.wait(60)  # set by a timer, which needs the GIL back once the session is dropped

    def localize(start_cb):
        # There is no map, so it fails, but its callbacks still run on the SLAM thread
        session[-1].localize_in_map("/nonexistent/cuvslam/map", 0, vslam.Pose(), images, settings, start_cb,
                             lambda pose, error: None)

    localize(first)
    if not started.wait(60):
        raise RuntimeError("the first callback did not run")
    for _ in range(10):
        localize(lambda: None)
    threading.Timer(0.5, release.set).start()
    session.clear()  # drops the Python owners while callbacks are pending
    queue.put(callback_threads[0] != threading.get_ident())

plt = None

def visualize_frame(images, i, subplots):
    if not plt:
        return None

    # Create figure only once before the loop
    if i == 0:
        plt.figure(figsize=(12, 6))
        subplots = []
        for cam in range(len(images)):
            subplot = plt.subplot(1, len(images), cam + 1)
            subplots.append(subplot)
        plt.ion()  # Enable interactive mode

    # Update images for each camera
    for cam in range(len(images)):
        plt.sca(subplots[cam])
        plt.cla()  # Clear the current axes
        plt.imshow(images[cam], cmap='gray')
        plt.title(f'Camera {cam}')
        plt.axis('off')

    plt.suptitle(f'Frame {i}')
    plt.draw()
    plt.pause(0.1)  # Add small delay to allow visualization

    # Close figure only at the end of the loop
    if i == STEPS - 1:  # Last iteration
        plt.close()

    return subplots

def perturb_pose(pose, max_translation=0.5):
    # Perturb translation
    translation = np.array(pose.translation)
    translation += np.random.uniform(-max_translation, max_translation, size=3)

    # Random yaw rotation around Y axis (vertical in OpenCV convention)
    # Localization searches yaw (angular_step_rads) but not pitch/roll
    yaw = np.random.uniform(-np.pi, np.pi)
    rotation = [0, np.sin(yaw / 2), 0, np.cos(yaw / 2)]

    return vslam.Pose(translation=translation, rotation=rotation)


class PrinterTestResult(unittest.TestResult):
    """Custom test result class to print debug data if tests fail."""

    def addFailure(self, test, err):
        super().addFailure(test, err)
        debug_messages = getattr(test, 'debug_messages', [])
        for msg in debug_messages:
            print(msg)

class TestMap(unittest.TestCase):

    def setUp(self):
        cameras = data.generate_stereo_camera(W, H, BASELINE)
        self.num_cameras = len(cameras)
        imu = vslam.ImuCalibration()
        self.rig = vslam.Rig(cameras, [imu])
        self.debug_messages = []
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def add_debug(self, message):
        self.debug_messages.append(message)

    ### Helper methods for map creation and localization

    def get_localization_configs(self):
        """Get configurations for localization mode."""
        odom_cfg = vslam.Odometry.Config()
        odom_cfg.odometry_mode = vslam.Odometry.OdometryMode.Multicamera
        odom_cfg.rectified_stereo_camera = True
        odom_cfg.async_sba = False

        slam_cfg = vslam.Slam.Config()
        slam_cfg.sync_mode = True

        loc_settings = vslam.Slam.LocalizationSettings(
            horizontal_search_radius=0.25,
            vertical_search_radius=0.25,
            horizontal_step=0.0625,
            vertical_step=0.0625,
            angular_step_rads=0.03125
        )

        return odom_cfg, slam_cfg, loc_settings

    def map_path(self, name):
        return os.path.join(self.temp_dir, name)

    def create_map(
            self,
            img: data.ImageGenerator,
            map_name='temp_map',
            start=0,
            stop=STEPS,
            step=1,
            timestamp=0,
            visualize=False):
        """Create and save a map using image generator. Returns the ids of its landmarks."""

        # Configure tracker for map creation
        odom_cfg, slam_cfg, _ = self.get_localization_configs()
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        tracker.slam.get_landmarks(vslam.Slam.DataLayer.Map)  # the first call enables the layer

        subplots = None

        z0 = None
        # Track through selected steps to build the map
        for i in range(start, stop, step):
            images, z = img.generate_zoomed_images(i)
            if i == start:
                z0 = z

            # track
            pose_estimate, slam_pose = tracker.track(i * 1_000_000 + timestamp, images)
            self.assertTrue(pose_estimate.world_from_rig)

            odom_pose = pose_estimate.world_from_rig.pose
            self.add_debug(f"{i}: gt, slam: {z}, {slam_pose.translation[2]}")

            self.add_debug(f"{i}: odom_pose={odom_pose}")
            assert_array_almost_equal(np.array(odom_pose.translation), [0, 0, z - z0], decimal=1)
            assert_array_almost_equal(np.array(odom_pose.rotation), [0, 0, 0, 1.0], decimal=2)

            self.add_debug(f"{i}: slam_pose={slam_pose}")
            assert_array_almost_equal(np.array(slam_pose.translation), [0, 0, z], decimal=1)
            assert_array_almost_equal(np.array(slam_pose.rotation), [0, 0, 0, 1.0], decimal=2)

            if visualize:
                subplots = visualize_frame(images, i, subplots)

        landmark_ids = {landmark.id for landmark in tracker.slam.get_landmarks(vslam.Slam.DataLayer.Map).landmarks}

        # Save the map
        map_saved = False
        def save_callback(success):
            nonlocal map_saved
            map_saved = success
        tracker.slam.save_map(self.map_path(map_name), save_callback)
        self.assertTrue(map_saved)
        return landmark_ids

    def try_localize(
            self,
            img: data.ImageGenerator,
            map_name: str,
            step: int,
            perturb: bool = False) -> vslam.Pose | None:
        """
        Try to localize in the map created by create_map.
        Returns True if localization was successful.
        """
        odom_cfg, slam_cfg, loc_settings = self.get_localization_configs()
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)

        images, z = img.generate_zoomed_images(step)
        timestamp = step * 1_000_000
        _, _ = tracker.track(timestamp, images)

        guess_pose = vslam.Pose(translation=[0, 0, z], rotation=[0, 0, 0, 1.0])
        if perturb:
            guess_pose = perturb_pose(guess_pose, 0.125)
        self.add_debug(f"{step}: guess_pose={guess_pose}")

        localization_complete = threading.Event()
        result_pose = None

        def localization_start_cb():
            self.add_debug("Localization started")

        def localization_finish_cb(pose, error_message):
            nonlocal result_pose
            self.add_debug(f"Localization result: {pose}, {error_message}")
            result_pose = pose
            localization_complete.set()

        tracker.slam.localize_in_map(self.map_path(map_name), timestamp, guess_pose, images, loc_settings,
                                     localization_start_cb, localization_finish_cb)
        while not localization_complete.wait(timeout=0.1):
            # Simulate time passing; this is necessary for the async callback to be processed
            timestamp += 1_000
            _, _ = tracker.track(timestamp, images)

        self.add_debug(f"{step}: result_pose={result_pose}")
        return result_pose

    def map_layer_after_localizing(self, img: data.ImageGenerator, map_name: str, step: int) -> set[int]:
        """Landmark ids of the Map layer a new session publishes once it localizes in the map at `step`."""
        odom_cfg, slam_cfg, loc_settings = self.get_localization_configs()
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        tracker.slam.get_landmarks(vslam.Slam.DataLayer.Map)  # the first call enables the layer
        images, z = img.generate_zoomed_images(step)
        timestamp = step * 1_000_000
        tracker.track(timestamp, images)
        poses = []
        guess_pose = vslam.Pose(translation=[0, 0, z], rotation=[0, 0, 0, 1.0])
        tracker.slam.localize_in_map(self.map_path(map_name), timestamp, guess_pose, images, loc_settings,
                                     lambda: None, lambda pose, error_message: poses.append(pose))
        self.assertIsNotNone(poses[0])
        for i in range(step + 1, STEPS):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)
            layer = tracker.slam.get_landmarks(vslam.Slam.DataLayer.Map)
            if layer.timestamp_ns > timestamp:
                return {landmark.id for landmark in layer.landmarks}
        self.fail(f"no Map layer after localizing in {map_name}")

    ### Test cases for map creation and localization

    def test_map(self):

        img = data.ImageGenerator(self.rig.cameras, 30)

        self.create_map(img, 'map')

        _, _, loc_settings = self.get_localization_configs()
        self.add_debug(f"Localization settings: {loc_settings}")

        self.assertTrue(any(self.try_localize(img, 'map', i, perturb=False) for i in range(0, 10)))

        self.assertTrue(any(self.try_localize(img, 'map', i, perturb=True) for i in range(0, 10)))

        self.assertTrue(any(self.try_localize(img, 'map', i, perturb=False) for i in range(10, 20)))
        # Localization in the end of the map isn't working well
        # self.assertTrue(any(self.try_localize(img, 'map', i, perturb=False) for i in range(20, 30)))

    def test_localized_session_keeps_the_saved_landmarks(self):
        img = data.ImageGenerator(self.rig.cameras, 30)
        saved = self.create_map(img, 'map')
        self.assertTrue(saved)
        localizing = [i for i in range(10) if self.try_localize(img, 'map', i)]
        self.assertTrue(localizing)
        step = localizing[0]

        # A new session localizes in the map half a second later and saves its map right away
        odom_cfg, slam_cfg, loc_settings = self.get_localization_configs()
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        tracker.slam.get_landmarks(vslam.Slam.DataLayer.Map)  # the first call enables the layer
        offset = 500_000_000
        images, z = img.generate_zoomed_images(step)
        timestamp = step * 1_000_000 + offset
        tracker.track(timestamp, images)
        poses = []
        guess_pose = vslam.Pose(translation=[0, 0, z], rotation=[0, 0, 0, 1.0])
        tracker.slam.localize_in_map(self.map_path('map'), timestamp, guess_pose, images, loc_settings,
                                     lambda: None, lambda pose, error_message: poses.append(pose))
        self.assertEqual(len(poses), 1)  # sync mode answers right away
        self.assertIsNotNone(poses[0])
        saved_again = []
        tracker.slam.save_map(self.map_path('resaved'), saved_again.append)
        self.assertEqual(saved_again, [True])

        # The Map layer published after localizing holds the saved landmarks
        for i in range(step + 1, STEPS):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000 + offset, images)
            layer = tracker.slam.get_landmarks(vslam.Slam.DataLayer.Map)
            if layer.timestamp_ns > timestamp:
                break
        self.assertGreater(layer.timestamp_ns, timestamp)
        kept = saved & {landmark.id for landmark in layer.landmarks}
        self.add_debug(f"kept {len(kept)} of {len(saved)} saved landmarks")
        self.assertGreaterEqual(len(kept), 0.9 * len(saved))

        # The map saved after localizing localizes wherever the first one does
        self.assertEqual([i for i in localizing if self.try_localize(img, 'resaved', i)], localizing)

    def test_flush_map_saves_the_map_at_map_path(self):
        img = data.ImageGenerator(self.rig.cameras, 30)
        odom_cfg, slam_cfg, _ = self.get_localization_configs()
        attached = self.map_path('attached')
        slam_cfg.map_path = attached  # a string_view: `attached` keeps the str alive
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        for i in range(STEPS):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)

        results = []
        tracker.slam.save_map(self.map_path('map'), results.append)
        tracker.slam.flush_map(results.append)
        self.assertEqual(results, [False, True])  # a map at map_path is flushed, not saved

        # While the session goes on, the flushed map file is a whole map
        shutil.copytree(attached, self.map_path('copy'))
        self.assertTrue(any(self.try_localize(img, 'copy', i) for i in range(0, 10)))

    def test_localize_in_map_with_attach_maps_into_the_loaded_map(self):
        img = data.ImageGenerator(self.rig.cameras, 30)
        saved = self.create_map(img, 'map', stop=20)
        step = next((i for i in range(10) if self.try_localize(img, 'map', i)), None)
        self.assertIsNotNone(step)

        # A new session localizes in the map and keeps mapping into it
        odom_cfg, slam_cfg, loc_settings = self.get_localization_configs()
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)
        tracker.slam.get_landmarks(vslam.Slam.DataLayer.Map)  # the first call enables the layer
        offset = 500_000_000
        images, z = img.generate_zoomed_images(step)
        timestamp = step * 1_000_000 + offset
        tracker.track(timestamp, images)
        poses = []
        guess_pose = vslam.Pose(translation=[0, 0, z], rotation=[0, 0, 0, 1.0])
        tracker.slam.localize_in_map(self.map_path('map'), timestamp, guess_pose, images, loc_settings,
                                     lambda: None, lambda pose, error_message: poses.append(pose), attach=True)
        self.assertEqual(len(poses), 1)  # sync mode answers right away
        self.assertIsNotNone(poses[0])
        for i in range(step + 1, STEPS):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000 + offset, images)
        added = {landmark.id for landmark in tracker.slam.get_landmarks(vslam.Slam.DataLayer.Map).landmarks} - saved
        self.add_debug(f"the attached session added {len(added)} landmarks to the {len(saved)} saved ones")
        self.assertTrue(added)

        results = []
        tracker.slam.save_map(self.map_path('copy'), results.append)
        tracker.slam.flush_map(results.append)
        self.assertEqual(results, [False, True])  # the attached map is written in place

        # The map file has the landmarks added after attaching, and it still localizes
        shutil.copytree(self.map_path('map'), self.map_path('snapshot'))
        found = added & self.map_layer_after_localizing(img, 'snapshot', step)
        self.add_debug(f"the flushed map file has {len(found)} of them")
        self.assertGreaterEqual(len(found), 0.9 * len(added))
        self.assertTrue(any(self.try_localize(img, 'snapshot', i) for i in range(0, 10)))

    def test_flush_map_and_attach_in_async_mode(self):
        img = data.ImageGenerator(self.rig.cameras, 30)
        self.create_map(img, 'map', stop=20)
        step = next((i for i in range(10) if self.try_localize(img, 'map', i)), None)
        self.assertIsNotNone(step)

        # Callbacks run on the SLAM thread, which also drops them
        odom_cfg, slam_cfg, loc_settings = self.get_localization_configs()
        odom_cfg.async_sba = True
        slam_cfg.sync_mode = False
        own = self.map_path('own')
        slam_cfg.map_path = own  # a string_view: `own` keeps the str alive
        tracker = vslam.Tracker(self.rig, vslam.Tracker.Mode.OdometryWithSlamRealtime, odom_cfg, slam_cfg)

        def flush():
            done = threading.Event()
            results = []
            tracker.slam.flush_map(lambda success: (results.append(success), done.set()))
            self.assertTrue(done.wait(timeout=60))
            return results

        for i in range(20):
            images, _ = img.generate_zoomed_images(i)
            tracker.track(i * 1_000_000, images)
            # Waits for SLAM too: a realtime session maps a keyframe only while its images are the latest
            self.assertEqual(flush(), [True])

        localized = threading.Event()
        poses = []
        def on_localized(pose, error_message):
            self.add_debug(f"async localization: {pose}, {error_message}")
            poses.append(pose)
            localized.set()
        images, z = img.generate_zoomed_images(step)
        guess_pose = vslam.Pose(translation=[0, 0, z], rotation=[0, 0, 0, 1.0])
        tracker.slam.localize_in_map(self.map_path('map'), step * 1_000_000, guess_pose, images, loc_settings,
                                     lambda: None, on_localized, attach=True)
        self.assertTrue(localized.wait(timeout=120))
        self.assertIsNotNone(poses[0])
        self.assertEqual(flush(), [True])
        del tracker

        # The session wrote its own map file before it moved to the loaded one, which it wrote too
        self.assertTrue(any(self.try_localize(img, 'own', i) for i in range(0, 10)))
        self.assertTrue(any(self.try_localize(img, 'map', i) for i in range(0, 10)))

    def end_with_pending_callbacks(self, use_tracker, owners):
        # In a child process, so that a deadlock fails the test instead of hanging it
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        child = context.Process(target=end_with_pending_callbacks, args=(use_tracker, queue))
        child.start()
        child.join(timeout=120)
        if child.is_alive():
            child.kill()
            child.join()
            self.fail(f"destroying {owners} with callbacks pending did not return")
        self.assertEqual(child.exitcode, 0)
        on_slam_thread = queue.get(timeout=10)
        self.assertTrue(on_slam_thread, "the callback did not run on the SLAM thread")

    def test_destroying_slam_with_callbacks_pending_returns(self):
        self.end_with_pending_callbacks(False, "slam")

    def test_destroying_tracker_with_callbacks_pending_returns(self):
        self.end_with_pending_callbacks(True, "slam, tracker")

    def check_raising_callbacks_are_reported(self, sync_mode):
        slam = vslam.Slam(self.rig, [0], vslam.Slam.Config(sync_mode=sync_mode))
        images = [np.zeros((H, W), dtype=np.uint8)] * self.num_cameras
        _, _, settings = self.get_localization_configs()
        done = threading.Event()

        def fail(*args):
            raise ValueError("callback failed")

        def fail_last(*args):
            done.set()
            raise ValueError("callback failed")

        reported = []
        with mock.patch('sys.unraisablehook', lambda report: reported.append((type(report.exc_value), report.object))):
            # There is no map, so localizing and flushing fail, but their callbacks still run
            slam.localize_in_map("/nonexistent/cuvslam/map", 0, vslam.Pose(), images, settings, fail, fail)
            slam.flush_map(fail)
            slam.save_map(self.map_path('map'), fail_last)
            self.assertTrue(done.wait(60), "the callbacks did not run")
            del slam  # joins the SLAM thread, which reports the last exception after setting done
        self.assertEqual(sorted(reported, key=lambda r: r[1]), [
            (ValueError, "Slam.flush_map callback"),
            (ValueError, "Slam.localize_in_map finish_cb"),
            (ValueError, "Slam.localize_in_map start_cb"),
            (ValueError, "Slam.save_map callback"),
        ])

    def test_raising_callbacks_are_reported_in_async_mode(self):
        self.check_raising_callbacks_are_reported(sync_mode=False)

    def test_raising_callbacks_are_reported_in_sync_mode(self):
        self.check_raising_callbacks_are_reported(sync_mode=True)


if __name__ == "__main__":
    # using test runner with custom result class to print additional debug messages
    unittest.main(testRunner=unittest.TextTestRunner(resultclass=PrinterTestResult))
