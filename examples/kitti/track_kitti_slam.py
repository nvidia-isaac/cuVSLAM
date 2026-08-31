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
import threading
import time
from PIL import Image
from numpy import loadtxt, asarray, array_equal as np_array_equal, savetxt, timedelta64
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
import cuvslam

# Dataset sequence to track and visualize
sequence_path = os.path.join(
    os.path.dirname(__file__),
    "dataset/sequences/06"
)

# Lambda to convert quaternion [x, y, z, w] to 3x3 rotation matrix (as list of lists)
quaternion_to_rotation_matrix = lambda q: R.from_quat(q).as_matrix().tolist()

# Lambda to multiply two quaternions [x, y, z, w] * [x, y, z, w]
quaternion_multiply = lambda q1, q2: (R.from_quat(q1) * R.from_quat(q2)).as_quat()

# Lambda to rotate a 3D vector using a 3x3 rotation matrix
rotate_vector = lambda vector, rotation_matrix: R.from_matrix(rotation_matrix).apply(vector)


def combine_poses(initial_pose, relative_pose):
    """
    Combine initial pose with relative pose to get absolute pose.

    Args:
        initial_pose: cuvslam.Pose object representing initial pose
        relative_pose: cuvslam.Pose object representing relative pose

    Returns:
        cuvslam.Pose object representing combined absolute pose
    """
    # Get rotation matrix from initial pose quaternion
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)

    # Rotate relative translation by initial pose rotation
    rotated_rel_t = rotate_vector(relative_pose.translation, rotation_matrix)

    # Add initial translation
    absolute_translation = [
        initial_pose.translation[0] + rotated_rel_t[0],
        initial_pose.translation[1] + rotated_rel_t[1],
        initial_pose.translation[2] + rotated_rel_t[2]
    ]

    # Multiply quaternions
    absolute_rotation = quaternion_multiply(initial_pose.rotation, relative_pose.rotation)

    return cuvslam.Pose(translation=absolute_translation, rotation=absolute_rotation)


def transform_landmarks(landmarks, initial_pose):
    """
    Transform landmarks by initial pose (rotation + translation).

    Args:
        landmarks: list of 3D landmark coordinates
        initial_pose: cuvslam.Pose object representing initial pose

    Returns:
        List of transformed 3D landmark coordinates
    """
    rotation_matrix = quaternion_to_rotation_matrix(initial_pose.rotation)
    transformed_landmarks = []

    for landmark in landmarks:
        # Rotate landmark by initial pose rotation
        rotated_landmark = rotate_vector(landmark, rotation_matrix)

        # Add initial translation
        transformed_landmark = [
            initial_pose.translation[0] + rotated_landmark[0],
            initial_pose.translation[1] + rotated_landmark[1],
            initial_pose.translation[2] + rotated_landmark[2]
        ]
        transformed_landmarks.append(transformed_landmark)

    return transformed_landmarks


# Save callback to set map_saved to True if map saving is successful
def save_callback(success):
    global map_saved
    map_saved = success

def localization_start_cb():
    print("Localization started")

# Localization callback to set slam_initial_pose and trigger localization_complete event
def localization_finish_cb(pose, error_message):
    global slam_initial_pose  # Declare slam_initial_pose as global
    print(f"Localization result: {pose}, {error_message}")
    slam_initial_pose = pose
    localization_complete.set()


# Generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier):
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


KITTI_3D_VIEW_EYE = rrb.EyeControls3D(
    kind=rrb.Eye3DKind.Orbital,
    position=(12.0, -9.0, -30.0),
    look_target=(0.0, 0.0, 8.0),
    eye_up=(0.0, -1.0, 0.0),
    speed=25.0,
)


# Setup rerun visualizer
rr.init('kitti', strict=True, spawn=True)  # launch re-run instance

# Setup rerun views
rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Vertical(
        row_shares=[0.6, 0.4],
        contents=[
            rrb.Spatial3DView(
                origin='view_anchor',
                name='3D',
                contents=['+ /**'],
                eye_controls=KITTI_3D_VIEW_EYE
            ),
            rrb.Spatial2DView(origin='car/cam0')
        ]
    )
))

# Setup coordinate basis for root, cuvslam uses right-hand system with X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Draw arrays in origin X-red, Y-green, Z-blue
rr.log("xyz", rr.Arrows3D(
    vectors=[[50, 0, 0], [0, 50, 0], [0, 0, 50]],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=['[x]', '[y]', '[z]']
), static=True)

SLAM_SYNC_MODE = False  # async slam thread is enabled
IDX = 700  # starting index of the sequence after localization
max_wait_time = 10.0  # seconds

# Load KITTI dataset calibration and initilize cameras
intrinsics = loadtxt(
    os.path.join(sequence_path, 'calib.txt'),
    usecols=range(1, 13)
)[:4].reshape(4, 3, 4)

size = Image.open(os.path.join(sequence_path, 'image_0', '000001.png')).size

cameras = [cuvslam.Camera(), cuvslam.Camera()]
for i in [0, 1]:
    cameras[i].size = size
    cameras[i].principal = [intrinsics[i][0][2], intrinsics[i][1][2]]
    cameras[i].focal = [intrinsics[i].diagonal()[0], intrinsics[i].diagonal()[1]]
cameras[1].rig_from_camera.translation[0] = -intrinsics[1][0][3] / intrinsics[1][0][0]

# Set Odometry and SLAM Configs and initialize the cuvslam tracker
odom_cfg = cuvslam.Odometry.Config(
    async_sba=False,
    enable_final_landmarks_export=True,
    rectified_stereo_camera=True
)
slam_cfg = cuvslam.Slam.Config(sync_mode=SLAM_SYNC_MODE)
tracker = cuvslam.Tracker(cuvslam.Rig(cameras), odom_cfg, slam_cfg)

# Get timestamps from times.txt file
timestamps = [
    int(10 ** 9 * float(sec_str))
    for sec_str in open(os.path.join(sequence_path, 'times.txt')).readlines()
]

# Check if map folder and trajectory file exist
map_path = os.path.join(sequence_path, 'map')
trajectory_file = os.path.join(sequence_path, 'trajectory_tum.txt')

if not os.path.exists(map_path):
    print(f"Map folder not found at {map_path}")

# Define these variables before the callback function
localization_complete = threading.Event()
slam_initial_pose = None
guess_pose = None
map_saved = False

# Define localization settings
loc_settings = cuvslam.Slam.LocalizationSettings(
    horizontal_search_radius=8.,
    vertical_search_radius=2.,
    horizontal_step=0.5,
    vertical_step=0.2,
    angular_step_rads=0.03
    )

# Define guess pose from trajectory file if it exists
if os.path.exists(trajectory_file) and os.path.exists(map_path):
    trajectory_data = loadtxt(trajectory_file)
    if IDX >= len(trajectory_data):
        raise IndexError(
            f"IDX ({IDX}) is out of bounds for loaded trajectory of length {len(trajectory_data)}"
        )
    guess_tum_pose = trajectory_data[IDX]
    guess_pose = cuvslam.Pose(translation=guess_tum_pose[:3], rotation=guess_tum_pose[3:])

# If guess pose is not None, localize in map
if os.path.exists(map_path) and (guess_pose is not None):
    timestamp = timestamps[IDX]
    init_images = [
        asarray(Image.open(os.path.join(sequence_path, f'image_{cam}', f'{IDX:0>6}.png')))
        for cam in [0, 1]
    ]
    _, _ = tracker.track(timestamp, init_images)

    tracker.slam.localize_in_map(map_path, timestamp, guess_pose, init_images, loc_settings, localization_start_cb,
                                 localization_finish_cb)

    wait_time = 0
    wait_timestamp_ns = timestamp
    # Wait for localization to complete with proper tracking, but only up to max_wait_time seconds
    while not SLAM_SYNC_MODE and not localization_complete.wait(timeout=0.5) and wait_time < max_wait_time:
        print(f"Waiting for localization... timestamp: {wait_time}")
        # call track() every 0.5 seconds only for the first 5 seconds
        if wait_time < 5.0:
            # Increment timestamp as track() requires strictly increasing timestamps.
            # Uses 1ms steps which won't reach the next real frame timestamp (KITTI timestamps are ~100ms apart).
            # In the real application sequencial frames with actual timestamps should be fed in parallel with localization.
            wait_timestamp_ns += 1_000_000
            _, slam_pose = tracker.track(wait_timestamp_ns, init_images)
            print(f"  slam_pose.t: {[f'{x:.3f}' for x in slam_pose.translation]}")
        wait_time += 0.5
    if not localization_complete.is_set():
        print(f"Localization did not complete within {max_wait_time} seconds")
    IDX += 1

if slam_initial_pose is not None:
    print(f"Localized pose: {slam_initial_pose}")
    wait_time = 0
    while not SLAM_SYNC_MODE and wait_time < max_wait_time:
        time.sleep(0.5)
        wait_timestamp_ns += 1_000_000
        _, slam_pose = tracker.track(wait_timestamp_ns, init_images)
        print(f"  slam_pose.t: {[f'{x:.3f}' for x in slam_pose.translation]}")
        identity_t = all(abs(x) < 1e-6 for x in slam_pose.translation)
        identity_r = all(abs(x) < 1e-6 for x in slam_pose.rotation[:3]) and abs(slam_pose.rotation[3] - 1.0) < 1e-6
        if not (identity_t and identity_r):
            break
        wait_time += 0.5
else:
    print("Warning: slam_initial_pose is None, set initial pose to zero, starting frame to 0, ignore map if exists")
    IDX = 0
    slam_initial_pose = cuvslam.Pose(translation=[0, 0, 0], rotation=[0, 0, 0, 1])

trajectory = []
trajectory_slam = []
trajectory_tum = []
loop_closure_poses = []

# Track each frames in the dataset sequence
for frame in range(IDX, len(timestamps)):
    time.sleep(0.01) # sleep 10ms to let SLAM thread catch up

    # Load grayscale pixels as array for left and right absolute image paths
    images = [
        asarray(Image.open(os.path.join(sequence_path, f'image_{cam}', f'{frame:0>6}.png')))
        for cam in [0, 1]
    ]

    # Do visual odometry and slam tracking
    odometry_pose_estimate, slam_pose = tracker.track(timestamps[frame], images)

    if odometry_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame}")
        continue

    # Get current pose and observations for the main camera and gravity in rig frame
    odom_pose = odometry_pose_estimate.world_from_rig.pose

    # transform odometry pose properly relative to the initial pose
    current_pose = combine_poses(slam_initial_pose, odom_pose)

    # Get visualization data
    observations = tracker.odometry.get_last_observations(0)  # get observation from left camera
    landmarks = tracker.odometry.get_last_landmarks()

    # Transform final landmarks by the initial pose
    raw_final_landmarks = list(tracker.odometry.get_final_landmarks().values())
    final_landmarks = transform_landmarks(raw_final_landmarks, slam_initial_pose)

    # Prepare visualization data
    observations_uv = [[o.u, o.v] for o in observations]
    observations_colors = [color_from_id(o.id) for o in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]

    trajectory.append(current_pose.translation)  # odometry trajectory in world frame
    trajectory_slam.append(slam_pose.translation)  # slam trajectory in world frame
    trajectory_tum.append(list(slam_pose.translation) + list(slam_pose.rotation))  # slam trajectory in tum format

    # Get loop closure poses
    current_lc_poses = tracker.slam.get_loop_closure_poses()
    if (current_lc_poses and
        (not loop_closure_poses or
         not np_array_equal(current_lc_poses[-1].pose.translation, loop_closure_poses[-1]))):
        loop_closure_poses.append(current_lc_poses[-1].pose.translation)

    # Send results to rerun for visualization
    rr.set_time('timestamp', duration=timedelta64(timestamps[frame], 'ns'))
    rr.log('view_anchor', rr.Transform3D(translation=current_pose.translation))
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('trajectory_slam', rr.LineStrips3D(trajectory_slam))
    rr.log('final_landmarks', rr.Points3D(final_landmarks, radii=0.1))
    rr.log('loop_closure_poses', rr.Points3D(
        loop_closure_poses, radii=0.6, colors=[[255, 0, 0]]
    ))
    rr.log('car', rr.Transform3D(
        translation=current_pose.translation,
        quaternion=current_pose.rotation
    ))
    rr.log('car/body', rr.Boxes3D(centers=[0, 1.65 / 2, 0], sizes=[[1.6, 1.65, 2.71]]))
    rr.log('car/landmarks_center', rr.Points3D(
        landmark_xyz, radii=0.25, colors=landmarks_colors
    ))
    rr.log('car/landmarks_lines', rr.Arrows3D(
        vectors=landmark_xyz, radii=0.05, colors=landmarks_colors
    ))
    rr.log('car/cam0', rr.Pinhole(
        image_plane_distance=1.68,
        image_from_camera=intrinsics[0][:3, :3],
        width=size[0],
        height=size[1]
    ))
    rr.log('car/cam0/image', rr.Image(images[0]).compress(jpeg_quality=80))
    rr.log('car/cam0/observations', rr.Points2D(
        observations_uv, radii=5, colors=observations_colors
    ))

print(f"Number of loop closure poses: {len(loop_closure_poses)}")

if guess_pose is None:
    # Save map and trajectory
    os.makedirs(map_path, exist_ok=True)
    print(f"Saving trajectory to {trajectory_file} of length {len(trajectory_tum)}")
    savetxt(trajectory_file, trajectory_tum)

    tracker.slam.save_map(map_path, save_callback)

    # Wait for map saving to complete
    start_time = time.time()
    while not map_saved and (time.time() - start_time) < max_wait_time:
        time.sleep(0.1)
        print(f"Waiting for map saving to complete... {time.time() - start_time} seconds")

    if map_saved:
        print("Map saved successfully")
    else:
        print("WARNING: Map saving may not have completed")


print("Cleaning up resources...")
try:
    del trajectory
    del trajectory_slam
    del trajectory_tum
    del loop_closure_poses
    del tracker
    del cameras
    del odom_cfg
    del slam_cfg
except Exception as e:
    print(f"Warning during cleanup: {e}")

print("Script completed")
