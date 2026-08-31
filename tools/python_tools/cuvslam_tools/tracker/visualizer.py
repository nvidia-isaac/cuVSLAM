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

"""Trajectory plotting and Rerun visualization helpers for tracker runs."""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from typing import List, Dict, Optional, Sequence
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

import cuvslam as vslam
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_trajectory(poses: Dict[int, vslam.Pose],
                    loop_closures: Dict[int, vslam.Pose],
                    gt_poses: Optional[List[np.ndarray]] = None,
                    visualize_plot: bool = False,
                    save_path: Optional[str] = None,
                    gt_from_shuttle: bool = False,
                    title: str = "Trajectory (X vs Z)"):
    """Plot trajectory and ground truth if available.

    Args:
        poses: Dictionary of poses
        gt_poses: List of ground truth poses
        save_path: Path to save plot image. If None, display plot instead
    """
    # Get translations and rotations
    #trajectory = [poses[frame_id].translation for frame_id in sorted(poses.keys())]
    #rotations = [poses[frame_id].rotation for frame_id in sorted(poses.keys())]

    trajectory = [poses[frame_id].translation
                  for frame_id in sorted(poses.keys()) if poses[frame_id] is not None]
    rotations = [poses[frame_id].rotation
                 for frame_id in sorted(poses.keys()) if poses[frame_id] is not None]
    lc = [loop_closures[frame_id].translation
          for frame_id in sorted(loop_closures.keys())]

    # Convert quaternions to Euler angles (in degrees)
    euler_angles = [Rotation.from_quat([q[0], q[1], q[2], q[3]]).as_euler('xyz', degrees=True)
                    for q in rotations]
    euler_zipped = list(zip(*euler_angles))

    # Get translations
    trans_zipped = list(zip(*trajectory))
    lc_zipped = list(zip(*lc))

    # Get ground truth if available
    if gt_poses:
        gt_trajectory = [gt_pose[:3, 3] for gt_pose in gt_poses]
        gt_rotations = [Rotation.from_matrix(gt_pose[:3, :3]).as_euler('xyz', degrees=True)
                        for gt_pose in gt_poses]
        gt_trans_zipped = list(zip(*gt_trajectory))
        gt_euler_zipped = list(zip(*gt_rotations))
    else:
        gt_trans_zipped = []
        gt_euler_zipped = []

    # Create figure with custom grid
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])
    fig = plt.figure(figsize=(20, 10))

    # Translation plot
    f1 = fig.add_subplot(gs[0, 0])
    f1.set_title('Translation (X, Y, Z) vs. Time')
    f1.plot(trans_zipped[0], label='x')
    f1.plot(trans_zipped[1], label='y')
    f1.plot(trans_zipped[2], label='z')
    if len(gt_trans_zipped):
        f1.plot(gt_trans_zipped[0], label='gt_x')
        f1.plot(gt_trans_zipped[1], label='gt_y')
        f1.plot(gt_trans_zipped[2], label='gt_z')
    f1.set_xlabel('Time')
    f1.set_ylabel('meters')
    f1.legend()

    # Rotation plot
    f2 = fig.add_subplot(gs[1, 0])
    f2.set_title('Rotation (X, Y, Z) vs. Time')
    f2.plot(euler_zipped[0], label='roll')
    f2.plot(euler_zipped[1], label='pitch')
    f2.plot(euler_zipped[2], label='yaw')
    if len(gt_euler_zipped):
        f2.plot(gt_euler_zipped[0], label='gt_roll')
        f2.plot(gt_euler_zipped[1], label='gt_pitch')
        f2.plot(gt_euler_zipped[2], label='gt_yaw')
    f2.set_xlabel('Time')
    f2.set_ylabel('degrees')
    f2.legend()

    # Bird's eye view plot
    f3 = fig.add_subplot(gs[:, 1])
    f3.set_title(title)
    if len(gt_trans_zipped):
        if gt_from_shuttle:
            f3.plot(trans_zipped[0], trans_zipped[2], label='Backward pass')
            f3.plot(gt_trans_zipped[0], gt_trans_zipped[2], label='Forward pass', linewidth=2)
        else:
            f3.plot(trans_zipped[0], trans_zipped[2], label='VO')
            f3.plot(gt_trans_zipped[0], gt_trans_zipped[2], label='GT')
    else:
        f3.plot(trans_zipped[0], trans_zipped[2], label='VO')
    if lc_zipped:
        f3.scatter(lc_zipped[0], lc_zipped[2], label='LC', c='green', s=20)

    # Get data range
    x_min, x_max = min(trans_zipped[0]), max(trans_zipped[0])
    z_min, z_max = min(trans_zipped[2]), max(trans_zipped[2])

    # Calculate range and center
    range_x = x_max - x_min
    range_z = z_max - z_min
    max_range = max(range_x, range_z)
    center_x = (x_max + x_min) / 2
    center_z = (z_max + z_min) / 2

    # Set equal limits with padding
    padding = 0.1  # 10% padding
    max_range *= (1 + padding)
    f3.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
    f3.set_ylim(center_z - max_range / 2, center_z + max_range / 2)

    f3.set_xlabel('x')
    f3.set_ylabel('z')
    f3.legend()
    f3.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if visualize_plot:
        plt.show(block=True)
    plt.close()

def _color_from_id(identifier):
    """Generate pseudo-random colour from integer identifier for visualization."""
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]

class RerunVisualizer:
    """Stream tracker state, landmarks, and trajectories to Rerun."""

    def __init__(self):
        """Initialize rerun visualizer."""
        rr.init("cuVSLAM Visualizer", spawn=True)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state="collapsed"),
                rrb.Horizontal(
                    column_shares=[0.5, 0.5],
                    contents=[
                        rrb.Spatial2DView(origin='world/camera_0'),
                        rrb.Spatial3DView(origin='world'),

                    ]
                )
            ),
            make_active=True
        )
        self.odom_trajectory = []
        self.slam_trajectory = []

    def _log_rig_pose(self, rotation_quat: ArrayLike,
                      translation: ArrayLike) -> None:
        """Log rig pose to Rerun."""
        scale = 1
        rr.log(
            "world/camera_0",
            rr.Transform3D(translation=translation, quaternion=rotation_quat),
            rr.Arrows3D(
                vectors=np.eye(3) * scale,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ axes
            )
        )

    def _log_observations(self, observations_main_cam: List[vslam.Observation],
                          image: np.ndarray) -> None:
        """Log 2D observations for a specific camera with consistent colors per track."""
        if not observations_main_cam:
            rr.log(
                "world/camera_0/observations",
                rr.Image(image).compress()
            )
            return

        points = np.array([[obs.u, obs.v] for obs in observations_main_cam])
        colors = np.array([_color_from_id(obs.id) for obs in observations_main_cam])

        rr.log(
            "world/camera_0/observations",
            rr.Points2D(positions=points, colors=colors, radii=3.0),
            rr.Image(image).compress()
        )

    def _log_landmarks(self, landmarks: List[vslam.Landmark]) -> None:
        """Log landmarks to Rerun."""
        rr.log(
            "world/camera_0/last_landmarks",
            rr.Points3D([ls.coords for ls in landmarks], colors=[_color_from_id(ls.id) for ls in landmarks]))

    def _log_slam_landmarks(self, layer: str, landmarks: vslam.Slam.Landmarks,
                            color: Optional[List[int]] = None, ui_radius: Optional[float] = None) -> None:
        """Log landmarks to Rerun."""
        if not landmarks or not landmarks.landmarks:
            return
        rr.log(
            f"world/{layer}_landmarks",
            rr.Points3D([ls.coords for ls in landmarks.landmarks],
            colors=[color] if color else [_color_from_id(ls.id) for ls in landmarks.landmarks],
            radii=rr.Radius.ui_points(ui_radius) if ui_radius else None))

    def _log_gravity(self, gravity: np.ndarray) -> None:
        """Log gravity vector to Rerun."""
        radius = 0.02
        rr.log(
            "world/gravity",
            rr.Arrows3D(vectors=gravity, colors=[[255, 0, 0]], radii=radius)
        )

    def _log_pose_graph(self, pose_graph: vslam.Slam.PoseGraph) -> None:
        """Log pose graph to Rerun."""
        if not pose_graph or (not pose_graph.nodes and not pose_graph.edges):
            return

        # Collect node ids and positions
        node_ids = []
        node_positions = []
        for node in pose_graph.nodes:
            node_ids.append(node.id)
            node_positions.append(node.node_pose.translation)

        # if node_positions:

        # Map id -> index in node_positions
        id_to_index = {nid: idx for idx, nid in enumerate(node_ids)}

        # Build edge segments from node positions
        edge_segments = []
        for edge in pose_graph.edges:
            if edge.node_from in id_to_index and edge.node_to in id_to_index:
                p_from = node_positions[id_to_index[edge.node_from]]
                p_to = node_positions[id_to_index[edge.node_to]]
                edge_segments.append([p_from, p_to])

        if edge_segments:
            rr.log(
                "world/pose_graph",
                rr.LineStrips3D(edge_segments, colors=[128, 128, 128]),
                # rr.Points3D(node_positions),
            )

    def visualize_frame(self, frame_id: int, images: Sequence[np.ndarray],
                        odom_pose: Optional[vslam.Pose], slam_pose: Optional[vslam.Pose],
                        observations_0: List[vslam.Observation],
                        last_landmarks: List[vslam.Landmark],
                        loop_closures: Dict[int, vslam.Pose],
                        final_landmarks: Dict[int, ArrayLike],
                        pose_graph: Optional[vslam.Slam.PoseGraph],
                        timestamp: int,
                        map_landmarks: Optional[vslam.Slam.Landmarks] = None,
                        lc_landmarks: Optional[vslam.Slam.Landmarks] = None,
                        gravity: Optional[np.ndarray] = None) -> None:
        """Visualize current frame state using Rerun."""
        rr.set_time_sequence("frame", sequence=frame_id)
        if odom_pose is not None:
            self.odom_trajectory.append(odom_pose.translation)
            rr.log("world/odom_trajectory", rr.LineStrips3D(self.odom_trajectory))
        rig_pose = odom_pose
        if slam_pose is not None:
            rig_pose = slam_pose
            self.slam_trajectory.append(slam_pose.translation)
            rr.log("world/slam_trajectory", rr.LineStrips3D(self.slam_trajectory))
        if rig_pose is None:
            return
        self._log_rig_pose(rig_pose.rotation, rig_pose.translation)
        self._log_observations(observations_0, images[0])
        self._log_landmarks(last_landmarks)
        self._log_slam_landmarks("map", map_landmarks, color=[[128, 128, 255]])
        self._log_slam_landmarks("lc", lc_landmarks, color=[[0, 255, 0]], ui_radius=2.5)
        rr.log("world/landmarks", rr.Points3D(final_landmarks.values()))
        rr.log('world/loop_closure_poses', rr.Points3D(
            [lc.translation for lc in loop_closures.values()], colors=[[255, 0, 0]], radii=rr.Radius.ui_points(2.5)
        ))
        if pose_graph:
            self._log_pose_graph(pose_graph)
        if gravity is not None:
            self._log_gravity(gravity)
        rr.log("world/timestamp", rr.TextLog(str(timestamp)))
