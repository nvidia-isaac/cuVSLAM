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

"""Dataset reader for replaying a video file through the tracker."""

import os
import json
import cv2
import numpy as np
from typing import List, Optional

from cuvslam_tools.tracker import conversions as conv
from cuvslam_tools.tracker.dataset_reader import DatasetReader, Processing


class VideoReader(DatasetReader):
    """Replay video frames with calibration loaded from a stereo.edex file."""

    def __init__(self, video_path: str, stereo_edex: Optional[str] = None, num_loops: int = 0,
                 repeat_type: str = "none", gt_path: Optional[str] = None,
                 gt_from_shuttle: bool = False):
        """Load calibration and preload all video frames into memory."""
        super().__init__(video_path, stereo_edex, num_loops, repeat_type, gt_path=gt_path,
                         gt_from_shuttle=gt_from_shuttle)
        self.replay_forward = True
        self.buffer = []  # Store frames in memory

        if not stereo_edex:
            stereo_edex = os.path.join(os.path.dirname(video_path), 'stereo.edex')

        if os.path.exists(stereo_edex):
            with open(stereo_edex, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Config file not found: {stereo_edex}")
        self.rig = self.parse_config(config_data)

        # Pre-load all frames
        print("Pre-loading video frames...")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.rig.cameras[0].size = (width, height)
        cap.release()
        print(f"Loaded {len(self.buffer)} frames")

        self.total_frames = len(self.buffer)
        self.fps = 30  # default FPS
        self.interval_ns = int(1e9 / self.fps)
        self.first_ts = 0
        self.last_ts = int((self.total_frames - 1) * self.interval_ns)
        self.sequence_duration = self.last_ts - self.first_ts
        self.max_ts = max(self.last_ts, self.first_ts)

    def replay(self, processor: Processing):
        """Replay video frames to a Processing implementation."""
        assert self.rig is not None
        images = [np.array([])] * len(self.rig.cameras)
        masks = [np.array([])] * len(self.rig.cameras)
        timestamps = [0] * len(self.rig.cameras)

        frame_id = 0
        while self.check_end_of_sequence():
            # Get frame from buffer
            images[0] = self.buffer[self.current_frame]
            raw_ts = self.current_frame * self.interval_ns
            timestamps[0] = self.adjust_timestamp(raw_ts)

            processor.process_images(frame_id, timestamps, images, masks)

            # Store pose from first forward run as ground truth
            if self.gt_from_shuttle and self.replay_forward and self.current_loop == 0:
                if hasattr(processor, 'get_camera_pose'):
                    pose = processor.get_camera_pose(frame_id)
                    if pose is not None:
                        self.gt_transforms.append(conv.pose_to_transform(pose))
                    else:
                        raise ValueError(f"Camera pose not found for frame {frame_id}")

            # Store metadata about when this frame was processed
            if hasattr(processor, 'set_frame_metadata'):
                processor.set_frame_metadata(frame_id, {
                    'loop': self.current_loop,
                    'forward': self.replay_forward
                })

            # Update frame counter based on direction
            self.current_frame = self.current_frame + (1 if self.replay_forward else -1)
            frame_id += 1
