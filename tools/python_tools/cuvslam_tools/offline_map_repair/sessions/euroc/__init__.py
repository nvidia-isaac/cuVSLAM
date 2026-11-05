"""Everything that knows the EuRoC on-disk layout, and nothing that does not.

Swapping datasets means providing another package exposing these four names; no
module outside here reads a recording, so nothing else has to change.
"""

from cuvslam_tools.offline_map_repair.sessions.euroc.frames import frame_metadata
from cuvslam_tools.offline_map_repair.sessions.euroc.ground_truth import ground_truth
from cuvslam_tools.offline_map_repair.sessions.euroc.reader import (
    calibration_source,
    get_rig,
    load_frame,
    read_csv_data,
)

__all__ = ["calibration_source", "frame_metadata", "get_rig", "ground_truth", "load_frame", "read_csv_data"]
