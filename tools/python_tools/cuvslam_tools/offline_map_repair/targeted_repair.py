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

"""Conservative follow-up: edit only poses with large independent-replay disagreement."""

import argparse
import asyncio
import json
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from cuvslam_tools.offline_map_repair.map_codec import MapData
from cuvslam_tools.offline_map_repair.mcp_client import with_session
from cuvslam_tools.offline_map_repair.repair_map import (
    commit,
    distant_single_observation_landmarks,
)
from cuvslam_tools.offline_map_repair.workspace import HERE


async def run(s):
    original = pickle.loads((HERE / "baseline/records.pkl").read_bytes())
    original_map = MapData(original)
    reference = MapData(pickle.loads((HERE / "repaired/records.pkl").read_bytes()))
    selected = []
    for i in original_map.keyframes:
        a, b = original_map.poses[i], reference.poses[i]
        displacement = np.linalg.norm(a[:3, 3] - b[:3, 3])
        angle = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude() * 180 / np.pi
        if displacement > 0.5 or angle > 10:
            original_map.poses[i] = b
            selected.append(i)
    updated = original_map.encode(reindex=True, update_edges=True)
    await commit(
        s,
        "baseline/map",
        "targeted/map",
        original,
        updated,
        "Conservative exploratory repair: replace only active poses differing from inertial replay by >0.5 m or >10 deg. "
        "Preserve other original poses; rebuild spatial cells and coherent graph transforms. No ground truth used to calculate edits.",
    )
    (HERE / "targeted/selection.json").write_text(
        json.dumps(
            {
                "selected_keyframes": selected,
                "translation_threshold_m": 0.5,
                "rotation_threshold_deg": 10,
                "exploratory_followup": True,
            },
            indent=2,
        )
    )
    m = MapData(updated)
    removed = distant_single_observation_landmarks(m)
    m.remove_landmarks(removed)
    await commit(
        s,
        "targeted/map",
        "targeted_maintained/map",
        updated,
        m.encode(),
        f"Maintenance: remove {len(removed)} singly observed landmarks beyond 15 m, their descriptors and graph/cell references.",
    )


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    asyncio.run(with_session(run))


if __name__ == "__main__":
    main()
