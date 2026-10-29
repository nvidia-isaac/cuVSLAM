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


"""Policy: replace only the poses where the two runs disagree substantially.

``full_pose_repair`` rewrites every keyframe, which also rewrites the ones the stereo
run already had right - and the replay is itself an estimate, so that can import new
error. This conservative variant keeps the baseline pose wherever the two runs
already agree and only adopts the replay where the disagreement is large enough that
one of them must be wrong.

It reads ``repaired/records.pkl`` purely as a convenient cache of the gauge-aligned
replay poses, so ``cuvslam_map_repair repair`` must have run first.
"""

import argparse
import asyncio
import json
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from cuvslam_tools.offline_map_repair.core.commit import commit
from cuvslam_tools.offline_map_repair.core.map_codec import MapData
from cuvslam_tools.offline_map_repair.core.mcp_client import with_session
from cuvslam_tools.offline_map_repair.core.workspace import HERE
from cuvslam_tools.offline_map_repair.policies.pruning import (
    SINGLE_OBSERVATION_RANGE_M,
    distant_single_observation_landmarks,
)

# Above these the two runs cannot both be describing the same pose, so prefer the
# inertially constrained one. Below them the difference is within normal VO noise.
TRANSLATION_THRESHOLD_M = 0.5
ROTATION_THRESHOLD_DEG = 10.0


async def run(session, baseline="baseline", repaired="repaired"):
    original = pickle.loads((HERE / baseline / "records.pkl").read_bytes())
    original_map = MapData(original)
    reference = MapData(pickle.loads((HERE / repaired / "records.pkl").read_bytes()))
    selected = []
    for i in original_map.keyframes:
        a, b = original_map.poses[i], reference.poses[i]
        displacement = np.linalg.norm(a[:3, 3] - b[:3, 3])
        angle = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude() * 180 / np.pi
        if displacement > TRANSLATION_THRESHOLD_M or angle > ROTATION_THRESHOLD_DEG:
            original_map.poses[i] = b
            selected.append(i)
    # Even a partial pose change invalidates the spatial index globally, because cells
    # are keyed by observer position: rebuild all of it, not just the moved keyframes.
    updated = original_map.encode(reindex=True, update_edges=True)
    await commit(
        session,
        f"{baseline}/map",
        "targeted/map",
        original,
        updated,
        f"Conservative exploratory repair: replace only active poses differing from the '{repaired}' estimate by "
        f">{TRANSLATION_THRESHOLD_M:g} m or >{ROTATION_THRESHOLD_DEG:g} deg. Preserve other original poses; rebuild "
        "spatial cells and coherent graph transforms. No ground truth used to calculate edits.",
    )
    (HERE / "targeted/selection.json").write_text(
        json.dumps(
            {
                "selected_keyframes": selected,
                "translation_threshold_m": TRANSLATION_THRESHOLD_M,
                "rotation_threshold_deg": ROTATION_THRESHOLD_DEG,
                "exploratory_followup": True,
            },
            indent=2,
        )
    )
    # Same landmark cleanup as the full repair, kept as a separate clone so the two
    # effects stay separable when the candidates are evaluated.
    m = MapData(updated)
    removed = distant_single_observation_landmarks(m)
    m.remove_landmarks(removed)
    await commit(
        session,
        "targeted/map",
        "targeted_maintained/map",
        updated,
        m.encode(),
        f"Maintenance: remove {len(removed)} singly observed landmarks beyond {SINGLE_OBSERVATION_RANGE_M:g} m, "
        "their descriptors and graph/cell references.",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="baseline", help="Session holding the map to repair")
    ap.add_argument("--repaired", default="repaired", help="Candidate supplying the gauge-aligned reference poses")
    args = ap.parse_args(argv)
    asyncio.run(with_session(lambda s: run(s, args.baseline, args.repaired)))


if __name__ == "__main__":
    main()
