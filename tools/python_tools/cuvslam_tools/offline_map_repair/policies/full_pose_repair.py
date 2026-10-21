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


"""Policy: replace every active keyframe pose with the stereo-inertial replay estimate.

The premise is that a stereo-only run can drift badly on a hard sequence while a
stereo-inertial run of the *same* recording stays metric. The visual content of the
baseline map - descriptors, observation rays, graph topology - is still good; only
where the keyframes sit in the world is wrong. So this policy keeps the former and
replaces the latter.

No ground truth is read here. The replay supplies poses, the original map supplies
everything else, and the two are tied together at one shared keyframe.
"""

import argparse
import asyncio
import json
import pickle

import numpy as np

from cuvslam_tools.offline_map_repair.core.commit import commit
from cuvslam_tools.offline_map_repair.core.map_codec import MapData
from cuvslam_tools.offline_map_repair.core.mcp_client import with_session
from cuvslam_tools.offline_map_repair.core.workspace import HERE
from cuvslam_tools.offline_map_repair.policies.pruning import (
    SINGLE_OBSERVATION_RANGE_M,
    distant_single_observation_landmarks,
)
from cuvslam_tools.offline_map_repair.sessions.trajectory import interpolate, load_trajectory


async def run(session):
    original = pickle.loads((HERE / "baseline/records.pkl").read_bytes())
    map_data = MapData(original)
    ts, replay = load_trajectory(HERE / "replay_inertial/optimized_trajectory.json")

    # Both runs share timestamps and the cam0 frame, but each SLAM session picks its
    # own world origin. Anchoring at the earliest keyframe gives a single rigid
    # transform that carries replay poses into the baseline map's gauge - it is one
    # pose alignment, not a least-squares fit, and it never touches ground truth.
    first = min(map_data.keyframes, key=lambda k: map_data.keyframes[k]["timestamp_ns"])
    replay_at_first = interpolate(ts, replay, [map_data.keyframes[first]["timestamp_ns"]])[0]
    common_gauge = map_data.poses[first] @ np.linalg.inv(replay_at_first)

    # Keyframe timestamps rarely coincide with replay samples, so interpolate the
    # replay at each one (Slerp for rotation, linear for translation).
    ids = list(map_data.keyframes)
    corrected = common_gauge @ interpolate(ts, replay, [map_data.keyframes[k]["timestamp_ns"] for k in ids])
    deltas = [float(np.linalg.norm(map_data.poses[k][:3, 3] - p[:3, 3])) for k, p in zip(ids, corrected)]
    for k, p in zip(ids, corrected):
        # Native poses are float32; round-trip now so the encoded bytes are exact.
        map_data.poses[k] = p.astype(np.float32)

    # reindex=True is mandatory after moving poses: a spatial cell id is derived from
    # the observer position and view direction, both of which just changed. Leaving
    # the old cells would make the native relocalizer look up landmarks in the wrong
    # places. update_edges=True keeps relative edge transforms consistent with the
    # new absolute poses; it is bookkeeping, not graph optimization.
    repaired = map_data.encode(reindex=True, update_edges=True)
    reason = (
        "Replace active keyframe poses using the same sequence's stereo-inertial replay, anchored at the first shared pose; "
        "update edge relative transforms to that replay estimate and rebuild landmark/cell links. Keep original "
        "landmark observation coordinates, descriptor bytes, graph topology and edge covariances. No ground truth."
    )
    await commit(session, "baseline/map", "repaired/map", original, repaired, reason)
    (HERE / "repaired/pose_changes.json").write_text(
        json.dumps(
            {
                "active_keyframes": len(ids),
                "translation_change_median_m": float(np.median(deltas)),
                "translation_change_max_m": max(deltas),
                "reference_to_map": common_gauge.tolist(),
                "ground_truth_used": False,
            },
            indent=2,
        )
    )

    # Second, independent candidate: the same repair plus a landmark cleanup. It is
    # committed as its own clone so the pose fix can be evaluated without the pruning.
    # The threshold was chosen a priori, not tuned against the localization scores.
    map_data = MapData(repaired)
    removed = distant_single_observation_landmarks(map_data)
    map_data.remove_landmarks(removed)
    maintained = map_data.encode()
    await commit(
        session,
        "repaired/map",
        "maintained/map",
        repaired,
        maintained,
        f"Prune single-observation landmarks farther than {SINGLE_OBSERVATION_RANGE_M:g} m from their observation "
        "origin; remove associated descriptors, keyframe references and cell memberships. Preserve keyframes and "
        "edges. Evaluation queries not consulted.",
    )
    (HERE / "maintained/pruning.json").write_text(
        json.dumps(
            {"removed_landmarks": len(removed), "remaining_landmarks": len(map_data.landmarks), "ids": removed},
            indent=2,
        )
    )


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    asyncio.run(with_session(run))


if __name__ == "__main__":
    main()
