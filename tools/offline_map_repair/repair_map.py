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

"""Agent-selected experimental edits applied exclusively through basic MCP tools.

No ground-truth files or query manifest are read. The second replay supplies poses;
the original map supplies landmarks, descriptors, observations and graph topology.
"""

import asyncio
import json
import pickle
from pathlib import Path

import numpy as np
from map_codec import MapData, changes_between
from mcp_client import call, with_session
from trajectory_utils import interpolate, load_trajectory
from workspace import HERE


async def commit(session, source, destination, old, new, reason):
    issues = MapData(new).validate()
    if issues:
        raise ValueError(issues[:20])
    await call(session, "clone_map", {"source": source, "destination": destination})
    changes = changes_between(old, new)
    result = await call(session, "write_records", {"map_name": destination, "changes": changes, "reason": reason})
    parent = HERE / Path(destination).parent
    (parent / "records.pkl").write_bytes(pickle.dumps(new))
    (parent / "edit.json").write_text(
        json.dumps(
            {
                "source": source,
                "reason": reason,
                **result,
                "validation": issues,
                "record_changes_by_table": {t: sum(c["table"] == t for c in changes) for t in old},
            },
            indent=2,
        )
    )
    print(json.dumps(result), flush=True)


async def run(session):
    original = pickle.loads((HERE / "baseline/records.pkl").read_bytes())
    map_data = MapData(original)
    ts, replay = load_trajectory(HERE / "replay_inertial/optimized_trajectory.json")
    first = min(map_data.keyframes, key=lambda k: map_data.keyframes[k]["timestamp_ns"])
    replay_at_first = interpolate(ts, replay, [map_data.keyframes[first]["timestamp_ns"]])[0]
    # Both replays use the same timestamps and camera frame, but have independent
    # world origins. Match one shared pose without fitting to ground truth.
    common_gauge = map_data.poses[first] @ np.linalg.inv(replay_at_first)
    ids = list(map_data.keyframes)
    corrected = common_gauge @ interpolate(ts, replay, [map_data.keyframes[k]["timestamp_ns"] for k in ids])
    deltas = [float(np.linalg.norm(map_data.poses[k][:3, 3] - p[:3, 3])) for k, p in zip(ids, corrected)]
    for k, p in zip(ids, corrected):
        map_data.poses[k] = p.astype(np.float32)
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
    # Maintenance policy chosen without localization scores: remove distant, singly observed points.
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
        "Prune single-observation landmarks farther than 15 m from their observation origin; remove associated "
        "descriptors, keyframe references and cell memberships. Preserve keyframes and edges. Evaluation queries not consulted.",
    )
    (HERE / "maintained/pruning.json").write_text(
        json.dumps(
            {"removed_landmarks": len(removed), "remaining_landmarks": len(map_data.landmarks), "ids": removed},
            indent=2,
        )
    )


def distant_single_observation_landmarks(map_data):
    """Experimental pruning policy: a single observation at a range over 15 m."""
    return [
        identifier
        for identifier, landmark in map_data.landmarks.items()
        if len(landmark["observations"]) == 1
        and np.linalg.norm(landmark["observations"][0]["xyz"] - landmark["observations"][0]["eye"]) > 15
    ]


if __name__ == "__main__":
    asyncio.run(with_session(run))
