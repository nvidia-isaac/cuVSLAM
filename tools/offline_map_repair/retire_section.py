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

"""Demonstrate explicit section retirement, not automatic detection of an aged scene."""

import asyncio
import json
import pickle

from map_codec import MapData
from mcp_client import with_session
from repair_map import commit
from workspace import HERE


async def run(session):
    source = pickle.loads((HERE / "targeted_maintained/records.pkl").read_bytes())
    map_data = MapData(source)
    cutoff = max(f["timestamp_ns"] for f in map_data.keyframes.values()) - 15_000_000_000
    removed = [
        i
        for i, lm in map_data.landmarks.items()
        if lm["observations"]
        and all(map_data.keyframes[o["keyframe"]]["timestamp_ns"] >= cutoff for o in lm["observations"])
    ]
    map_data.remove_landmarks(removed)
    await commit(
        session,
        "targeted_maintained/map",
        "retired_tail/map",
        source,
        map_data.encode(),
        "Section-retirement experiment: remove landmarks observed exclusively by retained keyframes in the last 15 s, "
        "plus their descriptors and all references. Keep graph connectivity. The cutoff is an explicit policy, not evidence of scene aging.",
    )
    (HERE / "retired_tail/retirement.json").write_text(
        json.dumps(
            {
                "cutoff_timestamp_ns": cutoff,
                "removed_landmarks": len(removed),
                "remaining_landmarks": len(map_data.landmarks),
                "scope": "visual landmarks exclusive to a trailing time section; pose graph retained",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(with_session(run))
