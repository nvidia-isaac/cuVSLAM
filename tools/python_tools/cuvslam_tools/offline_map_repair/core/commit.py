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


"""Apply a decided edit to a fresh map clone, atomically, through the MCP tools.

Every policy in ``policies`` ends here. The sequence is deliberately fixed:

1. Validate the *intended* record set before touching the database, so a policy that
   would leave dangling references never reaches the server.
2. ``clone_map`` a new candidate, because the server refuses writes to any map
   without the ``agent-copy.json`` marker. The baseline is therefore never edited.
3. ``write_records`` the full diff in one compare-and-swap transaction. A single
   stale hash aborts the whole batch, so a candidate is either fully edited or
   untouched - there is no partially repaired map to mistake for a good one.
4. Cache the new records and an ``edit.json`` provenance file next to the candidate,
   so later policies can build on this one without re-reading the database.
"""

import json
import pickle
from pathlib import Path

from cuvslam_tools.offline_map_repair.core.map_codec import MapData, changes_between
from cuvslam_tools.offline_map_repair.core.mcp_client import call
from cuvslam_tools.offline_map_repair.core.workspace import HERE


async def commit(session, source, destination, old, new, reason):
    """Clone ``source`` into ``destination`` and write the ``old`` -> ``new`` record diff.

    ``reason`` is required by the server and is stored verbatim in the map's durable
    ``agent_audit`` table alongside the pre-edit bytes, so every candidate map carries
    its own undo information and a statement of intent.
    """
    # Decode the intended result and refuse to clone anything structurally broken.
    issues = MapData(new).validate()
    if issues:
        raise ValueError(issues[:20])
    # Compute the diff before cloning. The server rejects an empty batch, and a
    # policy that selected nothing produces exactly that - pruning zero landmarks
    # re-encodes to identical bytes. Cloning first would leave an orphaned
    # destination directory that makes every retry fail on mkdir.
    changes = changes_between(old, new)
    if not changes:
        raise ValueError(f"No record changes for {destination}; the policy selected nothing to edit")
    await call(session, "clone_map", {"source": source, "destination": destination})
    try:
        result = await call(session, "write_records", {"map_name": destination, "changes": changes, "reason": reason})
    except Exception as error:
        # The clone exists but is unedited, and there is no delete tool by design.
        raise RuntimeError(f"Write failed; remove the orphaned clone {destination} before retrying") from error
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
