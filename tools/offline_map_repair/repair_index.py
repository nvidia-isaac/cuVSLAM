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

"""Control experiment: rebuild lookup cells while retaining poses, edges, and appearance."""

import asyncio
import pickle

from map_codec import MapData
from mcp_client import with_session
from repair_map import commit
from workspace import HERE


async def run(session):
    old = pickle.loads((HERE / "baseline/records.pkl").read_bytes())
    new = MapData(old).encode(reindex=True, update_edges=False)
    assert old["main"] == new["main"] and old["descriptors"] == new["descriptors"]
    await commit(
        session,
        "baseline/map",
        "index_only/map",
        old,
        new,
        "Control: rebuild spatial cells and landmark cell references from original poses; "
        "remove stale cell entries. Preserve all poses, edge measurements, landmarks and descriptors.",
    )


if __name__ == "__main__":
    asyncio.run(with_session(run))
