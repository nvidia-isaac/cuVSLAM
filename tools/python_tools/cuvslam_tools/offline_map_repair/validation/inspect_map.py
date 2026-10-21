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

"""Read a map through MCP, cache its records, and report what is in it.

This is the first step of any repair: it pulls every record over the transport,
pickles them next to the map as ``records.pkl``, and writes an ``inspection.json``
summary. The repair policies then work from that cache instead of re-reading the
database, which keeps each policy's input fixed and reproducible.

``roundtrip_equal`` in the summary is the important field. It says whether decoding
and re-encoding the map reproduces it byte for byte, i.e. whether the codec actually
understands this map. A False entry there means no edit should be trusted.
"""

import argparse
import asyncio
import json
import pickle
from pathlib import Path

import numpy as np

from cuvslam_tools.offline_map_repair.core.map_codec import MapData
from cuvslam_tools.offline_map_repair.core.mcp_client import call, read_map, with_session
from cuvslam_tools.offline_map_repair.core.workspace import HERE


async def inspect(session, name):
    summary = await call(session, "describe_map", {"map_name": name})
    data = await read_map(session, name)
    map_data = MapData(data)
    # The cache is a tool-output artifact only; every actual write still goes through
    # MCP with compare-and-swap tokens read from the live database.
    cache = HERE / Path(name).parent / "records.pkl"
    cache.write_bytes(pickle.dumps(data))
    roundtrip = map_data.encode()
    summary.update(
        {
            "poses": len(map_data.poses),
            "edges": len(map_data.edges),
            "cell_size": map_data.cell_size,
            "validation": map_data.validate()[:30],
            "roundtrip_equal": {t: roundtrip[t] == data[t] for t in data},
            "keyframe_id_range": [min(map_data.poses), max(map_data.poses)],
        }
    )
    coords = np.array([map_data.xyz(lm) for lm in map_data.landmarks.values()])
    obs = [len(lm["observations"]) for lm in map_data.landmarks.values()]
    summary.update(
        {
            "landmark_bbox": [coords.min(0).tolist(), coords.max(0).tolist()],
            "observation_count_quantiles": np.quantile(obs, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        }
    )
    cache.with_name("inspection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("map_name")
    args = ap.parse_args(argv)
    asyncio.run(with_session(lambda s: inspect(s, args.map_name)))


if __name__ == "__main__":
    main()
