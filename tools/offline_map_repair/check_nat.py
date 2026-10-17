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

"""NAT-specific protocol, schema, concurrency, and root boundary checks."""

import asyncio
import json
import tempfile
from pathlib import Path

from mcp_client import call, with_session
from workspace import HERE, map_name


async def run(session):
    checks = []
    names = {t.name: t for t in (await session.list_tools()).tools}
    assert set(names) == {"describe_map", "clone_map", "read_records", "write_records"}
    assert names["read_records"].inputSchema["properties"]["limit"]["maximum"] == 5000
    checks.append("NAT exposes exactly four tools with typed Pydantic input schemas")
    replies = await asyncio.gather(*(call(session, "describe_map", {"map_name": "baseline/map"}) for _ in range(8)))
    assert all(r == replies[0] for r in replies)
    checks.append("Eight concurrent reads of the same LMDB map complete consistently")
    with tempfile.TemporaryDirectory(prefix="cuvslam-nat-boundary-") as folder:
        (Path(folder) / "data.mdb").touch()
        result = await session.call_tool("describe_map", {"map_name": folder})
        assert result.isError
        assert "below the configured root" in str(result)
    checks.append("Existing data.mdb outside configured root is rejected")
    for args in [
        dict(map_name=map_name("baseline/map"), table="main", limit=5001),
        dict(map_name=map_name("baseline/map"), table="main", offset=-1),
    ]:
        result = await session.call_tool("read_records", args)
        assert result.isError
    checks.append("Out-of-range read pagination rejected at MCP schema boundary")
    (HERE / "nat_checks.json").write_text(json.dumps({"passed": checks}, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    asyncio.run(with_session(run))
