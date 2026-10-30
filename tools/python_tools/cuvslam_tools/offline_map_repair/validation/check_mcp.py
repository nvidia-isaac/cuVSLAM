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

"""Integration checks of the MCP contract, run over the real transport.

These are not unit tests of ``store.py`` (those live in ``cuvslam_tools/tests``); they
exercise the guarantees the repair workflow actually relies on, end to end:

* the server exposes exactly the four intended tools and nothing else,
* the original map cannot be written to, only clones can,
* a path outside the configured root is rejected,
* one stale compare-and-swap token rolls back the whole batch, data and audit alike,
* a binary value survives write, read and restore unchanged.

Needs a workspace containing ``baseline/map``, and creates ``mcp_checks/map``; use a
fresh workspace for a repeat run, since ``clone_map`` refuses an existing destination.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import tempfile
from pathlib import Path

import lmdb

from cuvslam_tools.offline_map_repair.core.mcp_client import call, with_session
from cuvslam_tools.offline_map_repair.core.workspace import HERE, MAP_ROOT


async def run(s):
    passed = []
    tools = await s.list_tools()
    assert {t.name for t in tools.tools} == {"describe_map", "clone_map", "read_records", "write_records"}
    passed.append("MCP initialization and four-tool discovery")
    page = await call(s, "read_records", {"map_name": "baseline/map", "table": "descriptors", "limit": 1})
    record = page["records"][0]
    old = base64.b64decode(record["value_base64"])
    new = old[:-1] + bytes([old[-1] ^ 1])
    change = {
        "table": "descriptors",
        "key_hex": record["key_hex"],
        "expected_sha256": record["sha256"],
        "value_base64": base64.b64encode(new).decode(),
    }

    async def rejects(name, args):
        try:
            await call(s, name, args)
        except RuntimeError:
            return
        raise AssertionError("Expected rejection")

    await rejects(
        "write_records", {"map_name": "baseline/map", "changes": [change], "reason": "Check baseline protection"}
    )
    passed.append("Baseline write protection")
    # The path must hold a real data.mdb, otherwise map_path rejects it for being a
    # missing database and the check would pass with the containment guard removed.
    with tempfile.TemporaryDirectory(prefix="cuvslam-mcp-outside-", dir=MAP_ROOT.parent) as folder:
        (Path(folder) / "data.mdb").touch()
        result = await s.call_tool("describe_map", {"map_name": folder})
        assert result.isError
        assert "below the configured root" in str(result)
    passed.append("Root confinement")
    await call(s, "clone_map", {"source": "baseline/map", "destination": "mcp_checks/map"})
    stale = {**change, "key_hex": (999999999).to_bytes(8, "little").hex(), "expected_sha256": "wrong"}
    await rejects(
        "write_records", {"map_name": "mcp_checks/map", "changes": [change, stale], "reason": "Rollback check"}
    )
    after = await call(s, "read_records", {"map_name": "mcp_checks/map", "table": "descriptors", "limit": 1})
    assert after["records"][0] == record
    passed.append("Stale-write rejection rolls back earlier changes in the transaction")
    await call(
        s,
        "write_records",
        {"map_name": "mcp_checks/map", "changes": [change], "reason": "Round-trip integration check"},
    )
    restore = {**change, "expected_sha256": hashlib.sha256(new).hexdigest(), "value_base64": record["value_base64"]}
    await call(
        s, "write_records", {"map_name": "mcp_checks/map", "changes": [restore], "reason": "Restore integration check"}
    )
    after = await call(s, "read_records", {"map_name": "mcp_checks/map", "table": "descriptors", "limit": 1})
    assert after["records"][0] == record
    passed.append("Binary write/read/restore round trip")
    with lmdb.open(str(HERE / "mcp_checks/map"), readonly=True, create=False, max_dbs=10) as env:
        db = env.open_db(b"agent_audit")
        with env.begin(db=db) as txn:
            assert txn.stat()["entries"] == 2
    passed.append("Only committed transactions appear in the durable audit")
    (HERE / "mcp_checks/results.json").write_text(json.dumps({"passed": passed}, indent=2))
    print(json.dumps(passed, indent=2))


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    asyncio.run(with_session(run))


if __name__ == "__main__":
    main()
