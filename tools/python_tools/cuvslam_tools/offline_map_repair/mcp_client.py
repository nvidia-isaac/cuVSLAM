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

"""Exercise NAT's Streamable HTTP MCP endpoint and retain a compact transcript."""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from cuvslam_tools.offline_map_repair.mcp_server.store import TABLES
from cuvslam_tools.offline_map_repair.workspace import HERE, map_name

URL = os.environ.get("CUVSLAM_MCP_URL", "http://127.0.0.1:9906/mcp")


async def with_session(action):
    async with streamablehttp_client(URL, timeout=300, sse_read_timeout=300) as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            initialized = await session.initialize()
            with (HERE / "mcp_transcript.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "initialize",
                            "url": URL,
                            "server": initialized.serverInfo.model_dump(),
                            "protocol": initialized.protocolVersion,
                        }
                    )
                    + "\n"
                )
            return await action(session)


async def call(session, tool, args):
    args = dict(args)
    for field in ("map_name", "source", "destination"):
        if field in args:
            args[field] = map_name(args[field])
    result = await session.call_tool(tool, args)
    blocks = [c.text for c in result.content if c.type == "text"]
    text = "\n".join(blocks)
    record = {
        "time_ns": time.time_ns(),
        "tool": tool,
        "args": args
        if tool != "write_records"
        else {
            "map_name": args["map_name"],
            "reason": args["reason"],
            "change_count": len(args["changes"]),
            "changes_sha256": hashlib.sha256(json.dumps(args["changes"]).encode()).hexdigest(),
        },
        "is_error": result.isError,
        "result_bytes": len(text),
        "result_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    with (HERE / "mcp_transcript.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if result.isError:
        raise RuntimeError(text)
    return json.loads(text)


async def read_map(session, name):
    data = {}
    for table in TABLES:
        rows, offset = {}, 0
        while True:
            page = await call(
                session, "read_records", {"map_name": name, "table": table, "offset": offset, "limit": 5000}
            )
            rows.update({bytes.fromhex(r["key_hex"]): base64.b64decode(r["value_base64"]) for r in page["records"]})
            offset = page["next_offset"]
            if offset >= page["total"]:
                break
        data[table] = rows
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=["describe_map", "clone_map", "read_records", "write_records"])
    parser.add_argument("arguments", type=Path, nargs="?", help="JSON file containing tool arguments")
    args = parser.parse_args(argv)
    arguments = json.loads(args.arguments.read_text()) if args.arguments else {}
    print(json.dumps(asyncio.run(with_session(lambda session: call(session, args.tool, arguments))), indent=2))


if __name__ == "__main__":
    main()
