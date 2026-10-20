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

"""Offline LMDB record operations shared by the NAT tools.

Application blobs remain opaque. Mutation policy belongs to the MCP client.
"""

import base64
import hashlib
import json
import time
from pathlib import Path

import lmdb

TABLES = ("main", "landmarks", "descriptors", "spatial_cells")


def digest(value):
    return hashlib.sha256(value).hexdigest() if value is not None else None


class MapStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def map_path(self, name):
        p = (self.root / name).resolve()
        if not p.is_relative_to(self.root) or not (p / "data.mdb").is_file():
            raise ValueError("Map must be an existing database below the configured root")
        return p

    def open_map(self, name, write=False):
        # The marker is a workflow guard, not an authentication mechanism. Native
        # cuVSLAM uses MDB_NOLOCK, so even a marked copy must be offline to edit.
        p = self.map_path(name)
        if write and not (p / "agent-copy.json").is_file():
            raise ValueError("Writes are allowed only to copies created by clone_map")
        return lmdb.open(
            str(p),
            readonly=not write,
            create=False,
            lock=True,
            max_dbs=10,
            map_size=max(1024**3, (p / "data.mdb").stat().st_size * 4),
        )

    def table_handle(self, env, name):
        if name not in TABLES:
            raise ValueError("Unknown table")
        return env.open_db(name.encode(), create=False)

    def describe_map(self, map_name: str) -> dict:
        """List tables and record counts. Does not interpret application blobs."""
        with self.open_map(map_name) as env:
            handles = {name: self.table_handle(env, name) for name in TABLES}
            with env.begin() as txn:
                counts = {name: txn.stat(db=h)["entries"] for name, h in handles.items()}
        return {
            "map": map_name,
            "tables": counts,
            "writable": (self.map_path(map_name) / "agent-copy.json").is_file(),
            "data_bytes": (self.map_path(map_name) / "data.mdb").stat().st_size,
        }

    def clone_map(self, source: str, destination: str) -> dict:
        """Create a compact independent database copy; never overwrite a destination."""
        src = self.map_path(source)
        dst = (self.root / destination).resolve()
        if not dst.is_relative_to(self.root) or dst == self.root:
            raise ValueError("Destination must be below the configured root")
        dst.mkdir(parents=True, exist_ok=False)
        with self.open_map(source) as env:
            env.copy(str(dst), compact=True)
        (dst / "agent-copy.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "source_sha256": digest((src / "data.mdb").read_bytes()),
                    "created_ns": time.time_ns(),
                }
            )
        )
        return self.describe_map(destination)

    def read_records(self, map_name: str, table: str, offset: int = 0, limit: int = 500) -> dict:
        """Read a page of binary key/value records with SHA256 tokens for conditional writes."""
        if offset < 0 or not 1 <= limit <= 5000:
            raise ValueError("offset >= 0 and 1 <= limit <= 5000 required")
        rows = []
        with self.open_map(map_name) as env:
            h = self.table_handle(env, table)
            with env.begin(db=h) as txn:
                total = txn.stat()["entries"]
                for i, (key, value) in enumerate(txn.cursor()):
                    if i < offset:
                        continue
                    if len(rows) == limit:
                        break
                    rows.append(
                        {
                            "key_hex": key.hex(),
                            "value_base64": base64.b64encode(value).decode(),
                            "sha256": digest(value),
                        }
                    )
        return {"records": rows, "total": total, "next_offset": offset + len(rows)}

    def write_records(self, map_name: str, changes: list[dict], reason: str) -> dict:
        """Atomically put/delete records in a copy. Each change needs table, key_hex,
        expected_sha256 (null for absent), and value_base64 (null deletes).
        A mismatched token aborts the entire batch. Audit is committed in the same transaction.
        Application consistency is the caller's responsibility.
        """
        if not reason.strip() or not changes:
            raise ValueError("A reason and nonempty changes are required")
        seen = set()
        audit_changes = []
        with self.open_map(map_name, write=True) as env:
            handles = {name: self.table_handle(env, name) for name in TABLES}
            audit_db = env.open_db(b"agent_audit", create=True)
            # Compare-and-swap tokens prevent edits based on stale reads. Any
            # failure rolls back both the data edits and their undo/audit record.
            with env.begin(write=True) as txn:
                for change in changes:
                    table = change["table"]
                    if table not in TABLES:
                        raise ValueError("Unknown table")
                    key = bytes.fromhex(change["key_hex"])
                    if len(key) != 8 or (table, key) in seen:
                        raise ValueError("Keys must be unique 64-bit keys")
                    seen.add((table, key))
                    old = txn.get(key, db=handles[table])
                    if digest(old) != change["expected_sha256"]:
                        raise ValueError(f"Stale record: {table}/{key.hex()}")
                    encoded = change["value_base64"]
                    new = base64.b64decode(encoded, validate=True) if encoded is not None else None
                    if new is None:
                        txn.delete(key, db=handles[table])
                    else:
                        txn.put(key, new, db=handles[table])
                    audit_changes.append(
                        {
                            "table": table,
                            "key_hex": key.hex(),
                            "before_sha256": digest(old),
                            "after_sha256": digest(new),
                            "before_base64": base64.b64encode(old).decode() if old is not None else None,
                        }
                    )
                audit_id = str(time.time_ns())
                txn.put(
                    audit_id.encode(), json.dumps({"reason": reason, "changes": audit_changes}).encode(), db=audit_db
                )
            env.sync()
        return {"map": map_name, "records_changed": len(changes), "audit_id": audit_id}
