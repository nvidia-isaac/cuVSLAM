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

"""Version-specific offline decoder for cuVSLAM main 2a0882c, little-endian x86_64.

Kept outside the MCP server: an agent can inspect and edit the raw representation
using normal code. Descriptor bytes and unknown/padding fields are preserved.
"""

import base64
import copy
import hashlib
import struct

import numpy as np


def key(i):
    return struct.pack("<Q", i)


def vector(ids):
    return key(len(ids)) + np.asarray(ids, dtype="<u8").tobytes()


def matrix_bytes(m):
    # Native Eigen matrices are float32, column-major, including their bottom row.
    return np.asarray(m, dtype="<f4").tobytes(order="F")


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise ValueError("Truncated blob or invalid count")
        result = self.data[self.pos : self.pos + n]
        self.pos += n
        return result

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))

    def u64(self):
        return self.unpack("Q")[0]

    def u32(self):
        return self.unpack("I")[0]

    def string(self):
        return self.take(self.u64()).decode()

    def ids(self):
        return np.frombuffer(self.take(self.u64() * 8), dtype="<u8").tolist()

    def matrix(self):
        return np.frombuffer(self.take(64), dtype="<f4").reshape(4, 4, order="F").copy()

    def done(self):
        if self.pos != len(self.data):
            raise ValueError(f"Unparsed data: {len(self.data) - self.pos} bytes")


class MapData:
    """Decode the supported native schema while preserving unedited byte ranges.

    `poses` can include reduced graph nodes no longer present in `keyframes`.
    Landmarks retain observations in keyframe coordinates, not a single world XYZ.
    """

    def __init__(self, records):
        self.records = records
        reader = Reader(records["main"][key(3)])
        if reader.string() != "PoseGraphHypothesis v0.0":
            raise ValueError("Unsupported pose hypothesis")
        self.hypothesis_prefix = reader.data[: reader.pos]
        self.poses = {}
        for _ in range(reader.u64()):
            identifier = reader.u64()
            self.poses[identifier] = reader.matrix()
        reader.done()
        reader = Reader(records["main"][key(2)])
        if reader.string() != "PoseGraph v0.00":
            raise ValueError("Unsupported graph")
        reader.take(24)  # next keyframe, next edge, graph head
        self.graph_prefix = reader.data[: reader.pos]
        self.keyframes = {}
        for _ in range(reader.u64()):
            start = reader.pos
            if reader.u32() != 2:
                raise ValueError("Unsupported keyframe")
            identifier = reader.u64()
            prefix = reader.data[start : reader.pos]
            ids = reader.ids()
            tail = reader.pos
            reader.take(reader.u64() * 64)  # merged transforms
            reader.string()  # frame metadata
            if reader.u32() != 0:
                raise ValueError("Unsupported keyframe metadata")
            timestamp = reader.unpack("q")[0]
            reader.take(16)  # legacy Gravity struct incl padding
            self.keyframes[identifier] = {
                "timestamp_ns": timestamp,
                "landmarks": ids,
                "prefix": prefix,
                "suffix": reader.data[tail : reader.pos],
            }
        self.edges = []
        for _ in range(reader.u64()):
            start = reader.pos
            if reader.u32() != 0:
                raise ValueError("Unsupported edge")
            identifier, a, b = reader.unpack("QQQ")
            transform = reader.matrix()
            covariance = np.frombuffer(reader.take(144), dtype="<f4").reshape(6, 6, order="F").copy()
            tracks, error = reader.unpack("Qf")
            self.edges.append(
                {
                    "id": identifier,
                    "a": a,
                    "b": b,
                    "transform": transform,
                    "covariance": covariance,
                    "tracks": tracks,
                    "error": error,
                    "raw": reader.data[start : reader.pos],
                }
            )
        reader.done()
        reader = Reader(records["main"][key(1)])
        if reader.string() != "LSIGrid v0.03":
            raise ValueError("Unsupported spatial index")
        self.cell_size = reader.unpack("f")[0]
        reader.take(16)
        reader.done()
        self.landmarks = {}
        for binary_key, blob in records["landmarks"].items():
            reader = Reader(blob)
            if reader.u32() != 1:
                raise ValueError("Unsupported landmark")
            obs = []
            for _ in range(reader.u32()):
                frame, link, xyz_valid = reader.unpack("Q??")
                xyz = np.array(reader.unpack("3f"), dtype=np.float32)
                uv = reader.unpack("2f")
                eye = np.array(reader.unpack("3f"), dtype=np.float32)
                obs.append({"keyframe": frame, "link": link, "has_xyz": xyz_valid, "xyz": xyz, "uv": uv, "eye": eye})
            prefix = blob[: reader.pos]
            cells = list(reader.unpack("Q" * reader.u32()))
            probes = reader.unpack("4i")
            reader.done()
            self.landmarks[int.from_bytes(binary_key, "little")] = {
                "observations": obs,
                "cells": cells,
                "probes": probes,
                "prefix": prefix,
            }

    def xyz(self, landmark):
        """Match native landmark reconstruction: average linked observations in world space."""
        values = [
            self.poses[o["keyframe"]][:3, :3] @ o["xyz"] + self.poses[o["keyframe"]][:3, 3]
            for o in landmark["observations"]
            if o["link"] and o["keyframe"] in self.poses
        ]
        return np.mean(values, axis=0) if values else np.zeros(3, dtype=np.float32)

    def encode(self, reindex=False, update_edges=False):
        """Encode edits and rebuild cell membership from surviving landmarks.

        Reindex after changing poses: cells encode the observer position and view
        direction, so moving only the pose graph leaves localization lookups stale.
        Even with reindex=False, stale cell members and removed descriptors disappear.
        """
        out = copy.deepcopy(self.records)
        out["main"][key(3)] = (
            self.hypothesis_prefix
            + key(len(self.poses))
            + b"".join(key(i) + matrix_bytes(m) for i, m in self.poses.items())
        )
        frames = []
        for info in self.keyframes.values():
            frames.append(info["prefix"] + vector(info["landmarks"]) + info["suffix"])
        edges = []
        for edge in self.edges:
            raw = edge["raw"]
            if update_edges:
                # Experimental policy: T_a_b = inverse(T_world_a) @ T_world_b.
                # Preserve covariance/tracks/error bytes; this is not graph optimization.
                transform = np.linalg.inv(self.poses[edge["a"]]) @ self.poses[edge["b"]]
                raw = raw[:28] + matrix_bytes(transform) + raw[92:]
            edges.append(raw)
        out["main"][key(2)] = (
            self.graph_prefix + key(len(frames)) + b"".join(frames) + key(len(edges)) + b"".join(edges)
        )
        out["landmarks"] = {}
        cells = {}
        for identifier, lm in self.landmarks.items():
            if reindex:
                xyz = self.xyz(lm)
                lm["cells"] = sorted(
                    {
                        cell_id(
                            self.poses[o["keyframe"]][:3, :3] @ o["eye"] + self.poses[o["keyframe"]][:3, 3],
                            xyz,
                            self.cell_size,
                        )
                        for o in lm["observations"]
                        if o["keyframe"] in self.poses
                    }
                )
            for c in lm["cells"]:
                cells.setdefault(c, []).append(identifier)
            out["landmarks"][key(identifier)] = (
                lm["prefix"]
                + struct.pack("<I", len(lm["cells"]))
                + b"".join(key(c) for c in lm["cells"])
                + struct.pack("<4i", *lm["probes"])
            )
        out["spatial_cells"] = {key(c): vector(sorted(set(ids))) for c, ids in cells.items()}
        out["descriptors"] = {
            k: v for k, v in out["descriptors"].items() if int.from_bytes(k, "little") in self.landmarks
        }
        return out

    def remove_landmarks(self, identifiers):
        ids = set(identifiers)
        for i in ids:
            del self.landmarks[i]
        for frame in self.keyframes.values():
            frame["landmarks"] = [i for i in frame["landmarks"] if i not in ids]

    def validate(self):
        issues = []
        # cuVSLAM retains pose hypotheses for graph nodes removed by reduction.
        if not set(self.keyframes).issubset(self.poses):
            issues.append("Active keyframes lack pose hypotheses")
        for identifier, m in self.poses.items():
            if (
                not np.isfinite(m).all()
                or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-4)
            ):
                issues.append(f"Invalid pose {identifier}")
        for edge in self.edges:
            if edge["a"] not in self.poses or edge["b"] not in self.poses:
                issues.append(f"Dangling edge {edge['id']}")
        for i, lm in self.landmarks.items():
            if key(i) not in self.records["descriptors"]:
                issues.append(f"Missing descriptor {i}")
            for o in lm["observations"]:
                if o["keyframe"] not in self.poses:
                    issues.append(f"Dangling landmark {i}")
        for k, blob in self.records["spatial_cells"].items():
            reader = Reader(blob)
            for i in reader.ids():
                if i not in self.landmarks:
                    issues.append(f"Dangling cell member {i}")
            reader.done()
        return issues


def cell_id(eye, xyz, size):
    """Exact layout of LSIGrid::GetKeyframeCells / grid::CellId on this main."""
    # Bits 0..56: three 19-bit spatial coordinates. Bits 57..59: cube face.
    # Bits 60..61: two one-bit view-direction bins on that face.
    position = (np.floor(np.asarray(eye, dtype=np.float64) / size).astype(np.int64) + (1 << 18)) & ((1 << 19) - 1)
    direction = np.asarray(xyz - eye, dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Degenerate landmark direction")
    direction /= norm
    axis = int(np.argmax(np.abs(direction)))
    plane = axis + (3 if direction[axis] < 0 else 0)
    uv = np.delete(direction, axis)
    uv = np.minimum(np.floor((uv * np.float32(0.5) + np.float32(0.5)) * 2).astype(int), 1)
    return (
        int(position[0])
        | (int(position[1]) << 19)
        | (int(position[2]) << 38)
        | (plane << 57)
        | (int(uv[0]) << 60)
        | (int(uv[1]) << 61)
    )


def changes_between(old, new):
    changes = []
    for table in old:
        for k in sorted(old[table].keys() | new[table].keys()):
            before, after = old[table].get(k), new[table].get(k)
            if before != after:
                changes.append(
                    {
                        "table": table,
                        "key_hex": k.hex(),
                        "expected_sha256": hashlib.sha256(before).hexdigest() if before is not None else None,
                        "value_base64": base64.b64encode(after).decode() if after is not None else None,
                    }
                )
    return changes
