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

"""Prove a candidate map is what was intended, structurally and to the native reader.

Four independent checks per candidate:

1. **Readback** - re-read every record over MCP and compare with the cached intent,
   so the committed database matches what the policy decided, not just what it sent.
2. **References** - no dangling keyframe, descriptor or cell references. The baseline
   is exempt from the cell check: native exports can ship stale cell members, which
   is recorded as a preexisting condition and required to be absent in candidates.
3. **Codec round trip** - decode/encode is byte identical, so the diff was complete.
4. **Native agreement** - run the C++ ``map_extractor`` and compare its reconstructed
   landmark positions with this codec's. This is the check that catches a
   misunderstanding of the binary schema rather than a mistake in the edit.

Finally it re-hashes ``baseline/map`` and compares with the hash recorded at mapping
time, proving the source map was never modified by any of this.
"""

import argparse
import asyncio
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np

from cuvslam_tools.offline_map_repair.core.map_codec import MapData
from cuvslam_tools.offline_map_repair.core.mcp_client import read_map, with_session
from cuvslam_tools.offline_map_repair.core.workspace import HERE


async def run(session, args):
    results = {}
    for name in args.variants:
        expected = pickle.loads((HERE / name / "records.pkl").read_bytes())
        actual = await read_map(session, name + "/map")
        equal = {t: actual[t] == expected[t] for t in expected}
        assert all(equal.values()), (name, equal)
        map_data = MapData(actual)
        issues = map_data.validate()
        encoded = map_data.encode()
        roundtrip = {t: encoded[t] == actual[t] for t in actual}
        # Decode what came back and insist it is self consistent.
        if name == "baseline":
            # Record existing native-export index leftovers; do not alter the control.
            assert all(i.startswith("Dangling cell member ") for i in issues), issues[:5]
            assert all(v for t, v in roundtrip.items() if t != "spatial_cells")
        else:
            assert not issues, (name, issues[:5])
            assert all(roundtrip.values()), (name, roundtrip)
        for f in map_data.keyframes.values():
            assert set(f["landmarks"]).issubset(map_data.landmarks)
        native_dir = HERE / f"native_{name}"
        native_dir.mkdir(exist_ok=True)
        with (native_dir / "extractor.log").open("w") as log:
            subprocess.run(
                [str(args.extractor.resolve()), "--map_path", str(HERE / name / "map")],
                cwd=native_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        # Cross-check against the native C++ decoder: same landmark ids, same geometry.
        native = json.loads((native_dir / "map.json").read_text())
        native_lm = {r["id"]: r for r in native["landmarks"]}
        assert native_lm.keys() == map_data.landmarks.keys()
        delta = max(
            np.linalg.norm(map_data.xyz(lm) - [native_lm[i]["pose"][c] for c in ["x", "y", "z"]])
            for i, lm in map_data.landmarks.items()
        )
        # NumPy/BLAS and Eigen can accumulate float32 transforms in different
        # orders. Allow 0.1 mm reconstruction roundoff, retaining the measured error.
        assert np.isfinite(delta) and delta <= 1e-4, (name, delta)
        results[name] = {
            "mcp_readback_equal": equal,
            "references_valid": not issues,
            "preexisting_dangling_cell_references": len(issues),
            "codec_roundtrip_equal": roundtrip,
            "native_max_landmark_error_m": float(delta),
            "native_landmark_tolerance_m": 1e-4,
            "landmarks": len(map_data.landmarks),
        }
        if args.compare_to:
            previous = pickle.loads((args.compare_to / name / "records.pkl").read_bytes())
            results[name]["previous_implementation_equal"] = {t: previous[t] == actual[t] for t in actual}
            assert all(results[name]["previous_implementation_equal"].values())
        print(name, results[name], flush=True)
    baseline_hash = hashlib.sha256((HERE / "baseline/map/data.mdb").read_bytes()).hexdigest()
    assert baseline_hash == json.loads((HERE / "baseline/summary.json").read_text())["map_sha256"]
    (HERE / "verification.json").write_text(
        json.dumps({"maps": results, "baseline_unchanged_sha256": baseline_hash}, indent=2)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variants", nargs="*", default=["baseline", "repaired", "targeted", "targeted_maintained", "retired_tail"]
    )
    parser.add_argument("--extractor", type=Path, required=True, help="Native map_extractor executable")
    parser.add_argument("--compare-to", type=Path, help="Previous workspace for byte-for-byte regression checks")
    args = parser.parse_args(argv)
    asyncio.run(with_session(lambda session: run(session, args)))


if __name__ == "__main__":
    main()
