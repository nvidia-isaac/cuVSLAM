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

"""Launch the map tools through NVIDIA NeMo Agent Toolkit's MCP server."""

import argparse
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="Directory containing offline map snapshots")
    ap.add_argument("--port", type=int, default=9906)
    ap.add_argument(
        "--nat-python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable in the separate NeMo server environment",
    )
    args = ap.parse_args(argv)
    # Keep host ROS Python packages out of the isolated NAT runtime.
    os.environ.pop("PYTHONPATH", None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["CUVSLAM_MAP_ROOT"] = str(args.root.resolve())
    nat = args.nat_python.absolute().parent / "nat"
    if not nat.is_file():
        raise SystemExit(
            "Install the mcp_server project in a separate environment and pass its Python with --nat-python."
        )
    cmd = [
        str(nat),
        "mcp",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--runner_class",
        "cuvslam_tools.offline_map_repair.mcp_server.worker.MapMCPWorker",
    ]
    for tool in ("describe_map", "clone_map", "read_records", "write_records"):
        cmd.extend(["--tool_names", tool])
    config = resources.files("cuvslam_tools.offline_map_repair.mcp_server").joinpath("configs/map_records.yaml")
    # Keep extracted resources alive until NAT exits, including in a wheel install.
    with resources.as_file(config) as config_path:
        raise SystemExit(subprocess.call([*cmd, "--config_file", str(config_path)]))


if __name__ == "__main__":
    main()
