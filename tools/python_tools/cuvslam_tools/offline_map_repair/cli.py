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

"""Console entry point for offline map repair workflows."""

import argparse
from importlib import import_module

COMMANDS = {
    "serve": ("mcp_server.server", "Serve raw map records through NeMo MCP"),
    "inspect": ("inspect_map", "Inspect a map and cache its MCP records"),
    "repair": ("repair_map", "Replace poses from an inertial replay and make a pruned copy"),
    "targeted": ("targeted_repair", "Replace only poses with large replay disagreement"),
    "reindex": ("repair_index", "Rebuild the spatial index without replacing poses or edges"),
    "retire": ("retire_section", "Remove landmarks exclusive to the trailing time section"),
    "run-sequence": ("run_sequence", "Create a stereo or inertial EuRoC SLAM session"),
    "evaluate": ("evaluate_localization", "Prepare queries or evaluate native relocalization"),
    "verify": ("verify_maps", "Compare MCP records with cached and native map representations"),
    "check-mcp": ("check_mcp", "Check MCP transactions on a disposable map copy"),
    "check-nat": ("check_nat", "Check NeMo schemas, concurrent reads, and root confinement"),
    "call": ("mcp_client", "Call a raw MCP tool with arguments from a JSON file"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        # Each workflow owns its arguments and help. Import it only after dispatch
        # so listing commands does not require NeMo, LMDB, or the cuVSLAM binding.
        subparsers.add_parser(name, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    module_name = COMMANDS[args.command][0]
    module = import_module(f"cuvslam_tools.offline_map_repair.{module_name}")
    return module.main(remaining)


if __name__ == "__main__":
    main()
