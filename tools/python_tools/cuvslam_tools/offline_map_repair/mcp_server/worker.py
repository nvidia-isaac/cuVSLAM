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

"""NAT server worker allowing bounded whole-map edit transactions over HTTP."""

from nat.plugins.mcp.server.front_end_plugin_worker import MCPFrontEndPluginWorker


class MapMCPWorker(MCPFrontEndPluginWorker):
    async def create_mcp_server(self):
        server = await super().create_mcp_server()
        # Pose edits can touch every spatial cell. Keep the whole batch atomic;
        # splitting it to fit the default HTTP limit could expose a partial map.
        server.settings.max_request_body_size = 64 * 1024 * 1024
        return server
