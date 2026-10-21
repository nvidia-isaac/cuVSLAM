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

"""Resolve experiment data locations independently of the source checkout.

Two directories, deliberately separate:

``CUVSLAM_EXPERIMENT_DIR`` (``HERE``) is the workspace holding one dataset's
``baseline/``, ``replay_inertial/`` and candidate maps, plus every JSON artifact the
tools write. ``CUVSLAM_MAP_ROOT`` is the directory the MCP server was started with,
and the server refuses to touch anything outside it.

They are usually the same. They differ when one server serves several dataset
workspaces: then ``MAP_ROOT`` is the common parent and ``map_name`` rewrites a
workspace-relative name such as ``baseline/map`` into the root-relative name the
server expects. Keeping both in one place means no other module has to know this.
"""

import os
from pathlib import Path

# One workspace contains baseline/, replay_inertial/, and the candidate maps.
# Multiple workspaces can share a server whose root is their common parent.
HERE = Path(os.environ.get("CUVSLAM_EXPERIMENT_DIR", Path.cwd())).resolve()
MAP_ROOT = Path(os.environ.get("CUVSLAM_MAP_ROOT", HERE)).resolve()


def map_name(name):
    """Translate a workspace-relative map name into a server-root-relative name.

    Raises if the map lies outside the server root, which catches a mismatched
    ``CUVSLAM_MAP_ROOT`` here rather than as a confusing server-side rejection.
    """
    return str((HERE / name).resolve().relative_to(MAP_ROOT))
