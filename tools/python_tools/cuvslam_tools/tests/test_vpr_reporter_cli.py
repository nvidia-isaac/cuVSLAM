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

import contextlib
import io
import unittest

from cuvslam_tools.vpr_reporter import cli


class TestVprReporterCli(unittest.TestCase):
    def test_parser_builds_without_the_tracker_loop_closure_flags(self):
        # The reporter has its own VPR flags; the tracker's would clash or be ignored.
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--help"])

        self.assertEqual(cm.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--vpr_model_path", help_text)
        self.assertNotIn("--loop_closure_mode", help_text)
        self.assertNotIn("--use_vpr", help_text)
        self.assertNotIn("--vpr_modes", help_text)


if __name__ == "__main__":
    unittest.main()
