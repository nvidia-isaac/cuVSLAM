#!/usr/bin/env python3
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

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuvslam_benchmark_report import parse_gtest_xml, render_markdown  # noqa: E402


def write_xml(directory: str, body: str) -> Path:
    path = Path(directory) / "benchmarks.xml"
    path.write_text(body)
    return path


class CuvslamBenchmarkReportTest(unittest.TestCase):
    def test_parses_metrics_failures_and_suppressed_tests(self):
        xml = """\
<testsuites tests="3" failures="1">
  <testsuite name="Cuda" tests="3" failures="1">
    <testcase name="SpeedUpPass" classname="Cuda" status="run" result="completed" time="1.25">
      <properties>
        <property name="iterations" value="100"/>
        <property name="cpu_ns_per_iteration" value="5000000"/>
        <property name="gpu_ns_per_iteration" value="1000000"/>
        <property name="speedup" value="5.0"/>
      </properties>
    </testcase>
    <testcase name="SpeedupFail" classname="Cuda" status="run" result="completed" time="2.5">
      <properties>
        <property name="iterations" value="100"/>
        <property name="cpu_ns_per_iteration" value="1000000"/>
        <property name="gpu_ns_per_iteration" value="2000000"/>
        <property name="speedup" value="0.5"/>
      </properties>
      <failure message="GPU was slower"/>
    </testcase>
    <testcase name="DISABLED_SpeedUp" classname="Cuda" status="notrun" result="suppressed" time="0"/>
  </testsuite>
</testsuites>
"""
        with tempfile.TemporaryDirectory() as directory:
            report = parse_gtest_xml(write_xml(directory, xml), expected_count=2)

        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertTrue(report["summary"]["complete"])
        self.assertEqual(report["benchmarks"][0]["cpu_ns_per_iteration"], 5_000_000)
        self.assertEqual(report["benchmarks"][1]["failure_message"], "GPU was slower")

    def test_marks_missing_metrics_incomplete(self):
        xml = """\
<testsuites>
  <testsuite name="Cuda">
    <testcase name="SpeedUpMissing" classname="Cuda" status="run" result="completed" time="0.1">
      <properties>
        <property name="iterations" value="100"/>
      </properties>
    </testcase>
  </testsuite>
</testsuites>
"""
        with tempfile.TemporaryDirectory() as directory:
            report = parse_gtest_xml(write_xml(directory, xml), expected_count=1)

        self.assertEqual(report["summary"]["errors"], 1)
        self.assertFalse(report["summary"]["complete"])
        self.assertIn("cpu_ns_per_iteration", report["benchmarks"][0]["metric_error"])

    def test_renders_configuration_and_metrics(self):
        report = {
            "summary": {
                "expected_count": 1,
                "total": 1,
                "passed": 1,
                "failed": 0,
                "errors": 0,
                "complete": True,
            },
            "metadata": {
                "config": "orin-cuda12.6.3-ubuntu22.04",
                "git_sha": "abc123",
                "source_modified": True,
                "jetpack": "6.1",
            },
            "benchmarks": [
                {
                    "name": "Cuda.SpeedUpPass",
                    "status": "pass",
                    "elapsed_seconds": 1.25,
                    "iterations": 100,
                    "cpu_ns_per_iteration": 5_000_000,
                    "gpu_ns_per_iteration": 1_000_000,
                    "speedup": 5.0,
                    "failure_message": "",
                    "metric_error": "",
                }
            ],
        }

        markdown = render_markdown(report)

        self.assertIn("### orin-cuda12.6.3-ubuntu22.04", markdown)
        self.assertIn("| PASS | Cuda.SpeedUpPass | 5.000000 | 1.000000 | 5.000x |", markdown)
        self.assertIn("Commit: `abc123-modified`", markdown)


if __name__ == "__main__":
    unittest.main()
