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

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


def _load_package_tool():
    module_path = Path(__file__).parents[1] / "packaging" / "build_packages.py"
    spec = importlib.util.spec_from_file_location("cuvslam_build_packages", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load package tool from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_tool = _load_package_tool()
SOURCE_ROOT = Path(__file__).parents[2]
PACKAGE_VERSION = (SOURCE_ROOT / "VERSION").read_text(encoding="utf-8").strip()


def _write_wheel(path: Path, distribution: str, files):
    dist_info = distribution.replace("-", "_") + f"-{PACKAGE_VERSION}.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.4\nName: {distribution}\nVersion: {PACKAGE_VERSION}\n",
        )
        for filename, contents in files.items():
            wheel.writestr(filename, contents)


class TestPackageMetadata(unittest.TestCase):
    def test_backend_metadata_uses_public_version_and_required_cuda_dependency(self):
        pyproject = package_tool.render_backend_pyproject(SOURCE_ROOT, "12")
        self.assertIn('name = "cuvslam-cu12"', pyproject)
        self.assertIn(f'version = "{PACKAGE_VERSION}"', pyproject)
        self.assertNotIn("+cu12", pyproject)
        self.assertIn(f"cuvslam-common=={PACKAGE_VERSION}", pyproject)
        self.assertIn("cuda-toolkit[cublas,cudart,cusolver,cusparse]==12.*", pyproject)
        self.assertNotIn("[project.optional-dependencies]", pyproject)
        self.assertIn('license = "LicenseRef-NVIDIA-Community-License"', pyproject)
        self.assertIn('license-files = ["LICENSE", "THIRD-PARTY.txt"]', pyproject)
        self.assertIn("[project.urls]", pyproject)
        self.assertIn("scikit-build-core>=1.0.3,<2", pyproject)
        self.assertIn("nanobind>=2.10.2,<3", pyproject)

    def test_cuda_backends_have_distinct_install_rpaths(self):
        cu12 = package_tool.backend_rpath(SOURCE_ROOT, "12")
        cu13 = package_tool.backend_rpath(SOURCE_ROOT, "13")
        self.assertIn("nvidia/cublas/lib", cu12)
        self.assertIn("nvidia/cu13/lib", cu13)
        self.assertNotEqual(cu12, cu13)

    def test_common_and_default_projects_are_staged(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backend = root / f"cuvslam_cu12-{PACKAGE_VERSION}-py3-none-any.whl"
            _write_wheel(
                backend,
                "cuvslam-cu12",
                {
                    "cuvslam/cu12/pycuvslam.pyi": "class Odometry: ...\n",
                    "cuvslam/cu12/pycuvslam.so": "",
                },
            )
            common = root / "common"
            meta = root / "meta"
            package_tool.stage_common(SOURCE_ROOT, backend, common)
            package_tool.stage_meta(SOURCE_ROOT, meta)

            self.assertTrue((common / "src" / "cuvslam" / "__init__.py").is_file())
            self.assertTrue((common / "src" / "cuvslam" / "__init__.pyi").is_file())
            self.assertIn(
                'name = "cuvslam-common"',
                (common / "pyproject.toml").read_text(encoding="utf-8"),
            )
            meta_project = (meta / "pyproject.toml").read_text(encoding="utf-8")
            self.assertIn('name = "cuvslam"', meta_project)
            self.assertIn(f"cuvslam-cu13=={PACKAGE_VERSION}", meta_project)
            self.assertFalse((meta / "src").exists())


class TestWheelOwnership(unittest.TestCase):
    def test_disjoint_distributions_pass(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            common = root / "common.whl"
            backend = root / "backend.whl"
            _write_wheel(common, "cuvslam-common", {"cuvslam/__init__.py": ""})
            _write_wheel(
                backend,
                "cuvslam-cu12",
                {"cuvslam/cu12/pycuvslam.so": ""},
            )
            package_tool.verify_ownership([common, backend])

    def test_payload_outside_distribution_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            backend = root / "backend.whl"
            _write_wheel(
                backend,
                "cuvslam-cu12",
                {"cuvslam/__init__.py": "backend must not own this"},
            )
            with self.assertRaisesRegex(ValueError, "distribution ownership"):
                package_tool.verify_ownership([backend])

    def test_full_distribution_set_has_disjoint_payloads(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wheels = {
                "cuvslam": root / "meta.whl",
                "cuvslam-common": root / "common.whl",
                "cuvslam-cu12": root / "cu12.whl",
                "cuvslam-cu13": root / "cu13.whl",
            }
            _write_wheel(wheels["cuvslam"], "cuvslam", {})
            _write_wheel(
                wheels["cuvslam-common"],
                "cuvslam-common",
                {
                    "cuvslam/__init__.py": "",
                    "cuvslam/_backend.py": "",
                    "cuvslam/utils.py": "",
                },
            )
            _write_wheel(
                wheels["cuvslam-cu12"],
                "cuvslam-cu12",
                {"cuvslam/cu12/pycuvslam.so": ""},
            )
            _write_wheel(
                wheels["cuvslam-cu13"],
                "cuvslam-cu13",
                {"cuvslam/cu13/pycuvslam.so": ""},
            )

            manifest = root / "manifest.json"
            package_tool.release_manifest(
                list(wheels.values()),
                manifest,
                100 * 1024 * 1024,
                {},
                PACKAGE_VERSION,
                {name: 1 for name in wheels},
            )
            entries = json.loads(manifest.read_text(encoding="utf-8"))["artifacts"]
            self.assertEqual(
                {entry["distribution"] for entry in entries},
                set(wheels),
            )

    def test_release_manifest_hashes_artifacts_and_enforces_size_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wheel = root / "backend.whl"
            manifest = root / "manifest.json"
            _write_wheel(
                wheel,
                "cuvslam-cu12",
                {"cuvslam/cu12/pycuvslam.so": "binary"},
            )

            with self.assertRaisesRegex(ValueError, "file-size gate failed"):
                package_tool.release_manifest([wheel], manifest, 1, {})

            package_tool.release_manifest(
                [wheel],
                manifest,
                1,
                {"cuvslam-cu12": wheel.stat().st_size},
                PACKAGE_VERSION,
                {"cuvslam-cu12": 1},
            )
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["artifacts"][0]["distribution"], "cuvslam-cu12")
            self.assertEqual(len(data["artifacts"][0]["sha256"]), 64)

            with self.assertRaisesRegex(ValueError, "distribution counts"):
                package_tool.release_manifest(
                    [wheel],
                    manifest,
                    wheel.stat().st_size,
                    {},
                    PACKAGE_VERSION,
                    {"cuvslam-cu12": 2},
                )


if __name__ == "__main__":
    unittest.main()
