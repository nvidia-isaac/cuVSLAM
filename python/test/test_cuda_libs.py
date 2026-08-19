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
import os
import subprocess
import sys
import tempfile
import unittest


def load_cuda_libs():
    """Load ``cuvslam/_cuda_libs.py`` as a standalone module, without importing ``cuvslam``.

    Importing the package runs ``preload()`` and loads the extension module, which is exactly the side effect
    test_import_resolved_the_cuda_math_libraries has to observe in an interpreter that has not done it yet.
    """
    package_root = importlib.util.find_spec('cuvslam').submodule_search_locations[0]
    spec = importlib.util.spec_from_file_location('_cuda_libs_under_test',
                                                  os.path.join(package_root, '_cuda_libs.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cuda_libs = load_cuda_libs()

# Run in a fresh interpreter by test_import_resolved_the_cuda_math_libraries.
IMPORT_SIDE_EFFECT_PROBE = """
import cuvslam  # noqa: F401  (imported for its side effect on the process' loaded libraries)

with open('/proc/self/maps') as maps_file:
    maps = maps_file.read()

missing = [soname for soname in ('libcublas.so.', 'libcusolver.so.', 'libcusparse.so.') if soname not in maps]
if missing:
    raise SystemExit('not loaded after "import cuvslam": ' + ', '.join(missing))
"""


def make_pip_cuda_layout(root, components):
    """Create a fake <site-packages>/nvidia layout next to a fake cuvslam package directory."""
    package_root = os.path.join(root, 'cuvslam')
    os.makedirs(package_root)
    for component, libraries in components.items():
        lib_dir = os.path.join(root, 'nvidia', component, 'lib')
        os.makedirs(lib_dir)
        for library in libraries:
            # Not valid ELF: enough to be discovered, never loadable.
            open(os.path.join(lib_dir, library), 'wb').close()
    return package_root


class TestCudaLibs(unittest.TestCase):
    def test_no_pip_cuda_packages_installed(self):
        # The libraries then come from the system CUDA Toolkit and there is nothing to preload.
        with tempfile.TemporaryDirectory() as root:
            package_root = os.path.join(root, 'cuvslam')
            os.makedirs(package_root)
            self.assertEqual(_cuda_libs.candidate_libraries(package_root), [])
            self.assertEqual(_cuda_libs.preload(package_root), [])

    def test_candidates_are_discovered_in_dependency_order(self):
        with tempfile.TemporaryDirectory() as root:
            package_root = make_pip_cuda_layout(root, {
                'cusolver': ['libcusolver.so.11'],
                'cublas': ['libcublas.so.12', 'libcublasLt.so.12'],
                'cusparse': ['libcusparse.so.12'],
                'nvjitlink': ['libnvJitLink.so.12']})
            found = [os.path.basename(path) for path in _cuda_libs.candidate_libraries(package_root)]
            self.assertEqual(found, [
                'libnvJitLink.so.12', 'libcublas.so.12', 'libcublasLt.so.12', 'libcusparse.so.12',
                'libcusolver.so.11'])

    def test_unrelated_components_are_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            package_root = make_pip_cuda_layout(root, {
                'cudnn': ['libcudnn.so.9'],
                'cusparse': ['libcusparse.so.12']})
            found = [os.path.basename(path) for path in _cuda_libs.candidate_libraries(package_root)]
            self.assertEqual(found, ['libcusparse.so.12'])

    def test_unloadable_libraries_do_not_raise(self):
        # A library that cannot be loaded is left to the extension import to report, and the retry loop that
        # tolerates an unknown load order must still terminate.
        with tempfile.TemporaryDirectory() as root:
            package_root = make_pip_cuda_layout(root, {
                'cublas': ['libcublas.so.12'],
                'cusolver': ['libcusolver.so.11']})
            self.assertEqual(_cuda_libs.preload(package_root), [])

    def test_import_resolved_the_cuda_math_libraries(self):
        # Whatever provided them, the CUDA math libraries libcuvslam.so links against are loaded once cuvslam
        # imports: this is the regression guard for wheels that neither bundle nor declare them. It runs in a
        # fresh interpreter, because a cuvslam already imported by another test would load them regardless of
        # whether importing it still does.
        probe = subprocess.run([sys.executable, '-c', IMPORT_SIDE_EFFECT_PROBE],
                               capture_output=True, text=True, check=False)
        self.assertEqual(probe.returncode, 0, probe.stderr)


if __name__ == "__main__":
    unittest.main()
