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

import os
import tempfile
import unittest

from cuvslam import _cuda_libs


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
        # imports: this is the regression guard for wheels that neither bundle nor declare them.
        import cuvslam  # noqa: F401  (imported for its side effect on the process' loaded libraries)
        with open('/proc/self/maps') as maps_file:
            maps = maps_file.read()
        for soname in ('libcublas.so.', 'libcusolver.so.', 'libcusparse.so.'):
            self.assertIn(soname, maps)


if __name__ == "__main__":
    unittest.main()
