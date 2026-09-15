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
import unittest
from unittest import mock

from cuvslam import _backend


class TestBackendSelection(unittest.TestCase):
    def setUp(self):
        self._original_backend = os.environ.pop("CUVSLAM_CUDA_MAJOR", None)
        _backend.load_backend.cache_clear()

    def tearDown(self):
        _backend.load_backend.cache_clear()
        if self._original_backend is not None:
            os.environ["CUVSLAM_CUDA_MAJOR"] = self._original_backend

    def _load(self, available):
        sentinel = object()
        with mock.patch.object(_backend, "_available_backends", return_value=available):
            with mock.patch.object(_backend, "import_module", return_value=sentinel) as importer:
                loaded = _backend.load_backend()
        return loaded, importer

    def test_local_backend_is_preferred_for_source_install(self):
        loaded, importer = self._load(
            {
                "local": "cuvslam.pycuvslam",
                "12": "cuvslam.cu12.pycuvslam",
            }
        )
        self.assertIsNotNone(loaded)
        importer.assert_called_once_with("cuvslam.pycuvslam")

    def test_single_release_backend_is_selected(self):
        _, importer = self._load({"12": "cuvslam.cu12.pycuvslam"})
        importer.assert_called_once_with("cuvslam.cu12.pycuvslam")

    def test_multiple_backends_require_explicit_selection(self):
        with mock.patch.object(
            _backend,
            "_available_backends",
            return_value={
                "12": "cuvslam.cu12.pycuvslam",
                "13": "cuvslam.cu13.pycuvslam",
            },
        ):
            with self.assertRaisesRegex(ImportError, "CUVSLAM_CUDA_MAJOR"):
                _backend.load_backend()

    def test_environment_selects_installed_backend(self):
        with mock.patch.dict(os.environ, {"CUVSLAM_CUDA_MAJOR": "cu13"}):
            _, importer = self._load(
                {
                    "12": "cuvslam.cu12.pycuvslam",
                    "13": "cuvslam.cu13.pycuvslam",
                }
            )
        importer.assert_called_once_with("cuvslam.cu13.pycuvslam")

    def test_missing_requested_backend_has_install_hint(self):
        with mock.patch.dict(os.environ, {"CUVSLAM_CUDA_MAJOR": "12"}):
            with mock.patch.object(_backend, "_available_backends", return_value={}):
                with self.assertRaisesRegex(ImportError, "pip install cuvslam-cu12"):
                    _backend.load_backend()

    def test_invalid_environment_value_is_rejected(self):
        with mock.patch.dict(os.environ, {"CUVSLAM_CUDA_MAJOR": "11"}):
            with self.assertRaisesRegex(ImportError, "must be 12 or 13"):
                _backend.load_backend()


if __name__ == "__main__":
    unittest.main()
