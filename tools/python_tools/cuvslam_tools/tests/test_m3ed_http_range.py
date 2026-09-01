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

"""The HTTP range reader that lets h5py read M3ED sources without downloading them.

A fake connection stands in for the network, so these exercise the range
arithmetic, the block cache, the direct-read path, redirects and retries without
touching S3.
"""

import http.client
import io
import unittest

from cuvslam_tools.dataset_preparation.m3ed_spot import http_range
from cuvslam_tools.dataset_preparation.m3ed_spot.http_range import HttpRangeError, HttpRangeFile

PAYLOAD = bytes(index % 251 for index in range(50_000))


class FakeResponse:
    def __init__(self, status, headers, body, will_close=False):
        self.status = status
        self._headers = headers
        self._body = body
        self.will_close = will_close

    def getheaders(self):
        return list(self._headers.items())

    def read(self):
        return self._body


class FakeConnection:
    """Serves ranges out of a bytes object and records what was asked for."""

    def __init__(self, payload=PAYLOAD, etag="abc", redirect_to=None, fail_times=0):
        self.payload = payload
        self.etag = etag
        self.redirect_to = redirect_to
        self.fail_times = fail_times
        self.requests = []
        self.paths = []
        self.closed = 0
        self._pending = None

    def request(self, method, path, headers=None):
        headers = headers or {}
        self.requests.append((method, headers.get("Range")))
        self.paths.append(path)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise http.client.HTTPException("boom")
        if self.redirect_to is not None:
            location, self.redirect_to = self.redirect_to, None
            self._pending = FakeResponse(302, {"Location": location}, b"")
            return
        if method == "HEAD":
            self._pending = FakeResponse(
                200,
                {"Content-Length": str(len(self.payload)), "ETag": f'"{self.etag}"'},
                b"",
            )
            return
        span = headers["Range"].removeprefix("bytes=")
        start, end = (int(value) for value in span.split("-"))
        self._pending = FakeResponse(206, {}, self.payload[start : end + 1])

    def getresponse(self):
        if self._pending is None:
            raise http.client.HTTPException("no request issued")
        response, self._pending = self._pending, None
        return response

    def close(self):
        self.closed += 1


def _open(connection, **kwargs):
    return HttpRangeFile(
        "https://example.invalid/bucket/object.h5",
        connection_factory=lambda scheme, host, port: connection,
        **kwargs,
    )


class TestHttpRangeFile(unittest.TestCase):
    def test_reports_size_and_etag_from_head(self):
        connection = FakeConnection()
        handle = _open(connection)
        self.assertEqual(handle.size, len(PAYLOAD))
        self.assertEqual(handle.etag, "abc")
        self.assertEqual(connection.requests[0][0], "HEAD")

    def test_small_reads_come_from_the_block_cache(self):
        connection = FakeConnection()
        handle = _open(connection, block_size=4096)
        self.assertEqual(handle.read(16), PAYLOAD[:16])
        # Two more reads inside the same block must not issue a request.
        before = handle.request_count
        handle.seek(32)
        self.assertEqual(handle.read(16), PAYLOAD[32:48])
        handle.seek(100)
        self.assertEqual(handle.read(16), PAYLOAD[100:116])
        self.assertEqual(handle.request_count, before)

    def test_large_reads_fetch_an_exact_range(self):
        connection = FakeConnection()
        handle = _open(connection, block_size=4096, direct_read_threshold=8192)
        handle.seek(10_000)
        data = handle.read(9000)
        self.assertEqual(data, PAYLOAD[10_000:19_000])
        # Exactly the requested span, not a block-aligned superset.
        self.assertEqual(connection.requests[-1][1], "bytes=10000-18999")
        self.assertEqual(handle.bytes_read, 9000)
        self.assertEqual(handle.request_count, 1)

    def test_direct_reads_are_not_cached(self):
        connection = FakeConnection()
        handle = _open(connection, block_size=4096, direct_read_threshold=8192)
        for _ in range(2):
            handle.seek(0)
            self.assertEqual(handle.read(9000), PAYLOAD[:9000])
        self.assertEqual(handle.request_count, 2)

    def test_read_spanning_blocks_is_reassembled(self):
        connection = FakeConnection()
        handle = _open(connection, block_size=1024)
        handle.seek(1000)
        self.assertEqual(handle.read(2048), PAYLOAD[1000:3048])

    def test_cache_is_bounded(self):
        connection = FakeConnection()
        handle = _open(connection, block_size=1024, cache_blocks=2)
        for position in (0, 2048, 4096):
            handle.seek(position)
            handle.read(8)
        # Re-reading the evicted first block issues a new request.
        before = handle.request_count
        handle.seek(0)
        handle.read(8)
        self.assertEqual(handle.request_count, before + 1)

    def test_reads_are_clamped_at_end_of_file(self):
        connection = FakeConnection()
        handle = _open(connection, block_size=4096)
        handle.seek(len(PAYLOAD) - 10)
        self.assertEqual(handle.read(100), PAYLOAD[-10:])
        handle.seek(len(PAYLOAD))
        self.assertEqual(handle.read(10), b"")

    def test_seek_modes_and_negative_rejection(self):
        connection = FakeConnection()
        handle = _open(connection, block_size=4096)
        self.assertEqual(handle.seek(10), 10)
        self.assertEqual(handle.seek(5, io.SEEK_CUR), 15)
        self.assertEqual(handle.seek(-4, io.SEEK_END), len(PAYLOAD) - 4)
        with self.assertRaises(OSError):
            handle.seek(-1)
        with self.assertRaises(ValueError):
            handle.seek(0, 99)

    def test_transport_errors_are_retried_then_reported(self):
        # Two failures then success: the read completes.
        connection = FakeConnection(fail_times=2)
        handle = _open(connection, block_size=4096, retries=5)
        self.assertEqual(handle.read(8), PAYLOAD[:8])
        self.assertGreaterEqual(connection.closed, 2)

        # Failing more often than the retry budget surfaces one clear error.
        connection = FakeConnection(fail_times=10)
        with self.assertRaisesRegex(HttpRangeError, "attempts failed"):
            _open(connection, block_size=4096, retries=2)

    def test_redirects_are_followed(self):
        connection = FakeConnection(redirect_to="https://example.invalid/moved/object.h5")
        handle = _open(connection, block_size=4096)
        self.assertEqual(handle.size, len(PAYLOAD))
        self.assertEqual(connection.paths[-1], "/moved/object.h5")

    def test_short_range_response_is_rejected(self):
        class Truncating(FakeConnection):
            def getresponse(self):
                response = super().getresponse()
                if response.status == 206:
                    return FakeResponse(206, {}, response.read()[:-1])
                return response

        handle = _open(Truncating(), block_size=4096)
        with self.assertRaisesRegex(HttpRangeError, "expected"):
            handle.read(8)

    def test_unexpected_status_is_rejected(self):
        class Forbidden(FakeConnection):
            def request(self, method, path, headers=None):
                self.requests.append((method, (headers or {}).get("Range")))
                self._pending = FakeResponse(403, {}, b"")

        with self.assertRaisesRegex(HttpRangeError, "HTTP 403"):
            _open(Forbidden())

    def test_invalid_configuration_is_rejected(self):
        connection = FakeConnection()
        for kwargs in (
            {"block_size": 0},
            {"cache_blocks": 0},
            {"retries": 0},
            {"direct_read_threshold": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    _open(connection, **kwargs)

    def test_non_http_scheme_is_rejected(self):
        with self.assertRaisesRegex(HttpRangeError, "only http and https"):
            HttpRangeFile("s3://bucket/object.h5")


if __name__ == "__main__":
    unittest.main()
