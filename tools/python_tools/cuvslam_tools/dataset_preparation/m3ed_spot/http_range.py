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

"""A seekable read-only file over HTTP range requests, for reading remote HDF5.

M3ED publishes the OVC stereo images only inside the per-sequence ``_data.h5``,
which also carries the event, LiDAR and IMU streams and so runs 25-42 GB. The
compressed image chunks are about 6% of that by span, sitting 3-4 MB apart, and
HDF5 reaches them through ordinary seeks. Handing h5py one of these objects
therefore transfers only the stereo data: roughly 3 GB per sequence instead of a
25-42 GB download, with no staging file.

Two properties matter for that to be fast:

- Reads land where they are asked. A chunk arrives in one read of roughly 0.85
  MB, which is fetched as an exact range rather than through the block cache;
  block-aligned fetches would pull the neighbouring event data on either side.
- The connection is reused. Each range request is small, so a fresh TLS
  handshake per request dominates the transfer; measured against this bucket, a
  new connection per read cost about 1.7 s against roughly 0.1 s of payload.

The small block cache remains for HDF5's B-tree traversal, which issues many
reads of a few hundred bytes.
"""

import http.client
import io
import time
import urllib.parse
from typing import Dict, Optional, Tuple

# Kept near the compressed chunk size so metadata traversal is cheap without
# pulling event data around each image.
DEFAULT_BLOCK_SIZE = 1 << 20
DEFAULT_CACHE_BLOCKS = 32

# Reads at least this large are fetched as an exact range and not cached: the
# converter reads each frame once.
DIRECT_READ_THRESHOLD = 256 << 10

DEFAULT_RETRIES = 5
DEFAULT_TIMEOUT_SECONDS = 300
MAX_REDIRECTS = 5


class HttpRangeError(RuntimeError):
    """Raised when a remote read cannot be completed."""


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable view of an HTTP resource that supports range requests.

    Not thread safe: one instance holds one connection and one file position.
    """

    def __init__(
        self,
        url: str,
        block_size: int = DEFAULT_BLOCK_SIZE,
        cache_blocks: int = DEFAULT_CACHE_BLOCKS,
        retries: int = DEFAULT_RETRIES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        connection_factory=None,
        direct_read_threshold: int = DIRECT_READ_THRESHOLD,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if cache_blocks <= 0:
            raise ValueError("cache_blocks must be positive")
        if retries <= 0:
            raise ValueError("retries must be positive")
        if direct_read_threshold <= 0:
            raise ValueError("direct_read_threshold must be positive")
        self.url = url
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.direct_read_threshold = direct_read_threshold
        self.retries = retries
        self.timeout = timeout
        self._connection_factory = connection_factory or self._default_connection
        self._connection = None
        self._position = 0
        # Insertion-ordered, so the oldest block is the first key.
        self._cache: Dict[int, bytes] = {}
        self.request_count = 0
        self.bytes_read = 0
        self._host, self._path = self._split_url(url)
        self.size, self.etag = self._head()

    @staticmethod
    def _split_url(url: str) -> Tuple[Tuple[str, str, Optional[int]], str]:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise HttpRangeError(f"{url}: only http and https are supported")
        if not parsed.hostname:
            raise HttpRangeError(f"{url}: no host")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return (parsed.scheme, parsed.hostname, parsed.port), path

    def _default_connection(self, scheme: str, host: str, port: Optional[int]):
        if scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=self.timeout)
        return http.client.HTTPConnection(host, port, timeout=self.timeout)

    def _connect(self):
        if self._connection is None:
            scheme, host, port = self._host
            self._connection = self._connection_factory(scheme, host, port)
        return self._connection

    def _drop_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
            self._connection = None

    def _perform(self, method: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, str], bytes]:
        """Issue one request on the shared connection, retrying on transport errors."""
        path = self._path
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                connection = self._connect()
                connection.request(method, path, headers=headers)
                response = connection.getresponse()
                # The body must be drained even for HEAD, or the connection
                # cannot serve the next request.
                payload = response.read()
                if response.will_close:
                    self._drop_connection()
                return response.status, dict(response.getheaders()), payload
            except (http.client.HTTPException, OSError) as exc:
                last_error = exc
                self._drop_connection()
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 30))
        raise HttpRangeError(f"{self.url}: {self.retries} attempts failed: {last_error}")

    def _request(self, method: str, headers: Dict[str, str]) -> Tuple[Dict[str, str], bytes]:
        for _ in range(MAX_REDIRECTS):
            status, response_headers, payload = self._perform(method, headers)
            if status in (301, 302, 303, 307, 308):
                location = response_headers.get("Location")
                if not location:
                    raise HttpRangeError(f"{self.url}: redirect without a Location header")
                self._drop_connection()
                self._host, self._path = self._split_url(
                    urllib.parse.urljoin(self.url, location)
                )
                continue
            if status not in (200, 206):
                raise HttpRangeError(f"{self.url}: HTTP {status}")
            return response_headers, payload
        raise HttpRangeError(f"{self.url}: too many redirects")

    def _head(self) -> Tuple[int, Optional[str]]:
        headers, _ = self._request("HEAD", {})
        length = headers.get("Content-Length")
        if length is None:
            raise HttpRangeError(f"{self.url}: server did not report Content-Length")
        etag = headers.get("ETag")
        return int(length), etag.strip('"') if etag else None

    def _fetch(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = start + length - 1
        _, payload = self._request("GET", {"Range": f"bytes={start}-{end}"})
        if len(payload) != length:
            raise HttpRangeError(
                f"{self.url}: range {start}-{end} returned {len(payload)} bytes, expected {length}"
            )
        self.request_count += 1
        self.bytes_read += len(payload)
        return payload

    def _block(self, index: int) -> bytes:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        start = index * self.block_size
        if start >= self.size:
            return b""
        payload = self._fetch(start, min(self.block_size, self.size - start))
        self._cache[index] = payload
        while len(self._cache) > self.cache_blocks:
            self._cache.pop(next(iter(self._cache)))
        return payload

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise OSError("negative seek position")
        self._position = position
        return self._position

    def readinto(self, buffer) -> int:
        wanted = min(len(buffer), max(self.size - self._position, 0))
        if wanted >= self.direct_read_threshold:
            payload = self._fetch(self._position, wanted)
            buffer[: len(payload)] = payload
            self._position += len(payload)
            return len(payload)
        written = 0
        while written < wanted:
            position = self._position + written
            index = position // self.block_size
            block = self._block(index)
            if not block:
                break
            offset = position - index * self.block_size
            chunk = block[offset : offset + (wanted - written)]
            buffer[written : written + len(chunk)] = chunk
            written += len(chunk)
        self._position += written
        return written

    def close(self) -> None:
        self._drop_connection()
        self._cache.clear()
        super().close()
