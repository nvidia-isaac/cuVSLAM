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
import sys
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

# A chunk read is under a megabyte and completes in about a second, so this is
# ample headroom while still bounding a socket that has silently died. The
# previous 300 s meant one dead connection could stall a conversion for five
# minutes with no output.
DEFAULT_TIMEOUT_SECONDS = 60

MAX_REDIRECTS = 5

_REDIRECT_STATUSES = (301, 302, 303, 307, 308)


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
        # ``url`` is what the caller asked for and is what errors and
        # provenance report; redirects retarget ``_effective_url`` instead.
        self.url = url
        self._effective_url = url
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
        self.retry_count = 0
        self._host, self._path = self._split_url(url)
        self.size, self._if_range = self._head()
        # Provenance records the entity tag without the quoting the header needs.
        self.etag = self._if_range.strip('"')

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

    def _check_range_response(
        self, status: int, headers: Dict[str, str], span: Tuple[int, int]
    ) -> None:
        """Reject a ranged GET that was not answered with the range asked for.

        An origin that ignores ``Range`` answers 200 with the whole object,
        which here is 25-42 GB. This runs before the body is read, so such a
        reply costs one request instead of that transfer.
        """
        start, end = span
        if status != 206:
            self._drop_connection()
            # Every range read carries If-Range, so a full response means the
            # object was republished mid-conversion or Range was ignored.
            detail = " (the object changed, or Range was ignored)" if status == 200 else ""
            raise HttpRangeError(
                f"{self.url}: range {start}-{end} answered with HTTP {status}, "
                f"expected 206{detail}"
            )
        content_range = headers.get("content-range")
        if content_range is None or not content_range.startswith(f"bytes {start}-{end}/"):
            self._drop_connection()
            raise HttpRangeError(
                f"{self.url}: range {start}-{end} answered with Content-Range {content_range!r}"
            )

    def _perform(
        self, method: str, headers: Dict[str, str], span: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Issue one request on the shared connection, retrying on transport errors.

        ``span`` names the byte range a GET asked for. The response then has to
        be a 206 covering exactly that range, and at most one byte past it is
        read, so a server that answers with more than was asked for cannot pull
        the whole object into memory.
        """
        path = self._path
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                connection = self._connect()
                connection.request(method, path, headers=headers)
                response = connection.getresponse()
                status = response.status
                # Field names are case-insensitive and getheaders() reports them
                # as the server cased them.
                response_headers = {name.lower(): value for name, value in response.getheaders()}
                if span is not None and status not in _REDIRECT_STATUSES:
                    self._check_range_response(status, response_headers, span)
                # The body must be drained even for HEAD, or the connection
                # cannot serve the next request.
                if span is None:
                    payload = response.read()
                else:
                    length = span[1] - span[0] + 1
                    payload = response.read(length + 1)
                    if len(payload) > length:
                        # The rest of the body is still in flight, so the
                        # connection cannot serve another request.
                        self._drop_connection()
                if response.will_close:
                    self._drop_connection()
                return status, response_headers, payload
            except (http.client.HTTPException, OSError) as exc:
                last_error = exc
                self._drop_connection()
                self.retry_count += 1
                # Retries were previously silent, which made a slow transfer and
                # a retry storm look identical from the outside.
                print(
                    f"warning: {method} {headers.get('Range', '')} failed "
                    f"(attempt {attempt + 1}/{self.retries}): {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 30))
        raise HttpRangeError(f"{self.url}: {self.retries} attempts failed: {last_error}")

    def _request(
        self, method: str, headers: Dict[str, str], span: Optional[Tuple[int, int]] = None
    ) -> Tuple[Dict[str, str], bytes]:
        # One request per redirect, plus the one that answers.
        for _ in range(MAX_REDIRECTS + 1):
            status, response_headers, payload = self._perform(method, headers, span)
            if status in _REDIRECT_STATUSES:
                location = response_headers.get("location")
                if not location:
                    raise HttpRangeError(f"{self.url}: redirect without a Location header")
                self._drop_connection()
                # A relative Location resolves against the URL that served it,
                # which after the first hop is no longer the URL passed in.
                self._effective_url = urllib.parse.urljoin(self._effective_url, location)
                self._host, self._path = self._split_url(self._effective_url)
                continue
            if status not in (200, 206):
                raise HttpRangeError(f"{self.url}: HTTP {status}")
            return response_headers, payload
        raise HttpRangeError(f"{self.url}: too many redirects")

    def _head(self) -> Tuple[int, str]:
        """Return the object's length and the strong entity tag that pins it."""
        headers, _ = self._request("HEAD", {})
        length = headers.get("content-length")
        if length is None:
            raise HttpRangeError(f"{self.url}: server did not report Content-Length")
        etag = (headers.get("etag") or "").strip()
        # A conversion reads one object over hours, and M3ED does republish
        # files. Without a strong validator to send back on every range read, a
        # file replaced mid-conversion would be spliced into the output
        # unnoticed, so there is nothing safe to do but refuse. A weak validator
        # can compare equal across representations and so cannot pin one.
        if len(etag) < 3 or not etag.startswith('"') or not etag.endswith('"'):
            raise HttpRangeError(
                f"{self.url}: no strong ETag (got {etag or 'none'}), so the object cannot be "
                "pinned for the length of a conversion"
            )
        return int(length), etag

    def _fetch(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        end = start + length - 1
        _, payload = self._request(
            "GET",
            {"Range": f"bytes={start}-{end}", "If-Range": self._if_range},
            (start, end),
        )
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
