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

"""Convert the 19 M3ED SPOT stereo sequences to portable cuVSLAM data.

The OVC stereo images live inside the published ``_data.h5``, which also carries
the event, LiDAR and IMU streams and so runs 25-42 GB per sequence. Those images
are read straight out of S3 over range requests, so a conversion transfers only
the stereo datasets and never stores the source file. ``--raw-dir`` switches to
already-downloaded files, which is what the tests and offline reruns use.
"""

import argparse
import contextlib
from pathlib import Path
from typing import List, Optional, Sequence

from cuvslam_tools.dataset_preparation.common import (
    PreparationError,
    add_common_arguments,
    require_nonempty_files,
    resolve_output_dir,
    run_preparation,
)

from . import convert_m3ed_spot
from .http_range import HttpRangeError, HttpRangeFile

DATASET_NAME = convert_m3ed_spot.DATASET_ID

BUCKET_URL = "https://m3ed-dist.s3.us-west-2.amazonaws.com"
OBJECT_PREFIX = "processed"

DATASET_ARTIFACTS = (
    "dataset_metadata.json",
    "m3ed_spot-vo.cfg",
    "m3ed_spot-slam.cfg",
    "m3ed_spot-vo_slam.cfg",
)
SEQUENCE_ARTIFACTS = (
    "stereo.edex",
    "frame_metadata.jsonl",
    "gt.txt",
    "00/000000.png",
    "01/000000.png",
)


def object_url(published: str, kind: str) -> str:
    """Return the public URL of one source file."""
    return f"{BUCKET_URL}/{OBJECT_PREFIX}/{published}/{published}_{kind}.h5"


def local_path(raw_dir: Path, published: str, kind: str) -> Path:
    """Return the local path of one source file under a raw directory."""
    return Path(raw_dir) / published / f"{published}_{kind}.h5"


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by environment, not tests
        raise PreparationError(
            "h5py is required to convert M3ED; install the cuvslam-tools package"
        ) from exc
    return h5py


@contextlib.contextmanager
def _open_remote(published: str, kind: str):
    h5py = _require_h5py()
    url = object_url(published, kind)
    try:
        reader = HttpRangeFile(url)
    except HttpRangeError as exc:
        raise PreparationError(str(exc)) from exc
    provenance = {"url": url, "size": reader.size, "etag": reader.etag}
    try:
        with h5py.File(reader, "r") as handle:
            yield handle, provenance
    except HttpRangeError as exc:
        raise PreparationError(f"{url}: {exc}") from exc
    finally:
        provenance["bytes_read"] = reader.bytes_read
        provenance["range_requests"] = reader.request_count
        reader.close()


def _open_local_factory(raw_dir: Path):
    @contextlib.contextmanager
    def open_local(published: str, kind: str):
        h5py = _require_h5py()
        path = local_path(raw_dir, published, kind)
        if not path.is_file():
            raise PreparationError(f"missing source file: {path}")
        provenance = {"path": str(path), "size": path.stat().st_size}
        # Hashing a 40 GB file to record provenance would cost more than the
        # conversion; the small ground-truth file is cheap and worth pinning.
        if kind == "pose_gt":
            provenance["sha256"] = convert_m3ed_spot.sha256(path)
        with h5py.File(path, "r") as handle:
            yield handle, provenance

    return open_local


def prepare(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sequences: Optional[Sequence[str]] = None,
    force_download: bool = False,
    download_only: bool = False,
    frame_limit: Optional[int] = None,
    skip_existing: bool = False,
) -> Path:
    """Convert the selected sequences and return the prepared root.

    Reads from S3 unless ``raw_dir`` names a directory of downloaded files.
    ``sequences`` defaults to all 19 published SPOT sequences.
    """
    output_dir = resolve_output_dir(output_dir)
    selected = list(sequences) if sequences is not None else None

    if download_only:
        raise PreparationError(
            "--download-only is not supported: the source images are read from the bucket "
            "during conversion rather than downloaded first"
        )
    if force_download:
        # Accepted so the provisioning workflow can pass it uniformly. Nothing is
        # cached between runs, so every conversion already re-reads the source.
        print("note: --force-download has no effect; the source is never cached locally")

    if raw_dir is not None:
        opener = _open_local_factory(Path(raw_dir))
        print(f"Source     : {raw_dir}")
    else:
        opener = _open_remote
        print(f"Source     : {BUCKET_URL}/{OBJECT_PREFIX}")
    print(f"Output dir : {output_dir}")
    print()

    dataset_dir = output_dir / DATASET_NAME
    try:
        convert_m3ed_spot.convert(
            dataset_dir,
            selected,
            open_sequence=opener,
            frame_limit=frame_limit,
            skip_existing=skip_existing,
        )
    except convert_m3ed_spot.ConversionError as exc:
        raise PreparationError(str(exc)) from exc

    require_nonempty_files(dataset_dir, DATASET_ARTIFACTS, "converter")
    for sequence in selected or convert_m3ed_spot.ALL_SEQS:
        require_nonempty_files(dataset_dir / sequence, SEQUENCE_ARTIFACTS, "converter")

    print()
    print(f"done — portable M3ED SPOT dataset ready at {dataset_dir}")
    return dataset_dir


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command-line arguments and prepare the M3ED SPOT dataset."""
    parser = argparse.ArgumentParser(
        prog="prepare_m3ed_spot",
        description=(
            "Convert the 19 M3ED SPOT stereo sequences to portable cuVSLAM EDEX data, "
            "reading the source HDF5 directly from the public bucket."
        ),
    )
    add_common_arguments(
        parser,
        DATASET_NAME,
        label="M3ED SPOT",
        force_download_help=argparse.SUPPRESS,
        download_only_help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=convert_m3ed_spot.ALL_SEQS,
        metavar="SEQUENCE",
        help="Convert an explicit sequence subset, such as skatepark_2. The default is all 19.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Leave sequences that already converted completely alone. Converting all 19 takes "
            "hours of network reads, so an interrupted run can be resumed."
        ),
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Convert at most this many frames per sequence. For local validation only.",
    )
    args = parser.parse_args(argv)

    return run_preparation(
        lambda: prepare(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            sequences=args.sequences,
            force_download=args.force_download,
            download_only=args.download_only,
            frame_limit=args.frame_limit,
            skip_existing=args.skip_existing,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
