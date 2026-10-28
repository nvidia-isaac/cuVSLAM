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

"""Console entry point for fitting a cuVSLAM place recognition vocabulary."""

import argparse
import bisect
import itertools
import os
from typing import Optional


def select_frames(frame_counts: list[int], capacity: int) -> list[tuple[int, int]]:
    """Spread up to `capacity` picks evenly over the frames of all sequences, as (sequence, frame position)."""
    total = sum(frame_counts)
    count = min(total, capacity)
    ends = list(itertools.accumulate(frame_counts))
    picks = []
    for pick in range(count):
        # Midpoint of the pick's bin, so count == total picks every frame.
        position = (2 * pick + 1) * total // (2 * count)
        sequence = bisect.bisect_right(ends, position)
        picks.append((sequence, position - (ends[sequence] - frame_counts[sequence])))
    return picks


def fit_vocabulary(fitter, readers: list, camera: int, output: str) -> list[int]:
    """Add evenly spread images of `camera` to `fitter`, save the vocabulary to `output`, and count them per reader."""
    for reader in readers:
        if camera >= len(reader.rig.cameras):
            raise ValueError(f"--camera {camera} is not a camera of {reader.edex_dir}, whose rig has "
                             f"{len(reader.rig.cameras)}")
    frame_ids = [sorted(reader.frames) for reader in readers]
    counts = [0] * len(readers)
    for sequence, position in select_frames([len(ids) for ids in frame_ids], fitter.capacity):
        fitter.add_image(readers[sequence].load_camera_image(frame_ids[sequence][position], camera))
        counts[sequence] += 1
    fitter.save(output)
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    """Parse CLI arguments, fit the vocabulary, and save it."""
    parser = argparse.ArgumentParser(
        prog="cuvslam_vpr_vocabulary",
        description="Fit the place recognition vocabulary of an operating domain to images spread evenly over "
                    "EDEX sequences of it.",
        allow_abbrev=False,
    )
    parser.add_argument("sequences", nargs="+", help="EDEX sequence folders of the operating domain.")
    parser.add_argument("--output", type=str, required=True, help="Vocabulary file to write.")
    parser.add_argument("--vpr_model_path", type=str, default=os.environ.get("CUVSLAM_ANYLOC_ENGINE", ""),
                        help="AnyLoc TensorRT engine, built by the anyloc_engine CMake target "
                             "(default: $CUVSLAM_ANYLOC_ENGINE).")
    parser.add_argument("--edex_filename", type=str, default="stereo.edex",
                        help="EDEX file of each sequence (default: stereo.edex).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera whose images are fitted: the first primary camera, which SLAM's place "
                             "recognition uses (default: 0).")

    args = parser.parse_args(argv)
    if not os.path.isfile(args.vpr_model_path):
        parser.error("--vpr_model_path must name the AnyLoc TensorRT engine the anyloc_engine CMake target builds")
    for sequence in args.sequences:
        if not os.path.isfile(os.path.join(sequence, args.edex_filename)):
            parser.error(f"{sequence} has no {args.edex_filename}")
    if args.camera < 0:
        parser.error("--camera must be a camera index, 0 or more")
    if os.path.isdir(args.output):
        parser.error(f"--output {args.output} is a folder; name the vocabulary file to write")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(output_dir):
        parser.error(f"--output folder does not exist: {output_dir}")

    try:
        import cuvslam as vslam
    except ImportError as exc:
        parser.error(f"cuvslam_vpr_vocabulary requires the cuVSLAM Python binding to be installed: {exc}")
    from cuvslam_tools.tracker.edex_reader import EdexReader

    try:
        readers = [EdexReader(sequence, stereo_edex=os.path.join(sequence, args.edex_filename))
                   for sequence in args.sequences]
        fitter = vslam.Slam.VprVocabularyFitter(
            vslam.Slam.VprVocabularyFitter.Config(vpr_model_path=args.vpr_model_path))
        counts = fit_vocabulary(fitter, readers, args.camera, args.output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    for sequence, count in zip(args.sequences, counts):
        print(f"{sequence}: {count} images")
    print(f"Saved the vocabulary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
