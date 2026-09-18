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

"""Map build, query, and scoring passes behind the cuvslam_vpr_reporter CLI."""

import argparse
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

CUVSLAM_VPR_REPORTER_DEPENDENCY_ERROR = "cuvslam_vpr_reporter requires the cuVSLAM Python binding to be installed"

DEFAULT_EDEX_FILE = "stereo.edex"
DEFAULT_GT_FILE = "gt.txt"
DEFAULT_SUCCESS_RADIUS_M = 10.0
DEFAULT_VPR_MODES = ("Simple",)
KNOWN_VPR_MODES = ("Simple", "Bow", "DBoW2", "AnyLoc")

TRUE_POSITIVE = "true_positive"
FALSE_POSITIVE = "false_positive"
UNRECOGNIZED = "unrecognized"
SELF_MATCH = "self_match"

# The three live classes were picked by a data-viz palette validator against a white surface with
# every pair in play: OKLab Delta E under simulated protanopia and deuteranopia (worst pair 12.5),
# normal vision (worst pair 17.1), a shared lightness band and a chroma floor. Amber sits at 2.17:1
# against white, which the method allows only when the same information is readable elsewhere, so
# the legend and the metrics table both have to stay. Self matches cannot occur (see below) and are
# drawn in a color outside that triple so a regression stands out instead of blending in.
_CLASS_STYLE = {
    TRUE_POSITIVE: ("#0d6b0d", "Recognized (true positive)"),
    UNRECOGNIZED: ("#e66767", "Unrecognized"),
    FALSE_POSITIVE: ("#eda100", "False positive"),
    SELF_MATCH: ("#6a3d9a", "Self match in query session"),
}

_MAP_INK = "#898781"
_AXIS_INK = "#c3c2b7"
_GRID_INK = "#e1e0d9"
_TEXT_INK = "#55534d"

_RIBBON_WIDTH_PT = 3.0
# Below this many queried frames the ribbon is too short to read as a route, so the frames are also
# drawn as markers; above it they would smear back into the blob the ribbon replaced.
_MARKER_QUERY_LIMIT = 200
_MARKER_AREA_PT2 = 64.0

# Inches and dots per inch of the plot. Font sizes are in points, so the side length sets how large
# the labels are relative to the route, and the PDF scales the whole image down to the page: a
# larger figure at the same dpi buys no detail and only shrinks the text a reader ends up with.
_PLOT_SIDE_IN = 7.0
_PLOT_MAX_SIDE_IN = 11.0
_PLOT_DPI = 150

# Corner of the plot box each legend location owns, as (x range, z range) in axes fractions, sized
# to hold the widest legend this plot can produce. Ordered by preference, so ties keep the first.
_LEGEND_CORNERS = {
    "upper right": ((0.55, 1.0), (0.72, 1.0)),
    "upper left": ((0.0, 0.45), (0.72, 1.0)),
    "lower right": ((0.55, 1.0), (0.0, 0.28)),
    "lower left": ((0.0, 0.45), (0.0, 0.28)),
}

# Thumbnails of the query sequence, drawn as one strip per (mode, pair) under its trajectory plot.
_STRIP_COUNT = 20
_STRIP_THUMB_PX = 160
_STRIP_BORDER_PX = 4
_STRIP_DPI = 150
# Width of one thumbnail in inches, and the band under it that carries the frame index. The page
# scales the whole strip to fit its box, so what a reader ends up with is the ratio between these
# two, not either on its own: the label is kept at a tenth of a thumbnail's width whatever grid the
# frames are laid out in, which lands it between 6 and 10 pt on the page.
_STRIP_THUMB_IN = 1.05
_STRIP_LABEL_IN = 0.10
_STRIP_LABEL_PT = 5.5
# Thumbnails held back while the strip's frames are being picked, as a multiple of the strip's own
# length. Four keeps every drawn frame within a quarter of a strip step of where it belongs.
_STRIP_BUFFER = 4


@dataclass
class VprPair:
    """One map sequence and the query sequence replayed against the map built from it."""

    title: str
    map_sequence: str
    query_sequence: str
    edex_file: str = DEFAULT_EDEX_FILE
    map_gt: str = DEFAULT_GT_FILE
    query_gt: str = DEFAULT_GT_FILE


@dataclass
class VprStat:
    """Scored result of one (VPR mode, sequence pair) evaluation."""

    title: str = ""
    vpr_mode: str = ""
    n_query_frames: int = 0
    n_map_frames: int = 0
    # Bytes the place recognition map itself occupies on disk, and that spread over the frames that
    # were mapped, which is what a caller budgets against a route length.
    map_bytes: int = 0
    map_mb_per_frame: float = 0.0
    n_queries: int = 0
    n_query_errors: int = 0
    n_true_positives: int = 0
    n_false_positives: int = 0
    n_unrecognized: int = 0
    n_self_matches: int = 0
    n_skipped: int = 0
    n_timestamp_fallbacks: int = 0
    success_rate: float = 0.0
    false_positive_rate: float = 0.0
    unrecognized_rate: float = 0.0
    self_match_rate: float = 0.0
    precision: float = 0.0
    median_error_m: float = 0.0
    mean_error_m: float = 0.0
    mean_score: float = 0.0
    map_build_time_s: float = 0.0
    # Per-frame costs, each averaged over the frames it was measured on: the whole map pass over the
    # mapped frames, one `add_frame_to_vpr_map` over the same frames, one `recognize_place_by_frame`
    # over the queried ones.
    map_build_time_ms_per_frame: float = 0.0
    mean_add_time_ms: float = 0.0
    mean_search_time_ms: float = 0.0
    # Distance between the world pose a match carries and the map pose of the frame its timestamp
    # resolved to. It scores the timestamp lookup itself, not place recognition.
    node_pose_gap_m: float = 0.0
    plot_path: str = ""
    strip_path: str = ""
    errors_m: list = field(default_factory=list)


# The file Slam::SaveMap writes for the place recognition part of a map, next to the landmark
# database. Its name is fixed by VprMap::kFileName in libs/slam/vpr/vpr_map.h.
VPR_MAP_FILE = "vpr_map.bin"


def _vpr_map_bytes(map_dir: str) -> int:
    """Size of the saved place recognition map, ignoring the landmark database beside it.

    What a backend costs to keep is its descriptors, not the SLAM map it happens to be stored with,
    so a size that included the LMDB would be the same number for all four backends.
    """
    try:
        return os.path.getsize(os.path.join(map_dir, VPR_MAP_FILE))
    except OSError:
        return 0


@dataclass
class _MapBuild:
    """What phase 1 leaves behind for the query and scoring phases."""

    n_frames: int = 0
    build_time_s: float = 0.0
    add_time_s: float = 0.0
    map_bytes: int = 0
    frame_index_from_ts: dict = field(default_factory=dict)
    sorted_timestamps: Any = None
    world_from_rig: dict = field(default_factory=dict)


@dataclass
class _QueryHit:
    """One place recognition answer, tagged with the query frame that asked for it."""

    frame_index: int
    found: bool
    # False when the match is a keyframe of the query session rather than an entry of the loaded
    # map. A map loaded from disk is read only (VprMap::ReadOnly, libs/slam/vpr/vpr_map.h), so the
    # session's own keyframes never enter the search and this cannot happen; such a match would
    # carry a query timestamp, which names nothing in the map sequence and cannot be scored, so it
    # is counted to catch the day that guarantee breaks.
    imported: bool
    node_id: int
    score: float
    timestamp_ns: int
    position: Any


@dataclass
class _QueryRun:
    """What phase 2 leaves behind for the scoring and plotting phases."""

    hits: list = field(default_factory=list)
    n_frames: int = 0
    n_queries: int = 0
    n_errors: int = 0
    search_time_s: float = 0.0
    thumbnails: list = field(default_factory=list)


class _ReplayLimitReached(Exception):
    """Stops a reader replay from inside a processor callback once --frame_limit is reached."""


def _thumbnail(image: np.ndarray) -> np.ndarray:
    """Downscale one camera image to the strip's thumbnail width, keeping its aspect ratio."""
    height, width = image.shape[:2]
    thumb_height = max(1, round(height * _STRIP_THUMB_PX / width))
    resized = Image.fromarray(image).resize((_STRIP_THUMB_PX, thumb_height), Image.BILINEAR)
    return np.asarray(resized)


class _ThumbnailCollector:
    """Keep a bounded, evenly spread sample of the left images of the queried frames.

    How many frames the query pass will ask about is not known until it ends, so frames are kept on
    a stride that doubles whenever the buffer fills. That bounds this to a few dozen thumbnails
    instead of the whole sequence, which at 3.8 MB per CODa frame does not fit in memory, and still
    leaves what is kept spread evenly over the pass, within half a stride of the wanted spacing.
    """

    def __init__(self, count: int = _STRIP_COUNT):
        """Sample enough frames to draw `count` of them."""
        self.count = max(1, count)
        self.stride = 1
        self.n_seen = 0
        self.kept: list[tuple[int, np.ndarray]] = []
        self.last: Optional[tuple[int, Any]] = None

    def offer(self, frame_index: int, image: np.ndarray) -> None:
        """Take one more queried frame, downscaling it only when the stride keeps it."""
        if self.n_seen % self.stride == 0:
            self.kept.append((frame_index, _thumbnail(image)))
            if len(self.kept) > _STRIP_BUFFER * self.count:
                # Dropping every second entry doubles the spacing and keeps the first frame.
                self.kept = self.kept[::2]
                self.stride *= 2
        self.n_seen += 1
        # The reader hands out a fresh array per frame, so holding the newest one costs one frame
        # and saves downscaling every frame only to find out it was not the last after all.
        self.last = (frame_index, image)

    def select(self) -> list[tuple[int, np.ndarray]]:
        """Pick the frames to draw: evenly spaced, with the first and last queried frame included."""
        if not self.kept:
            return []
        entries = list(self.kept)
        positions = [index * self.stride for index in range(len(entries))]
        if self.last is not None and self.last[0] != entries[-1][0]:
            entries.append((self.last[0], _thumbnail(self.last[1])))
            positions.append(self.n_seen - 1)

        chosen: list[int] = []
        for step in range(self.count):
            target = round(step * positions[-1] / (self.count - 1)) if self.count > 1 else 0
            nearest = min(range(len(positions)), key=lambda index: abs(positions[index] - target))
            if not chosen or nearest != chosen[-1]:
                chosen.append(nearest)
        return [entries[index] for index in chosen]


def _slug(text: str) -> str:
    """Turn a pair title and mode into a file name safe for the plots folder."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "pair"


def normalize_modes(names) -> list[str]:
    """Resolve user spellings of backend names onto Slam.VprMode member names, keeping order."""
    lookup = {name.lower(): name for name in KNOWN_VPR_MODES}
    modes: list[str] = []
    for raw in names:
        key = str(raw).strip().lower()
        if key not in lookup:
            raise ValueError(f"Unknown VPR mode {raw!r}; expected one of {', '.join(KNOWN_VPR_MODES)}")
        if lookup[key] not in modes:
            modes.append(lookup[key])
    if not modes:
        raise ValueError("No VPR modes selected")
    return modes


def _resolve_sequence(path: str, datasets_root: str) -> str:
    """Resolve a config sequence path against the datasets root; absolute paths are used as is."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(datasets_root or ".", path))


def parse_config(config: Any, datasets_root: str) -> dict:
    """Validate a VPR reporter config and return its pairs plus the defaults it overrides."""
    if not isinstance(config, dict):
        raise ValueError("VPR reporter config must be a JSON object")

    raw_pairs = config.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("VPR reporter config missing required key: pairs (a non-empty list)")

    pairs = []
    for index, entry in enumerate(raw_pairs):
        if not isinstance(entry, dict):
            raise ValueError(f"VPR reporter config pairs[{index}] must be an object")
        for key in ("title", "map_sequence", "query_sequence"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f"VPR reporter config pairs[{index}] missing required string key: {key}")
        pairs.append(VprPair(
            title=entry["title"],
            map_sequence=_resolve_sequence(entry["map_sequence"], datasets_root),
            query_sequence=_resolve_sequence(entry["query_sequence"], datasets_root),
            edex_file=entry.get("edex_file", DEFAULT_EDEX_FILE),
            map_gt=entry.get("map_gt", DEFAULT_GT_FILE),
            query_gt=entry.get("query_gt", DEFAULT_GT_FILE),
        ))

    success_radius_m = config.get("success_radius_m")
    if success_radius_m is not None:
        if not isinstance(success_radius_m, (int, float)) or isinstance(success_radius_m, bool):
            raise ValueError("VPR reporter config key success_radius_m must be a number")
        if success_radius_m <= 0:
            raise ValueError("VPR reporter config key success_radius_m must be positive")

    vpr_modes = config.get("vpr_modes")
    if vpr_modes is not None:
        if not isinstance(vpr_modes, list):
            raise ValueError("VPR reporter config key vpr_modes must be a list")
        vpr_modes = normalize_modes(vpr_modes)

    return {
        "pairs": pairs,
        "success_radius_m": float(success_radius_m) if success_radius_m is not None else None,
        "vpr_modes": vpr_modes,
    }


def pair_from_args(args: argparse.Namespace, datasets_root: str) -> VprPair:
    """Build the single pair described by --map_sequence and --query_sequence."""
    map_sequence = _resolve_sequence(args.map_sequence, datasets_root)
    query_sequence = _resolve_sequence(args.query_sequence, datasets_root)
    title = args.pair_title or f"{os.path.basename(map_sequence)} to {os.path.basename(query_sequence)}"
    gt_file = args.gt_path or DEFAULT_GT_FILE
    return VprPair(
        title=title,
        map_sequence=map_sequence,
        query_sequence=query_sequence,
        edex_file=args.edex_filename or DEFAULT_EDEX_FILE,
        map_gt=gt_file,
        query_gt=gt_file,
    )


def _load_cuvslam():
    """Import the binding late so config parsing and --help work without it."""
    try:
        import cuvslam as vslam
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        if getattr(exc, "name", None) == "cuvslam" or "cuvslam" in str(exc):
            raise RuntimeError(CUVSLAM_VPR_REPORTER_DEPENDENCY_ERROR) from exc
        raise
    return vslam


def _load_gt_positions(sequence_dir: str, gt_file: str) -> np.ndarray:
    """Read KITTI-format absolute poses and keep the translation of each."""
    from cuvslam_tools.tracker.ground_truth import load_gt_transforms

    path = gt_file if os.path.isabs(gt_file) else os.path.join(sequence_dir, gt_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    transforms = load_gt_transforms(path)
    if not transforms:
        raise ValueError(f"Ground-truth file holds no poses: {path}")
    return np.array([transform[:3, 3] for transform in transforms])


def _open_sequence(sequence_dir: str, pair: VprPair, args: argparse.Namespace):
    """Open an EDEX sequence with its own ground truth left out; scoring reads it separately."""
    from cuvslam_tools.tracker.edex_reader import EdexReader

    reader = EdexReader(sequence_dir,
                        stereo_edex=os.path.join(sequence_dir, pair.edex_file),
                        cache_uncompressed=getattr(args, "cache_uncompressed", False))
    if not reader.validate_rig():
        raise ValueError(f"Rig parameters are invalid: {os.path.join(sequence_dir, pair.edex_file)}")
    return reader


def _odometry_config(vslam, args: argparse.Namespace):
    """Odometry configuration shared by the map and query passes."""
    from cuvslam_tools.tracker import conversions as conv

    cfg = vslam.Odometry.Config()
    cfg.odometry_mode = (conv.str2odometry_mode(args.odometry_mode)
                         if isinstance(args.odometry_mode, str) else args.odometry_mode)
    cfg.multicam_mode = (conv.str2multicam_mode(args.multicam_mode)
                         if isinstance(args.multicam_mode, str) else args.multicam_mode)
    cfg.use_gpu = args.use_gpu
    cfg.use_motion_model = args.use_motion_model
    cfg.use_denoising = args.use_denoising
    cfg.rectified_stereo_camera = args.rectified_stereo_camera
    cfg.max_frame_delta_s = args.max_frame_delta_s
    # SLAM runs in the calling thread, so a query already sees the keyframe the previous frame
    # created, and it needs the odometry exports to create keyframes at all.
    cfg.async_sba = False
    cfg.enable_observations_export = True
    cfg.enable_landmarks_export = True
    return cfg


def _slam_config(vslam, args: argparse.Namespace, mode_name: str, vpr_map_path: str = ""):
    """SLAM configuration for one VPR mode, optionally starting from a saved map.

    `vpr_map_path` is a std::string_view on the caller's str, so that str has to outlive the
    Tracker built from this config.
    """
    cfg = vslam.Slam.Config()
    cfg.use_gpu = args.use_gpu
    cfg.sync_mode = True
    cfg.max_map_size = 0
    cfg.vpr_mode = getattr(vslam.Slam.VprMode, mode_name)
    cfg.vpr_score_threshold = args.vpr_score_threshold
    if args.vpr_model_path:
        cfg.vpr_model_path = args.vpr_model_path
    if vpr_map_path:
        cfg.vpr_map_path = vpr_map_path
    return cfg


def _rig_pose(odom_pose, slam_pose):
    """Prefer the SLAM pose of a frame and fall back to odometry, as the tracker CLI does."""
    if slam_pose is not None:
        return slam_pose
    if odom_pose is not None and odom_pose.world_from_rig is not None:
        return odom_pose.world_from_rig.pose
    return None


class _MapBuilder:
    """Track the map sequence and offer every tracked frame to the place recognition map."""

    def __init__(self, tracker, frame_limit: int):
        """Store the tracker to drive and the frame budget to stop at."""
        self.tracker = tracker
        self.frame_limit = frame_limit
        self.frame_index_from_ts: dict[int, int] = {}
        self.world_from_rig: dict[int, Any] = {}
        self.n_frames = 0
        self.build_time_s = 0.0
        self.add_time_s = 0.0

    def process_images(self, frame_id, timestamps, images, masks, depths=None):
        """Track one frame, then hand it to the map."""
        if self.frame_limit and self.n_frames >= self.frame_limit:
            raise _ReplayLimitReached
        timestamp = max(timestamps)
        start = time.perf_counter()
        odom_pose, slam_pose = self.tracker.track(timestamp, images, masks, depths)
        add_start = time.perf_counter()
        self.tracker.slam.add_frame_to_vpr_map(images, timestamp)
        done = time.perf_counter()
        self.add_time_s += done - add_start
        self.build_time_s += done - start
        self.frame_index_from_ts[timestamp] = frame_id
        pose = _rig_pose(odom_pose, slam_pose)
        if pose is not None:
            self.world_from_rig[frame_id] = pose
        self.n_frames += 1

    def process_imu(self, timestamp, linear_accelerations, angular_velocities):
        """Ignore IMU samples: place recognition is scored on images alone."""

    def get_camera_pose(self, frame_id):
        """Return the pose tracked for a frame, as the reader protocol expects."""
        return self.world_from_rig.get(frame_id)

    def set_frame_metadata(self, frame_id, metadata):
        """Ignore replay metadata: every sequence is replayed once, forward."""


class _MapQuerier:
    """Track the query sequence and ask the loaded map where each frame was taken."""

    def __init__(self, tracker, frame_limit: int, query_stride: int, track_frames: bool):
        """Store the tracker to drive, the frame budget, how often to query, and whether to track."""
        self.tracker = tracker
        self.frame_limit = frame_limit
        self.query_stride = max(1, query_stride)
        self.track_frames = track_frames
        self.hits: list[_QueryHit] = []
        self.thumbnails = _ThumbnailCollector()
        self.n_frames = 0
        self.n_queries = 0
        self.n_errors = 0
        self.last_error = ""
        self.search_time_s = 0.0

    def process_images(self, frame_id, timestamps, images, masks, depths=None):
        """Track one frame when asked to, and recognize the place it shows on the queried frames."""
        if self.frame_limit and self.n_frames >= self.frame_limit:
            raise _ReplayLimitReached
        timestamp = max(timestamps)
        if self.track_frames:
            self.tracker.track(timestamp, images, masks, depths)
        self.n_frames += 1
        if frame_id % self.query_stride != 0:
            return
        answers: list[tuple[Any, str]] = []
        start = time.perf_counter()
        self.tracker.slam.recognize_place_by_frame(
            images, timestamp, lambda place, error_message: answers.append((place, error_message)))
        self.search_time_s += time.perf_counter() - start
        self.n_queries += 1
        self.thumbnails.offer(frame_id, images[0])
        # The search is asynchronous and only Slam.Config.sync_mode makes it finish before the call
        # returns. This tool sets that flag, so an unanswered call is a broken assumption, not a
        # frame to skip: reporting rates over whichever frames happened to answer would hide it.
        if not answers:
            raise RuntimeError(f"recognize_place_by_frame returned before answering frame {frame_id}; "
                               "the reporter needs Slam.Config.sync_mode to collect its result")
        place, error_message = answers[0]
        if error_message or place is None:
            self.n_errors += 1
            self.last_error = error_message or "no result and no error message"
            return
        self.hits.append(_QueryHit(
            frame_index=frame_id,
            found=place.found,
            imported=place.imported,
            node_id=place.node_id,
            score=place.score,
            timestamp_ns=place.timestamp_ns,
            position=np.array(place.pose.translation),
        ))

    def process_imu(self, timestamp, linear_accelerations, angular_velocities):
        """Ignore IMU samples: place recognition is scored on images alone."""

    def get_camera_pose(self, frame_id):
        """Query poses are not scored, so nothing is kept for the reader to read back."""
        return None

    def set_frame_metadata(self, frame_id, metadata):
        """Ignore replay metadata: every sequence is replayed once, forward."""


def _build_vpr_map(pair: VprPair, mode_name: str, args: argparse.Namespace, map_dir: str) -> _MapBuild:
    """Phase 1: track the map sequence, collect its map, and write it to `map_dir`."""
    vslam = _load_cuvslam()

    reader = _open_sequence(pair.map_sequence, pair, args)
    odom_cfg = _odometry_config(vslam, args)
    slam_cfg = _slam_config(vslam, args, mode_name)
    tracker = vslam.Tracker(reader.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)

    builder = _MapBuilder(tracker, args.frame_limit)
    try:
        reader.replay(builder)
    except _ReplayLimitReached:
        pass

    # Loop closures and pose graph optimization move poses after the frame that produced them was
    # tracked, and the map is saved with the optimized ones, so take those here too.
    for slam_pose in tracker.slam.get_all_slam_poses() or []:
        frame_index = builder.frame_index_from_ts.get(slam_pose.timestamp_ns)
        if frame_index is not None:
            builder.world_from_rig[frame_index] = slam_pose.pose

    # The place recognition map is part of the SLAM map, so this writes the landmark database too.
    saved: list[bool] = []
    tracker.slam.save_map(map_dir, saved.append)
    if not saved:
        raise RuntimeError(f"save_map returned before saving the {mode_name} map to {map_dir}; "
                           "the reporter needs Slam.Config.sync_mode to know whether it worked")
    if not saved[0]:
        raise RuntimeError(f"Failed to write the {mode_name} map to {map_dir}")

    build = _MapBuild(
        n_frames=builder.n_frames,
        build_time_s=builder.build_time_s,
        add_time_s=builder.add_time_s,
        map_bytes=_vpr_map_bytes(map_dir),
        frame_index_from_ts=builder.frame_index_from_ts,
        sorted_timestamps=np.array(sorted(builder.frame_index_from_ts), dtype=np.int64),
        world_from_rig=builder.world_from_rig,
    )
    # The query pass builds its own tracker; drop this one first so both never hold GPU memory.
    del tracker
    return build


def _query_vpr_map(pair: VprPair, mode_name: str, args: argparse.Namespace, map_dir: str) -> _QueryRun:
    """Phase 2: replay the query sequence against a fresh tracker holding the saved map."""
    vslam = _load_cuvslam()

    reader = _open_sequence(pair.query_sequence, pair, args)
    odom_cfg = _odometry_config(vslam, args)
    # Keep the path in a local: Slam.Config.vpr_map_path only borrows this str.
    vpr_map_path = str(map_dir)
    slam_cfg = _slam_config(vslam, args, mode_name, vpr_map_path)
    tracker = vslam.Tracker(reader.rig, vslam.Tracker.Mode.OdometryWithSlamOffline, odom_cfg, slam_cfg)

    querier = _MapQuerier(tracker, args.frame_limit, args.query_stride, args.query_tracking)
    try:
        reader.replay(querier)
    except _ReplayLimitReached:
        pass

    if querier.n_errors and querier.n_errors == querier.n_queries:
        raise RuntimeError(f"place recognition failed on all {querier.n_errors} queried frames: {querier.last_error}")
    if querier.n_errors:
        print(f"WARNING: {pair.title} [{mode_name}]: place recognition returned an error for "
              f"{querier.n_errors} of {querier.n_queries} queried frames, which are left out of "
              f"every rate below; last error: {querier.last_error}")

    run = _QueryRun(hits=querier.hits, n_frames=querier.n_frames, n_queries=querier.n_queries,
                    n_errors=querier.n_errors, search_time_s=querier.search_time_s,
                    thumbnails=querier.thumbnails.select())
    del tracker
    return run


def _lookup_map_index(timestamp_ns: int, build: _MapBuild) -> tuple[Optional[int], bool]:
    """Resolve a match timestamp to a map frame index, falling back to the nearest timestamp."""
    index = build.frame_index_from_ts.get(timestamp_ns)
    if index is not None:
        return index, False
    if build.sorted_timestamps is None or len(build.sorted_timestamps) == 0:
        return None, True
    position = int(np.searchsorted(build.sorted_timestamps, timestamp_ns))
    candidates = [c for c in (position - 1, position) if 0 <= c < len(build.sorted_timestamps)]
    nearest = min(candidates, key=lambda c: abs(int(build.sorted_timestamps[c]) - timestamp_ns))
    return build.frame_index_from_ts[int(build.sorted_timestamps[nearest])], True


def _score_pair(pair: VprPair, mode_name: str, build: _MapBuild, hits: list[_QueryHit],
                gt_map: np.ndarray, gt_query: np.ndarray, success_radius_m: float) -> tuple[VprStat, list]:
    """Phase 3: classify every query frame and reduce the classifications to one row of metrics."""
    stat = VprStat(title=pair.title, vpr_mode=mode_name, n_map_frames=build.n_frames,
                   map_build_time_s=build.build_time_s, map_bytes=build.map_bytes)
    if build.n_frames:
        stat.map_build_time_ms_per_frame = 1e3 * build.build_time_s / build.n_frames
        stat.mean_add_time_ms = 1e3 * build.add_time_s / build.n_frames
        stat.map_mb_per_frame = build.map_bytes / 1e6 / build.n_frames
    classified = []
    scores = []
    node_pose_gaps = []

    for hit in hits:
        if hit.frame_index >= len(gt_query):
            stat.n_skipped += 1
            continue
        if not hit.found:
            stat.n_unrecognized += 1
            classified.append((hit.frame_index, UNRECOGNIZED))
            continue
        if not hit.imported:
            stat.n_self_matches += 1
            classified.append((hit.frame_index, SELF_MATCH))
            continue
        matched_index, fell_back = _lookup_map_index(hit.timestamp_ns, build)
        if fell_back:
            stat.n_timestamp_fallbacks += 1
        if matched_index is None or matched_index >= len(gt_map):
            stat.n_skipped += 1
            continue
        scores.append(hit.score)
        map_pose = build.world_from_rig.get(matched_index)
        if map_pose is not None:
            node_pose_gaps.append(float(np.linalg.norm(hit.position - np.array(map_pose.translation))))
        error_m = float(np.linalg.norm(gt_map[matched_index] - gt_query[hit.frame_index]))
        if error_m <= success_radius_m:
            stat.n_true_positives += 1
            stat.errors_m.append(error_m)
            classified.append((hit.frame_index, TRUE_POSITIVE))
        else:
            stat.n_false_positives += 1
            classified.append((hit.frame_index, FALSE_POSITIVE))

    stat.n_query_frames = (stat.n_true_positives + stat.n_false_positives + stat.n_unrecognized
                           + stat.n_self_matches)
    if stat.n_query_frames:
        stat.success_rate = 100.0 * stat.n_true_positives / stat.n_query_frames
        stat.false_positive_rate = 100.0 * stat.n_false_positives / stat.n_query_frames
        stat.unrecognized_rate = 100.0 * stat.n_unrecognized / stat.n_query_frames
        stat.self_match_rate = 100.0 * stat.n_self_matches / stat.n_query_frames
    recognized = stat.n_true_positives + stat.n_false_positives
    if recognized:
        stat.precision = stat.n_true_positives / recognized
    if stat.errors_m:
        stat.median_error_m = float(np.median(stat.errors_m))
        stat.mean_error_m = float(np.mean(stat.errors_m))
    if scores:
        stat.mean_score = float(np.mean(scores))
    if node_pose_gaps:
        stat.node_pose_gap_m = float(np.median(node_pose_gaps))
    return stat, classified


def _legend_corner(drawn: np.ndarray, low: np.ndarray, high: np.ndarray) -> str:
    """Pick the corner of the plot box holding the fewest drawn points, so the legend covers none.

    matplotlib's loc='best' cannot do this here: it reads the offsets of a collection, not its
    paths, so the query ribbon is invisible to it and only the map line is avoided.
    """
    fractions = (drawn - low) / np.maximum(high - low, 1e-9)
    counts = {}
    for location, (x_range, z_range) in _LEGEND_CORNERS.items():
        inside = ((fractions[:, 0] >= x_range[0]) & (fractions[:, 0] <= x_range[1])
                  & (fractions[:, 1] >= z_range[0]) & (fractions[:, 1] <= z_range[1]))
        counts[location] = int(inside.sum())
    return min(counts, key=lambda location: counts[location])


def _figure_size(span_x: float, span_z: float) -> tuple[float, float]:
    """Give the figure the route's own aspect, so a wide route is not letterboxed into a square."""
    aspect = span_x / span_z
    width = min(_PLOT_SIDE_IN * max(1.0, aspect), _PLOT_MAX_SIDE_IN)
    height = min(_PLOT_SIDE_IN / min(1.0, aspect), _PLOT_MAX_SIDE_IN)
    return width, height


def _save_pair_plot(stat: VprStat, gt_map: np.ndarray, gt_query: np.ndarray,
                    classified: list, output_dir: str) -> str:
    """Draw the bird's eye view of the map trajectory and the classified query trajectory."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{_slug(stat.title)}_{_slug(stat.vpr_mode)}.png")

    map_xz = gt_map[:, [0, 2]]
    query_xz = gt_query[[index for index, _ in classified]][:, [0, 2]] if classified else np.empty((0, 2))
    drawn = np.vstack((map_xz, query_xz))
    low, high = drawn.min(axis=0), drawn.max(axis=0)
    span = np.maximum(high - low, 1.0)
    pad = 0.05 * span

    fig, ax = plt.subplots(figsize=_figure_size(span[0], span[1]), layout="constrained")
    ax.plot(map_xz[:, 0], map_xz[:, 1], color=_MAP_INK, linewidth=1.0, solid_capstyle="round", zorder=2)

    # One segment per consecutive pair of queried frames, colored by the later frame: on a long
    # sequence this reads as a ribbon, where a point per frame smears into a blob.
    if len(query_xz) > 1:
        segments = np.stack((query_xz[:-1], query_xz[1:]), axis=1)
        colors = [_CLASS_STYLE[classification][0] for _, classification in classified[1:]]
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=_RIBBON_WIDTH_PT,
                                         capstyle="round", joinstyle="round", zorder=3))

    present = [label for label in _CLASS_STYLE
               if any(classification == label for _, classification in classified)]
    if len(query_xz) <= _MARKER_QUERY_LIMIT:
        for label in present:
            points = query_xz[[position for position, (_, classification) in enumerate(classified)
                               if classification == label]]
            ax.scatter(points[:, 0], points[:, 1], s=_MARKER_AREA_PT2, c=_CLASS_STYLE[label][0],
                       edgecolors="white", linewidths=0.8, zorder=4)

    handles = [Line2D([], [], color=_MAP_INK, linewidth=1.5, label="Map sequence")]
    handles += [Line2D([], [], color=_CLASS_STYLE[label][0], linewidth=_RIBBON_WIDTH_PT,
                       label=_CLASS_STYLE[label][1]) for label in present]
    legend = ax.legend(handles=handles, loc=_legend_corner(drawn, low - pad, high + pad), fontsize=10,
                       framealpha=0.93, edgecolor=_AXIS_INK, labelcolor=_TEXT_INK)
    legend.get_frame().set_linewidth(0.8)

    ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
    ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID_INK, linewidth=0.8, linestyle="-")
    for spine in ax.spines.values():
        spine.set_color(_AXIS_INK)
        spine.set_linewidth(0.8)
    ax.tick_params(color=_AXIS_INK, labelcolor=_TEXT_INK, labelsize=9, width=0.8)
    ax.set_xlabel("x [m]", fontsize=11, color=_TEXT_INK)
    ax.set_ylabel("z [m]", fontsize=11, color=_TEXT_INK)

    ax.set_title(f"{stat.title} - {stat.vpr_mode}", fontsize=14, pad=22)
    # Anchored to the axes rather than to the figure: an equal-aspect box can be much shorter than
    # the figure, and a figure-level subtitle would then float far above the plot.
    ax.annotate(f"success {stat.success_rate:.1f}%  false positive {stat.false_positive_rate:.1f}%  "
                f"unrecognized {stat.unrecognized_rate:.1f}%",
                xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, color=_TEXT_INK)

    fig.savefig(plot_path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _strip_grid(count: int, thumb_aspect: float) -> tuple:
    """Rows and columns to lay `count` thumbnails out in, for a strip that fits a report page.

    One long row is the obvious layout and the wrong one: the page scales the strip to its own
    width, so twenty frames across an A4 page leave each one a few millimetres wide and unreadable.
    The strip now sits beside the trajectory in a box about 12.5 cm wide and 15.5 cm tall, and a
    thumbnail is largest when the grid has the same shape as that box: any other shape leaves the
    box empty along one axis. So pick the factorization whose overall shape is closest to it, which
    for wide driving images means a few tall columns rather than one squat banner.
    """
    target = 12.5 / 15.5
    best = (1, count)
    best_error = None
    for rows in range(1, count + 1):
        if count % rows:
            continue
        columns = count // rows
        aspect = (columns / rows) / max(thumb_aspect, 1e-6)
        error = abs(math.log(aspect / target))
        if best_error is None or error < best_error:
            best, best_error = (rows, columns), error
    return best


def _save_strip_plot(stat: VprStat, thumbnails: list, classified: list, output_dir: str) -> str:
    """Draw the sampled query frames in a row, bordered by whether the robot knew where it was."""
    if not thumbnails:
        return ""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    strip_path = os.path.join(plots_dir, f"{_slug(stat.title)}_{_slug(stat.vpr_mode)}_frames.png")

    classification = dict(classified)
    tallest = max(thumb.shape[0] / thumb.shape[1] for _, thumb in thumbnails)
    rows, columns = _strip_grid(len(thumbnails), tallest)
    label_in = _STRIP_LABEL_IN
    label_pt = _STRIP_LABEL_PT
    width_in = _STRIP_THUMB_IN * columns
    height_in = (_STRIP_THUMB_IN * tallest + label_in) * rows
    fig = plt.figure(figsize=(width_in, height_in), dpi=_STRIP_DPI)
    grid = fig.add_gridspec(rows, columns, wspace=0.06, hspace=label_in / _STRIP_THUMB_IN,
                            left=0.002, right=0.998, bottom=label_in / height_in, top=0.995)
    # Spine widths are in points, and the border is specified in pixels of the image written here.
    border_pt = _STRIP_BORDER_PX * 72.0 / _STRIP_DPI

    for position, (frame_index, thumb) in enumerate(thumbnails):
        ax = fig.add_subplot(grid[position // columns, position % columns])
        if thumb.ndim == 2:
            ax.imshow(thumb, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(thumb)
        ax.set_xticks([])
        ax.set_yticks([])
        # Green only for a frame that was recognized correctly: a false positive left the robot as
        # lost as an unrecognized frame did, so both are red.
        recognized = classification.get(frame_index) == TRUE_POSITIVE
        for spine in ax.spines.values():
            spine.set_color(_CLASS_STYLE[TRUE_POSITIVE if recognized else UNRECOGNIZED][0])
            spine.set_linewidth(border_pt)
        ax.set_xlabel(str(frame_index), fontsize=label_pt, color=_TEXT_INK, labelpad=1)

    fig.savefig(strip_path, dpi=_STRIP_DPI)
    plt.close(fig)
    return strip_path


def evaluate_pair(pair: VprPair, mode_name: str, args: argparse.Namespace,
                  success_radius_m: float, output_dir: str) -> VprStat:
    """Run the build, query, score, and plot phases for one (mode, pair) combination."""
    gt_map = _load_gt_positions(pair.map_sequence, pair.map_gt)
    gt_query = _load_gt_positions(pair.query_sequence, pair.query_gt)

    map_dir = tempfile.mkdtemp(prefix=f"vpr_map_{_slug(mode_name)}_")
    try:
        build = _build_vpr_map(pair, mode_name, args, map_dir)
        print(f"{pair.title} [{mode_name}]: mapped {build.n_frames} frames")
        run = _query_vpr_map(pair, mode_name, args, map_dir)
    finally:
        # The folder holds the whole SLAM map, landmark database included, so it is far larger than
        # the place recognition map alone and is deleted unless the run asked to keep it.
        if args.keep_vpr_maps:
            print(f"{pair.title} [{mode_name}]: map kept at {map_dir}")
        else:
            shutil.rmtree(map_dir, ignore_errors=True)

    stat, classified = _score_pair(pair, mode_name, build, run.hits, gt_map, gt_query, success_radius_m)
    stat.n_queries = run.n_queries
    stat.n_query_errors = run.n_errors
    if run.n_queries:
        stat.mean_search_time_ms = 1e3 * run.search_time_s / run.n_queries
    # Draw only the part of the map sequence that was actually mapped, which --frame_limit cuts short.
    stat.plot_path = os.path.abspath(
        _save_pair_plot(stat, gt_map[:build.n_frames], gt_query, classified, output_dir))
    strip_path = _save_strip_plot(stat, run.thumbnails, classified, output_dir)
    stat.strip_path = os.path.abspath(strip_path) if strip_path else ""
    print(f"{pair.title} [{mode_name}]: queried {run.n_queries} of {run.n_frames} frames, "
          f"success {stat.success_rate:.1f}%, false positives {stat.false_positive_rate:.1f}%, "
          f"unrecognized {stat.unrecognized_rate:.1f}%, "
          f"median error {stat.median_error_m:.2f} m, map pose gap {stat.node_pose_gap_m:.2f} m, "
          f"{stat.n_timestamp_fallbacks} timestamp fallbacks, {stat.n_skipped} skipped")
    # The loaded map is read only, so the query session cannot match its own keyframes and this
    # count is always zero. It is still counted, and shouted about, because a non-zero value means
    # the read-only guarantee broke and every rate in this row is measuring the wrong thing.
    if stat.n_self_matches:
        print(f"WARNING: {pair.title} [{mode_name}]: {stat.n_self_matches} query frames matched a "
              f"keyframe of the query session ({stat.self_match_rate:.1f}%); a loaded place "
              f"recognition map is supposed to be read only, so these rates are not trustworthy")
    return stat


def evaluate_pairs(pairs: list[VprPair], modes: list[str], args: argparse.Namespace,
                   success_radius_m: float, output_dir: str) -> list[VprStat]:
    """Evaluate every (mode, pair) combination, one at a time so each owns the GPU alone."""
    stats = []
    failures = []
    for mode_name in modes:
        for pair in pairs:
            try:
                stats.append(evaluate_pair(pair, mode_name, args, success_radius_m, output_dir))
            except RuntimeError as exc:
                if str(exc) == CUVSLAM_VPR_REPORTER_DEPENDENCY_ERROR:
                    raise
                failures.append(f"{pair.title} [{mode_name}]: {exc}")
    if failures:
        raise RuntimeError("VPR evaluation failed for: " + "; ".join(failures))
    return stats
