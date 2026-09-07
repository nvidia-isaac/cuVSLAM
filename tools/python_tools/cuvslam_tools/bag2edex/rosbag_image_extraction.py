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

"""Image extraction and synchronization utilities for ROS bag to EDEX conversion."""

import concurrent.futures
import json
import logging
import os
import pathlib
import queue
import shutil
import sys
import threading
import time
from typing import Any, Optional

import av
import numpy as np
import pandas as pd
from rosbags import highlevel

from cuvslam_tools.common import edex
from . import (
    Config,
    get_first_message,
    log_rosbag_info,
    open_rosbag_reader,
)


DISTORTION_MODEL_ROS2EDEX = {
    "pinhole": edex.DistortionModel.PINHOLE,
    "equidistant": edex.DistortionModel.FISHEYE,
    # plumb_bob is the odd one out: its coefficients need reordering, see get_distortion_model().
    "plumb_bob": edex.DistortionModel.BROWN5K,
    "rational_polynomial": edex.DistortionModel.POLYNOMIAL,
}

# ROS plumb_bob is [k1, k2, p1, p2, k3], edex brown5k is [k1, k2, k3, p1, p2].
_PLUMB_BOB_TO_BROWN5K = [0, 1, 4, 2, 3]


def progress_bar(
    iteration: int, total: int, prefix="", suffix="", line_length=80, fill="█"
):
    """Render a single-line text progress bar to stdout."""
    length = line_length - len(prefix) - len(suffix)
    if total <= 0:
        percent = "0.0"
        filled_length = 0
    else:
        iteration = max(0, min(iteration, total))
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()


def pyav_format_from_ros_encoding(encoding: str) -> tuple[str, int]:
    """Convert a ros encoding to a pyav format string and num color channels."""
    ros_to_pyav = {
        "mono8": ("gray8", 1),
        "bgr8": ("bgr24", 3),
        "rgb8": ("rgb24", 3),
    }
    if encoding not in ros_to_pyav:
        raise ValueError(f"Unknown ROS image encoding: '{encoding}'. Supported: {list(ros_to_pyav)}")
    return ros_to_pyav[encoding]


def pyav_codec_from_ros_format(fmt: str) -> str:
    """Convert a ros format to a pyav codec string."""
    if "jpeg" in fmt:
        return "mjpeg"
    elif "png" in fmt:
        return "png"
    elif fmt == "h264":
        return "h264"
    elif fmt == "hevc" or fmt == "h265":
        return "hevc"
    else:
        raise ValueError(f"Unknown ROS image message format: '{fmt}'.")


def get_image_path(base_path: pathlib.Path, topic: str, frame_idx: int) -> pathlib.Path:
    """Get the path to the image with the given index in the given camera."""
    # Remove leading slash
    topic_clean = topic.lstrip("/")
    if topic_clean == "":
        return base_path / f"{frame_idx:06d}.png"
    return base_path / f"{topic_clean}/{frame_idx:06d}.png"


def _producer(
    reader: highlevel.AnyReader,
    topics: list[str],
    width: Optional[int],
    height: Optional[int],
    pixel_format: Optional[str],
    images_base_path: pathlib.Path,
    frame_queue: queue.Queue,
    shutdown_event: threading.Event,
) -> pd.DataFrame:
    """A function that fills a queue with frames that should be written to disk."""
    if width or height:
        assert width and height, "Both width and height must be specified."

    logging.debug("Started producer thread.")
    logging.info(f"Writing images to '{images_base_path}'.")

    # Setup required helper objects.
    timestamps: dict[str, list[int]] = {}
    camera_indices: dict[str, int] = {}
    for idx, topic in enumerate(topics):
        timestamps[topic] = []
        camera_indices[topic] = idx
        # If the parent directory of the images does not exist it will fail silently.
        get_image_path(images_base_path, topic, 0).parent.mkdir(
            parents=True, exist_ok=True
        )

    # Create a new reader instance here because some reader classes (e.g. .db3), are not thread safe
    with highlevel.AnyReader(
        paths=reader.paths,
        default_typestore=reader.default_typestore,
    ) as reader:
        # Incrementally extract the images from the image streams.
        connections = [c for c in reader.connections if c.topic in topics]
        logging.info("Starting to extract images from rosbag.")

        num_messages = sum(c.msgcount for c in connections)

        # Decoders are only initialized for compressed images.
        decoders: dict[str, av.CodecContext] = {}

        for idx, (connection, _, rawdata) in enumerate(reader.messages(connections)):
            if idx % 100 == 0:
                progress_bar(
                    idx, num_messages, prefix="Extracting images...", suffix="Done"
                )

            # Deserialize the ROS message.
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.msgtype == "sensor_msgs/msg/Image":
                # Directly use the uncompressed frame.
                av_format, num_color_channels = pyav_format_from_ros_encoding(msg.encoding)
                if num_color_channels == 1:
                    decoded_frame = av.VideoFrame.from_ndarray(
                        msg.data.reshape(msg.height, msg.width),
                        format=av_format,
                    )
                else:
                    decoded_frame = av.VideoFrame.from_ndarray(
                        msg.data.reshape(msg.height, msg.width, num_color_channels),
                        format=av_format,
                    )
            elif connection.msgtype == "sensor_msgs/msg/CompressedImage":
                if topic not in decoders:
                    decoders[topic] = av.CodecContext.create(
                        pyav_codec_from_ros_format(msg.format), "r"
                    )
                    logging.info(
                        f"Using codec '{decoders[topic].codec.name}' for topic '{topic}'."
                    )
                # Decode the frame.
                encoded_frame_bytes = msg.data.tobytes()
                try:
                    encoded_packet = av.packet.Packet(encoded_frame_bytes)
                    decoded_frames = decoders[topic].decode(encoded_packet)
                except av.error.InvalidDataError:
                    logging.warning(
                        f"Skipping message {idx} for topic '{topic}' due to InvalidDataError."
                    )
                    continue

                # H264 decoders may buffer frames, so decoded_frames can be empty
                if len(decoded_frames) == 0:
                    logging.warning(
                        f"Skipping message {idx} for topic '{topic}' due to 0 decoded frames."
                    )
                    continue
                elif len(decoded_frames) == 1:
                    decoded_frame = decoded_frames[0]
                else:
                    raise ValueError(
                        f"Expected 1 decoded frame in message {idx} for topic '{topic}', but got {len(decoded_frames)}"
                    )
            else:
                raise ValueError(
                    f"Unknown message type '{connection.msgtype}' in message {idx} for topic '{topic}'."
                )

            decoded_frame = decoded_frame.reformat(
                width=width or decoded_frame.width,
                height=height or decoded_frame.height,
                format=pixel_format or decoded_frame.format,
            )

            # Store the timestamp corresponding to the frame.
            frame_idx = len(timestamps[topic])
            timestamps[topic].append(
                msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            )

            frame_queue.put((images_base_path, topic, frame_idx, decoded_frame))

    # Print new line to finish progress bar.
    print("")

    shutdown_event.set()
    logging.info("Finished extracting images from rosbag.")

    # Append -1 to all timestamps lists to make them the same length.
    max_len = max(len(timestamps_list) for timestamps_list in timestamps.values())
    for timestamps_list in timestamps.values():
        timestamps_list += [-1] * (max_len - len(timestamps_list))

    timestamp_df = pd.DataFrame(timestamps)
    timestamp_df.to_csv(images_base_path / "raw_timestamps.csv")

    return timestamp_df


def _consumer(
    thread_id: int, frame_queue: queue.Queue, shutdown_event: threading.Event
) -> None:
    """A function that consumes the next frame from the queue and writes it to disk."""
    while True:
        try:
            images_base_path, topic, frame_idx, frame = frame_queue.get(timeout=1)
            image_path = get_image_path(images_base_path, topic, frame_idx)
            logging.debug(f"Writing frame {topic}/{frame_idx} to {image_path}.")
            frame.to_image().save(str(image_path))
        except queue.Empty:
            # If the queue is empty and the shutdown event is set the producer is done, thus we can
            # stop.
            if shutdown_event.is_set():
                break


def run_executor(
    reader: highlevel.AnyReader,
    topics: list[str],
    width: Optional[int],
    height: Optional[int],
    pixel_format: Optional[str],
    images_base_path: pathlib.Path,
    num_workers: int = -1,
) -> pd.DataFrame:
    """
    Extract all images from a rosbag with a worker pool.
    """
    if num_workers == -1:
        num_workers = 2 * (os.cpu_count() or 1)

    if num_workers < 2:
        logging.warning(
            f"Need at least 2 workers, but "
            f"only has {num_workers} workers. Increasing num_workers"
        )
        num_workers = 2

    shutdown_event = threading.Event()
    frame_queue: queue.Queue[tuple] = queue.Queue(maxsize=num_workers * 2)

    start = time.time()
    logging.info(f"Starting thread pool with {num_workers} workers.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Start 1 thread for the producer.
        logging.info("Starting producer thread.")
        producer_future = executor.submit(
            _producer,
            reader,
            topics,
            width,
            height,
            pixel_format,
            images_base_path,
            frame_queue,
            shutdown_event,
        )

        # Use all remaining resources for the consumers.
        logging.info(f"Starting {num_workers - 1} consumer threads.")
        consumer_futures = [
            executor.submit(_consumer, i, frame_queue, shutdown_event)
            for i in range(num_workers - 1)
        ]

        # Wait for the producers and consumers to finish.
        all_futures = [producer_future] + consumer_futures
        try:
            for future in concurrent.futures.as_completed(all_futures):
                future.result()
        except Exception as e:
            logging.error(f"Error extracting images: {e}")
            shutdown_event.set()
            raise e
        timestamps_df = producer_future.result()

    end = time.time()
    duration_s = end - start
    logging.info(f"Finished extracting all images. Took {duration_s} seconds.")
    return timestamps_df


def synchronize_images(
    timestamps_df: pd.DataFrame, images_base_path: pathlib.Path, sync_threshold_ns: int
) -> pd.DataFrame:
    """
    Synchronize the images based on their timestamp. This will also modify/move the images on
    disk. Returns a dataframe with the synchronized timestamps.
    """
    # Our strategy is to iterate through the timestamps from the front. If all front stamps are
    # inside of the threshold we have a match. Else we increment the index of the earliest
    # timestamp in the front set.

    # Setup helper objects.
    topics = timestamps_df.columns
    front_idx = {topic: 0 for topic in topics}
    frame_idx = 0
    synced_timestamps: dict[str, list[int]] = {topic: [] for topic in topics}

    # Iterate until we reach the end of one image stream.
    while all(front_idx[topic] < timestamps_df.shape[0] for topic in topics):
        # Update the front values list.
        front = [timestamps_df[topic][idx] for topic, idx in front_idx.items()]

        # Values are -1 if we reached the end of an image stream.
        if any(v < 0 for v in front):
            break

        argmin = np.argmin(front)
        argmax = np.argmax(front)
        if front[argmax] - front[argmin] < sync_threshold_ns:
            # Rename images on disk.
            for topic, old_frame_idx in front_idx.items():
                old_path = get_image_path(images_base_path, topic, old_frame_idx)
                new_path = get_image_path(images_base_path, topic, frame_idx)
                logging.debug(f"Renaming {old_path} to {new_path}.")
                os.rename(old_path, new_path)
                synced_timestamps[topic].append(timestamps_df[topic][old_frame_idx])
            # Bump all frame indices.
            front_idx = {topic: front_idx[topic] + 1 for topic in topics}
            frame_idx += 1
        else:
            # Skip the oldest front image.
            path = get_image_path(
                images_base_path, topics[argmin], front_idx[topics[argmin]]
            )
            path.unlink(missing_ok=True)
            front_idx[topics[argmin]] += 1

    # Remove the leftover images.
    for topic in topics:
        for old_frame_idx in range(front_idx[topic], timestamps_df.shape[0]):
            old_path = get_image_path(images_base_path, topic, old_frame_idx)
            old_path.unlink(missing_ok=True)

    # Store the synchronized timestamps.
    synced_timestamp_df = pd.DataFrame(synced_timestamps)
    synced_timestamp_df.to_csv(images_base_path / "synced_timestamps.csv")
    return synced_timestamp_df


def extract_frame_metadata(synced_timestamps_df: pd.DataFrame, config: Config) -> int:
    """Extract all frame metadata and store to disk. Returns the number of found frames."""
    topics = synced_timestamps_df.columns
    num_frames = synced_timestamps_df.shape[0]

    out_lines = []
    for frame_idx in range(num_frames):
        timestamps = synced_timestamps_df.iloc[frame_idx]
        cams_list = []
        for camera_idx, topic in enumerate(topics):
            path = get_image_path(pathlib.Path("images"), topic, frame_idx)
            cams_list.append(
                {
                    "id": camera_idx,
                    "filename": str(path),
                    "timestamp": int(timestamps.iloc[camera_idx]),
                }
            )
        out_lines.append({"frame_id": frame_idx, "cams": cams_list})

    with (config.output_path / "frame_metadata.jsonl").open("w") as f:
        for out_line in out_lines:
            json.dump(out_line, f)
            f.write("\n")

    return num_frames


def run_image_extraction(
    reader: highlevel.AnyReader,
    config: Config,
) -> int:
    """Run the image extraction pipeline."""
    timestamps_df = run_executor(
        reader=reader,
        topics=config.image_topics,
        width=config.output_width,
        height=config.output_height,
        pixel_format=config.output_format,
        images_base_path=config.output_path / "images",
        num_workers=config.num_workers,
    )
    synced_timestamps_df = synchronize_images(
        timestamps_df=timestamps_df,
        images_base_path=config.output_path / "images",
        sync_threshold_ns=config.sync_threshold_ns,
    )
    num_frames = extract_frame_metadata(synced_timestamps_df, config)
    logging.info(f"Number of synced frames: {num_frames}")
    if num_frames == 0:
        counts = {topic: len(timestamps_df[topic].dropna()) for topic in timestamps_df.columns}
        counts_str = ", ".join(f"{t}={n}" for t, n in counts.items())
        logging.error(
            f"No synchronized frames found (pairs=0, per-topic counts: {counts_str}). "
            "Check that image topics are correct and timestamps overlap within "
            f"sync_threshold_ns={config.sync_threshold_ns}."
        )
        sys.exit(1)
    return num_frames


def get_distortion_model(
    distortion_model: str, distortion_params: np.ndarray
) -> tuple[edex.DistortionModel, np.ndarray]:
    """Map ROS camera distortion metadata to an EDEX distortion model."""
    assert (
        distortion_model in DISTORTION_MODEL_ROS2EDEX
    ), f"Unrecognized distortion model: '{distortion_model}'"

    if np.all(distortion_params == 0):
        logging.info("All distortion parameters are zero. Using pinhole model.")
        return edex.DistortionModel.PINHOLE, np.array([], dtype=np.float32)

    edex_model = DISTORTION_MODEL_ROS2EDEX[distortion_model]
    if edex_model == edex.DistortionModel.BROWN5K:
        # equidistant and rational_polynomial already match the edex order, plumb_bob does not.
        assert (
            len(distortion_params) == 5
        ), f"plumb_bob needs 5 coefficients, got {len(distortion_params)}"
        distortion_params = np.asarray(distortion_params)[_PLUMB_BOB_TO_BROWN5K]
    return edex_model, distortion_params


def get_camera_intrinsics(
    camera_msg: Any,
    config: Config,
) -> edex.Intrinsics:
    """Get the camera intrinsics needed for the edex file."""
    distortion_model, distortion_params = get_distortion_model(
        camera_msg.distortion_model, camera_msg.d
    )

    width_ratio = config.output_width / camera_msg.width if config.output_width else 1.0
    height_ratio = (
        config.output_height / camera_msg.height if config.output_height else 1.0
    )
    if width_ratio != height_ratio:
        logging.warning(
            "The resized images do not have the same aspect ratio. This may lead "
            "to incorrect results."
        )

    sx, sy = int(width_ratio * camera_msg.width), int(height_ratio * camera_msg.height)

    # Focal length and principal point of the raw camera.
    #     [fx  0 cx]
    # K = [ 0 fy cy]
    #     [ 0  0  1]
    fx, fy = width_ratio * camera_msg.k[0], height_ratio * camera_msg.k[4]
    cx, cy = width_ratio * camera_msg.k[2], height_ratio * camera_msg.k[5]

    projection = camera_msg.p.reshape(3, 4).copy()
    projection[0, :] *= width_ratio
    projection[1, :] *= height_ratio

    rectification = camera_msg.r.reshape(3, 3).copy()

    return edex.Intrinsics(
        distortion_model=distortion_model,
        distortion_params=distortion_params,
        focal=np.array([fx, fy], dtype=np.float32),
        principal=np.array([cx, cy], dtype=np.float32),
        resolution=np.array([sx, sy], dtype=np.int32),
        projection=projection,
        rectification=rectification,
    )


def extract_edex_metadata(
    reader: highlevel.AnyReader,
    config: Config,
    num_frames: int,
):
    """Create the partial edex metadata file with only intrinsics."""
    camera_info_msgs = get_first_message(reader, config.camera_info_topics)
    cameras_metadata = []
    for msg in camera_info_msgs:
        camera = edex.Camera(
            intrinsics=get_camera_intrinsics(msg, config),
            transform=None,
        )
        cameras_metadata.append(camera)

    sequence_paths = [
        get_image_path(pathlib.Path("images"), topic, 0)
        for topic in config.image_topics
    ]

    edex_header = edex.EDEXHeader(
        version="0.9",
        frame_start=0,
        frame_end=num_frames,
        cameras=cameras_metadata,
    )
    edex_body = edex.EDEXBody(
        frame_metadata="frame_metadata.jsonl",
        sequence=sequence_paths,
    )
    edex_metadata = edex.EDEXMetadata(header=edex_header, body=edex_body)

    edex_metadata_path = config.output_path / "edex"
    logging.info(
        f"Writing partial edex metadata with intrinsics to '{edex_metadata_path}'."
    )
    edex_metadata.write(edex_metadata_path)


def extract_images(config: Config):
    """Extract the synchronized images and camera intrinsics from the rosbag."""
    # Create output path.
    shutil.rmtree(config.output_path / "images", ignore_errors=True)
    (config.output_path / "edex").unlink(missing_ok=True)
    (config.output_path / "frame_metadata.jsonl").unlink(missing_ok=True)
    config.output_path.mkdir(parents=True, exist_ok=True)

    with open_rosbag_reader(config.rosbag_path, config.ros_distribution) as reader:
        log_rosbag_info(reader)

        # Do some quick checks that all the required data is present in the rosbag.
        # If not we ignore the inexistent topics.
        bag_topics = [c.topic for c in reader.connections]
        image_topics = []
        camera_info_topics = []
        for image_topic, info_topic in zip(
            config.image_topics, config.camera_info_topics
        ):
            if image_topic not in bag_topics:
                logging.warning(
                    f"Could not find topic '{image_topic}' in rosbag. Ignoring it."
                )
            elif info_topic not in bag_topics:
                logging.warning(
                    f"Could not find topic '{info_topic}' in rosbag. Ignoring it."
                )
            else:
                image_topics.append(image_topic)
                camera_info_topics.append(info_topic)

        config.image_topics = image_topics
        config.camera_info_topics = camera_info_topics

        # Extract the images and create the edex metadata file with only intrinsics.
        num_frames = run_image_extraction(reader, config)
        extract_edex_metadata(reader, config, num_frames)

    logging.info(f"Finished extracting images to '{config.output_path}'.")
