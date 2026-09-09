# bag2edex — Convert ROS 2 Bag to EDEX

Converts a ROS 2 bag into an EDEX sequence directory (images + `edex` + `frame_metadata.jsonl` + optional `imu.jsonl`) that the `tracker` tool can consume directly.

Uses the `cuvslam-tools` Python package (`tools/python_tools/`). Supports all ROS 2 distros (Foxy through Jazzy/Kilted), multiple cameras, config-driven topic/frame configuration, optional image resizing, and automatic extrinsic extraction from `/tf_static`.

---

## Install

```bash
cd tools/python_tools
./create_env.sh
source .env/bin/activate
```

Or manually:

```bash
cd tools/python_tools
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

---

## Configuration

Create a YAML config file (or use one of the provided presets in `tools/python_tools/cuvslam_tools/bag2edex/configs/`):

```yaml
# Topic names for CameraInfo messages
camera_info_topics:
  - /camera/infra1/camera_info
  - /camera/infra2/camera_info

# Topic names for image messages (same order as camera_info_topics)
image_topics:
  - /camera/infra1/image_rect_raw
  - /camera/infra2/image_rect_raw

# Rig frame for extrinsic extraction (run rosbag_extract_urdf to see available frames)
rig_frame: camera_link

# IMU topic (optional — omit for VO-only mode)
imu_topic: /camera/imu

# ROS distribution of the bag (foxy, humble, iron, jazzy, kilted — default: humble)
ros_distribution: humble

# Optional: resize output images
# output_width: 960
# output_height: 600
# output_format: png

# Optional: timestamp sync tolerance in nanoseconds (default: 1ms)
# sync_threshold_ns: 1000000

# Optional: number of parallel workers (default: number of CPU cores)
# num_workers: 8
```

### Provided sensor configs

| Config | Sensor |
|---|---|
| `configs/realsense.yaml` | RealSense (configurable topics) |
| `configs/nova_hawk.yaml` | NVIDIA Nova + HAWK stereo |
| `configs/oak6.yaml` | OAK-6 camera |

---

## Usage

**Extract a full EDEX dataset:**

```bash
rosbag_extract_edex \
    -c tools/python_tools/cuvslam_tools/bag2edex/configs/realsense.yaml \
    -r path/to/bag_folder \
    -o path/to/edex_folder
```

**List available TF frames** (to find the right `rig_frame`):

```bash
rosbag_extract_urdf \
    -r path/to/bag_folder \
    -o /tmp/urdf_out \
    -d humble
```

**Extract images only** (without extrinsics):

```bash
rosbag_extract_images \
    -c tools/python_tools/cuvslam_tools/bag2edex/configs/realsense.yaml \
    -r path/to/bag_folder \
    -o path/to/output_folder
```

**Extract videos only** (H.264 encoded bags):

```bash
rosbag_extract_videos \
    -c tools/python_tools/cuvslam_tools/bag2edex/configs/realsense.yaml \
    -r path/to/bag_folder \
    -o path/to/output_folder
```

### CLI flag reference

All commands (`rosbag_extract_edex`, `rosbag_extract_images`, `rosbag_extract_videos`) share:

| Flag | Short | Description |
|---|---|---|
| `--config_path` | `-c` | Path to YAML config file (required) |
| `--rosbag_path` | `-r` | Path to input rosbag directory (required; overrides `rosbag_path` in config) |
| `--output_path` | `-o` | Path for output directory (required; overrides `output_path` in config) |

`rosbag_extract_urdf` does not take a config file:

| Flag | Short | Description |
|---|---|---|
| `--rosbag_path` | `-r` | Path to input rosbag directory (required) |
| `--output_path` | `-o` | Output directory for the URDF file (required) |
| `--ros_distribution` | `-d` | ROS distribution (default: `humble`) |

---

## What it produces

| File | Contents |
|---|---|
| `edex` | Camera intrinsics, extrinsics, IMU transform, sequence frame list |
| `images/<topic>/NNNNN.png` | Extracted camera frames (one subdirectory per topic) |
| `frame_metadata.jsonl` | Per-frame metadata with filenames and timestamps |
| `imu.jsonl` | IMU samples (if `imu_topic` is set in config) |
| `robot.urdf` | URDF extracted from `/tf_static` |

---

## Troubleshooting

**`AssertionError: Failed to parse <type>`** — set `ros_distribution: jazzy` (or the correct distro) in the config. The `rosbags` library uses distro-specific type stores to parse message definitions.

If setting the correct `ros_distribution` does not fix it, the bag may have been recorded with a Jazzy rosbag2 recorder that leaves `type_description_hash` blank in the SQLite schema (all types in the `message_definitions` table have an empty `type_description_hash` and some have an empty `encoding`). rosbags 0.11.x unconditionally asserts the stored hash matches the computed one, so blank hashes always fail. Workaround — patch the installed rosbags in the venv:

```python
# Run once after activating the venv:
import pathlib, sys

candidates = [p for p in pathlib.Path(sys.prefix).rglob('storage_sqlite3.py')
              if 'rosbags' in str(p)]
assert candidates, 'rosbags not found in venv'
p = candidates[0]
text = p.read_text()

# 1. Skip types with empty encoding
text = text.replace(
    "            for typ in msgtypes:\n"
    "                assert typ['encoding'] == 'ros2msg'\n",
    "            for typ in msgtypes:\n"
    "                if not typ['encoding']:\n"
    "                    continue\n"
    "                assert typ['encoding'] == 'ros2msg'\n",
)
# 2. Skip hash check when digest is empty
text = text.replace(
    "                assert typ['digest'] == store.hash_rihs01(\n"
    "                    typ['name'],\n"
    "                ), f'Failed to parse {typ[\"name\"]}'\n",
    "                if typ['digest']:\n"
    "                    assert typ['digest'] == store.hash_rihs01(\n"
    "                        typ['name'],\n"
    "                    ), f'Failed to parse {typ[\"name\"]}'\n",
)
# 3. Guard get_msgdef against empty encoding
text = text.replace(
    "            if msgtype := next((x for x in msgtypes if x['name'] == name), None):\n"
    "                return MessageDefinition(fmtmap[msgtype['encoding']], msgtype['msgdef'])\n",
    "            if msgtype := next((x for x in msgtypes if x['name'] == name), None):\n"
    "                if not msgtype['encoding']:\n"
    "                    return MessageDefinition(MessageDefinitionFormat.NONE, '')\n"
    "                return MessageDefinition(fmtmap[msgtype['encoding']], msgtype['msgdef'])\n",
)

p.write_text(text)
print('patched', p)
```

**`KeyError: "Unknown frame 'camera_imu_optical_frame'"`** — the `frame_id` in the bag's IMU messages does not exist in the `/tf_static` tree. This happens when the bag was recorded without a combined IMU frame (e.g. RealSense bags that publish separate gyro/accel frames). Always run `inspect_rosbag.py --vio` on the bag first to see which frames are in `/tf_static`, then set `imu_frame` explicitly in the config to the matching optical frame (e.g. `imu_frame: camera_gyro_optical_frame` for a RealSense D455).

**`Rig frame 'X' not found`** — run `rosbag_extract_urdf` to list all TF frames in the bag, then update `rig_frame` in the config.

**Missing topics** — the tool warns and skips topics not present in the bag. Check topic names with `ros2 bag info <bag_folder>`.

**Wrong extrinsics (zero stereo baseline)** — if the output EDEX has zero translation for all cameras, the likely cause is that the `camera_info` messages for the non-reference cameras have the wrong `header.frame_id`. This is a known RealSense recording bug where `/camera/infra2/camera_info` publishes `frame_id: camera_infra1_optical_frame` instead of `camera_infra2_optical_frame`. The tool uses this frame_id to look up the TF chain, so both cameras resolve to the same frame and produce an identity (zero-baseline) extrinsic. Fix: explicitly set `camera_optical_frames` in the config to override the message frame_ids:

```yaml
camera_optical_frames:
  - camera_infra1_optical_frame
  - camera_infra2_optical_frame
```

To detect this before converting, read the first message of each `camera_info` topic and check `header.frame_id`:

```python
from rosbags import highlevel
from cuvslam_tools.bag2edex import get_typestore_from_ros_distribution, get_first_message
import pathlib

bag_path = pathlib.Path('/path/to/bag')
topics = ['/camera/infra1/camera_info', '/camera/infra2/camera_info']
with highlevel.AnyReader(paths=[bag_path],
        default_typestore=get_typestore_from_ros_distribution('humble')) as reader:
    for topic, msg in zip(topics, get_first_message(reader, topics)):
        print(topic, '->', msg.header.frame_id)
```

If any topic prints a frame_id belonging to a different camera, set `camera_optical_frames` explicitly.

**Wrong extrinsics (other causes)** — extrinsics are derived from `/tf_static` in the bag. Verify the TF tree is complete and the `rig_frame` is correct.

**Wrong intrinsics** — intrinsics are read from `camera_info` messages in the bag. Verify the published `camera_info` matches the actual camera before converting.

**Mismatched frame counts** — only synchronized frame pairs (within `sync_threshold_ns`) are saved. Increase `sync_threshold_ns` in the config if too many frames are dropped.

**No synchronized frames found** — the tool exits with an error listing per-topic frame counts. Check that image topics are correct and timestamps overlap within `sync_threshold_ns`.

**Missing IMU data** — if the bag has no IMU topic, `imu.jsonl` will be absent. The EDEX will still work for VO-only mode (`multicamera`) but not inertial mode.
