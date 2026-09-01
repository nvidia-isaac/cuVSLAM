# Dataset Guides

Detailed setup for each public dataset supported by cuVSLAM examples.

## Table of Contents
- [KITTI](#kitti)
- [EuRoC MAV](#euroc-mav)
- [TUM RGB-D](#tum-rgb-d)
- [Multi-Camera Datasets](#multi-camera-datasets)
- [Multisensor](#multisensor)
- [Distortion Models](#distortion-models)
- [EDEX File Format](#edex-file-format)
- [Converting ROS Bags](#converting-ros-bags)

---

## KITTI

**What:** Stereo grayscale benchmark for visual odometry (22 GB).

**Download:** http://www.cvlibs.net/datasets/kitti/eval_odometry.php (requires registration)

**Setup:**
```bash
cd examples/kitti
unzip data_odometry_gray.zip
# Produces: dataset/sequences/00..20/image_0/*.png, image_1/*.png, calib.txt, times.txt
```

**Run stereo odometry:**
```bash
python3 track_kitti.py
```

**Run SLAM (mapping + loop closure + localization):**
```bash
python3 track_kitti_slam.py
# Outputs: trajectory_tum.txt, map/data.mdb
```

**Localization from arbitrary frame:** After mapping, edit `track_kitti_slam.py` to set `IDX = 700` and enable the localization path. cuVSLAM matches visual features in the loaded map near the initial pose hint.

**Dynamic masks with PyTorch:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # or cu130
pip install transformers==5.2.0
python3 track_kitti_masks.py
```
Uses NVIDIA Segformer to segment cars in real-time and exclude them from feature detection.

**Performance tip:** Use `Odometry.MulticameraMode.Performance` and provide masks only for left-camera images to reduce compute overhead.

---

## EuRoC MAV

**What:** Stereo + IMU dataset from a micro aerial vehicle.

**Download:**
1. Go to https://doi.org/10.3929/ethz-b-000690084
2. Download "Machine Hall Datasets" (ZIP, ~12 GB)
3. Extract `MH_01_easy/mav0/` to `examples/euroc/dataset/mav0/`

**Copy recalibrated parameters (recommended for best accuracy):**
```bash
cd examples/euroc
cp sensor_cam0.yaml dataset/mav0/cam0/sensor_recalibrated.yaml
cp sensor_cam1.yaml dataset/mav0/cam1/sensor_recalibrated.yaml
cp sensor_imu0.yaml dataset/mav0/imu0/sensor_recalibrated.yaml
```

**Run stereo-inertial odometry:**
```bash
python3 track_euroc.py
```

**Tracking modes available on EuRoC:**
- Stereo: `OdometryMode(0)` / `OdometryMode.Multicamera`
- Stereo-Inertial: `OdometryMode(1)` / `OdometryMode.Inertial` (default in example)
- Mono: `OdometryMode(3)` / `OdometryMode.Mono`

Change mode by editing `euroc_tracking_mode` in `track_euroc.py`.

**C++ version:**
```bash
# From repo root
cmake -S . -B build
cmake --build build --target track_euroc
./build/bin/track_euroc /path/to/euroc/mav0
```
With Rerun: `cmake -S . -B build -DUSE_RERUN=ON`

---

## TUM RGB-D

**What:** RGB-D dataset for mono-depth visual odometry.

**Download:**
```bash
cd examples/tum
mkdir -p dataset
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz -O dataset/fr3.tgz
tar -xzf dataset/fr3.tgz -C dataset && rm dataset/fr3.tgz
cp freiburg3_rig.yaml dataset/rgbd_dataset_freiburg3_long_office_household/
```

**Run mono-depth odometry:**
```bash
python3 track_tum.py
```

**Depth settings:**
- Images: grayscale or RGB, `uint8`
- Depth maps: `uint16`
- `depth_scale_factor`: maps pixel values → meters
- `depth_camera_id`: camera index providing the visual image for the depth map

**Masking for depth borders:**
```python
cam = cuvslam.Camera()
cam.border_top = 20
cam.border_bottom = 20
cam.border_left = 10
cam.border_right = 50
```
Depth quality degrades near edges — mask to avoid unreliable features.

---

## Multi-Camera Datasets

### Tartan Ground (6 stereo pairs)

```bash
cd examples/multicamera_edex
pip install tartanair    # x86_64 only; on Jetson, download on x86 and transfer
python3 download_tartan.py
python3 track_multicamera_tartan.py
```
~8 GB download, ~17 GB on disk. Delete `.zip` files after extraction to reclaim space.

### R2B Galileo (4 stereo pairs, from ROS bag)

Requires Isaac ROS Docker environment:
1. Download R2B rosbag from https://registry.ngc.nvidia.com/orgs/nvidia/teams/isaac/resources/r2bdataset2024/files
2. Extract EDEX using `isaac_ros_rosbag_utils`:
```bash
ros2 run isaac_ros_rosbag_utils extract_edex \
  --config_path config/edex_extraction_nova.yaml \
  --rosbag_path <rosbag_path> \
  --edex_path <output_edex_path>
```
3. Move extracted folder to `examples/multicamera_edex/dataset/r2b_galileo_edex`
4. Run: `python3 track_multicamera_r2b.py`

---

## Multisensor

The TartanGround example exercises multiple RGB-D cameras with an optional IMU:

```bash
cd examples/multisensor
pip install tartanair    # x86_64 only; on Jetson, download on x86 and transfer
python3 download_tartan.py
python3 track_multisensor_tartan.py
python3 track_multisensor_tartan.py --no-imu
```

Multisensor requires `USE_CUNLS=ON`, pinhole cameras, and at least one RGB-D camera or one overlapping camera pair.
Multisensor tracking is experimental and may be inaccurate or fail for some sensor configurations and scenes.

---

## Distortion Models

| Model | Coefficients | Enum |
|-------|-------------|------|
| Pinhole | 0 (no distortion) | `Distortion.Model.Pinhole` / `Model(0)` |
| Fisheye (equidistant) | 4: k1,k2,k3,k4 | `Distortion.Model.Fisheye` / `Model(1)` |
| Brown | 5: k1,k2,k3,p1,p2 | `Distortion.Model.Brown` / `Model(2)` |
| Polynomial | 8: k1,k2,p1,p2,k3,k4,k5,k6 | `Distortion.Model.Polynomial` / `Model(3)` |

Ensure correct number and order of coefficients. If tracking is poor with unrectified images, test with `OdometryMode.Mono` first — it produces smooth trajectories and accurate rotation when intrinsics are correct.

---

## EDEX File Format

EDEX (cuVSLAM Data EXchange) is the standard JSON scene file. It describes camera rig, image/depth sequences, optional IMU data, and can hold tracking output (poses, observations, landmarks).

Minimal stereo example:
```json
[{
  "version": "0.9",
  "frame_start": 1,
  "frame_end": 4541,
  "cameras": [
    {
      "intrinsics": {
        "distortion_model": "pinhole",
        "distortion_params": [],
        "focal": [718.856, 718.856],
        "principal": [607.193, 185.216],
        "size": [1241, 376]
      },
      "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    },
    {
      "intrinsics": { "...same..." },
      "transform": [[1,0,0,0.537],[0,1,0,0],[0,0,1,0]]
    }
  ]
},
{
  "fps": 10,
  "sequence": [["00/00.0.0001.png"], ["01/00.1.0001.png"]]
}]
```

---

## Converting ROS Bags

**bag2edex** (offline, frame-by-frame):
```bash
cd tools/ros/bag2edex
python3 -m venv .env && source .env/bin/activate
pip install -r requirements.txt
python3 bag_to_edex.py <rosbag_path> <output_edex_path>
```

**edex2bag** (reverse):
```bash
cd tools/ros/edex2bag
pip install -r requirements.txt
python3 edex_to_bag.py <edex_path> <rosbag_path>
```

**Isaac ROS extract_edex** (preferred for NVIDIA robot datasets):
```bash
ros2 run isaac_ros_rosbag_utils extract_edex \
  --config_path <yaml> --rosbag_path <bag> --edex_path <out>
```
