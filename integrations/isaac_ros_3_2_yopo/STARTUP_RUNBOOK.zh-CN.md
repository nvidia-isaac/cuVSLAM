# Jetson Isaac ROS / cuVSLAM / YOPO 系统启动手册

本文档是 Jetson 机载感知与规划链路的长期启动手册。每完成并验收一个阶段，就把该阶段的启动、验证和停止命令追加到这里，使后续使用者可以按顺序直接启动系统。

项目阶段、复选框、坐标系与 SO3 后续任务统一维护在 [`PROJECT_TASKFLOW.zh-CN.md`](PROJECT_TASKFLOW.zh-CN.md)。

当前文档覆盖：

- Isaac ROS 开发容器的首次创建、冷启动和重复进入；
- Intel RealSense D435i 正常运行节点的独立启动；
- NVIDIA Isaac ROS Visual SLAM（cuVSLAM）的独立启动；
- D435i 双红外与 PX4/MAVROS 飞控 IMU 的 odometry-only 统一启动；
- 相机、IMU、cuVSLAM 输出的运行检查；
- 正确停止顺序和常见故障；
- 飞控 IMU、状态适配器和 YOPO PASSIVE 的后续章节边界。

本文档不是标定录包手册。正常运行时不要启动 `d435_ir_calibration.launch.py` 或 `d435_fcu_imu_record.launch.py`。

## 当前状态

最后更新：2026-07-21

| 模块 | 当前状态 | 本文档是否提供启动命令 |
| --- | --- | --- |
| Isaac ROS 3.2 开发容器 | 已验证 | 是 |
| D435i + 内置 IMU 正常运行源 | 官方组合链路已验证；拆分命令已按同一源码参数整理 | 是 |
| cuVSLAM + D435i 内置 IMU | 官方组合链路已验证；拆分命令需在 Jetson 再次复核 | 是 |
| cuVSLAM + PX4/MAVROS 飞控 IMU | mapping-on 首次 Jetson 联合验收通过；odometry-only 后续版本待 Jetson A/B，Allan 为可选调优 | 是 |
| cuVSLAM 到 `/state/odom` 适配 | 尚未完成联合验收 | 否 |
| YOPO PASSIVE | 尚未完成联合验收 | 否 |
| YOPO 控制输出 | 禁止启用 | 否 |

固定环境：

| 项目 | 值 |
| --- | --- |
| Jetson 工作空间 | `$HOME/workspaces/isaac_ros_3_2` |
| 容器工作空间 | `/workspaces/isaac_ros-dev` |
| ROS 版本 | ROS 2 Humble |
| Isaac ROS Visual SLAM | `v3.2-15` / `e31f4cc1d41a329a01946e5fe63669f8b15da677` |
| ROS Domain | `42` |
| D435i 序列号 | `243622070369` |
| D435i 固件 | `5.15.1.55` |
| 双目配置 | `640x360@90 Hz` |

## 当前推荐日常启动

当前已经完成真机验证的主路径是 NVIDIA 官方 RealSense + cuVSLAM 组合启动：

1. 在 `H1` 按 1.1 使用已有镜像创建并进入容器；
2. 在 `C1` 按 1.4 初始化 ROS 环境；
3. 在 `C1` 按第 2 节启动官方组合 launch，并保持终端运行；
4. 在 `H2` 按 1.2 进入同一个容器，在 `C2` 按第 5 节检查运行状态。

第 3、4 节提供“相机与 cuVSLAM 分终端启动”的候选命令。它们已按 NVIDIA 源码逐项核对，但必须先通过一次 Jetson 冒烟测试，才能提升为日常主路径。

第 8 节提供飞控 IMU 联合链路。mapping-on 首次真机联合验收已经通过；当前源码后续版本固定为 odometry-only，仍需按第 8.7 节完成 Jetson A/B 后才能替换上述恢复基线。当前 Kalibr 四项噪声权重已获项目运行批准；Allan 只作为独立噪声测量和后续调优手段。

## 执行位置与终端约定

本文统一使用以下标签。执行命令前必须先确认所在环境。

| 标签 | 执行位置 | 用途 |
| --- | --- | --- |
| `H1` | Jetson 宿主机终端 1 | 创建或启动容器 |
| `C1` | 由 H1 进入的主容器终端 | 启动并保持 RealSense 相机运行 |
| `H2` | Jetson 宿主机终端 2 | 进入已经运行的容器 |
| `C2` | 由 H2 进入的附加容器终端 | 启动并保持 cuVSLAM 运行 |
| `H3` | Jetson 宿主机终端 3 | 再次进入已经运行的容器 |
| `C3` | 由 H3 进入的附加容器终端 | 检查话题、频率和 RViz |

宿主机提示符通常类似：

```text
nvidia@tegra-ubuntu:~$
```

容器提示符通常类似：

```text
admin@tegra-ubuntu:/workspaces/isaac_ros-dev$
```

> [!IMPORTANT]
> 不要在同一个代码块中混用宿主机和容器命令。看到 `admin@...:/workspaces/isaac_ros-dev$` 后，才表示已经进入容器。

## 安全边界

- 同一时刻只能运行一份 `realsense2_camera_node`。重复启动会导致 `RS2_USB_STATUS_BUSY`。
- 本手册的正常运行链路与标定录包链路互斥。
- 拆分启动模式下，不要再运行 `isaac_ros_visual_slam_realsense.launch.py`；该官方启动文件会再次启动一份 RealSense 驱动。
- 所有终端必须使用相同的 `ROS_DOMAIN_ID=42`。
- 每个新容器终端都必须重新加载 ROS 环境。
- 在 `source /opt/ros/humble/setup.bash` 前先执行 `set +u`。
- 当前不得启动 YOPO 控制输出、PX4 OFFBOARD、解锁或起飞命令。

## 1. 创建或进入 Isaac ROS 容器

### 1.1 使用已有镜像创建并进入新容器

执行位置：`H1`，Jetson 宿主机。

这是当前日常冷启动命令。适用情况：容器当前没有运行，并且已验证的开发镜像已经存在。

```bash
conda deactivate 2>/dev/null || true
unset PYTHONHOME

export ISAAC_ROS_WS="$HOME/workspaces/isaac_ros_3_2"
export ROS_DOMAIN_ID=42

cd "$ISAAC_ROS_WS/src/isaac_ros_common"
./scripts/run_dev.sh -b -d "$ISAAC_ROS_WS"
```

`-b` 是 `--skip_image_build`，含义是跳过镜像构建，不是构建镜像。该命令使用已有镜像创建一个新的临时容器并进入 `C1`，不是通过 `docker start` 重启旧容器。

工作空间通过 bind mount 保存在 Jetson 主机上。容器内 `/workspaces/isaac_ros-dev` 之外的临时修改不会自动保存到下一次新容器中。

### 1.2 从新终端进入已经运行的容器

执行位置：`H2` 或 `H3`，Jetson 宿主机。

保持 H1/C1 不退出，然后在新的宿主机终端执行：

```bash
conda deactivate 2>/dev/null || true
unset PYTHONHOME

export ISAAC_ROS_WS="$HOME/workspaces/isaac_ros_3_2"
export ROS_DOMAIN_ID=42

cd "$ISAAC_ROS_WS/src/isaac_ros_common"
./scripts/run_dev.sh -d "$ISAAC_ROS_WS"
```

正常复用时应先看到：

```text
Attaching to running container: ...
Docker workspace: /workspaces/isaac_ros-dev
```

`run_dev.sh` 会自动通过 `docker exec` 进入同一个容器，不会重新构建镜像，也不会创建第二个容器。如果第二个终端显示的是 `Launching Isaac ROS Dev container`，说明原容器不存在或容器名称配置已变化，不要继续把它当成 `C2/C3`。

`-d` 只在创建新容器时决定主机目录的挂载位置。容器已经运行时，脚本不会重新挂载工作空间。若要切换工作空间，必须先按第 6 节停止原容器，再使用新的 `-d` 创建容器。运行期间不要修改 `CONFIG_CONTAINER_NAME_SUFFIX`。

不要在正式启动流程中硬编码：

```text
docker exec ... isaac_ros_dev-aarch64-container ...
```

容器名可能受到 `CONFIG_CONTAINER_NAME_SUFFIX` 影响。统一使用 `run_dev.sh` 可以避免多数 `No such container` 问题。

### 1.3 完整镜像构建（维护操作，非日常启动）

当前完整 cuVSLAM 开发环境包含已经制备的依赖和工作空间构建结果。仅运行基础镜像构建流程不能保证复现全部环境，因此本节不是日常启动步骤。

需要维护镜像时，在 `H1` 执行：

```bash
conda deactivate 2>/dev/null || true
unset PYTHONHOME

export ISAAC_ROS_WS="$HOME/workspaces/isaac_ros_3_2"
export ROS_DOMAIN_ID=42

cd "$ISAAC_ROS_WS/src/isaac_ros_common"
./scripts/run_dev.sh -d "$ISAAC_ROS_WS"
```

只有在同名容器未运行，并且 `SKIP_DOCKER_BUILD` 与 `CONFIG_SKIP_IMAGE_BUILD` 均未启用时，不带 `-b` 才会执行镜像构建。配置可能来自 `isaac_ros_common/scripts/.isaac_ros_common-config` 或 `~/.isaac_ros_common-config`。

构建后必须重新验证 GPU、RealSense、NITROS、cuVSLAM 动态库、工作空间 overlay 和本手册第 2 节的完整链路。新机器的完整环境制备仍需单独安装/构建文档支持。

### 1.4 初始化每个容器终端

执行位置：每一个新打开的容器终端 `C1`、`C2`、`C3`。

```bash
set +u
set +e
set -o pipefail
export ROS_DOMAIN_ID=42

source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash

cd /workspaces/isaac_ros-dev
```

快速确认：

```bash
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
ros2 pkg prefix realsense2_camera
ros2 pkg prefix isaac_ros_visual_slam
test "$(ros2 pkg prefix isaac_ros_visual_slam)" = \
  /workspaces/isaac_ros-dev/install/isaac_ros_visual_slam
```

最后一项必须成功，确保当前使用的是工作空间 overlay，而不是错误的系统包。如果 `/workspaces/isaac_ros-dev/install/setup.bash` 不存在，或包前缀不匹配，应先停止，不要继续启动传感器。

## 2. 当前推荐：官方组合启动

这是当前已经完成 Jetson 真机验证的日常主路径。

执行位置：`C1`。

保持该终端持续运行：

```bash
ros2 launch isaac_ros_visual_slam \
  isaac_ros_visual_slam_realsense.launch.py
```

该命令会同时启动：

- 一份 `realsense2_camera_node`；
- D435i 双红外和内置 IMU；
- 一份 cuVSLAM 节点。

运行该命令前不得已有 `/camera/camera`，否则会争用 USB 设备。该官方基线不使用飞控 IMU，并且启动文件本身不提供相机序列号参数；连接多台 RealSense 时不能保证选择 `243622070369`。

需要检查运行状态时，在 `H2` 按 1.2 进入同一个容器，得到 `C2`，再按第 5 节执行检查。

## 3. 拆分启动候选：正常 D435i 相机

> [!WARNING]
> 第 3、4 节是按 NVIDIA 官方参数拆分出的候选路径，尚需在 Jetson 完成一次冒烟测试。当前日常运行优先使用第 2 节。

本节启动正常运行用的 D435i 数据源，不启动标定节点，不录包。参数与 NVIDIA Isaac ROS 3.2 的 RealSense Visual SLAM 基线一致：

- IR1、IR2：启用，`640x360@90 Hz`；
- D435i 陀螺仪、加速度计：启用，`200 Hz`；
- 合并 IMU：启用，发布 `/camera/imu`；
- 彩色、深度：关闭；
- 红外发射器：关闭。

执行位置：`C1`。

保持该终端持续运行：

```bash
ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __node:=camera \
  -r __ns:=/camera \
  -p serial_no:="'243622070369'" \
  -p enable_infra1:=true \
  -p enable_infra2:=true \
  -p enable_color:=false \
  -p enable_depth:=false \
  -p depth_module.emitter_enabled:=0 \
  -p depth_module.profile:="'640x360x90'" \
  -p enable_gyro:=true \
  -p enable_accel:=true \
  -p gyro_fps:=200 \
  -p accel_fps:=200 \
  -p unite_imu_method:=2
```

启动成功后应存在以下主要输入话题：

```text
/camera/infra1/image_rect_raw
/camera/infra2/image_rect_raw
/camera/infra1/camera_info
/camera/infra2/camera_info
/camera/imu
/tf_static
```

> [!NOTE]
> 这条命令是正常运行相机命令，不是 `image_bag_recorder` 中的标定相机命令。

## 4. 拆分启动候选：cuVSLAM

当前拆分启动命令使用 D435i 内置 IMU，仅用于复现 NVIDIA 官方 RealSense 基线。飞控 IMU 的受控运行入口已经在第 8 节单独实现；不能在本命令中把 `/camera/imu` 手工替换成 `/mavros/imu/data_raw`，否则会绕过时间偏移、frame、TF、数据质量和运行健康门禁。

### 4.1 进入第二个容器终端

在 `H2` 按 1.2 的方法进入同一个容器，得到 `C2`，然后按 1.4 初始化 ROS 环境。

### 4.2 启动 cuVSLAM 节点

执行位置：`C2`。

先确认独立可执行文件已安装：

```bash
ros2 pkg executables isaac_ros_visual_slam | \
  grep -F 'isaac_ros_visual_slam isaac_ros_visual_slam'
```

保持该终端持续运行：

```bash
ros2 run isaac_ros_visual_slam isaac_ros_visual_slam --ros-args \
  -r __node:=visual_slam_node \
  -p enable_image_denoising:=false \
  -p rectified_images:=true \
  -p num_cameras:=2 \
  -p enable_imu_fusion:=true \
  -p gyro_noise_density:=0.000244 \
  -p gyro_random_walk:=0.000019393 \
  -p accel_noise_density:=0.001862 \
  -p accel_random_walk:=0.003 \
  -p calibration_frequency:=200.0 \
  -p image_jitter_threshold_ms:=22.0 \
  -p base_frame:=camera_link \
  -p imu_frame:=camera_gyro_optical_frame \
  -p camera_optical_frames:="['camera_infra1_optical_frame','camera_infra2_optical_frame']" \
  -p enable_localization_n_mapping:=true \
  -p enable_slam_visualization:=true \
  -p enable_landmarks_view:=true \
  -p enable_observations_view:=true \
  -r visual_slam/image_0:=/camera/infra1/image_rect_raw \
  -r visual_slam/camera_info_0:=/camera/infra1/camera_info \
  -r visual_slam/image_1:=/camera/infra2/image_rect_raw \
  -r visual_slam/camera_info_1:=/camera/infra2/camera_info \
  -r visual_slam/imu:=/camera/imu
```

该命令复现 NVIDIA `isaac_ros_visual_slam_realsense.launch.py` 中 VisualSlamNode 的节点名、参数与话题映射，但不会再次启动 RealSense 驱动。独立可执行文件使用自身的多线程 executor，而官方 launch 使用 component container，因此仍需进行 Jetson 冒烟测试。

## 5. 检查运行状态

### 5.1 进入检查终端

官方组合模式：在 `H2` 按 1.2 进入同一个容器，得到 `C2`。

拆分候选模式：由于 `C2` 正在运行 cuVSLAM，在 `H3` 按 1.2 进入同一个容器，得到 `C3`。

进入后按 1.4 初始化 ROS 环境。

### 5.2 检查节点和发布端

执行位置：官方组合模式使用 `C2`；拆分候选模式使用 `C3`。

```bash
echo "========== Nodes =========="
ros2 node list | sort

CAMERA_NODE_COUNT="$(
  ros2 node list 2>/dev/null |
  grep -cx '/camera/camera' || true
)"
echo "camera_node_count=$CAMERA_NODE_COUNT"

echo "========== cuVSLAM node =========="
ros2 node info /visual_slam_node
ros2 param get /visual_slam_node camera_optical_frames

echo "========== Camera inputs =========="
ros2 topic info -v /camera/infra1/image_rect_raw
ros2 topic info -v /camera/infra2/image_rect_raw
ros2 topic info -v /camera/infra1/camera_info
ros2 topic info -v /camera/infra2/camera_info
ros2 topic info -v /camera/imu
ros2 topic info -v /tf_static

echo "========== cuVSLAM outputs =========="
ros2 topic info -v /visual_slam/status
ros2 topic info -v /visual_slam/tracking/odometry
```

必须确认：

- `camera_node_count=1`；
- 左右 Image、两路 CameraInfo 和 `/camera/imu` 均为一个发布端；
- 上述五个输入话题都能看到 cuVSLAM 的一个订阅端；
- `/visual_slam_node` 的订阅已映射到预期的 `/camera/...` 话题；
- `camera_optical_frames` 正确解析为左右两个光学坐标系；
- `/visual_slam/status` 和 `/visual_slam/tracking/odometry` 有发布端。

### 5.3 检查 CameraInfo 和 TF

执行位置：官方组合模式使用 `C2`；拆分候选模式使用 `C3`。

```bash
timeout 8s ros2 topic echo \
  /camera/infra1/camera_info --once || true

timeout 8s ros2 topic echo \
  /camera/infra2/camera_info --once || true

timeout --signal=INT 5s ros2 run tf2_ros tf2_echo \
  camera_link camera_infra1_optical_frame || true

timeout --signal=INT 5s ros2 run tf2_ros tf2_echo \
  camera_link camera_infra2_optical_frame || true

timeout --signal=INT 5s ros2 run tf2_ros tf2_echo \
  camera_link camera_gyro_optical_frame || true
```

两路 CameraInfo 必须包含非零 `K/P`，三个 TF 查询必须得到有效变换。RealSense ROS 4.51.1 可能让右 CameraInfo 消息头误用左光学 frame；cuVSLAM 命令已通过 `camera_optical_frames` 显式固定左右物理坐标系，但 TF 本身仍必须存在。

### 5.4 检查频率

执行位置：官方组合模式使用 `C2`；拆分候选模式使用 `C3`。

```bash
timeout --signal=INT 8s ros2 topic hz /camera/infra1/image_rect_raw || true
timeout --signal=INT 8s ros2 topic hz /camera/infra2/image_rect_raw || true
timeout --signal=INT 8s ros2 topic hz /camera/imu || true
timeout --signal=INT 8s ros2 topic hz /visual_slam/tracking/odometry || true
```

参考值：

| 话题 | 参考频率 |
| --- | --- |
| `/camera/infra1/image_rect_raw` | 约 `89.8 Hz` |
| `/camera/infra2/image_rect_raw` | 约 `89.8 Hz` |
| `/camera/imu` | 约 `199.8 Hz` |
| `/visual_slam/tracking/odometry` | 约 `89.8 Hz` |

### 5.5 读取状态和里程计样本

执行位置：官方组合模式使用 `C2`；拆分候选模式使用 `C3`。

```bash
timeout 8s ros2 topic echo /visual_slam/status --once || true
timeout 8s ros2 topic echo /visual_slam/tracking/odometry --once || true
```

正常状态必须满足：

- `/visual_slam/status` 中 `vo_state: 1`；
- 里程计时间戳非零并持续更新；
- `header.frame_id` 为 `odom`；
- `child_frame_id` 为 `camera_link`。

只有图像、CameraInfo、TF 和 IMU 持续有效，并且 cuVSLAM 完成初始化后，里程计才会正常输出。

### 5.6 使用 RViz

执行位置：官方组合模式使用 `C2`；拆分候选模式使用 `C3`。需要可用的图形显示环境。

```bash
RVIZ_CONFIG="$(ros2 pkg prefix --share isaac_ros_visual_slam)/rviz/realsense.cfg.rviz"
rviz2 -d "$RVIZ_CONFIG"
```

### 5.7 使用 rqt 查看左右红外图像

执行位置：已初始化 ROS 环境且有图形显示的容器终端。左右 cuVSLAM 输入分别为：

```text
/camera/infra1/image_rect_raw
/camera/infra2/image_rect_raw
```

可以在两个终端各启动一个 Image View：

```bash
ros2 run rqt_image_view rqt_image_view \
  /camera/infra1/image_rect_raw
```

```bash
ros2 run rqt_image_view rqt_image_view \
  /camera/infra2/image_rect_raw
```

两幅图应同时、连续更新，均为 `640x360` rectified 红外图。rqt 只用于短时检查；正式运行和性能测试前关闭两个窗口，避免额外订阅、显示与拷贝负载。

## 6. 正确停止顺序

### 6.1 官方组合模式

1. 在持有官方组合 launch 的 `C1` 中按一次 `Ctrl+C`。
2. 等待 RealSense 与 cuVSLAM 节点一起退出并释放 USB 设备。
3. 关闭 RViz，在 `C2` 等附加容器终端执行 `exit`。
4. 确认不再需要容器后，最后在主容器终端 `C1` 执行 `exit`。

### 6.2 拆分候选模式

1. 在 `C2` 中按一次 `Ctrl+C`，等待 cuVSLAM 完全退出。
2. 在 `C1` 中按一次 `Ctrl+C`，等待 RealSense 节点释放 USB 设备。
3. 关闭 RViz，然后在 `C3`、`C2` 等附加容器终端执行 `exit`。
4. 确认不再需要容器后，最后在主容器终端 `C1` 执行 `exit`。

### 6.3 退出主容器前检查残留进程

执行位置：`C1`。

```bash
pgrep -af realsense2_camera_node || true
pgrep -af isaac_ros_visual_slam || true
```

确认没有残留的相机或 cuVSLAM 进程后，再退出主容器 shell。

`run_dev.sh` 使用 `docker run --rm` 创建主容器。退出主容器 shell 后，容器会停止并自动删除；所有附加容器终端和容器内进程也会结束。退出附加终端本身不会停止主容器。

## 7. 常见故障

### `No such container`

可能原因：

- 固定使用了过期或带不同后缀的容器名；
- 主容器终端 `C1` 已退出，`--rm` 已删除容器；
- H1 的 SSH 会话或终端被关闭；
- Docker 服务已经重启；
- H1 与 H2 读取了不同的 `CONFIG_CONTAINER_NAME_SUFFIX`；
- 只有名称相似的容器，触发了 `run_dev.sh` 的 Docker 名称过滤边界。

检查位置：Jetson 宿主机。

```bash
docker ps -a \
  --filter 'name=isaac_ros_dev-' \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

如果目标容器仍在运行，按 1.2 进入。如果容器不存在但已验证镜像仍在，按 1.1 创建新容器。如果 `run_dev.sh` 先显示 `Attaching to running container`，随后又报 `No such container`，重点检查是否只有名称相似而非完全相同的容器。

### `No built image found`

原因：使用 `-b` 跳过了镜像构建，但当前配置所对应的镜像不存在；修改容器后缀或镜像 key 也可能改变所需镜像名。

检查位置：Jetson 宿主机。

```bash
docker image ls 'isaac_ros_dev-*'
env | grep -E 'SKIP_DOCKER_BUILD|ISAAC_ROS_WS' || true
grep -H -E 'CONFIG_(IMAGE_KEY|SKIP_IMAGE_BUILD|CONTAINER_NAME_SUFFIX)' \
  "$HOME/.isaac_ros_common-config" \
  "$HOME/workspaces/isaac_ros_3_2/src/isaac_ros_common/scripts/.isaac_ros_common-config" \
  2>/dev/null || true
```

不要为了消除该报错而盲目重建并覆盖当前已验证镜像。确认配置后，按项目的镜像制备记录执行；只有明确进行环境维护时才使用 1.3。

### `RS2_USB_STATUS_BUSY` 或 `failed to claim usb interface`

原因：另一份 RealSense 节点已经占用 D435i。

检查位置：任一已经初始化的容器终端。

```bash
ros2 node list | grep -E '^/camera/camera$' || true
pgrep -af realsense2_camera_node || true
```

停止旧节点并确认 USB 释放后，只选择一种模式：运行第 2 节官方组合 launch，或者运行第 3、4 节拆分候选；不得同时运行。

### `Package '...' not found`

原因：新的容器终端尚未加载工作空间 overlay。

处理：重新执行 1.4 的完整环境初始化命令。

### `setup.bash` 因未定义变量退出

原因：在 source ROS 环境前启用了 `set -u`。

处理：先执行 `set +u`，再 source 两个 `setup.bash`。

### 相机有图像但没有里程计

依次检查：

```bash
ros2 topic info /camera/infra1/image_rect_raw
ros2 topic info /camera/infra2/image_rect_raw
ros2 topic info /camera/infra1/camera_info
ros2 topic info /camera/infra2/camera_info
ros2 topic info /camera/imu
ros2 topic echo /visual_slam/status --once
```

同时查看 `C2` 中是否存在 TF、时间戳、IMU 注册或 cuVSLAM 初始化错误。

## 8. 飞控 IMU odometry-only 运行链路

> [!CAUTION]
> mapping-on 首次真机联合验收已经通过。当前后续版本将统一入口固定为 odometry-only，待 Jetson A/B 复验；坐标系、状态适配和系统级安全尚未完成，当前仍禁止接入飞行控制。现有 `seeker_imu.yaml` 四项值是项目批准的 Kalibr 权重，不是 PX4 Allan 输出；Allan 不作为第一版部署的强制前置。

本链路固定为：

```text
[Jetson 宿主机 H4]
PX4 -> /dev/ttyTHS2:921600 -> MAVROS -> /mavros/imu/data_raw

[Isaac ROS 容器 C1]
/mavros/imu/data_raw
  -> aligned_fcu_imu_relay
  -> /fcu/imu/data_raw_aligned
  -> cuVSLAM

D435i IR1 + IR2 + CameraInfo + factory TF -> cuVSLAM
camera_infra1_optical_frame -> fcu_imu -> 标定静态 TF
```

宿主机只负责 UART/MAVROS；容器统一 launch 负责唯一一份 RealSense 驱动、时间对齐节点、标定 TF 和 cuVSLAM。该模式与第 2 节官方组合 launch、以及第 3、4 节拆分链路互斥。

### 8.1 固定标定合同

版本化配置：

```text
isaac_ros_yopo_bringup/config/d435i_243622070369_fcu_imu.yaml
```

运行时固定：

- 相机原厂 `K/D/R/P` 由 RealSense 两路 `CameraInfo` 提供，不硬编码进 cuVSLAM；
- 左右物理 frame 显式为 `camera_infra1_optical_frame`、`camera_infra2_optical_frame`；
- 原始飞控 IMU 保留 `/mavros/imu/data_raw` 与 `base_link` 不变；
- 对齐节点只把时间戳增加 `1,737,987 ns`，并把同一数值坐标轴命名为 `fcu_imu`；
- 原始 IMU stamp 与宿主 system time 的接收残差必须不超过 `250 ms`，PX4 boot-time 会被拒绝；
- 对齐节点不旋转、不滤波、不插值、不重采样，也不修改任何测量或 covariance；
- 静态 TF 发布 `camera_infra1_optical_frame -> fcu_imu`；
- cuVSLAM 只订阅 `/fcu/imu/data_raw_aligned`；
- `calibration_frequency=170.0` 使用实测频率，不使用 MAVLink 请求值 `200 Hz`。

这里有意选择“原厂 rectified CameraInfo + 使用同一相机模型求得的联合外参”。另一组自由估计内参的 Kalibr 结果为左目约 `[324.5671, 325.0486, 322.0116, 184.0997]`、右目约 `[324.9723, 325.4113, 321.3716, 184.1345]`，并估计了非零 radtan 畸变和 `49.9168 mm` 基线；它不能直接与当前 RealSense 已校正图像发布的原厂 `K/D/R/P` 混用。若以后选择该模型，必须同时实现并验收新的图像校正与 CameraInfo 发布链路，再重新生成匹配的联合外参。

两组联合结果的 IMU/相机旋转接近，但相机原点在 IMU 坐标系中的位置相差约 `6.83 mm`。该差异来自不同相机模型，不能解释为 factory-rectified 标定失败；当前 factory-rectified 结果已经通过真机联合运行并获项目批准。独立重复标定仍可用于后续量化可重复性。

Kalibr 时移定义为 `t_imu = t_cam + timeshift_cam_imu`，其中 `timeshift_cam_imu=-0.001737986760008108 s`。保持相机时间戳不变时，正确实现就是给原始 IMU 时间戳增加 `1,737,987 ns`，不能再次使用负号。

### 8.2 `H4`：启动并保持 MAVROS

执行位置：Jetson 宿主机新终端 `H4`，不在容器内。

```bash
conda deactivate 2>/dev/null || true
unset PYTHONHOME
set +u
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42

FCU_DEV=/dev/ttyTHS2
FCU_BAUD=921600

if [ ! -c "$FCU_DEV" ]; then
  echo "[STOP] $FCU_DEV does not exist"
  exit 1
fi

if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
  echo "[STOP] current user is not in dialout"
  exit 1
fi

if fuser "$FCU_DEV" >/dev/null 2>&1; then
  echo "[STOP] $FCU_DEV is already in use"
  fuser -v "$FCU_DEV"
  exit 1
fi

ros2 launch mavros px4.launch \
  fcu_url:="${FCU_DEV}:${FCU_BAUD}"
```

保持 `H4` 持续运行。不要传空的 `gcs_url:=""`；Humble 会把它解析成 malformed launch argument。正常日志必须出现 `Got HEARTBEAT, connected. FCU: PX4 Autopilot`。

### 8.3 `H5`：请求 HIGHRES_IMU 并检查原始流

执行位置：Jetson 宿主机另一个终端 `H5`，不在容器内。

```bash
conda deactivate 2>/dev/null || true
unset PYTHONHOME
set +u
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42

STATE="$(
  timeout 10s ros2 topic echo \
    /mavros/state mavros_msgs/msg/State --once
)"
printf '%s\n' "$STATE"
if ! printf '%s\n' "$STATE" | grep -q '^connected: true$'; then
  echo "[STOP] MAVROS is not connected to PX4"
  exit 1
fi

ros2 service call \
  /mavros/set_message_interval \
  mavros_msgs/srv/MessageInterval \
  "{message_id: 105, message_rate: 200.0}"

IMU_TYPE="$(ros2 topic type /mavros/imu/data_raw 2>/dev/null || true)"
if [ "$IMU_TYPE" != sensor_msgs/msg/Imu ]; then
  echo "[STOP] unexpected or missing FCU IMU type: $IMU_TYPE"
  exit 1
fi

ros2 topic info -v /mavros/imu/data_raw

timeout 8s ros2 topic echo \
  /mavros/imu/data_raw sensor_msgs/msg/Imu \
  --once --qos-reliability best_effort

timeout --signal=INT 10s ros2 topic hz \
  /mavros/imu/data_raw || true

timeout 8s ros2 topic echo \
  /mavros/timesync_status mavros_msgs/msg/TimesyncStatus \
  --once --qos-reliability best_effort
```

必须确认 service 返回 `success=True`、IMU 约 `170 Hz`、stamp 非零且递增、静止时加速度模长约 `9.8 m/s^2`，并且 timesync 持续稳定。请求 `200 Hz` 不代表实际一定达到 `200 Hz`。

### 8.4 `C1`：构建和核对运行包

执行位置：已按 1.1、1.4 进入并初始化的容器主终端 `C1`。

先确认 NVIDIA wrapper 的 v3.2-15 时间戳补丁已应用，且新包能被 colcon 发现：

```bash
export VSLAM_SOURCE=/workspaces/isaac_ros-dev/src/isaac_ros_visual_slam
export YOPO_ADAPTER=/workspaces/isaac_ros-dev/src/cuvslam-yopo-adapter/integrations/isaac_ros_3_2_yopo
export VSLAM_PATCH="$YOPO_ADAPTER/patches/isaac_ros_visual_slam_v3_2_15_imu_timestamp.patch"

if ! git -C "$VSLAM_SOURCE" apply --reverse --check "$VSLAM_PATCH"; then
  echo "[STOP] expected Isaac ROS Visual SLAM timestamp patch is not applied"
  exit 1
fi

if ! colcon list | grep -q '^isaac_ros_yopo_bringup[[:space:]]'; then
  echo "[STOP] isaac_ros_yopo_bringup is not visible to colcon"
  exit 1
fi

cd /workspaces/isaac_ros-dev
colcon build --packages-select isaac_ros_yopo_bringup
source /workspaces/isaac_ros-dev/install/setup.bash

if [ "$(ros2 pkg prefix isaac_ros_visual_slam)" != \
  /workspaces/isaac_ros-dev/install/isaac_ros_visual_slam ]; then
  echo "[STOP] patched visual_slam workspace overlay is not active"
  exit 1
fi

if ! grep -aFq \
  ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1 \
  /workspaces/isaac_ros-dev/install/isaac_ros_visual_slam/lib/libvisual_slam_node.so; then
  echo "[STOP] installed Visual SLAM binary was not rebuilt with the required patch"
  exit 1
fi

ros2 pkg prefix isaac_ros_yopo_bringup
ros2 launch isaac_ros_yopo_bringup \
  d435i_fcu_imu_cuvslam.launch.py --show-args
```

源码补丁本身的首次应用和 `isaac_ros_visual_slam` 重编译仍按 [`README_set_up.md`](README_set_up.md) 执行。仅编译本 bringup 包不会自动重编译 NVIDIA wrapper。

### 8.5 `C1`：正常启动 odometry-only 链路

执行位置：容器主终端 `C1`。运行前必须停止所有旧 RealSense 和 Visual SLAM 节点，并确保 `H4` 中 MAVROS 正常连接。

```bash
ros2 launch isaac_ros_yopo_bringup \
  d435i_fcu_imu_cuvslam.launch.py
```

默认标定记录和 IMU noise schema v2 均已获项目批准。noise 文件仍保留 `validated: false`，准确表示当前四项值来自 Kalibr 输入权重而不是独立 Allan 结果；这与 `project_status: approved` 相互独立，因此不需要临时放行参数。

launch 必须打印 `Operating mode: odometry-only`，并说明 mapping、loop closure、ground constraints、内部可视化和 `map -> odom` TF 均已关闭。该终端会同时保持 RealSense、aligned IMU、标定 TF、cuVSLAM 和运行健康监视节点。不要再单独启动相机节点或官方组合 launch。

### 8.6 `C1`：可选使用独立 Allan 参数

需要进行可选 Allan A/B 时，从 `px4_imu_noise_allan.template.yaml` 生成 schema v2 文件。`validated: true` 只表示来源文件和 SHA-256 通过核验，不会自动授予运行批准；完成项目审查后还必须把 `project_status` 从 `candidate` 改为 `approved`。

```bash
export PX4_IMU_NOISE_FILE=/absolute/path/to/px4_imu_allan.yaml

if [ -z "${PX4_IMU_NOISE_FILE:-}" ] || [ ! -s "$PX4_IMU_NOISE_FILE" ]; then
  echo "[STOP] approved PX4 Allan YAML does not exist"
else
  ros2 launch isaac_ros_yopo_bringup \
    d435i_fcu_imu_cuvslam.launch.py \
    imu_noise_file:="$PX4_IMU_NOISE_FILE"
fi
```

旧 schema v1 仅为兼容：`validated: true` 沿用旧版运行批准语义；`validated: false` 视为候选，不能启动。正常运行始终使用 schema v2。

### 8.7 `C2`：运行验收

在 `H2` 按 1.2 进入同一个容器，得到 `C2`，再按 1.4 初始化 ROS 环境。

先检查 D435i 内置 IMU 确实关闭：

```bash
ros2 param get /camera/camera enable_gyro
ros2 param get /camera/camera enable_accel
ros2 param get /camera/camera unite_imu_method

TOPICS="$(ros2 topic list)"
if printf '%s\n' "$TOPICS" | grep -qx /camera/imu; then
  if timeout 6s ros2 topic echo \
    /camera/imu sensor_msgs/msg/Imu \
    --once --qos-reliability best_effort \
    >/tmp/d435_runtime_imu_probe.log 2>&1
  then
    cat /tmp/d435_runtime_imu_probe.log
    echo "[STOP] D435i IMU produced a forbidden sample"
    exit 1
  else
    RC=$?
    if [ "$RC" -eq 124 ]; then
      echo "[PASS] /camera/imu exists but remained silent"
    else
      cat /tmp/d435_runtime_imu_probe.log
      echo "[STOP] D435i IMU silence probe failed with rc=$RC"
      exit 1
    fi
  fi
else
  echo "[PASS] /camera/imu is absent"
fi
```

三个参数必须依次为 `False`、`False`、`0`。RealSense ROS 4.51.1 可能保留一个没有实际样本的 `/camera/imu` publisher，因此必须检查 6 秒内是否收到消息，不能只看话题是否存在。

检查 raw/aligned IMU、参数与 TF：

```bash
timeout 8s ros2 topic echo \
  /camera/infra1/camera_info sensor_msgs/msg/CameraInfo --once
timeout 8s ros2 topic echo \
  /camera/infra2/camera_info sensor_msgs/msg/CameraInfo --once

ros2 topic info -v /mavros/imu/data_raw
ros2 topic info -v /fcu/imu/data_raw_aligned

ros2 param get /aligned_fcu_imu_relay imu_to_camera_offset_ns
ros2 param get /aligned_fcu_imu_relay output_frame_id
ros2 param get /visual_slam_node imu_frame
ros2 param get /visual_slam_node calibration_frequency

for parameter in \
  enable_localization_n_mapping \
  enable_ground_constraint_in_odometry \
  enable_ground_constraint_in_slam \
  enable_slam_visualization \
  enable_landmarks_view \
  enable_observations_view \
  publish_map_to_odom_tf
do
  echo "--- $parameter"
  ros2 param get /visual_slam_node "$parameter"
done

timeout 8s ros2 topic echo \
  /fcu/imu/data_raw_aligned sensor_msgs/msg/Imu \
  --once --qos-reliability best_effort

timeout --signal=INT 5s ros2 run tf2_ros tf2_echo \
  camera_infra1_optical_frame fcu_imu || true
```

RealSense 启动日志必须报告序列号 `243622070369` 和固件 `5.15.1.55`。两路 CameraInfo 必须为 `640x360`，并与版本化 YAML 中的 factory `K/D/R/P` 一致；右 CameraInfo 的消息头允许存在已知的左 frame 复用问题。还必须看到 offset `1737987`、两个 IMU frame 均为 `fcu_imu`、频率参数 `170.0`。七个 odometry-only 参数必须全部返回 `False`。TF 平移应接近 `[0.02736293, 0.05285189, -0.06214162] m`，四元数 xyzw 应接近 `[0.50184877, -0.50942523, 0.49064963, 0.49789224]`。

检查 cuVSLAM 只有一个正确 IMU 输入：

```bash
VSLAM_INFO="$(ros2 node info /visual_slam_node)"
printf '%s\n' "$VSLAM_INFO"

if ! printf '%s\n' "$VSLAM_INFO" | \
  grep -q '/fcu/imu/data_raw_aligned: sensor_msgs/msg/Imu'; then
  echo "[STOP] cuVSLAM is not subscribed to the aligned FCU IMU"
  exit 1
fi

if printf '%s\n' "$VSLAM_INFO" | \
  grep -Eq '/camera/imu|/mavros/imu/data_raw'; then
  echo "[STOP] cuVSLAM has a forbidden raw or D435i IMU subscription"
  exit 1
fi
```

检查频率、diagnostics 和里程计：

```bash
timeout --signal=INT 10s ros2 topic hz /camera/infra1/image_rect_raw || true
timeout --signal=INT 10s ros2 topic hz /camera/infra2/image_rect_raw || true
timeout --signal=INT 10s ros2 topic hz /camera/infra1/camera_info || true
timeout --signal=INT 10s ros2 topic hz /camera/infra2/camera_info || true
timeout --signal=INT 10s ros2 topic hz /fcu/imu/data_raw_aligned || true
timeout --signal=INT 10s ros2 topic hz /visual_slam/tracking/odometry || true

timeout 8s ros2 topic echo \
  /diagnostics diagnostic_msgs/msg/DiagnosticArray \
  --once --filter \
  "m.status[0].name == '/aligned_fcu_imu_relay: aligned FCU IMU'"

timeout 8s ros2 topic echo \
  /diagnostics diagnostic_msgs/msg/DiagnosticArray \
  --once --filter \
  "m.status[0].name == '/d435i_cuvslam_runtime_health_monitor: calibrated runtime'"

timeout 8s ros2 topic echo /visual_slam/status --once || true
timeout 8s ros2 topic echo /visual_slam/tracking/odometry --once || true
```

参考值为左右图像和 CameraInfo 约 `89.8 Hz`、aligned IMU 约 `170 Hz`、里程计约 `89.8 Hz`。两个 diagnostics 的 `level` 必须为 `0`；ROS 2 YAML 可能把该字节显示成 `"\0"`。aligned relay 的 `last_clock_residual_ms` 绝对值必须小于 `250`，累计 `zero_stamp`、`invalid_stamp`、`duplicate`、`nonmonotonic`、`frame_mismatch`、`clock_domain_mismatch`、`aligned_out_of_range`、`nonfinite_measurement` 都应为 `0`；单次 IMU 间隔门限约为 `29.4 ms`，持续频率或间隔异常仍会在三个诊断周期后退出。运行健康监视 diagnostics 也必须为 `OK`，odometry 必须使用 `odom -> camera_link`、非零严格递增时间戳和有限数值，`vo_state` 应为 `1`。持续 duplicate、跨时间域、非有限测量、异常频率、CameraInfo 不匹配、D435i IMU 实际出数或里程计中断时，相应守护节点必须非零退出并触发整套 launch 关闭。

### 8.8 停止顺序和验收边界

1. 在 `C1` 中按一次 `Ctrl+C`，等待健康监视、cuVSLAM、TF、aligned relay 和 RealSense 全部退出。
2. 在 `H4` 中按一次 `Ctrl+C`，等待 MAVROS 退出并释放 `/dev/ttyTHS2`。
3. 在 `C2` 等附加容器终端执行 `exit`，最后退出主容器终端。

源码与单元测试完成不等于真机生产验收。正式替换第 2 节日常主路径前，仍需：

- 可选：用 PX4 原始静止数据完成 Allan 标定并固化四项噪声参数及来源，用于独立噪声测量与调优；
- 可选：在相同 factory-rectified 模型下做一次独立重复联合标定，量化可重复性；当前约 `6.83 mm` 是不同相机模型结果的比较，不等同于标定失败；
- 在 Jetson 验证真实 ROS `Imu` 深拷贝、BEST_EFFORT QoS、拒绝计数和 output-stale diagnostics；
- 完成时间戳、轨迹质量、失跟恢复、资源占用和长时间稳定性测试。

## 9. cuVSLAM 状态适配器（待完成）

目标链路：

```text
/visual_slam/tracking/odometry + /visual_slam/status
  -> 状态与坐标适配器
  -> /state/odom
```

只有在 frame、位置、姿态、速度语义和 tracking-lost 门控完成真机验收后，才在本节加入正式启动命令。

## 10. YOPO PASSIVE（待联合验收）

本节当前不提供启动命令，避免把孤立可运行的 YOPO 命令误写成已经打通的完整系统入口。

加入命令前必须满足：

- `/state/odom` 的 frame 和速度语义已经验收；
- Depth 与 CameraInfo 的分辨率、单位、时间戳和话题契约已经固定；
- YOPO 配置明确设置 `output_enabled=false`；
- tracking lost、里程计超时或深度失效时，YOPO 不使用陈旧数据；
- 被动联合运行期间控制话题实际消息数为零；
- 完成长时间资源和稳定性测试。

完成后，本节按以下结构追加：

1. `[Jetson 宿主机]` YOPO 环境初始化；
2. YOPO PASSIVE 启动命令；
3. 输入、输出和零控制消息验证；
4. 正确停止顺序；
5. 常见故障。

## 11. 文档维护规则

后续每加入一个模块，都必须同时写明：

- 当前状态：已验证、待验证或禁止；
- 精确执行位置：Jetson 宿主机或 Isaac ROS 容器；
- 使用的终端编号；
- 完整环境初始化命令；
- 启动命令；
- 必须持续运行的终端；
- 输入输出话题和正常参考值；
- 验证命令；
- 停止顺序；
- 固定版本或提交；
- 已知限制和安全边界。

不得把尚未通过真机联合验收的命令写成可直接运行的正式步骤。

## 变更记录

| 日期 | 内容 |
| --- | --- |
| 2026-07-21 | 加入 D435i factory-rectified + PX4/MAVROS 外部 IMU + cuVSLAM 候选统一启动、噪声门禁、验收与停止流程。 |
| 2026-07-21 | 创建系统启动手册；加入容器生命周期、正常 D435i、cuVSLAM、验证、停止和后续 YOPO 章节边界。 |
| 2026-07-21 | 首次 Jetson 联合运行验收通过：双 IR/CameraInfo、aligned FCU IMU、标定 TF、唯一 IMU 订阅、两项 diagnostics、`vo_state=1` 与 `odom -> camera_link` 均满足合同，并连续运行约 17 分 48 秒；修正 `/camera/imu` 探测中的 `grep -q` 管道噪声。 |
| 2026-07-21 | 后续源码改为无临时放行参数的正常启动：项目批准与 Allan 来源状态分离，默认使用已批准 Kalibr 权重；统一 launch 固定 odometry-only，并关闭 cuVSLAM `map -> odom` TF；Jetson A/B 待完成。 |
