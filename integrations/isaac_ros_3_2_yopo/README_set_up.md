# Isaac ROS 3.2 外部 IMU 包装层补丁

中文日常运行与后续扩展手册：[`STARTUP_RUNBOOK.zh-CN.md`](STARTUP_RUNBOOK.zh-CN.md)

外部飞控 IMU 运行包：[`isaac_ros_yopo_bringup/README.md`](isaac_ros_yopo_bringup/README.md)

本目录提供 YOPO 集成所使用的 Isaac ROS Visual SLAM 包装层树外补丁。本集成不会修改、
构建或链接本仓库中的 cuVSLAM 实现；尤其是，`core17` 不属于本集成的一部分。

“树外（out-of-tree）”描述的是补丁的维护归属：该补丁维护在本 fork 中，而不是 NVIDIA
仓库中。应用补丁后，NVIDIA 源码检出目录中会产生一处已经审核的受 Git 跟踪修改。
这是 NVIDIA 源码通常只读策略中经过批准的唯一例外；不会在 NVIDIA 仓库中创建提交。

运行时仍使用 Isaac ROS 3.2 随附的 NVIDIA `libcuvslam.so`。本补丁只修改
[`COMPATIBILITY.md`](COMPATIBILITY.md) 中列出的精确版本所对应的开源 ROS 包装层。

## 为什么需要此补丁

Isaac ROS Visual SLAM 3.2 接收的 IMU 与图像时间戳以纳秒为单位，但未打补丁的包装层
存在两个相互独立的时间戳缺陷：

1. 包装层会计算每条 IMU 消息的时间戳（`imu_ts`），却将当前图像时间戳
   （`latest_ts`）传给 `CUVSLAM_RegisterImuMeasurement`。当两帧图像之间到达多条
   IMU 样本时，这些样本会以相同时间戳注册，而不是使用各自严格递增的时间戳。
2. `MessageStreamSequencer` 接收纳秒时间戳，而它的图像和 IMU 抖动阈值是以毫秒表示的
   ROS 参数。未打补丁的构造函数直接传入这些阈值，没有进行单位换算。

因此，本补丁将 `imu_ts` 传给 SDK，并将两个序列器阈值从毫秒转换为纳秒。两项修改
缺一不可，能够消除包装层中的两个阻塞问题，但仅凭这些修改还不能建立完整的外部 IMU
数据链路。

补丁还会在跟踪器成功初始化的日志中嵌入固定字符串
`ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1`。运行时启动包会在启动前检查已安装的
`libvisual_slam_node.so` 是否包含该 marker，并在运行期间等待同一 marker。这样既能
拒绝未打补丁的 overlay，也能拒绝源码虽已打补丁但尚未重新构建的情况。

SDK 契约已依据 Isaac ROS 3.2 容器中的以下头文件完成核验：

```text
/opt/ros/humble/share/isaac_ros_nitros/cuvslam/include/cuvslam.h
```

该头文件要求传给 `CUVSLAM_RegisterImuMeasurement` 的时间戳以纳秒为单位，并且必须
始终递增。

## 在容器中仅获取本集成

在 Isaac ROS 开发容器内执行以下命令：

```bash
cd /workspaces/isaac_ros-dev/src

git clone \
  --depth 1 \
  --branch u5-4/fcu-imu-cuvslam-integration \
  --filter=blob:none \
  --sparse \
  https://github.com/u5-4/cuVSLAM.git \
  cuvslam-yopo-adapter

git -C cuvslam-yopo-adapter sparse-checkout set \
  integrations/isaac_ros_3_2_yopo
```

以下三个导出路径仅在当前 shell 中有效。每次打开新的容器终端后都需要重新设置：

```bash
export VSLAM_SOURCE=/workspaces/isaac_ros-dev/src/isaac_ros_visual_slam
export YOPO_ADAPTER=/workspaces/isaac_ros-dev/src/cuvslam-yopo-adapter/integrations/isaac_ros_3_2_yopo
export VSLAM_PATCH="$YOPO_ADAPTER/patches/isaac_ros_visual_slam_v3_2_15_imu_timestamp.patch"
```

## 验证并应用补丁

验证脚本要求 NVIDIA 源码检出目录处于干净状态。脚本会应用补丁、检查两项修正、核验
已安装的 NITROS SDK 软件包及头文件，并在退出前反向撤销补丁：

```bash
"$YOPO_ADAPTER/scripts/verify_visual_slam_patch.sh" "$VSLAM_SOURCE"
```

`--source-only` 会跳过已安装 SDK 的检查，仅用于离线开发补丁。Jetson 验收时不得使用
该选项。

为实际构建应用补丁：

```bash
test -f /workspaces/isaac_ros-dev/install/setup.bash

# 修改或重新构建库之前，先停止正在运行的 Visual SLAM launch。
git -C "$VSLAM_SOURCE" apply --check \
  "$VSLAM_PATCH"

git -C "$VSLAM_SOURCE" apply \
  "$VSLAM_PATCH"
```

确认预期补丁是源码中唯一受 Git 跟踪的修改，然后仅重新构建受影响的 ROS 软件包。以下
按软件包选择的构建命令假定现有 Isaac ROS 工作空间 overlay 已经成功完成过构建：

```bash
cd /workspaces/isaac_ros-dev
set +u
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash

git -C "$VSLAM_SOURCE" diff --check
test "$(git -C "$VSLAM_SOURCE" status --short --untracked-files=no)" = \
  " M isaac_ros_visual_slam/src/impl/visual_slam_impl.cpp"
git -C "$VSLAM_SOURCE" apply --reverse --check "$VSLAM_PATCH"

colcon build \
  --packages-select isaac_ros_visual_slam \
  --cmake-args -DBUILD_TESTING=OFF

source /workspaces/isaac_ros-dev/install/setup.bash
test "$(ros2 pkg prefix isaac_ros_visual_slam)" = \
  /workspaces/isaac_ros-dev/install/isaac_ros_visual_slam

grep -aFq \
  ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1 \
  /workspaces/isaac_ros-dev/install/isaac_ros_visual_slam/lib/libvisual_slam_node.so
```

构建完成后重新启动 Visual SLAM launch，并确认其软件包前缀解析为
`/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam`。

确认打过补丁的包装层构建成功后，再构建运行时启动包：

```bash
cd /workspaces/isaac_ros-dev
colcon build --packages-select isaac_ros_yopo_bringup
source /workspaces/isaac_ros-dev/install/setup.bash
ros2 pkg prefix isaac_ros_yopo_bringup
```

软件包前缀必须解析到 `/workspaces/isaac_ros-dev/install/` 下。运行命令以及宿主机与
容器的执行边界记录在
[`STARTUP_RUNBOOK.zh-CN.md`](STARTUP_RUNBOOK.zh-CN.md) 第 8 节中。

克隆内容、补丁、构建结果和 `install/` 输出都位于宿主机绑定挂载的工作空间内，因此会
持久保存在宿主机上，但不会嵌入 `isaac_ros_dev-aarch64:latest` 镜像。制作生产镜像是
后续单独的任务。

## 更新本集成

当前补丁仍应用在 NVIDIA 源码上时，绝不能拉取新版适配器。应先停止 Visual SLAM，
使用当前版本的补丁文件反向撤销当前补丁，再依次执行拉取、验证、应用新补丁、重新构建
和重启：

```bash
git -C "$VSLAM_SOURCE" apply --reverse --check "$VSLAM_PATCH"
git -C "$VSLAM_SOURCE" apply --reverse "$VSLAM_PATCH"

git -C /workspaces/isaac_ros-dev/src/cuvslam-yopo-adapter pull --ff-only

"$YOPO_ADAPTER/scripts/verify_visual_slam_patch.sh" "$VSLAM_SOURCE"
git -C "$VSLAM_SOURCE" apply --check "$VSLAM_PATCH"
git -C "$VSLAM_SOURCE" apply "$VSLAM_PATCH"

cd /workspaces/isaac_ros-dev
set +u
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
colcon build --packages-select isaac_ros_visual_slam --cmake-args -DBUILD_TESTING=OFF
source /workspaces/isaac_ros-dev/install/setup.bash
```

### 从仅含补丁的分支一次性迁移

运行时分支在包装层补丁中增加了二进制 marker。如果 Jetson 当前应用的是早期
`u5-4/isaac-ros-3.2-yopo-adapter` 分支中的补丁，必须趁旧补丁文件仍在当前检出内容中，
在 fetch 或切换分支**之前**反向撤销它：

```bash
export VSLAM_SOURCE=/workspaces/isaac_ros-dev/src/isaac_ros_visual_slam
export ADAPTER_REPO=/workspaces/isaac_ros-dev/src/cuvslam-yopo-adapter
export OLD_PATCH="$ADAPTER_REPO/integrations/isaac_ros_3_2_yopo/patches/isaac_ros_visual_slam_v3_2_15_imu_timestamp.patch"

git -C "$VSLAM_SOURCE" apply --reverse --check "$OLD_PATCH"
git -C "$VSLAM_SOURCE" apply --reverse "$OLD_PATCH"

git -C "$ADAPTER_REPO" fetch origin \
  u5-4/fcu-imu-cuvslam-integration
git -C "$ADAPTER_REPO" switch -c \
  u5-4/fcu-imu-cuvslam-integration \
  --track origin/u5-4/fcu-imu-cuvslam-integration
```

随后根据新分支重新设置 `YOPO_ADAPTER`/`VSLAM_PATCH`，运行验证脚本，应用新补丁，并
重新构建 `isaac_ros_visual_slam` 和 `isaac_ros_yopo_bringup`。不得使用新的 marker
补丁反向撤销旧的双区块补丁；二者的补丁上下文本来就不同。

## 回滚

首先停止 Visual SLAM。反向撤销源码补丁不会替换已经安装或已经加载的库，因此必须
重新构建并重启：

```bash
git -C "$VSLAM_SOURCE" apply --reverse --check "$VSLAM_PATCH"
git -C "$VSLAM_SOURCE" apply --reverse "$VSLAM_PATCH"

cd /workspaces/isaac_ros-dev
set +u
set -eo pipefail
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash
colcon build --packages-select isaac_ros_visual_slam --cmake-args -DBUILD_TESTING=OFF
source /workspaces/isaac_ros-dev/install/setup.bash
```

只有在此次构建成功后，才能重新启动 Visual SLAM。

## 验收边界

构建成功只能证明包装层能够编译和链接。在接受外部飞控 IMU 融合方案之前，必须在
运行时验证以下事项：

- 飞控 IMU 话题使用预期的时钟，且时间戳严格递增；
- 重复、非单调、跨时钟域和过期的飞控时间戳会由 `aligned_fcu_imu_relay` 拒绝并计数，
  而不会转发给此包装层；
- Visual SLAM 节点恰好具有预期的飞控 `sensor_msgs/msg/Imu` 订阅或重映射，并且不再
  使用 D435i 的 `/camera/imu`；
- SDK 注册之前，已经按照文档规定的符号约定应用标定得到的相机与飞控固定时钟偏移；
- 所有 IMU 样本均使用预期坐标轴，且标定变换方向已经转换为文档规定的 ROS TF 链；
  对于所选的校正后双目输入，Kalibr 结果应被视为 `T_Crect0_I`，而不是原始光学相机
  变换；
- 噪声密度与随机游走数值来自选定的飞控 IMU，并通过一个可追溯的 Allan YAML 使用
  Isaac ROS 参数所要求的单位，而不是使用相互独立的 CLI 数值或 D435i/默认数值；
- Visual SLAM 日志中没有 IMU 注册失败记录；
- 使用本补丁后新录制的数据通过跟踪与时间戳检查。

此前 10 分钟的视觉跟踪测试使用的是 D435i 板载 IMU，且测试时间早于本补丁。该测试
不能证明飞控 IMU 测量已经以正确时间戳完成路由、标定和注册。

随附的验证脚本只检查源码和 SDK 契约，并不执行融合行为测试。生产验收仍需在 Jetson
上重新构建软件包，并使用新的飞控 IMU rosbag 或运行时数据完成验证。
