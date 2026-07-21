# D435i + PX4 IMU + Isaac ROS cuVSLAM 启动包

本 ROS 2 Humble 软件包把已完成的 factory-rectified 双目/飞控 IMU 标定结果接入 Isaac ROS Visual SLAM 3.2。当前标定被版本化为运行候选：内参、外参和时间偏移已经固定，但 PX4 Allan 噪声标定与独立重复性验收尚未完成。

完整的宿主机、容器、MAVROS、构建、启动、检查和停止顺序见父目录的 [`STARTUP_RUNBOOK.zh-CN.md`](../STARTUP_RUNBOOK.zh-CN.md) 第 8 节。

## 运行链路

```text
Jetson 宿主机
  PX4 -> /dev/ttyTHS2:921600 -> MAVROS -> /mavros/imu/data_raw

Isaac ROS 容器
  /mavros/imu/data_raw
    -> aligned_fcu_imu_relay
       - 拒绝零、重复、倒退和错误 frame 的时间戳
       - 拒绝与宿主 system time 相差超过 250 ms 的陈旧/boot-time 时间戳
       - stamp 固定增加 1,737,987 ns
       - base_link 同轴重命名为 fcu_imu
       - 不旋转、不滤波、不插值、不修改 covariance
    -> /fcu/imu/data_raw_aligned
    -> Isaac ROS cuVSLAM

  D435i IR1/IR2 + CameraInfo + factory TF
    -> Isaac ROS cuVSLAM

  camera_infra1_optical_frame -> fcu_imu
    -> 已标定静态 TF
```

统一 launch 会启动一份且仅一份 RealSense 驱动、时间对齐节点、标定静态 TF、官方 cuVSLAM component 和运行健康监视节点。D435i 的 gyro/accel 被明确关闭，cuVSLAM 的 IMU 订阅只映射到 `/fcu/imu/data_raw_aligned`。

relay 在 15 秒内收不到首条可发布样本，或运行中 raw/aligned 流持续中断超过 2 秒时会以错误退出；统一 launch 随即停止相机、TF 和 cuVSLAM，避免留下表面存活但不再融合 IMU 的进程。

频率或最大时间间隔单次越界先报告 WARN；连续 3 个诊断周期仍不健康时升级为 ERROR 并退出。最大间隔门限为实测周期的 5 倍，约 `29.4 ms`，为已录制数据的 `16.592 ms` 峰值保留 Jetson 负载余量。`/clock` 或 `use_sim_time` 不会替代该 system-time 新鲜度门禁。

统一 launch 还会检查已安装 `libvisual_slam_node.so` 中的 patch marker，并等待 patched cuVSLAM 打印 tracker 成功初始化 marker。常驻健康监视节点严格比对两路 CameraInfo 的尺寸、模型和 `K/D/R/P`，禁止 D435i 自带 IMU 产生实际样本，并要求 `/visual_slam/tracking/odometry` 首帧及后续数据持续在线；任何合同失效都会非零退出并关闭整套 launch。

## 标定来源

版本化记录位于 [`config/d435i_243622070369_fcu_imu.yaml`](config/d435i_243622070369_fcu_imu.yaml)：

- D435i：序列号 `243622070369`，固件 `5.15.1.55`；
- 图像：原厂 rectified `640x360@90 Hz`；
- 原厂内参：由运行中的 RealSense `CameraInfo` 提供，YAML 中的 `K/D/R/P` 只用于审计；
- 飞控 IMU：`/mavros/imu/data_raw`，实测约 `170 Hz`；
- 时移：`t_aligned = t_imu_raw + 1,737,987 ns`；
- TF：`camera_infra1_optical_frame -> fcu_imu`，使用 `T_Crect0_I`。

右侧 RealSense `CameraInfo.header.frame_id` 可能错误复用左光学 frame。launch 已显式设置左右 `camera_optical_frames`，不依赖该错误消息头推断右相机坐标系。

## 标定状态门禁

当前 YAML 状态为 `runtime_candidate_pending_allan_and_independent_repeatability`。launch 默认只接受 `approved`；现阶段真机验证必须明确传入 `allow_candidate_calibration:=true`。`rejected` 状态永远不能被命令行覆盖。

这个开关只承认“操作者知道当前仍是候选标定”，不会跳过 CameraInfo、TF、时间戳、IMU 数据质量或运行新鲜度检查，也不会把未验证的 IMU 噪声参数变成已验证参数。

## IMU 噪声门禁

launch 不接受四个彼此独立的 CLI 噪声数值，而是要求一个无默认值的 `imu_noise_file`。该 YAML 必须同时记录：

- `validated` 状态；
- PX4 IMU 硬件/传输 ID；
- Allan 数据采样率；
- 方法名 `allan_deviation`；
- 源数据文件名与 SHA-256；
- 四项参数及精确单位。

填写入口为 [`config/px4_imu_noise_allan.template.yaml`](config/px4_imu_noise_allan.template.yaml)。模板中的所有 `REPLACE_WITH_...` 都必须替换，不能把模板本身传给 launch。

四项单位为：

- `gyro_noise_density`：`rad/(s*sqrt(Hz))`；
- `gyro_random_walk`：`rad/(s^2*sqrt(Hz))`；
- `accel_noise_density`：`m/(s^2*sqrt(Hz))`；
- `accel_random_walk`：`m/(s^3*sqrt(Hz))`。

现有 `seeker_imu.yaml` 中的 `0.09 / 0.05 / 0.06 / 0.001` 是 Kalibr 联合标定的输入假设，不是 Allan 标定输出。D435i 官方默认四元组也不属于 PX4 IMU。仓库中的 `config/px4_imu_noise_unvalidated.yaml` 明确记录了旧值并设置 `validated: false`；launch 默认拒绝它，`allow_unvalidated_imu_noise:=true` 只用于验证接线，不能作为导航验收结果。真实 Allan YAML 引用的源数据文件必须实际存在，且记录的 SHA-256 必须与该文件字节完全一致。

## 本地逻辑测试

在本软件包目录执行：

```bash
python3 -m unittest discover -s test -v
```

测试固定了整数纳秒运算、ROS Time 边界、严格递增状态机、消息透明转发、标定矩阵/四元数一致性，以及 launch 的关键订阅和噪声门禁合同。

## Jetson 构建边界

本包不修改、构建或链接仓库中的 cuVSLAM 核心库。运行时仍使用 Isaac ROS 3.2 的 NVIDIA `libcuvslam.so`，并要求工作空间中的 `isaac_ros_visual_slam` 已应用父目录的 v3.2-15 IMU 时间戳补丁。

Jetson 上只构建本包：

```bash
cd /workspaces/isaac_ros-dev
set +u
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/install/setup.bash

colcon build --packages-select isaac_ros_yopo_bringup
source /workspaces/isaac_ros-dev/install/setup.bash
```

编译成功只能证明 ROS 包结构和依赖成立。真机验收必须继续检查频率、frame、TF、时间戳、diagnostics、cuVSLAM 唯一 IMU 订阅、`vo_state` 和长时间运行稳定性。
