# D435i + PX4 IMU + Isaac ROS cuVSLAM 启动包

本 ROS 2 Humble 软件包把已完成并获项目运行批准的 factory-rectified 双目/飞控 IMU 标定结果接入 Isaac ROS Visual SLAM 3.2。内参、外参、时间偏移和第一版 IMU 噪声权重均已版本化；PX4 Allan 噪声测量与独立重复标定保留为可选增强，不是启动前置。

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

  D435i native Depth
    -> ROS 2 DDS planner input

  camera_infra1_optical_frame -> fcu_imu
    -> 已标定静态 TF
```

统一 launch 会启动一份且仅一份 RealSense 驱动、时间对齐节点、标定静态 TF、官方 cuVSLAM component 和运行健康监视节点。D435i 的 gyro/accel 被明确关闭，cuVSLAM 的 IMU 订阅只映射到 `/fcu/imu/data_raw_aligned`。原生 Depth 与红外双目共用同一 `640x360@90 Hz` profile，emitter 固定关闭；Depth 只作为规划输入，不改变 cuVSLAM 的 IR1/IR2 图像输入。

生产入口固定为 odometry-only：`enable_localization_n_mapping=false`，两类 ground constraint、三类内部可视化以及 cuVSLAM 的 `map -> odom` TF 发布均关闭。该入口不提供命令行开关重新启用这些功能；需要建图或调试可视化时应使用单独的调试 launch。

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

当前 YAML 状态为 `approved`。launch 只接受获项目批准的标定记录，`candidate` 或 `rejected` 不能由命令行覆盖。项目批准来自本机首次联合验收；它不会跳过 CameraInfo、TF、时间戳、IMU 数据质量或运行新鲜度检查。

## IMU 噪声门禁

launch 不接受四个彼此独立的 CLI 噪声数值，而是从一个版本化 `imu_noise_file` 读取完整记录。默认文件为 `config/px4_imu_noise_unvalidated.yaml`；文件名中的 `unvalidated` 仅表示未声称 Allan 来源验证，不表示未获项目运行批准。

schema v2 将两个概念明确分开：

- `project_status`：是否允许用于本项目运行；
- `validated`：是否具有通过文件和 SHA-256 核验的独立 Allan 来源。

当前默认记录是 `project_status: approved`、`validated: false`，方法仍诚实记录为 `kalibr_input_assumption`。因此它可用于第一版 cuVSLAM，但不冒充 Allan 结果。schema v1 仅为旧文件兼容：旧 `validated: true` 视为已批准，旧 `validated: false` 视为候选且不能启动。

该 YAML 还必须记录：

- PX4 IMU 硬件/传输 ID；
- 数据采样率；
- 方法名及来源文件；
- 四项参数及精确单位。

可选 Allan 结果的填写入口为 [`config/px4_imu_noise_allan.template.yaml`](config/px4_imu_noise_allan.template.yaml)。模板默认 `project_status: candidate`；完成来源核验和项目审查后才能改为 `approved`。所有 `REPLACE_WITH_...` 都必须替换，不能把模板本身传给 launch。

四项单位为：

- `gyro_noise_density`：`rad/(s*sqrt(Hz))`；
- `gyro_random_walk`：`rad/(s^2*sqrt(Hz))`；
- `accel_noise_density`：`m/(s^2*sqrt(Hz))`；
- `accel_random_walk`：`m/(s^3*sqrt(Hz))`。

现有 `seeker_imu.yaml` 中的 `0.09 / 0.05 / 0.06 / 0.001` 是 Kalibr 联合标定使用并经本次运行验收接受的权重，不是 Allan 标定输出。D435i 官方默认四元组也不属于 PX4 IMU。真实 Allan YAML 引用的源数据文件必须实际存在，且记录的 SHA-256 必须与该文件字节完全一致。

正常启动不需要临时放行参数：

```bash
ros2 launch isaac_ros_yopo_bringup d435i_fcu_imu_cuvslam.launch.py
```

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
