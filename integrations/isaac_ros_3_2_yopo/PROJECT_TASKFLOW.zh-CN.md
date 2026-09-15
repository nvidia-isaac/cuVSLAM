# YOPO / cuVSLAM / PX4 Jetson 项目任务流

> 桌面主文件：`C:\Users\10416\Desktop\agent.md`
> 仓库镜像：`integrations/isaac_ros_3_2_yopo/PROJECT_TASKFLOW.zh-CN.md`
> 最近更新：2026-07-21
> 当前阶段：外部飞控 IMU cuVSLAM 首次联合运行验收完成，进入稳定化、图像质量与 odometry-only 验证
> 安全状态：仅允许被动感知与规划验证；禁止 OFFBOARD、解锁、起飞和有效控制输出

## 0. 版本取舍与历史保留说明

本文件由两份资料合并而成：

1. 桌面旧任务流 `agent.md`，主体形成于 2026-07-17，包含环境审计、官方 D435i 基线、P-001 至 P-010、D-001/D-002、早期两阶段路线和后续补充记录；
2. 2026-07-21 的 `PROJECT_TASKFLOW.zh-CN.md`，包含外部 FCU IMU 标定、wrapper 修复、统一 launch、坐标系、YOPO PASSIVE 和 SO3 后续任务。

取舍原则不是“新版本覆盖并删除旧版本”，而是：

- 已经被新证据更新的旧结论，在下表中说明替代关系；
- 仍然成立的环境基线、问题记录和安全规则继续生效；
- 旧路线、旧分支和旧镜像继续作为回退、A/B 或审计记录；
- 全部旧文档原文保留在第 15 节历史快照中，但其中标注为“当前”的内容不再覆盖第 1 至 14 节的现行状态。

### 0.1 关键版本取舍

| 主题 | 旧文档状态/路线 | 当前生效选择 | 取舍原因与历史用途 |
| --- | --- | --- | --- |
| 项目范围 | 当前两阶段不包含 PX4、MAVROS 和控制器 | 已加入 PX4/MAVROS 外部 IMU；SO3 只进入参数与接口审计，不启用控制 | 用户在完成官方基线后明确扩展任务范围；旧的被动安全边界继续保留 |
| cuVSLAM 集成分支 | 计划 `u5-4/isaac-ros-3.2-yopo-adapter` | 当前 `u5-4/fcu-imu-cuvslam-integration@9ca7190`，本地后续提交待发布 | 旧分支 `4a41339` 保留为仅含 wrapper 时间戳补丁的历史基线；`9ca7190` 完成首次联合验收，后续提交收敛批准语义和 odometry-only |
| integration 路径白名单 | 早期只预留 CMake adapter 目录结构 | 仍只允许修改 `integrations/isaac_ros_3_2_yopo/`，但加入 Python bringup、标定 YAML、patch、测试和中文文档 | 所有新增内容仍与 core17 隔离；早期目录草案作为设计历史保留，不再限制具体 ROS 包构建类型 |
| Git 签署 | 旧规则要求后续提交使用 `git commit -s` | 已发布的 `9ca7190` 没有 `Signed-off-by`，不重写已推送历史；后续提交恢复 `-s`，除非仓库规则另有明确决定 | 如远端将来启用 DCO，再通过后续合规流程处理，禁止为补签而 force-push |
| NVIDIA wrapper 策略 | 官方源码保持只读，发现必须修改时停止决策 | 允许一个经过固定 commit、apply/reverse verifier 和 marker 约束的树外补丁 | 实测确认 wrapper 把图像时间戳用于 IMU 注册，并存在 jitter 单位错误；SDK 二进制仍不修改、不上传官方仓库 |
| cuVSLAM core17 | 不加入 Isaac ROS 3.2 构建 | 继续不构建、不链接、不运行 | 旧决策保持不变；Isaac ROS 3.2 继续使用配套 SDK ABI |
| IMU 路线 | 官方 D435i 内置 IMU fusion | 官方路线保留为 A/B 和回退；当前候选路线使用 PX4 FCU IMU | 官方路线已经完成约 10 分钟稳定基线；外部 IMU 更符合最终机载硬件设计，当前只把 Jetson 联合运行作为必需验收，Allan 为可选调优 |
| 相机模型 | D435i EEPROM 原厂内外参 | 运行时继续使用原厂 rectified `K/D/R/P` | 自由估计 Kalibr 内参作为对比保留；它不能与当前 RealSense rectified CameraInfo 直接混用 |
| 相机-FCU IMU 标定 | 未完成 | `T_Crect0_I` 与 `+1,737,987 ns` 已固化，Kalibr 联合标定已完成并在首次 Jetson 联合运行后转为项目 `approved` | `9ca7190` 的 runtime candidate 命名作为已验收历史保留；后续 schema 将项目批准与 Allan/重复性来源状态拆开 |
| IMU 噪声策略 | `9ca7190` 把 Allan 来源作为默认生产门禁 | 当前项目接受 Kalibr 配置中的四项噪声权重用于联合标定和第一版 cuVSLAM；Allan 改为可选增强 | Kalibr 可在近似权重下迭代轨迹、bias、外参和时移并收敛；它不会自动重新估计四个噪声系数，但这不构成当前部署阻塞 |
| 里程计参考点 | 官方 `base_frame=camera_link` | 当前联合冒烟暂时保持；最终目标改为 `base_link` | 旧相机参考输出可用于基线；接入规划/控制前必须补齐 `base_link -> fcu_imu` 并形成单父 TF 树 |
| 定位模式 | 官方 localization + mapping | mapping-on 联合冒烟已通过；后续源码已固定为 odometry-only，等待 Jetson A/B | 先隔离外部 IMU 链路风险，再关闭 mapping/回环、ground constraint、可视化和 `map -> odom` TF |
| Depth 与 emitter | emitter 策略、深度合同待验证 | 问题继续保留到 YOPO PASSIVE 阶段 | 当前外部 IMU launch 关闭 Depth；不能因为 VIO 已稳定就视为深度链已验收 |
| 阶段结构 | 官方基线 + YOPO PASSIVE 两阶段 | 部署、坐标、状态适配、Depth/YOPO、SO3 五个连续阶段 | 新增外部 IMU、机体坐标和控制合同后，需要更细门禁；旧两阶段结果映射到新阶段而不是删除 |
| 开发镜像 | 记录 `jp6.2` 与 `vslam3.2-dev-20260717` 快照 | 两个镜像继续作为恢复锚点，当前开发工作区使用 overlay 增量构建 | 镜像历史仍有恢复价值；当前补丁和 bringup 尚未声明已经固化进生产镜像 |

### 0.2 历史结论的生效优先级

发生冲突时按以下顺序判断：

1. 当前 Jetson 的真实命令输出、日志和设备数据；
2. 本文件第 1 至 14 节的当前任务流；
3. `STARTUP_RUNBOOK.zh-CN.md` 中与当前 commit 匹配的运行命令；
4. 第 15 节旧文档快照和早期补充记录。

低优先级内容不会被删除，但不能覆盖后来已经验证的新事实。

## 1. 文档维护规则

- 桌面的 `agent.md` 是用户日常查询的主任务流。
- 仓库中的本文件是可审计、可提交的镜像，不替代桌面主文件。
- 每完成一个检查点，应同步更新日期、实际 commit、验证证据、复选框和下一步。
- 没有真实 Jetson 输出或日志证据的任务不得标记为 `[x]`。
- 部分完成的任务保持 `[ ]`，并在后面注明“部分完成”和缺少的证据。
- 运行命令集中维护在 [`STARTUP_RUNBOOK.zh-CN.md`](STARTUP_RUNBOOK.zh-CN.md)，本文件只维护目标、状态、依赖关系和验收门槛。

## 2. 最终目标

建立一条坐标、时间、单位和健康状态都明确的被动数据链：

```text
D435i IR1 + IR2 + CameraInfo
PX4 FCU IMU -> MAVROS
  -> Isaac ROS Visual SLAM / cuVSLAM
  -> 标准定位输出与健康状态
  -> YOPO PASSIVE

D435i Depth + CameraInfo
  -> YOPO PASSIVE

定位、深度和规划完成被动验收后
  -> 审计 SO3 状态/指令/动力学合同
```

当前不允许把定位结果回灌 PX4，不允许 YOPO 或 SO3 发布有效控制命令。

## 3. 固定版本与硬件基线

| 项目 | 当前固定值 | 状态 |
| --- | --- | --- |
| Jetson | Orin NX 16GB / JetPack 6.2 / L4T 36.4.3 | 已确认 |
| 系统 | Ubuntu 22.04 / ROS 2 Humble | 已确认 |
| Isaac ROS Common | `v3.2-15@fcf4d9e` | 已固定 |
| Isaac ROS Visual SLAM | `v3.2-15@e31f4cc1d41a329a01946e5fe63669f8b15da677` | 已固定 |
| cuVSLAM 运行时 | Isaac ROS 3.2 配套二进制 SDK，不构建 core17 | 已固定 |
| 集成分支 | `u5-4/fcu-imu-cuvslam-integration` | 已发布 |
| 集成提交 | `9ca71902ec8d50d87390072a213f8ec211b88817` | 已拉取到 Jetson |
| wrapper marker | `ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1` | 已在源码补丁验证 |
| D435i | 序列号 `243622070369` / 固件 `5.15.1.55` | 已固定 |
| RealSense | librealsense `2.55.1` / ROS `4.51.1` | 已确认 |
| 相机 VIO 输入 | IR1/IR2 rectified `640x360@90 Hz`，emitter 关闭 | 已固定 |
| 飞控桥 | MAVROS `2.14.0` / `/dev/ttyTHS2:921600` | 已确认 |
| FCU IMU | MAVLink `HIGHRES_IMU(105)` 请求 200 Hz，实测约 170 Hz | 已确认 |
| ROS Domain | `42` | 已固定 |

## 4. 已完成事实

### 4.1 环境与官方基线

- [x] 容器内确认 NVIDIA GPU、D435i USB3、librealsense 和 ROS 2 Humble。
- [x] 完成 IR1/IR2/Depth/合并 D435i IMU 的短时帧率、时间戳和同步基线。
- [x] 克隆 Visual SLAM 并精确固定到 `v3.2-15@e31f4cc`。
- [x] 确认运行时使用 Isaac ROS 3.2 配套 cuVSLAM SDK，不构建或链接仓库 core17。
- [x] 在容器中构建并运行 NVIDIA 官方 RealSense + D435i IMU fusion 基线。
- [x] 完成官方 IMU fusion 与视觉-only A/B 基线检查。
- [x] 官方链路连续运行约 10 分钟，`vo_state/status`、时间戳和 tracking 状态正常。
- [x] 当前 D435i 固件 `5.15.1.55` 已运行官方链路，不需要为兼容性立即刷写。

### 4.2 标定与录包

- [x] 建立独立 `image_bag_recorder` 标定录包流程。
- [x] 录制 D435i 原厂 rectified 双目内参包。
- [x] 录制 D435i 双目 + PX4/MAVROS FCU IMU 联合标定包。
- [x] 验证双目约 `89.8 Hz`、FCU IMU 约 `170 Hz`、时间戳非零递增和传输合同。
- [x] 完成 Kalibr 双目内参与相机-FCU IMU 联合标定。
- [x] 对比自由估计内参与 D435i 原厂 rectified CameraInfo，选择与实时图像链一致的原厂 rectified 模型。
- [x] 固化 `T_Crect0_I`、双目 baseline 和相机/IMU 时移。
- [x] 固化运行时补偿 `t_aligned = t_imu_raw + 1,737,987 ns`。
- [x] 项目确认当前 Kalibr 四项 IMU 噪声权重可用于第一版联合标定与 cuVSLAM 验证，不以 Allan 作为强制前置。
- [ ] 可选增强：完成 PX4 IMU Allan 噪声标定，用于独立测量噪声系数和后续调优。
- [ ] 可选增强：在相同 factory-rectified 模型下完成一次独立重复联合标定，用于量化可重复性。

### 4.3 外部飞控 IMU cuVSLAM 源码

- [x] 修复 Isaac ROS 3.2 wrapper 将图像时间戳误用于 IMU 注册的问题。
- [x] 修复 `MessageStreamSequencer` 抖动阈值毫秒/纳秒单位问题。
- [x] 增加编译产物 marker 和固定 NVIDIA commit 校验脚本。
- [x] 实现严格递增、时钟域检查、常量时移和有限数值门禁的 aligned IMU relay。
- [x] 实现版本化相机 `K/D/R/P`、外参、时移和 IMU 噪声 schema 校验。
- [x] 实现 D435i IMU 禁用、CameraInfo 核对和 cuVSLAM odometry 常驻健康监控。
- [x] 实现 RealSense、aligned IMU、标定 TF、cuVSLAM 和健康监控统一 launch。
- [x] 本地完成 `65/65` 逻辑测试、AST/XML/格式检查和补丁 apply/reverse 验证。
- [x] 创建并发布 `u5-4/fcu-imu-cuvslam-integration@9ca7190`。
- [x] Jetson 已拉取新分支并将新版 wrapper 补丁应用到固定 NVIDIA 源码。
- [x] 首次真机验收后完成本地后续源码：项目批准与 Allan 来源拆分、默认 noise YAML、固定 odometry-only 和 `map -> odom` 所有权边界；`72/72` 逻辑测试通过。

## 5. 当前检查点：Jetson 构建与联合启动

### 5.1 构建状态

- [x] Jetson 上确认 adapter 为 `9ca7190`。
- [x] 新补丁 verifier 通过并自动恢复干净源码。
- [x] 新补丁实际应用后，NVIDIA 源码仅修改 `visual_slam_impl.cpp`。
- [x] 源码中确认 `ISAAC_ROS_YOPO_IMU_TIMESTAMP_PATCH_V1`。
- [x] 重新构建 `isaac_ros_visual_slam`。（Jetson 实测完成，耗时 57 分 20 秒）
- [x] 构建 `isaac_ros_yopo_bringup`。
- [x] 确认安装后的 `libvisual_slam_node.so` 包含 marker。
- [x] 确认安装 `aligned_imu_relay` 和 `runtime_health_monitor` 两个可执行文件。
- [x] 确认统一 launch 的 `--show-args` 正常。

### 5.2 第一次联合冒烟

联合 launch 会独占启动一份 RealSense。不得同时运行官方 RealSense launch、标定录包 launch 或第二份相机节点。

- [x] 宿主机启动 MAVROS `/dev/ttyTHS2:921600`，确认 `connected: true`。
- [x] 请求 `HIGHRES_IMU(105)` 200 Hz，确认实际约 169–171 Hz、时间戳有效且 timesync RTT 约 1.31 ms。
- [x] 容器启动统一 `d435i_fcu_imu_cuvslam.launch.py`；RealSense、aligned relay、标定 TF、cuVSLAM 与 runtime monitor 均成功启动。
- [x] 确认 D435i IR1/IR2 和两路 CameraInfo 稳定在约 89.9–90.4 Hz；左 CameraInfo 单次订阅瞬态复测后稳定为 89.86 Hz。
- [x] 确认 D435i `enable_gyro=False`、`enable_accel=False`、`unite_imu_method=0`，且 `/camera/imu` 6 秒内没有实际样本。
- [x] 确认 `/fcu/imu/data_raw_aligned` 约 171.3 Hz、frame 为 `fcu_imu`。
- [x] 确认 cuVSLAM 只订阅 aligned FCU IMU，不订阅 raw IMU 或 D435i IMU。
- [x] 确认 `camera_infra1_optical_frame -> fcu_imu` 标定 TF；平移匹配标定值，四元数以等价的整体反号形式发布。
- [x] 确认 tracker marker、`/visual_slam/status` 和 tracking odometry；`vo_state=1`，状态 frame 为 `map`，odometry 为 `odom -> camera_link` 且数值有限。
- [x] 确认 aligned relay 与 runtime monitor diagnostics 均为 `OK`；170.821 Hz、最大间隔 6.542 ms、时钟残差 -5.661 ms，所有 IMU 拒绝计数及 `forbidden_camera_imu` 均为 0。
- [x] 完成约 17 分 48 秒首次联合运行；相机、aligned IMU、odometry 与两个健康守护持续在线，无 USB 断连、崩溃、CUDA OOM 或持续 tracking lost。

`9ca7190` 使用的两个显式确认开关已经完成其首次联合验收用途。当前本地后续源码把相机-FCU 标定设为项目 `approved`，noise schema v2 使用独立的 `project_status: approved` 与 `validated: false`，并移除两个临时放行参数；`validated: false` 仅表示当前 Kalibr 权重不声称独立 Allan 来源。该后续版本尚未在 Jetson 重建和 A/B，不能用本段本地状态覆盖上面的 `9ca7190` 真机证据。

## 6. 阶段一收尾：稳定部署与文档

- [x] 根据第一次联合冒烟修正 `/camera/imu` 检查的 BrokenPipe 噪声、批准语义和稳定启动文档。
- [x] 后续源码固定为 odometry-only：关闭建图/回环、两类 ground constraint、三类内部可视化和 cuVSLAM `map -> odom` TF。
- [ ] 在 Jetson 构建后续 `isaac_ros_yopo_bringup`，确认七个模式参数均为 `False`，重新完成频率、diagnostics、受控运动和 10 分钟 A/B。
- [x] emitter 关闭时用两个 `rqt_image_view` 目视确认 IR1/IR2 无黑屏、冻结、明显曝光差异或异常拖影。
- [ ] 归档 emitter 关闭时的代表性 IR 图像、曝光、模糊和特征质量。
- [x] 完成受控刚体运动响应测试：18 秒内最大位移约 1.368 m、最大转角约 179.99°，917 个不同位姿，`vo_state` 全程为 1，odometry 时间戳无逆序。
- [ ] 记录 emitter 关闭时的 VIO 频率、失跟、延迟和漂移结果。
- [x] 完成 IR1/IR2/FCU IMU 的首次 10 分钟压力测试；连续运行约 17 分 48 秒并补充受控大幅运动，无守护退出。
- [ ] 后续启用 Depth 后，完成 IR1/IR2/Depth/FCU IMU 的并发压力测试。
- [ ] 将最终宿主机、容器、MAVROS、构建、启动、验收和停止命令同步到启动手册。
- [ ] 更新桌面 `agent.md`、仓库任务流和变更记录。

## 7. 阶段二：统一坐标系与机体坐标

### 7.1 推荐坐标树

最终目标遵循 ROS REP-103/REP-105：

```text
map                       全局任务坐标，ENU，可接收低频全局校正
└── odom                  连续局部坐标，ENU，不允许跳变
    └── base_link         机体坐标，FLU，原点为最终控制参考点/质心
        ├── fcu_imu       飞控 IMU 物理坐标
        └── camera_link   D435i 机身坐标
            ├── camera_infra1_optical_frame
            ├── camera_infra2_optical_frame
            └── camera_depth_optical_frame
```

PX4 内部继续使用 `NED + FRD`。ROS 感知、规划和状态接口使用 `ENU + FLU`。转换只能集中在 MAVROS/控制网关边界，不能分别散落在 cuVSLAM、YOPO 和 SO3 中。

### 7.2 必须确认的事实

- [ ] 确定 `base_link` 原点：车辆质心、控制参考点还是估计器参考点。
- [ ] 确定 `base_link` 轴向为 FLU，并记录对应的 PX4 `base_frd` 关系。
- [ ] 查明 MAVROS `/mavros/imu/data_raw` 的实际轴向转换，不只依赖 `frame_id` 名称。
- [ ] 通过绕单轴和静态重力实验验证 FCU IMU 的 X/Y/Z 正负方向。
- [ ] 获取或测量 `base_link -> fcu_imu` 的平移和旋转。
- [ ] 使用已标定的 `camera_infra1_optical_frame -> fcu_imu` 推导并复核 `base_link -> camera_link`。
- [ ] 重构 TF 为单父节点树，禁止同时给 `fcu_imu` 发布两个父 frame。
- [ ] 将 cuVSLAM `base_frame` 从临时 `camera_link` 改为最终 `base_link`。
- [ ] 确认 cuVSLAM odometry 的 header frame、child frame、四元数方向和 twist 表达坐标系。
- [ ] 确定局部规划与 SO3 使用连续 `odom`，全局 `map` 只通过 `map -> odom` 提供校正。
- [ ] 形成一份坐标系表：frame、父 frame、原点、轴向、单位、来源、发布节点和验证方法。
- [ ] 将最终坐标树和转换公式明确写入桌面 `agent.md` 与启动/接口文档。

在这一步完成前，不得把 cuVSLAM odometry 直接当成无人机质心状态，也不得接入控制器。

## 8. 阶段三：定位状态适配器

目标接口：

```text
/visual_slam/tracking/odometry + /visual_slam/status + diagnostics
  -> localization/state adapter
  -> /localization/odometry
  -> YOPO remap /state/odom
```

- [ ] 确定 YOPO 对 pose、quaternion、linear velocity 和 angular velocity 的实际坐标语义。
- [ ] 保留一条符合 `nav_msgs/msg/Odometry` 标准语义的 `/localization/odometry`。
- [ ] 若 YOPO 历史接口要求世界系速度，必须在 adapter 中显式转换并单独记录该偏差。
- [ ] 实现 tracking lost、odometry stale、时间戳回退、frame 错误和非有限数值门禁。
- [ ] 确认 `/state/odom` 不发布陈旧或跨坐标系混合的数据。
- [ ] 使用已知直线距离、单轴旋转和静止测试验证尺度、符号和速度方向。
- [ ] 完成状态适配器单元测试、rosbag 回放和 Jetson 实时验收。

## 9. 阶段四：Depth 与 YOPO PASSIVE

当前统一外部 IMU launch 明确关闭 Depth。Depth 阶段只能扩展这一份 RealSense 节点，不能启动第二份驱动。

- [ ] 在唯一 RealSense 节点中启用 Depth，同时保留 VIO 所需 IR1/IR2。
- [ ] 固定 Depth topic、CameraInfo、分辨率、FPS、编码、单位和时间戳合同。
- [ ] 验证 `16UC1` 毫米或 `32FC1` 米，禁止隐式猜测单位。
- [ ] 比较 emitter 开/关对 VIO 和深度空洞率、有效距离、噪声的影响。
- [ ] 选择不会破坏 VIO 的 emitter 策略并记录应用场景边界。
- [ ] 将标准深度接口 remap/转换到 YOPO `/depth_image`。
- [ ] 保持 YOPO `output_enabled=false`。
- [ ] 验证 `/so3_control/pos_cmd` 实际消息数为零，而不是只检查 publisher 是否存在。
- [ ] 联合运行定位、Depth 和 YOPO PASSIVE 至少 30 分钟。
- [ ] 记录 GPU、CPU、内存、温度、功耗、频率、延迟和队列积压。

## 10. 阶段五：SO3 重力与动力学合同

只有坐标系和 YOPO PASSIVE 通过后才进入本阶段。该阶段首先是离线审计和被动观测，不启用控制。

### 10.1 坐标与消息合同

- [ ] 审查 SO3 源码实际要求的世界坐标、机体坐标和四元数方向。
- [ ] 确认状态输入使用 `ENU/FLU` 还是 `NED/FRD`。
- [ ] 确认位置、速度、加速度、jerk、yaw、yaw rate、body rate 和 thrust 的单位。
- [ ] 明确重力向量与加速度计 specific force 的区别。
- [ ] 固定唯一坐标转换边界，禁止 YOPO、MAVROS 和 SO3 重复转换。

### 10.2 车辆参数

- [ ] 车辆总质量与载荷配置。
- [ ] 质心位置和 `base_link` 原点。
- [ ] 完整惯量矩阵及其测量/辨识来源。
- [ ] 重力常数和世界坐标中的重力方向。
- [ ] 机臂长度、电机布局和旋转方向。
- [ ] 推力系数、反扭矩系数、推力曲线和电机时间常数。
- [ ] 最小/最大转速、总推力和力矩饱和范围。
- [ ] 悬停推力、气动阻力参数和电池电压影响。
- [ ] PX4 传感器位置、板载方向和估计器相关 offset 参数。
- [ ] 控制频率、状态延迟、命令延迟和超时门限。

### 10.3 安全验收

- [ ] 建立只记录不发布执行指令的 SO3 dry-run 模式。
- [ ] 对重力补偿、悬停推力和力矩符号做离线测试。
- [ ] 对状态丢失、时间戳异常和坐标不一致实施 fail-closed。
- [ ] 完成仿真、桨叶拆除测试、系留测试和独立安全审查后，才允许讨论主动控制。

## 11. 旧任务清单的当前判定

以下内容用于同步原桌面 `agent.md` 中的复选框：

```markdown
- [x] 容器内确认 NVIDIA GPU、D435i USB3、librealsense 和 ROS2 Humble。
- [x] 完成 IR1/IR2/Depth/合并 IMU 的短时帧率、时间戳和同步基线。
- [x] 克隆 Visual SLAM 并精确固定到 `v3.2-15@e31f4cc`；不链接 core17。
- [x] 在现有容器内解析依赖并构建官方 `isaac_ros_visual_slam` 与 interfaces 包。
- [x] 运行官方 RealSense + IMU fusion 基线并完成视觉-only A/B 检查。
- [ ] 使用 odometry-only 配置。（后续源码与本地测试已完成，Jetson A/B 待完成）
- [x] 验证当前固件 `5.15.1.55` 可以运行官方链路，无需立即刷写。
- [ ] 归档 emitter 关闭时的图像和 VIO 质量。（VIO 部分已完成，图像证据待归档）
```

其他阶段一验收项：

```markdown
- [ ] CameraInfo 长期队列积压和 PVA 设备节点已单独复核。（CameraInfo 时间戳已完成）
- [x] `/visual_slam/status` 正常，`/visual_slam/tracking/odometry` 连续输出。
- [ ] 已知平移/旋转动作的方向、尺度和姿态符号符合预期。
- [x] 官方 IR 双目 + D435i IMU 连续运行至少 10 分钟且没有 tracking lost。
- [ ] IR 双目 + Depth + FCU IMU 连续并发至少 10 分钟。
- [ ] 阶段日志、配置、commit 和未解决问题已经集中归档。
```

## 12. 当前下一步

当前只执行以下顺序，不并行进入 YOPO 或控制：

1. [x] Jetson 完成两个包的编译和安装验证。
2. [x] 宿主机启动 MAVROS 并确认 FCU IMU 约 170 Hz。
3. [x] 容器启动统一外部 IMU cuVSLAM launch。
4. [x] 完成短时话题、TF、订阅、diagnostics 和 odometry 验收。
5. [x] 完成至少 10 分钟联合稳定性测试。（首次连续运行约 17 分 48 秒）
6. [x] 修复首次真机发现的问题并更新后续稳定启动文档。
7. [ ] 进入 odometry-only A/B。
8. [ ] 进入统一坐标系和机体外参阶段。

## 13. 当前已知未解决项

| 编号 | 问题 | 当前影响 |
| --- | --- | --- |
| R-001 | PX4 IMU 四项噪声权重没有独立 Allan 来源 | 当前不阻塞部署；保留来源说明，运行质量不足时再执行 Allan 调优 |
| R-002 | factory-rectified 联合标定没有独立重复数据集 | 当前不阻塞部署；作为可选重复性量化，不能把“未重复”误写成“联合标定未完成” |
| R-003 | 当前 cuVSLAM `base_frame=camera_link` | 输出仍是相机参考点，不是最终机体/质心状态 |
| R-004 | 后续源码已设 `enable_localization_n_mapping=False`，但 Jetson A/B 尚未完成 | 不能把本地源码合同直接视为真机 odometry-only 验收 |
| R-005 | 当前统一外部 IMU launch 关闭 Depth | 尚不能作为 YOPO 完整感知输入 |
| R-006 | `base_link -> fcu_imu` 真实安装关系未固定 | 不能建立最终机体 TF 树 |
| R-007 | YOPO 对 odometry twist 的坐标语义待源码确认 | 不能直接发布最终 `/state/odom` |
| R-008 | SO3 车辆参数与坐标合同尚未审计 | 禁止启用控制 |

## 14. 变更记录

| 日期 | 更新 |
| --- | --- |
| 2026-07-21 | 将任务流同步到外部 FCU IMU 集成现状；记录 `9ca7190`、wrapper 补丁、标定候选、Jetson 当前构建检查点；新增坐标系、状态适配、Depth/YOPO 和 SO3 阶段。 |
| 2026-07-21 | 根据项目决定，将 Allan 和独立重复联合标定调整为可选增强；当前 Kalibr 联合标定保持已完成，第一版运行使用现有四项噪声权重。 |
| 2026-07-21 | Jetson 用时 57 分 20 秒完成 `isaac_ros_visual_slam` 重编译和 `isaac_ros_yopo_bringup` 构建；安装节点、二进制 marker 与统一 launch 参数验证通过，检查点推进到首次联合运行。 |
| 2026-07-21 | 首次 D435i 双 IR + PX4 aligned IMU + cuVSLAM 联合运行验收通过：话题频率、TF、唯一 IMU 订阅、diagnostics、`vo_state=1` 和 `odom -> camera_link` 均满足合同，并连续运行约 17 分 48 秒。 |
| 2026-07-21 | 本地后续源码将标定项目批准与 Allan provenance 拆分，移除两个临时放行参数并提供默认 noise YAML；生产 launch 固定为 odometry-only 且不发布 `map -> odom`，`72/72` 逻辑测试通过，Jetson A/B 待执行。 |

## 15. 桌面旧任务流完整内容快照

> 来源：`C:\Users\10416\Desktop\agent.md`
> 合并前 SHA-256：`699B49FB4CB0777E390FA711A0B1C4997EDF60BECB357097B649060C10CFEB2E`
> 快照用途：保留 2026-07-17 环境审计、问题/决策编号、早期阶段计划、镜像恢复信息和随后追加的讨论记录。
> 保留方式：文字、代码和顺序全部保留；仅规范化历史原文中的行尾空格，未覆盖桌面源文件。
> 注意：本快照中的“当前”“尚未完成”“不进入”等表述只代表当时状态；与第 0 节取舍表或第 1 至 14 节冲突时，以当前章节为准。

<details>
<summary>展开 2026-07-17 版 agent.md 完整原文</summary>

<!-- BEGIN DESKTOP_AGENT_HISTORY -->
# YOPO ROS2 与 cuVSLAM Jetson 部署工作文档

> 用途：记录项目边界、已验证事实、技术决策、问题和接下来两个阶段的任务。
> 最近更新：2026-07-17
> 当前原则：只推进被动部署，不进入控制、OFFBOARD 或飞行阶段。

## 1. 项目目标与当前位置

保留 YOPO 官方算法主体，完成 ROS2 Humble 接口和 Jetson Orin NX 被动部署，并验证以下真实数据链：

```text
D435i IR1 + IR2 + IMU
  -> Isaac ROS Visual SLAM / cuVSLAM
  -> 坐标、速度与健康状态适配
  -> /state/odom

D435i Depth + CameraInfo
  -> /depth_image
  -> YOPO

/state/odom + /depth_image + goal
  -> YOPO PASSIVE（output_enabled=false）
```

### 1.1 已完成

- [x] YOPO 官方算法主体的 ROS2 Humble 接口迁移。
- [x] `quadrotor_msgs` 与 `yopo_planner` 从 `~/catkin_ws` 完整构建成功。
- [x] `PositionCommand` 接口和 `yopo_node` 可执行文件验证成功。
- [x] Jetson 加载官方 `epoch50.pth`，PyTorch CUDA 推理初始化成功。
- [x] YOPO 以 `output_enabled=false` 被动启动成功。
- [x] VINS-Fusion-ROS2 在 Jetson Humble 上完成 Ceres 构建兼容修复，保留为回退方案。
- [x] JetPack、CUDA、Docker、NVIDIA Container Runtime 和 D435i USB3 链路完成基础审计。
- [x] Isaac ROS Common 固定到 `v3.2-15`，RealSense Dev 镜像已构建并固定标签。
- [x] NVIDIA GPU/PVA CDI 名称已注册，官方 `run_dev.sh -b` 可启动容器并识别 Orin GPU。
- [x] D435i IR1、IR2、Depth 和合并 IMU 的短时帧率、时间戳及同步基线通过。

### 1.2 尚未完成

- [ ] Isaac ROS Visual SLAM `v3.2-15` 固定版本导入、构建与官方运行基线。
- [ ] D435i IR 双目、IMU、深度至少 10 分钟并发流压力测试。
- [ ] cuVSLAM 里程计到项目 `/state/odom` 契约的适配。
- [ ] cuVSLAM 与 YOPO PASSIVE 联合运行和长期性能验收。

本文件当前只安排上述两类任务。ROS1/RflySim 仿真、PX4、MAVROS、控制器、TensorRT 和实飞均不属于当前两个阶段。

## 2. 工作范围与协作边界

### 2.1 代码、仓库与工作区

| 位置 | 当前状态 | 用途与规则 |
| --- | --- | --- |
| `D:\catkin_ws\src\YOPO` | `ros2-humble@80c0569`，干净 | 当前可部署 ROS2 基线，允许按真实问题修改 |
| `work\YOPO-phase2` | `jetson-passive-deployment@80c0569`，无 upstream | 阶段二本地草案克隆；`stash@{0}` 仅属于该克隆，不是可发布版本 |
| `D:\catkin_ws\src\YOPO-ROS2` | 第三方参考 | 只读参考，不覆盖官方算法主体 |
| `D:\catkin_ws\src\VINS-Fusion-ROS2` | `jetson-orin-build-fixes@11d64f1` | 已修改并推送的回退/A-B 基线，不是当前 cuVSLAM 集成对象 |
| `u5-4/cuVSLAM` | `main` 为 core 17；阶段二计划 `u5-4/isaac-ros-3.2-yopo-adapter` 分支 | 用户指定的适配代码托管仓库；core 保持只读，适配层与 core17 隔离 |
| Jetson `~/catkin_ws` | YOPO/VINS ROS2 工作区 | 已有包验证，不放入 Isaac ROS 全栈 |
| Jetson `~/workspaces/isaac_ros_3_2` | 已创建；Common 固定 `v3.2-15` | Isaac ROS 3.2 独立工作区；RealSense Dev 镜像已构建，Visual SLAM 源码尚未导入 |

`stash@{0}` 的序号会变化，不能作为长期版本标识。草案只有在审查、测试并提交后才能视为可部署代码。

### 2.2 职责分工

- Codex：阅读和修改 Windows 本地源码、设计接口、根据用户返回的真实日志修复问题。
- 用户：在 Jetson/WSL2 执行安装、构建和运行命令，并返回完整输出。
- 不通过 SSH 操作飞行计算机；不替用户启动传感器、控制器或飞行任务。
- Git 推送、分支发布或外部状态修改只在用户明确要求时执行。

### 2.3 安全门禁

- 不启动 RflySim、SITL、PX4、MAVROS、OFFBOARD、解锁或起飞脚本。
- 不启用 YOPO 控制输出；`output_enabled` 必须保持 `false`。
- 不将 cuVSLAM 输出回灌 PX4 external vision。
- 不在 Jetson 上运行 ROS1/Noetic 仿真；如需仿真，只能在 PC/WSL 隔离环境另行安排。
- 不盲目刷写 D435i 固件。
- 宿主机与容器不能同时启动两份 RealSense 驱动。
- 不为尚未出现的问题提前改代码、安装无关依赖或转换 TensorRT。

### 2.4 代码修改与上传边界

“适配 cuVSLAM”在当前项目中指适配其 ROS2 输出、坐标、速度、健康状态和 YOPO 接口，不等于修改或替换 cuVSLAM core。

| 仓库 | 允许做 | 禁止做 | 代码上传位置 |
| --- | --- | --- | --- |
| `u5-4/YOPO_ROS2` | 修复 YOPO ROS2 接口、被动安全和真实部署问题 | 覆盖官方算法主体、未经验证启用控制、直接改 `main` | 已验收的 YOPO 专用分支 |
| `u5-4/cuVSLAM` | 在隔离目录增加 ROS2 adapter、launch、配置、测试和兼容文档 | 修改 core17、根 CMake/README/LICENSE/VERSION、`libs/`、`python/`、`examples/` | `u5-4/isaac-ros-3.2-yopo-adapter` |
| `u5-4/VINS-Fusion-ROS2` | 维护回退/A-B 基线和真实 Jetson 构建修复 | 与 cuVSLAM 阶段混改或同时占用 D435i | `jetson-orin-build-fixes` |
| NVIDIA Isaac ROS 官方仓库 | 固定 `v3.2-15`、构建、运行和记录 commit | 修改、覆盖、向官方仓库上传或混用 4.x/main | 不上传；只作为固定外部依赖 |

`u5-4/cuVSLAM` 中允许新增的唯一路径为：

```text
integrations/isaac_ros_3_2_yopo/
  README.md
  COMPATIBILITY.md
  LICENSE                  # 仅覆盖本目录新增代码，具体许可证由用户确认
  config/
  launch/
  cuvslam_yopo_adapter/
    package.xml
    CMakeLists.txt
    include/
    src/
    test/
```

该 adapter 必须满足：

- 只通过 ROS2 topic 与外部 Isaac ROS Visual SLAM 通信。
- 不包含或链接 core17 的头文件、共享库、Python 包或 CMake target。
- 首选标准 ROS2 接口：`nav_msgs`、`diagnostic_msgs`、`tf2` 等。
- 输入 `/visual_slam/tracking/odometry` 与标准 `/diagnostics`，输出 `/state/odom` 和 adapter diagnostics。
- `/depth_image` 不属于该 adapter，由唯一 RealSense 驱动独立发布、remap 或转换。
- README 明确运行时依赖外部 Isaac ROS `v3.2-15` 与 cuVSLAM SDK `12.6`；仓库内 core `17.0.0` 不构建、不链接、不运行。

Git 与上传规则：

- `main` 始终跟随 NVIDIA 上游 core，不直接提交 YOPO 适配代码。
- 分支遵守仓库规则，计划名称为 `u5-4/isaac-ros-3.2-yopo-adapter`，不使用容易冒充 NVIDIA 发行线的 `release-3.2` 等名称。
- 任何 branch/commit 操作前先获得用户明确确认；提交使用 `git commit -s`。
- 只允许 `git add integrations/isaac_ros_3_2_yopo/...`，禁止 `git add .`。
- 提交前执行路径白名单检查，所有变更必须位于上述 integration 目录。
- 禁止提交权重、固件、设备序列号、运行日志、容器缓存或构建产物。
- 禁止 force-push、直接推送 `main` 或创建类似 NVIDIA release 的 tag。
- cuVSLAM 仓库自身 `AGENTS.md` 规定 Codex 不执行 `git push`；Codex准备本地差异和经授权的签署提交，由用户审核后推送到自己的 fork。
- 如果确认必须修改 Isaac ROS wrapper，本阶段立即停止并重新决策；不能把 wrapper patch 或 SDK12.6 二进制改动伪装成 core17 修改。

## 3. Git 与版本基线

| 组件 | 固定版本 | 说明 |
| --- | --- | --- |
| YOPO ROS2 | `ros2-humble@80c0569` | 当前 1.0 被动基线 |
| YOPO 阶段二草案 | `work\YOPO-phase2` 中的 `jetson-passive-deployment@80c0569` | 仅本地分支；尚无 upstream，stash 未发布 |
| VINS-Fusion-ROS2 | `jetson-orin-build-fixes@11d64f1` | 构建成功的回退版本 |
| Isaac ROS | `v3.2-15` | JetPack 6.2、Orin、Ubuntu 22.04、ROS2 Humble 维护线 |
| Isaac ROS Visual SLAM | 目标 `v3.2-15@e31f4cc` | 必须与 Common/NITROS 使用同一 3.2 更新代际；Jetson 尚未导入 |
| cuVSLAM SDK | Isaac ROS 3.2 配套 SDK `12.6` | 此处 12.6 是 SDK 版本，不是 CUDA 版本 |
| `u5-4/cuVSLAM` | core `17.0.0` 开发线 | 与 Isaac ROS 3.2 wrapper 不兼容，不加入当前构建 |
| cuVSLAM YOPO adapter | 计划 `u5-4/isaac-ros-3.2-yopo-adapter` | 只包含隔离 ROS2 integration 目录，不链接 core17 |

当前路线是：隔离评估 cuVSLAM，达标后才允许将其晋升为 YOPO 主里程计；VINS 保留为回退和 A/B 对照。

## 4. 已验证环境基线

### 4.1 Jetson

| 项目 | 当前值 | 状态 |
| --- | --- | --- |
| 设备 | Jetson Orin NX 16GB | 已确认 |
| JetPack / L4T | JetPack 6.2 / `36.4.3` | 与 Isaac ROS 3.2 Update 1+ 匹配 |
| Ubuntu | `22.04.5 LTS` | 已确认 |
| ROS2 | Humble | 当前目标；每个新终端仍需检查未混入 Noetic |
| CUDA | `12.6`，nvcc `12.6.68` | `/usr/local/cuda/bin/nvcc` 已验证 |
| cuDNN / TensorRT | `9.3` / `10.3` | 已确认 |
| YOPO PyTorch | `2.11.0`，Orin CUDA 可用 | 在 YOPO 运行环境验证成功 |
| Conda | 存在 `base`/`yopo` 环境 | 启动 Isaac ROS 脚本前先 `conda deactivate` |
| Docker | `29.4.3` | 已安装；用户已加入 `docker` 组，重启后的持久权限仍待复核 |
| Buildx | `0.34.0` | 已安装 |
| NVIDIA Container Toolkit | `1.16.2` | 已安装 |
| Docker Runtime | `nvidia` 已注册，默认 `runc` | 正常；无需改默认 Runtime |
| Isaac ROS Common | `v3.2-15@fcf4d9e` | 工作区和 RealSense Dev 镜像已完成；Visual SLAM 尚未构建 |
| RealSense Dev 镜像 | `isaac_ros_dev-aarch64:isaac3.2-rs2.55.1-jp6.2` | image ID `sha256:bceda07bd17756dc582693e00fc388fb5b2589c94d8267e6a253546640f8a933` |

环境诊断原始日志：`jetson_env_20260716_074237.log`。

### 4.2 D435i

| 项目 | 当前值 | 状态 |
| --- | --- | --- |
| USB ID | `8086:0b3a` | 已识别 |
| USB 链路 | D435i 所有接口 `5000M`；descriptor `3.2` | 已在 `lsusb -t` 与 librealsense 实测为 USB3 |
| 序列号 | `243622070369` | 已记录 |
| 固件 | `5.15.1.55` | 不属于 Isaac ROS 3.2 官方验证组合 |
| librealsense | `2.55.1` | 容器内 `pkg-config` 验证 |
| RealSense ROS | `4.51.1-0jammy` | camera、msgs、description 均已安装 |
| 图像流 | IR1 `29.99 Hz`、IR2 `29.99 Hz`、Depth `29.99 Hz` | 短时并发基线通过 |
| 惯性流 | Gyro `199.8 Hz`、合并 IMU `199.8 Hz` | `unite_imu_method=2`；加速度和角速度均已取样 |
| 时间戳 | IR1/IR2/Depth/IMU 均 `zero=0`、`nonmonotonic=0` | IR1-IR2、IR1-Depth 的 p95 均为 `0.000 ms` |
| 枚举与并发稳定性 | 短时通过 | 至少 10 分钟 Visual SLAM 联合压力测试仍待做 |

Isaac ROS 3.2 官方 Quickstart 要求：

```text
Firmware:       5.13.0.50
librealsense:   2.55.1
realsense-ros:  4.51.1-isaac
```

当前固件已通过枚举和短时多流测试，暂不刷写。它尚未证明 Visual SLAM 全链路兼容；只有在 USB、容器驱动和配置均确认正确后仍有明确版本证据，才评估有回退方案的固件变更。

`/camera/accel/sample` 在 `unite_imu_method=2` 下未独立发布不是故障；加速度已合入 `/camera/imu`，并已确认该消息同时含有效 `linear_acceleration` 与 `angular_velocity`。

### 4.3 YOPO 被动基线

- `quadrotor_msgs`、`yopo_planner` 已能从 `~/catkin_ws` 被 `colcon list` 发现和构建。
- `ros2 interface show quadrotor_msgs/msg/PositionCommand` 已通过。
- 官方 `epoch50.pth` 已在 Jetson CUDA 上加载。
- `/yopo_net` 的 `output_enabled` 已验证为 `false`。
- 当前实现即使在被动模式仍会创建 `/so3_control/pos_cmd` publisher 和控制 timer，只是在发布回调内返回。因此“ROS 图中存在 publisher”不等于“实际发布控制消息”。阶段二必须验证该 topic 实际消息数为零。

## 5. 已确认问题与技术决策

### P-001：`quadrotor_msgs` 曾无法发现

- 状态：已解决。
- 原因：曾在包目录内构建，未从 `~/catkin_ws` 工作区根目录执行。
- 验证：`colcon list`、消息接口和 `yopo_node` 均已通过。

### P-002：缺少 `policy/models/__init__.py`

- 状态：未解决，当前非阻塞。
- 现象：`setup.py` 声明 `policy.models`，文件实际不存在；`--symlink-install` 构建产生警告。
- 处理原则：阶段二先增加普通安装验证；只有确认影响部署时再修复并提交。

### P-003：CUDA 编译器路径

- 状态：已确认。
- `/usr/local/cuda/bin/nvcc` 为 CUDA `12.6.68`；是否加入 shell `PATH` 不是当前阻塞项。

### P-004：D435i 固件偏离官方组合

- 状态：短时 RealSense 基线通过；Visual SLAM 兼容性待验证。
- 当前固件 `5.15.1.55`，官方 3.2 Quickstart 要求 `5.13.0.50`。
- 当前固件已完成 USB3.2 枚举、IR 双目、Depth 和合并 IMU 短时并发验证。
- 禁止在 USB 不稳定或没有恢复方案时刷写。

### P-005：RealSense 发射器策略未确定

- 状态：待阶段一验证。
- NVIDIA Visual SLAM 文档要求 IR 跟踪时关闭 emitter，避免投射白点造成漂移；YOPO 深度在低纹理环境可能受益于主动纹理。
- 必须分别记录 emitter 关闭时的 VIO 与深度质量，不可默认选择。

### P-006：深度与相机契约尚未验收

- 状态：待阶段二验证。
- YOPO 当前支持 `16UC1` 毫米和 `32FC1` 米，错误单位可能造成 1000 倍尺度错误。
- 目标配置需记录实际 RealSense topic、编码、单位、`480x270`、目标 FPS、CameraInfo、FOV 和时间戳。
- `/depth_image` 由唯一的 RealSense 驱动独立 remap/转换，不由里程计适配节点生成。

### P-007：缺少 Git LFS 导致 `run_dev.sh` 静默退出

- 状态：已解决。
- 现象：`run_dev.sh` 只打印调用目录后退出，没有显示预期错误。
- 原因：Jetson 缺少 `git-lfs`；官方脚本启用 `set -e`，在静默执行 `git lfs`失败时提前退出，来不及执行后续错误提示。
- 处理：安装 Ubuntu `git-lfs 3.0.2` 并执行 `git lfs install --skip-repo`。
- 验证：重新运行后已进入 `aarch64.ros2_humble.realsense` 镜像构建流程。

### P-008：RealSense 镜像层未识别 librealsense `v2.55.1`

- 状态：已解决。
- 现象：`Dockerfile.realsense` 调用 `build-librealsense.sh -v v2.55.1`，最终报告该 tag 不可用并以 exit code 1 退出。
- 原因：Docker build 内 GitHub/TLS 传输中断；上游 `v2.55.1@e196cefa` 始终存在，“tag 不可用”是 clone 失败后的连锁误报。
- 处理：在官方源码外配置 Git transport，固定 HTTP/1.1、`http.maxRequests=1`、canonical `realsenseai/librealsense` URL rewrite，并在构建时显式传入 LAN 代理。
- 边界：未修改 NVIDIA Isaac ROS 官方 Dockerfile、脚本或仓库内容。
- 验证：镜像构建成功；容器内 `pkg-config --modversion realsense2` 返回 `2.55.1`，D435i 可由 `rs-enumerate-devices` 枚举。

### P-009：容器启动时缺少 GPU CDI 设备

- 状态：已解决。
- 现象：`run_dev.sh -b` 报 `unresolvable CDI devices nvidia.com/gpu=all`。
- 原因：宿主已有 `pva-allow-2` 提供的 PVA CDI，但缺少 GPU CDI spec。
- 处理：生成 `/etc/cdi/nvidia.yaml`，保留 `/etc/cdi/nvidia-pva.yaml`。
- 验证：`nvidia-ctk cdi list` 已包含 `nvidia.com/gpu={0,all}` 与 `nvidia.com/pva={0,all}`；`run_dev.sh -b` 可启动，容器内 `nvidia-smi` 识别 Orin，`/dev/nvidia0` 可见。
- 未过度声明：尚未单独证明容器内 `/dev/nvhost-pva*` 节点可见；PVA 节点仍需在后续运行检查中复核。

### P-010：Docker 默认 bridge 的 `iptables raw` 错误

- 状态：当前路线不适用，非阻塞。
- 现象：手工使用 Docker 默认 bridge 网络测试时报 `Unable to enable DIRECT ACCESS FILTERING`，宿主 legacy iptables 缺少 `raw` 表。
- 判断：官方 `run_dev.sh` 使用 host network，且实际容器已成功启动；该错误不影响当前 Isaac ROS 路线。
- 处理原则：不为无关的 bridge 测试修改 Jetson 内核或 iptables；只有未来明确需要 Docker bridge 时再单独处理。

### D-001：Isaac ROS 使用容器优先

- 固定 `isaac_ros_common`、`isaac_ros_visual_slam`、NITROS 到同一 `v3.2-15`。
- `v3.2-15` 是仓库 tag；开发镜像由对应版本 `run_dev.sh` 分层构建，不是一个可随意混用的 core17 镜像标签。
- 已构建镜像固定为 `isaac_ros_dev-aarch64:isaac3.2-rs2.55.1-jp6.2`，作为恢复锚点。`run_dev.sh -b` 实际读取 `isaac_ros_dev-aarch64:latest`，启动前需确认二者 image ID 一致；不一致时才将固定标签重新 tag 为 `latest`。
- 使用官方 `run_dev.sh`，不手写不完整的 `docker run`。脚本负责 NVIDIA Runtime、host network、IPC、设备和工作区挂载。
- Isaac ROS 官方源码在当前两阶段保持只读；配置和外部 adapter 保存在用户 fork 的隔离 integration 目录。

### D-002：容器与宿主暂时分离

- Isaac 容器：唯一一份 `realsense2_camera`、Visual SLAM，以及需要 Isaac 状态消息的适配逻辑。
- Jetson 宿主：现有 YOPO Python/CUDA 环境。
- 双方固定相同 `ROS_DOMAIN_ID`、RMW 和经过验证的 QoS。
- 第一版不把 YOPO 安装进 Isaac 系统 Python，避免 Torch、OpenCV、Ceres 和 GXF/NITROS 依赖冲突。

## 6. 当前推荐架构

```text
Isaac ROS 3.2 容器（仓库统一 v3.2-15）

D435i IR1 + IR2 + IMU
  -> isaac_ros_visual_slam（配套 cuVSLAM SDK 12.6）
  -> /visual_slam/tracking/odometry + /visual_slam/status
  -> 状态/坐标适配节点
  -> /state/odom

D435i Depth + CameraInfo
  -> 独立 remap/必要的编码转换
  -> /depth_image + /depth_image/camera_info

                    ↓ ROS2 DDS

Jetson 宿主
  -> YOPO（output_enabled=false）
  -> 仅记录候选轨迹、推理延迟和健康状态
```

cuVSLAM release-3.2 不消费 YOPO 的深度图。深度与 VIO 是同一 RealSense 驱动发布的两条并行数据链。

## 7. 后续仅保留两个阶段

### 阶段一：Isaac ROS 3.2 / D435i 官方基线

**状态：进行中（2026-07-17 启动）**
**当前检查点：1C - 构建 Visual SLAM `v3.2-15` 并验证真实里程计输出。**

检查点 1A 已于 2026-07-17 通过：

- ROS 环境仅包含 Humble：`AMENT_PREFIX_PATH=/opt/ros/humble`。
- `isaac_ros_common` HEAD 为 `fcf4d9e17f8f0a7f47f1d22d6a18421ce3768c01`。
- `v3.2-14` 与 `v3.2-15` 同时指向该提交；HEAD 已与 `refs/tags/v3.2-15^{commit}` 精确比对通过。
- `.isaac_ros_common-config` 已设置 `CONFIG_IMAGE_KEY=ros2_humble.realsense`。
- NVMe 根分区剩余约 `60G`，当前可进入容器构建检查点。

检查点 1B 已于 2026-07-17 通过：

- `ros2_humble.realsense` Dev 镜像已成功构建并固定为 `isaac_ros_dev-aarch64:isaac3.2-rs2.55.1-jp6.2`，image ID 为 `sha256:bceda07bd17756dc582693e00fc388fb5b2589c94d8267e6a253546640f8a933`。
- GPU/PVA CDI 名称已注册；`run_dev.sh -b` 成功进入容器，`nvidia-smi` 识别 Orin GPU。PVA CDI 存在，但容器内 PVA 节点仍待单独复核。
- 容器内 librealsense 为 `2.55.1`，RealSense ROS 为 `4.51.1-0jammy`，D435i 固件为 `5.15.1.55`，USB descriptor 为 `3.2`。
- IR1、IR2、Depth 均约 `29.99 Hz`，Gyro 与合并 IMU 约 `199.8 Hz`；被测消息时间戳非零、单调，双红外和红外/深度同步偏差 p95 均为 `0.000 ms`。
- `unite_imu_method=2` 下 `/camera/accel/sample` 不独立发布；加速度和角速度已在 `/camera/imu` 中确认。

检查点 1C 的源码预审结论：

- Visual SLAM `v3.2-15` 目标 commit 为 `e31f4cc1d41a329a01946e5fe63669f8b15da677`；它与 `v3.2-14` 共用 commit，不能只依赖 `git describe`，必须比对完整 SHA。
- Jetson 已将该仓库导入 `~/workspaces/isaac_ros_3_2/src/isaac_ros_visual_slam`；完整 SHA 比对通过，工作区源码固定完成。
- 官方 `isaac_ros_visual_slam_realsense.launch.py` 会独占启动一份 RealSense 节点，配置 IR1/IR2 `640x360x90`、Gyro/Accel `200 Hz`、`unite_imu_method=2`、emitter 关闭、Depth 关闭并启用 IMU fusion。
- 官方 launch 已 remap IR1、IR2、对应 CameraInfo 和 `/camera/imu`；启动前必须关闭现有 `/camera/camera`，禁止两份驱动争用 D435i。
- 该 launch 适合官方功能基线，但不是最终低负载配置：它开启三类 SLAM 可视化，并沿用 `enable_localization_n_mapping=true` 默认值。基线通过后再从外部 integration 目录提供 odometry-only launch，不修改 NVIDIA 仓库。
- wrapper 从 `isaac_ros_nitros` 配套资源链接 `libcuvslam.so` 并使用旧 `CUVSLAM_*` C API；不得链接 `u5-4/cuVSLAM` core17 C++ API。

#### 目标

在独立工作区中复现 NVIDIA 官方 RealSense Visual SLAM 基线，证明容器、D435i 双目、IMU 和 cuVSLAM SDK 在当前 JetPack 6.2 上可用。阶段一不接 YOPO，也不修改当前 YOPO/VINS 基线。

#### 任务

- [x] 新建 `~/workspaces/isaac_ros_3_2/src`。
- [x] 克隆 `isaac_ros_common` 并固定 `v3.2-15`。
- [x] 使用已配置的 `CONFIG_IMAGE_KEY=ros2_humble.realsense` 运行官方 `run_dev.sh` 并构建容器。
- [x] 在容器内确认 NVIDIA GPU、D435i USB3、librealsense 和 ROS2 Humble。
- [x] 完成 IR1/IR2/Depth/合并 IMU 的短时帧率、时间戳和同步基线。
- [x] 克隆 Visual SLAM 并精确固定到 `v3.2-15@e31f4cc`；不链接 `u5-4/cuVSLAM` core17。
- [x] 在现有容器内解析依赖并构建 `isaac_ros_visual_slam` 与 interfaces 包。
- [x] 先运行官方 RealSense + IMU fusion 基线并保存 topic、频率、状态和日志；随后再做关闭 IMU 的视觉-only A/B 检查。
- [ ] 使用 odometry-only 配置：关闭建图、回环、ground constraint 和非必要可视化。
- [ ] 验证当前固件 `5.15.1.55` 是否能运行官方链路；失败时先定位版本证据，不立即刷固件。
- [ ] 记录 emitter 关闭时的图像和 VIO 质量。

#### 验收标准

- [x] `isaac_ros_common` 的实际 tag/commit 已记录并精确固定到 `v3.2-15`。
- [x] Visual SLAM 的 Jetson 实际 tag/commit 已记录并固定到 `v3.2-15@e31f4cc`。
- [x] 容器能识别 NVIDIA Runtime、Orin GPU 和 D435i USB3 链路。
- [x] IR1、IR2、Depth、合并 IMU 的短时时间戳单调且实际频率稳定。
- [ ] CameraInfo 时间戳、长期队列积压和 PVA 设备节点已单独复核。
- [ ] `/visual_slam/status` 正常，`/visual_slam/tracking/odometry` 连续输出。
- [ ] 保持相机静止、前后左右上下和平移/旋转时，方向、尺度和姿态符号符合预期。
- [ ] IR 双目 + IMU 连续运行至少 10 分钟，无 USB disconnect、进程崩溃、CUDA OOM 或持续 tracking lost。
- [ ] 阶段一日志、配置和未解决问题已归档；未启动 YOPO、PX4 或任何控制组件。

#### 阶段一立即执行的两个动作

1. 重启或重新登录后复核 `docker` 组、CDI 清单，并确认 `latest` 与固定镜像 image ID 一致，再用 `run_dev.sh -b` 复用现有镜像；禁止触发无必要的 RealSense 重建。
2. 在现有容器内构建 Visual SLAM，随后验证 `IR1 + IR2 + /camera/imu -> /visual_slam/status + /visual_slam/tracking/odometry`。

### 阶段二：cuVSLAM 到 YOPO 的联合 PASSIVE 验收

#### 进入条件

阶段一全部通过，且固件、emitter、双目/IMU频率和 Visual SLAM 配置已有固定记录。

#### 目标

建立并验收以下被动闭环，不产生任何有效控制消息：

```text
/visual_slam/tracking/odometry + status
  -> adapter -> /state/odom

D435i depth + CameraInfo
  -> /depth_image

/state/odom + /depth_image + goal
  -> YOPO PASSIVE
```

#### 任务

- [ ] 在 `u5-4/cuVSLAM` 的计划分支 `u5-4/isaac-ros-3.2-yopo-adapter` 下，仅在 `integrations/isaac_ros_3_2_yopo/` 新建独立 ROS2 适配包。
- [ ] adapter 不构建或链接仓库 core17；明确输入、输出、frame、速度语义和健康门控。
- [ ] 将 cuVSLAM 任意初始化世界通过唯一对齐关系转换到项目 `map`。
- [ ] 输出 `map -> base_link` 语义，并验证四元数 `xyzw`。
- [ ] 查明 cuVSLAM odometry twist 的实际参考系；如为机体系，旋转成 YOPO 要求的世界系速度。
- [ ] 对 tracking lost、odometry 超时和时间戳回退实施阻断，禁止发布陈旧 `/state/odom`。
- [ ] 在唯一 RealSense 驱动中启用深度，固定实际 topic、`480x270`、目标 FPS、编码、单位和 CameraInfo。
- [ ] 验证 `16UC1` 为毫米或 `32FC1` 为米；禁止隐式单位猜测。
- [ ] 保持 YOPO `output_enabled=false`，联合运行 cuVSLAM、depth 和 YOPO。
- [ ] 记录 GPU、CPU、内存、温度、功耗、图像/IMU/odom频率、YOPO延迟和队列积压。
- [ ] 只从 `work\YOPO-phase2` 中选择经过真实日志证明需要的草案改动；审查后提交，不直接发布 stash。

#### 验收标准

- [ ] `/state/odom` 的 pose 位于 `map`，child 为 `base_link`，位置、姿态和世界系速度方向全部正确。
- [ ] D435i 深度编码、单位、分辨率、FOV、CameraInfo 和时间戳已归档并与 YOPO 配置一致。
- [ ] VIO、深度和 YOPO 联合运行至少 30 分钟，无 USB disconnect、崩溃、CUDA OOM、持续掉帧或热失控。
- [ ] tracking lost、里程计超时或深度失效时，YOPO 不使用陈旧状态。
- [ ] `output_enabled=false`；虽然 ROS 图中存在控制 publisher，但 `/so3_control/pos_cmd` 的实际消息数为零。
- [ ] 联合运行日志、配置、适配代码和固定 commit 已归档。
- [ ] 阶段二通过只代表“真机被动导航链可用”，不代表允许 PX4 回灌、OFFBOARD 或飞行。

## 8. 当前不进入的工作

- ROS1/RflySim 仿真基线：如后续确有必要，在 PC/WSL 独立安排，不占用当前两阶段。
- TensorRT：当前 PyTorch CUDA 已可运行；待被动数据链稳定后再评估，并必须在目标 Jetson 生成引擎。
- VINS 实机定位：保留为回退/A-B 基线，不与阶段一容器同时争用 D435i。
- PX4 external vision、MAVROS、控制器、OFFBOARD、拆桨、系留和实飞：全部延后，当前禁止启动。
- 训练新模型：不属于当前部署阻塞项。

## 9. 进度与问题记录模板

### 9.1 进度记录

```markdown
#### YYYY-MM-DD：任务标题

- 所属阶段：一 / 二
- 目标：
- 执行环境：
- 仓库、分支、tag、commit：
- 执行命令：
- 结果：成功 / 失败 / 部分成功
- 关键输出与日志路径：
- 是否满足验收标准：
- 下一步：
```

### 9.2 问题记录

```markdown
#### P-XXX：问题标题

- 状态：待处理 / 处理中 / 已解决 / 暂缓
- 所属阶段：
- 发现日期：
- 环境与版本：
- 复现步骤：
- 完整报错或日志路径：
- 原因分析：
- 最小处理方案：
- 验证结果：
```

## 10. 官方参考资料

- [Isaac ROS 3.2 Getting Started](https://nvidia-isaac-ros.github.io/v/release-3.2/getting_started/index.html)
- [Isaac ROS 3.2 Release Notes](https://nvidia-isaac-ros.github.io/v/release-3.2/releases/index.html)
- [Isaac ROS 3.2 Compute Setup](https://nvidia-isaac-ros.github.io/v/release-3.2/getting_started/hardware_setup/compute/index.html)
- [Isaac ROS 3.2 RealSense Setup](https://nvidia-isaac-ros.github.io/v/release-3.2/getting_started/hardware_setup/sensors/realsense_setup.html)
- [Isaac ROS 3.2 RealSense VSLAM](https://nvidia-isaac-ros.github.io/v/release-3.2/concepts/visual_slam/cuvslam/tutorial_realsense.html)







补充‘：

不是同一个东西，但关系很紧密：

```
isaac_ros_visual_slam
  = ROS2 封装层、节点、launch、话题接口
        ↓ 调用
libcuvslam.so
  = 真正执行视觉惯性里程计的 cuVSLAM SDK 引擎
```

我们现在要克隆的 `NVIDIA-ISAAC-ROS/isaac_ros_visual_slam@v3.2-15` 是 ROS2 封装源码，不是 cuVSLAM 算法核心源码。真正的 cuVSLAM SDK 12.6 已由 Isaac ROS 镜像配套提供。

而你的 `u5-4/cuVSLAM` 是更新的 core17 源码，使用新的 C++ API；Isaac ROS 3.2 wrapper 使用旧的 `CUVSLAM_*` C API，两者不能直接连接。

因此当前路线是：

```
D435i
  -> RealSense ROS
  -> isaac_ros_visual_slam v3.2-15
  -> 镜像内 cuVSLAM SDK 12.6
  -> /visual_slam/tracking/odometry
```

所以现在克隆 Visual SLAM wrapper 到 `~/workspaces/isaac_ros_3_2/src`；暂时不要把 `u5-4/cuVSLAM` core17 放进 `~/catkin_ws/src`。







镜像安排设计（待验证）

更适合量产和算法迭代的架构。不过建议不要把“规划”和“飞控执行”塞进同一个镜像，应进一步拆开安全边界。

```
固定基础层
┌─────────────────────────────────────┐
│ localization-runtime                │
│ RealSense + Isaac ROS + NITROS      │
│ Visual SLAM + cuVSLAM SDK 12.6      │
│ 统一里程计、TF、状态、深度接口       │
└─────────────────────────────────────┘
                    │
          标准 ROS2 DDS 接口
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  YOPO 镜像     EGO-Planner   Super Planner
                  镜像           镜像
```

对于需要稠密地图的规划器，再插入可选模块：

```
localization-runtime
        │
        ├── odometry
        └── depth/pointcloud
                   │
            mapping-runtime
          voxel / occupancy / ESDF
                   │
        EGO-Planner / Super Planner
```

典型组合就是：

```
YOPO：
localization + yopo

EGO-Planner：
localization + mapping + ego-planner

Super Planner：
localization + mapping + super-planner

自研：
localization + 可选 mapping + custom-planner
```

关键是固定一套与规划器无关的接口契约：

```
/localization/odometry       nav_msgs/msg/Odometry
/localization/status         diagnostic_msgs/msg/DiagnosticArray
/camera/depth/image          sensor_msgs/msg/Image
/camera/depth/camera_info    sensor_msgs/msg/CameraInfo
/tf                          map / odom / base_link / camera_link
```

各规划器通过 remap 转成自己需要的话题，例如 YOPO 的：

```
/localization/odometry -> /state/odom
/camera/depth/image     -> /depth_image
```

这样 localization 镜像完全不知道后面接的是 YOPO、EGO-Planner 还是其他算法。

量产镜像建议分成：

- `isaac-vslam-runtime`：唯一拥有 D435i，发布定位、深度和健康状态。
- `mapping-runtime`：可选，负责点云、体素地图、ESDF。
- `yopo-runtime`、`ego-planner-runtime`、`super-planner-runtime`：互斥选择一个规划器。
- `vehicle-control-gateway`：未来单独建立，负责 PX4/MAVROS/控制安全门；当前阶段不启用。

再通过 Docker Compose profiles 选择组合：

```
docker compose --profile yopo up
docker compose --profile ego up
docker compose --profile super up
```

这些镜像可以继承同一个 Jetson ROS2 基础镜像，因此公共层只保存一次，不会因为模块化就成倍占用磁盘。

需要特别明确：`isaac-vslam-runtime` 中使用的是 Isaac ROS 3.2 配套的 cuVSLAM SDK 12.6，不构建 `u5-4/cuVSLAM` core17。你的 fork 用于保存 adapter、Dockerfile、Compose、配置和兼容文档。

这个结构比一个大镜像更适合多规划器组合，也能让定位、建图、规划和控制分别升级、测试及回滚。









最终结果是 `[PASS]`。前面的 `[STOP]` 是粘贴过程混入的残留文本，实际统计显示：

- 全部 `53868` 条 `vo_state=1`
- 全部 `53867` 条 `vo_status=0`
- 时间戳零值和回退均为 0
- CameraInfo 完全同步
- 只有一次 `22.255 ms` gap
- `66.348 ms` 是一次计算尖峰，未导致 tracking lost

**当前用了什么标定**

你虽然没有手工标定，但 D435i 出厂时已经写入了一套标定数据：

| 参数                            | 当前来源                                 | 是否在线重算 |
| ------------------------------- | ---------------------------------------- | ------------ |
| IR1/IR2 内参 `fx/fy/cx/cy`      | D435i EEPROM，经 `CameraInfo` 发布       | 否           |
| 畸变、校正矩阵                  | D435i EEPROM，经 `CameraInfo D/R/P` 发布 | 否           |
| IR1 与 IR2 双目外参/基线        | D435i 出厂标定和 TF                      | 否           |
| 相机与 D435i 内置 IMU 外参      | RealSense 静态 TF                        | 否           |
| IMU bias、重力、速度、运动状态  | cuVSLAM 跟踪过程                         | 会在线估计   |
| D435i 到无人机 `base_link` 外参 | 当前没有                                 | 必须后续标定 |

wrapper 会在初始化时读取一次 CameraInfo 和 TF，构造固定 camera rig，见 [visual_slam_impl.cpp (line 195)](/C:/Users/10416/Documents/Codex/2026-07-16/w-m/work/isaac_ros_visual_slam-v3.2-15/isaac_ros_visual_slam/src/impl/visual_slam_impl.cpp:195)。初始化完成后，后续 CameraInfo 不再更新；内参转换逻辑见 [cuvslam_ros_conversion.cpp (line 157)](/C:/Users/10416/Documents/Codex/2026-07-16/w-m/work/isaac_ros_visual_slam-v3.2-15/isaac_ros_visual_slam/src/impl/cuvslam_ros_conversion.cpp:157)。

`calibration_frequency=200` 容易误解。它只是 IMU 模型的采样频率，与噪声密度、随机游走和固定 `rig_from_imu` 一起传入 cuVSLAM，并不是在线重新标定内外参。

**现在验证的是什么**

当前测试证明了：

- D435i 出厂标定与 cuVSLAM 能正常配合。
- 双目、IMU、时间戳、GPU 跟踪和 ROS 输出稳定。
- 连续 10 分钟没有 tracking lost。
- 当前固件无需为了兼容性立即刷写。

但它还不能证明“绝对定位很精准”，因为没有真值轨迹、已知距离或高精度姿态作为对照。

**后续必须标定的部分**

最重要的是 D435i 安装到无人机后的刚体外参：

```
camera_link -> base_link
```

当前官方 launch 把 `base_frame` 直接设置为 `camera_link`，所以输出：

```
odom -> camera_link
```

这表示当前测得的是相机位姿，不是无人机机体中心位姿。接入 YOPO 前必须得到相机到机体的平移和旋转，并验证 15°/0° 相机支架的真实安装误差。

建议顺序：

1. 保留 D435i 出厂内参、双目外参和相机-IMU 外参作为第一版。
2. 用现有 rosbag 中的 CameraInfo 与 `/tf_static` 归档这套出厂数据。
3. 标定或精确测量 `camera_link -> base_link`。
4. 使用已知直线距离和已知角度验证尺度、方向和漂移。
5. 只有发现双目重投影误差或 VIO 明显漂移时，才进行 Kalibr/RealSense 专用的相机-IMU或双目标定。

因此：cuVSLAM 有在线状态估计能力，但当前 Isaac ROS 3.2 wrapper不能替代完整的内外参标定。最关键的无人机安装外参仍需要我们后续单独完成。





开发容器快照

  --filter name=isaac_ros_dev-aarch64-container

echo "========== Preserved images =========="
docker image inspect \
  isaac_ros_dev-aarch64:latest \
  isaac_ros_dev-aarch64:isaac3.2-rs2.55.1-vslam3.2-dev-20260717 \
  isaac_ros_dev-aarch64:isaac3.2-rs2.55.1-jp6.2 \
  --format 'tags={{json .RepoTags}} id={{.Id}}'
========== Container ==========
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
========== Preserved images ==========
tags=["isaac_ros_dev-aarch64:isaac3.2-rs2.55.1-vslam3.2-dev-20260717","isaac_ros_dev-aarch64:latest"] id=sha256:ddd233211dc6cd29aaa6eaf6b317931e912f0ed057a762496e8df390904474dc
tags=["isaac_ros_dev-aarch64:isaac3.2-rs2.55.1-vslam3.2-dev-20260717","isaac_ros_dev-aarch64:latest"] id=sha256:ddd233211dc6cd29aaa6eaf6b317931e912f0ed057a762496e8df390904474dc
tags=["isaac_ros_dev-aarch64:isaac3.2-rs2.55.1-jp6.2"] id=sha256:bceda07bd17756dc582693e00fc388fb5b2589c94d8267e6a253546640f8a933









位置是对的。你没看到 `cuVSLAM` 源码是正常的，因为当前实际运行的是容器里的 cuVSLAM SDK 二进制，不是 `u5-4/cuVSLAM` 源码。

| 组件                    | 位置                                                       | 作用                                   |
| ----------------------- | ---------------------------------------------------------- | -------------------------------------- |
| `isaac_ros_common`      | `~/workspaces/isaac_ros_3_2/src/`                          | 容器和开发环境                         |
| `isaac_ros_visual_slam` | `~/workspaces/isaac_ros_3_2/src/`                          | ROS2 wrapper、launch、消息接口         |
| `libcuvslam.so`         | 容器 `/opt/ros/humble/share/isaac_ros_nitros/cuvslam/lib/` | 真正执行 VIO 的 cuVSLAM SDK 12.6       |
| `u5-4/cuVSLAM`          | 尚未克隆到 Jetson                                          | core17 源码及后续 integration 托管仓库 |

构建以后，SDK 还被复制到宿主工作区的 overlay：

```
ls -lh \
  ~/workspaces/isaac_ros_3_2/install/isaac_ros_visual_slam/lib/libcuvslam.so
```

容器内可以确认原始 SDK：

```
ls -lh \
  /opt/ros/humble/share/isaac_ros_nitros/cuvslam/lib/libcuvslam.so
```

所以当前结构没有缺文件：

```
isaac_ros_visual_slam 源码
        ↓ 调用
容器内 libcuvslam.so SDK 12.6
```

`u5-4/cuVSLAM` 不应直接克隆到 `isaac_ros_3_2/src`，否则容易让 core17 与 SDK 12.6 混入同一构建。下一阶段开始时，建议单独放在：

```
/home/nvidia/workspaces/cuVSLAM
```

只将其中的 `integrations/isaac_ros_3_2_yopo/` 作为 ROS2 集成代码参与 Isaac 工作区构建；core17 本身不构建、不链接。





前半部分正确，后半部分需要收紧一下。

Docker 镜像里已经包含 cuVSLAM SDK 12.6，`isaac_ros_visual_slam` wrapper 也已经构建并成功运行。因此不需要为了运行当前 cuVSLAM，再克隆和编译 `u5-4/cuVSLAM` 的 core17 源码。

当前状态是：

| 项目                               | 状态                     |
| ---------------------------------- | ------------------------ |
| cuVSLAM SDK 安装、GPU 运行         | 已完成                   |
| Isaac ROS Visual SLAM wrapper 构建 | 已完成                   |
| D435i 双目 + D435i IMU 官方基线    | 已完成                   |
| 里程计、状态、10 分钟稳定性        | 已完成                   |
| 飞控 IMU 接入                      | 未完成                   |
| 相机—飞控 IMU 空间外参             | 未完成                   |
| 相机—飞控时钟偏移和同步            | 未完成                   |
| 飞控坐标系转换、噪声参数           | 未完成                   |
| 外部 IMU 专用 launch               | 未完成                   |
| cuVSLAM 到 YOPO 的接口和健康门控   | 未完成                   |
| 可复现生产 Dockerfile              | 未完成，目前只是开发快照 |

所以准确结论是：

> cuVSLAM 的开发环境部署和 NVIDIA 官方 D435i 基线已经完成；面向实际飞行系统的适配尚未完成。

后续仍会克隆 `u5-4/cuVSLAM`，但目的只是新增：

```
integrations/isaac_ros_3_2_yopo/
```

里面放飞控 IMU 适配、外部 launch、标定配置和 YOPO adapter。不会编译或修改 core17。

而且剩余工作不只是一个外参矩阵，还包括时间同步、IMU 坐标转换、噪声参数、运行时话题接入以及重新验收。完成这些后，才能称“生产用途的 cuVSLAM 适配收尾”。







当前阶段：

标定

统一坐标系











我们可以写一个录包的launch 文件 可以仿照这个D:\catkin_ws\src\image_bag_recorder目录下的程序来另外写一个我们专门录用程序的（参考的那个是标定四目鱼眼时录制四目相机灰度图像用的）  然后内参的话D435出场就已经标定好了，所以我们只需要录包 飞控IMU和相机的应该录制的话题（灰度还是原画质）





ummary: 1 package finished [57.2s]

========== Overlay ==========

/workspaces/isaac_ros-dev/install/isaac_ros_visual_slam

========== Libraries ==========

-rw-r--r-- 1 admin admin 4.5M Oct 26  2025 install/isaac_ros_visual_slam/lib/libcuvslam.so

-rw-r--r-- 1 admin admin 5.4M Jul 20 02:38 install/isaac_ros_visual_slam/lib/libvisual_slam_node.so

========== Missing libraries ==========

[PASS] No missing dynamic libraries

[PASS] Patched Visual SLAM build completed

admin@tegra-ubuntu:/workspaces/isaac_ros-de    这个是CUslam的












<!-- END DESKTOP_AGENT_HISTORY -->

</details>
