# Piper Master-Slave Workspace (ROS1 + LeRobot)

This README includes both English and Chinese documentation.

## English

### 1. Scope

This repo is a ROS1 workspace for Piper dual-arm workflows:

- teleop + force feedback data collection
- rosbag replay
- bag -> LeRobot dataset conversion
- ACT training helpers
- ACT online deployment (2 arms + 3 cameras)

Main folders:

- `src/piper_ros`: low-level Piper ROS drivers / interfaces
- `src/piper_test`: launch/config/scripts for teleop, replay, and inference
- `src/piper_moveit`: MoveIt related packages/configs
- `lerobot-server`: conversion + training scripts
- `data`: local recording outputs

### 2. Environment and Build

Recommended base:

- Ubuntu 20.04
- ROS Noetic
- Python 3
- Intel RealSense for camera pipelines
- Conda env for LeRobot training/inference (for example `lerobot-mujoco`)

Build:

```bash
cd /workspace/piper_master_slave_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg piper_test
source devel/setup.bash
```

Dependency snapshots saved in this repo:

- `env/conda_lerobot-mujoco.yml`
- `env/conda_lerobot-mujoco_explicit.txt`
- `env/conda_lerobot-mujoco_pip_freeze.txt`
- `env/ros_noetic_apt_packages.txt`
- `env/ros_system_python_pip_freeze.txt`

Install before running (recommended baseline):

```bash
cd /workspace/piper_master_slave_ws

# 1) Conda env for training/inference
conda env create -f env/conda_lerobot-mujoco.yml || true
conda env update -n lerobot-mujoco -f env/conda_lerobot-mujoco.yml

# 2) ROS packages snapshot (requires sudo)
sudo apt-get update
xargs -a env/ros_noetic_apt_packages.txt sudo apt-get install -y

# 3) System python packages used with ROS (optional, depends on your host image)
/usr/bin/python3 -m pip install -r env/ros_system_python_pip_freeze.txt
```

Refresh snapshots later:

```bash
cd /workspace/piper_master_slave_ws
bash env/export_env_snapshots.sh lerobot-mujoco
```

### 3. Teleop Recording (bag)

Launch:

```bash
roslaunch piper_test teleop_raw_record_piperros_ff.launch
```

With custom config:

```bash
roslaunch piper_test teleop_raw_record_piperros_ff.launch \
  config:=$(rospack find piper_test)/config/teleop_raw_record_piperros_ff.yaml
```

Config file:

- `src/piper_test/config/teleop_raw_record_piperros_ff.yaml`

Default recording topics come from:

- each arm pair:
  - `/{master}/joint_states_single`
  - `/{slave}/joint_states_single`
- each camera:
  - if `rosbag.camera_transport: compressed` -> `/realsense_{name}/color/image_raw/compressed`
  - else -> `/realsense_{name}/color/image_raw`
  - depth topic is included only when camera depth is enabled and save_depth is true

Keyboard controls in current recorder:

- `SPACE`: start/stop recording (stop = save episode)
- `Q`: quit (if recording, current episode is stopped first)

Flexible recorder variant (generalized topic config + save/discard flow):

```bash
roslaunch piper_test teleop_raw_record_piperros_ff_flexible.launch
```

Flexible config:

- `src/piper_test/config/teleop_raw_record_piperros_ff_flexible.yaml`

Flexible key bindings:

- `SPACE`: start recording; if currently recording, stop and discard
- `S`: stop and save
- `D`: stop and discard
- `Q`: quit; if currently recording, discard current episode

Flexible topic controls (`rosbag` section in flexible yaml):

- `topic_mode`: `default` / `custom` / `default_plus_custom` / `all` / `all_plus_custom`
- `record_topics`: explicit topic list with template variables
- `additional_topics`: always appended topics
- `include_topic_regex`: include filter
- `exclude_topic_regex`: exclude filter

Output layout:

- `data/<session_name>/episode_XXX/episode.bag`
- `data/<session_name>/episode_XXX/metadata.json`
- `data/<session_name>/episode_XXX/rosbag_record.log`

Important knobs in config:

- `rosbag.session_prefix` / `rosbag.session_name`
- `rosbag.lz4`
- `rosbag.camera_transport`
- `arm_pairs` (master/slave namespaces and CAN names)
- `cameras` (serial, resolution, fps)

### 4. Replay

Launch:

```bash
roslaunch piper_test teleop_raw_replay.launch
```

Config:

- `src/piper_test/config/teleop_raw_replay.yaml`

Most useful replay fields:

- `replay.episode_file`
- `replay.session_name`
- `replay.episode_id` (`-1` means latest)
- `replay.speed_scale`
- `safety_reset.*` (pre-replay slow reset policy)

### 5. Convert bag to LeRobot

Scripts:

- `lerobot-server/bag_to_lerobot.py`
- config: `lerobot-server/config/bag_convert_config.yaml`

Phase 1 (ROS environment, bag -> v2.1):

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
source /opt/ros/noetic/setup.bash
python3 bag_to_lerobot.py --config config/bag_convert_config.yaml --phase 1
```

Phase 2 (LeRobot environment, v2.1 -> v3.0):

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
conda activate lerobot-mujoco
python bag_to_lerobot.py --config config/bag_convert_config.yaml --phase 2
```

Notes:

- `convert.session_dir` in `bag_convert_config.yaml` must point to your `episode_*/episode.bag` folder.
- `convert.clean_output: true` will clear old dataset output before conversion.
- Output is generated under `convert.output_root/convert.dataset_id`.

Streaming producer-consumer converter (convert while teleop is still recording):

- `lerobot-server/convert_stream.py`
- `lerobot-server/convert_stream.sh`

Run in another terminal while teleop recorder is active:

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
bash convert_stream.sh --config config/bag_convert_config.yaml
```

Useful flags:

- `--poll-sec 1.0`: faster directory polling
- `--settle-sec 2.0`: wait time before a bag is treated as closed
- `--once`: convert ready episodes once and exit
- `--idle-exit-sec 120`: exit after idle timeout
- `--phase2-on-exit`: run v2.1 -> v3.0 once on watcher exit (if enabled in config)

### 6. Training (ACT)

Script:

- `lerobot-server/run_train.py`
- config: `lerobot-server/config/train_config.yaml`

Dry run (recommended first):

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
python run_train.py --config config/train_config.yaml --dry-run
```

Run training:

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
python run_train.py --config config/train_config.yaml
```

Key config sections:

- `runtime.command`: training executable path (recommended absolute path)
- `runtime.env`: env overrides (for example `CUDA_VISIBLE_DEVICES`)
- `variables`: reusable template variables
- `train_args`: nested args auto-expanded to CLI flags
- `flags` / `raw_args`: extra direct CLI options

Resume training:

- set `train_args.resume: true`
- set `train_args.config_path` to a checkpoint `train_config.json`

### 7. Deploy (Online Inference)

Primary deploy path in this repo:

- launch: `src/piper_test/launch/act_infer_2arm3cam_direct.launch`
- node: `src/piper_test/scripts/act_infer_2arm3cam_direct.py`
- config: `src/piper_test/config/act_infer_2arm3cam_direct.yaml`

Run:

```bash
roslaunch piper_test act_infer_2arm3cam_direct.launch
```

Typical override when moving to another machine:

```bash
roslaunch piper_test act_infer_2arm3cam_direct.launch \
  conda_python:=/path/to/conda/env/bin/python \
  conda_site_packages:=/path/to/conda/env/lib/python3.10/site-packages \
  infer_config:=$(rospack find piper_test)/config/act_infer_2arm3cam_direct.yaml
```

Deploy checklist:

- update CAN names (`slave1_can`, `slave2_can`) in launch args
- update RealSense serial numbers in launch args
- set `model.checkpoint_dir` in inference config
- verify topics:
  - states: `/slave1/joint_states_single`, `/slave2/joint_states_single`
  - images: `/cam_left/color/image_raw`, `/cam_right/color/image_raw`, `/cam_top/color/image_raw`
  - command outputs defined by config topics (`left_cmd`, `right_cmd`)

Checkpoint expectation:

- inference checkpoint directory should contain at least:
  - `config.json`
  - `model.safetensors`

### 8. Git / Large Files

Large generated artifacts should stay ignored (already in `.gitignore`), e.g.:

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`
- `lerobot-server/act_data/`
- `lerobot-server/output/`
- rosbag / datasets / runtime outputs

## 中文

### 1. 项目范围

这个仓库是 Piper 双臂在 ROS1 下的工作空间，覆盖：

- 遥操作 + 力反馈数据采集
- rosbag 回放
- bag -> LeRobot 数据集转换
- ACT 训练辅助
- ACT 在线部署（双臂三相机）

核心目录：

- `src/piper_ros`：Piper 底层驱动与接口
- `src/piper_test`：teleop / replay / infer 的 launch、config、脚本
- `src/piper_moveit`：MoveIt 相关包
- `lerobot-server`：转换与训练脚本
- `data`：本地录制数据

### 2. 环境与编译

推荐环境：

- Ubuntu 20.04
- ROS Noetic
- Python 3
- 相机流程使用 Intel RealSense
- LeRobot 训练/推理建议使用 conda（例如 `lerobot-mujoco`）

编译：

```bash
cd /workspace/piper_master_slave_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg piper_test
source devel/setup.bash
```

仓库里已保存的依赖快照：

- `env/conda_lerobot-mujoco.yml`
- `env/conda_lerobot-mujoco_explicit.txt`
- `env/conda_lerobot-mujoco_pip_freeze.txt`
- `env/ros_noetic_apt_packages.txt`
- `env/ros_system_python_pip_freeze.txt`

运行前建议安装（基线）：

```bash
cd /workspace/piper_master_slave_ws

# 1) 训练/推理 conda 环境
conda env create -f env/conda_lerobot-mujoco.yml || true
conda env update -n lerobot-mujoco -f env/conda_lerobot-mujoco.yml

# 2) ROS apt 依赖快照（需要 sudo）
sudo apt-get update
xargs -a env/ros_noetic_apt_packages.txt sudo apt-get install -y

# 3) ROS 使用的系统 python 包（可选，按你的主机镜像决定）
/usr/bin/python3 -m pip install -r env/ros_system_python_pip_freeze.txt
```

后续刷新依赖快照：

```bash
cd /workspace/piper_master_slave_ws
bash env/export_env_snapshots.sh lerobot-mujoco
```

### 3. Teleop 录制（bag）

启动：

```bash
roslaunch piper_test teleop_raw_record_piperros_ff.launch
```

指定配置：

```bash
roslaunch piper_test teleop_raw_record_piperros_ff.launch \
  config:=$(rospack find piper_test)/config/teleop_raw_record_piperros_ff.yaml
```

配置文件：

- `src/piper_test/config/teleop_raw_record_piperros_ff.yaml`

默认会录制的话题来源：

- 每个 arm pair：
  - `/{master}/joint_states_single`
  - `/{slave}/joint_states_single`
- 每个相机：
  - 若 `rosbag.camera_transport: compressed` -> `/realsense_{name}/color/image_raw/compressed`
  - 否则 -> `/realsense_{name}/color/image_raw`
  - depth 只有在相机开启 depth 且 save_depth=true 时才录

当前 recorder 键盘逻辑：

- `SPACE`：开始/停止录制（停止即保存）
- `Q`：退出（若在录制会先停止当前段）

增强版 recorder（可泛化话题配置 + 保存/丢弃流程）：

```bash
roslaunch piper_test teleop_raw_record_piperros_ff_flexible.launch
```

增强版配置：

- `src/piper_test/config/teleop_raw_record_piperros_ff_flexible.yaml`

增强版按键：

- `SPACE`：开始录制；若正在录制则停止并丢弃
- `S`：停止并保存
- `D`：停止并丢弃
- `Q`：退出；若正在录制会丢弃当前段

增强版话题控制（配置文件 `rosbag` 段）：

- `topic_mode`：`default` / `custom` / `default_plus_custom` / `all` / `all_plus_custom`
- `record_topics`：显式录制话题列表（支持模板变量）
- `additional_topics`：总是附加的话题
- `include_topic_regex`：包含过滤
- `exclude_topic_regex`：排除过滤

输出目录结构：

- `data/<session_name>/episode_XXX/episode.bag`
- `data/<session_name>/episode_XXX/metadata.json`
- `data/<session_name>/episode_XXX/rosbag_record.log`

常改配置项：

- `rosbag.session_prefix` / `rosbag.session_name`
- `rosbag.lz4`
- `rosbag.camera_transport`
- `arm_pairs`（命名空间与 CAN）
- `cameras`（序列号、分辨率、帧率）

### 4. 回放

启动：

```bash
roslaunch piper_test teleop_raw_replay.launch
```

配置文件：

- `src/piper_test/config/teleop_raw_replay.yaml`

关键参数：

- `replay.episode_file`
- `replay.session_name`
- `replay.episode_id`（`-1` 表示最新）
- `replay.speed_scale`
- `safety_reset.*`（回放前慢速复位策略）

### 5. bag 转 LeRobot

脚本：

- `lerobot-server/bag_to_lerobot.py`
- 配置：`lerobot-server/config/bag_convert_config.yaml`

Phase 1（ROS 环境，bag -> v2.1）：

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
source /opt/ros/noetic/setup.bash
python3 bag_to_lerobot.py --config config/bag_convert_config.yaml --phase 1
```

Phase 2（LeRobot 环境，v2.1 -> v3.0）：

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
conda activate lerobot-mujoco
python bag_to_lerobot.py --config config/bag_convert_config.yaml --phase 2
```

说明：

- `bag_convert_config.yaml` 里的 `convert.session_dir` 必须指向包含 `episode_*/episode.bag` 的目录。
- `convert.clean_output: true` 会先清理旧输出再转换。
- 输出目录为 `convert.output_root/convert.dataset_id`。

边录边转（生产者-消费者）脚本：

- `lerobot-server/convert_stream.py`
- `lerobot-server/convert_stream.sh`

在录制终端之外另开一个终端运行：

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
bash convert_stream.sh --config config/bag_convert_config.yaml
```

常用参数：

- `--poll-sec 1.0`：更快轮询
- `--settle-sec 2.0`：bag 关闭稳定等待时间
- `--once`：只转换当前就绪 episode 后退出
- `--idle-exit-sec 120`：空闲超时退出
- `--phase2-on-exit`：watcher 退出时执行一次 v2.1 -> v3.0（需配置允许）

### 6. 训练（ACT）

脚本：

- `lerobot-server/run_train.py`
- 配置：`lerobot-server/config/train_config.yaml`

建议先 dry-run：

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
python run_train.py --config config/train_config.yaml --dry-run
```

正式训练：

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
python run_train.py --config config/train_config.yaml
```

关键配置块：

- `runtime.command`：训练可执行命令（建议绝对路径）
- `runtime.env`：环境变量（如 `CUDA_VISIBLE_DEVICES`）
- `variables`：模板变量
- `train_args`：嵌套参数会自动展开为 CLI 参数
- `flags` / `raw_args`：附加参数

断点续训：

- 设置 `train_args.resume: true`
- 设置 `train_args.config_path` 指向 checkpoint 的 `train_config.json`

### 7. 部署（在线推理）

当前主路径：

- launch：`src/piper_test/launch/act_infer_2arm3cam_direct.launch`
- 节点：`src/piper_test/scripts/act_infer_2arm3cam_direct.py`
- 配置：`src/piper_test/config/act_infer_2arm3cam_direct.yaml`

运行：

```bash
roslaunch piper_test act_infer_2arm3cam_direct.launch
```

跨机器常用覆盖参数：

```bash
roslaunch piper_test act_infer_2arm3cam_direct.launch \
  conda_python:=/path/to/conda/env/bin/python \
  conda_site_packages:=/path/to/conda/env/lib/python3.10/site-packages \
  infer_config:=$(rospack find piper_test)/config/act_infer_2arm3cam_direct.yaml
```

部署检查项：

- launch 里更新 CAN（`slave1_can`、`slave2_can`）
- launch 里更新相机序列号
- infer config 里设置 `model.checkpoint_dir`
- 确认话题：
  - 状态：`/slave1/joint_states_single`、`/slave2/joint_states_single`
  - 图像：`/cam_left/color/image_raw`、`/cam_right/color/image_raw`、`/cam_top/color/image_raw`
  - 输出命令话题由 infer config 的 `left_cmd`、`right_cmd` 指定

checkpoint 最低要求：

- 目录里至少有：
  - `config.json`
  - `model.safetensors`

### 8. Git 与大文件

大体积生成文件建议保持忽略（`.gitignore` 已包含），例如：

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`
- `lerobot-server/act_data/`
- `lerobot-server/output/`
- bag / 数据集 / 运行时输出
