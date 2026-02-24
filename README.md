# Piper Master-Slave Workspace (ROS1 + LeRobot)

English and Chinese documentation are both included in this file.

## English

### Overview

This repository contains a ROS1 workspace for Piper dual-arm workflows:

- Bimanual teleoperation with force feedback
- Rosbag recording and replay
- 2-arm 3-camera ACT policy inference
- Bag-to-LeRobot dataset conversion and training helpers

### Repository Layout

- `src/piper_ros`: Piper ROS drivers, messages, and hardware interfaces
- `src/piper_test`: launch/config/scripts for teleop, recording, replay, and inference
- `src/piper_moveit`: MoveIt configurations and related motion planning code
- `lerobot-server`: dataset conversion and training utility scripts
- `data`: recorded episodes and runtime outputs

### Environment

- Ubuntu 20.04 + ROS Noetic
- Python 3 with ROS Python dependencies
- CAN interfaces configured for your master/slave arms
- Intel RealSense cameras for camera-based workflows
- Optional conda env (for LeRobot/ACT), e.g. `lerobot-mujoco`

### Build

```bash
cd /workspace/piper_master_slave_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg piper_test
source devel/setup.bash
```

### Common Workflows

1. Teleop + rosbag recording:

```bash
roslaunch piper_test teleop_raw_record_piperros_ff.launch
```

Config file: `src/piper_test/config/teleop_raw_record_piperros_ff.yaml`

2. Replay recorded episodes:

```bash
roslaunch piper_test teleop_raw_replay.launch
```

Config file: `src/piper_test/config/teleop_raw_replay.yaml`

3. ACT direct inference (2 arms + 3 cameras):

```bash
roslaunch piper_test act_infer_2arm3cam_direct.launch
```

Before running, check paths in:

- `src/piper_test/launch/act_infer_2arm3cam_direct.launch`
- `src/piper_test/config/act_infer_2arm3cam_direct.yaml`

4. Convert bag data to LeRobot format:

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
bash convert.sh --config config/bag_convert_config.yaml
```

5. Run ACT training:

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
python run_train.py --config config/train_config.yaml
```

### Data Layout

Recorded episodes follow:

`data/<session_name>/episode_XXX/episode.bag`  
`data/<session_name>/episode_XXX/metadata.json`

### Git Notes

Large generated files are intentionally ignored in `.gitignore`, including model/data outputs such as:

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- `lerobot-server/act_data/`
- `lerobot-server/output/`

## 中文

### 项目简介

这个仓库是一个基于 ROS1 的 Piper 双臂工作空间，主要包含：

- 双臂主从遥操作（含力反馈）
- rosbag 录制与回放
- 双臂三相机 ACT 推理
- bag 到 LeRobot 数据集转换与训练辅助脚本

### 目录说明

- `src/piper_ros`：Piper ROS 驱动、消息定义和硬件接口
- `src/piper_test`：遥操作/录制/回放/推理相关 launch、config、scripts
- `src/piper_moveit`：MoveIt 配置和运动规划相关代码
- `lerobot-server`：数据集转换与训练工具脚本
- `data`：录制数据与运行时输出

### 环境要求

- Ubuntu 20.04 + ROS Noetic
- Python 3 与 ROS Python 依赖
- 已正确配置主从机械臂对应的 CAN 口
- 需要相机流程时，准备 Intel RealSense
- 如需 LeRobot/ACT，建议使用 conda 环境（例如 `lerobot-mujoco`）

### 编译

```bash
cd /workspace/piper_master_slave_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg piper_test
source devel/setup.bash
```

### 常用流程

1. 遥操作 + rosbag 录制：

```bash
roslaunch piper_test teleop_raw_record_piperros_ff.launch
```

配置文件：`src/piper_test/config/teleop_raw_record_piperros_ff.yaml`

2. 回放录制数据：

```bash
roslaunch piper_test teleop_raw_replay.launch
```

配置文件：`src/piper_test/config/teleop_raw_replay.yaml`

3. ACT 直连推理（双臂三相机）：

```bash
roslaunch piper_test act_infer_2arm3cam_direct.launch
```

运行前请先检查：

- `src/piper_test/launch/act_infer_2arm3cam_direct.launch`
- `src/piper_test/config/act_infer_2arm3cam_direct.yaml`

4. 将 bag 转换为 LeRobot 数据集：

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
bash convert.sh --config config/bag_convert_config.yaml
```

5. 启动 ACT 训练：

```bash
cd /workspace/piper_master_slave_ws/lerobot-server
python run_train.py --config config/train_config.yaml
```

### 数据目录结构

录制输出目录结构如下：

`data/<session_name>/episode_XXX/episode.bag`  
`data/<session_name>/episode_XXX/metadata.json`

### Git 说明

仓库默认会忽略大体积生成文件（权重/数据产物），包括：

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- `lerobot-server/act_data/`
- `lerobot-server/output/`
