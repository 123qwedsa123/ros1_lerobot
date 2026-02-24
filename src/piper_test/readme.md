# Piper MoveIt Click-Run (Real Arm)

这个方案基于 MoveIt，支持两种交互方式：

1. `拖动`：在 RViz 的 MotionPlanning 里拖动交互 marker，点 `Plan`/`Execute`。
2. `点点+Run`：在 RViz 用 `Publish Point` 点一个三维点，然后在弹出的 `Piper Click-Run` 小窗口点 `Run Clicked`。
3. `Run XYZ`：不需要找 `Publish Point`，直接在弹窗输入 XYZ 后点 `Run XYZ`。
4. `先预览再执行`：先点 `Preview XYZ` 或 `Preview Clicked`，在 RViz 看最终状态/路径，再点 `Run`。

两种方式都会走 MoveIt，可继续用于后续避障。

## 1. 准备 CAN

如果你把 CAN 接口重命名为 `slave3`（示例）：

```bash
cd ~/Desktop/piper_sdk/piper_sdk
bash can_activate.sh slave3 1000000 3-8:1.0
ip -br link show type can
```

确认看到 `slave3 UP`。

## 2. 编译

```bash
cd /home/jinhe/Desktop/piper_master_slave_ws/piper_master_slave_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg piper_test
source devel/setup.bash
```

## 3. 启动

```bash
roslaunch piper_test moveit_teaching_realtime.launch can_port:=slave3
```

参数说明：

- `can_port`：CAN 口名（比如 `slave3` 或 `can0`）
- `enable_click_run`：是否启用“点点+Run”节点（默认 true）
- `show_click_gui`：是否显示 Run 按钮窗口（默认 true）

例如无弹窗模式：

```bash
roslaunch piper_test moveit_teaching_realtime.launch can_port:=slave3 show_click_gui:=false
```

## 4. 操作方式

### A) 拖动模式（MoveIt 原生）

1. RViz 左上角选择 `MotionPlanning`。
2. 在末端交互球上拖动目标位姿。
3. 点 `Plan`，再点 `Execute`。

### B) 点击模式（点点+Run）

1. 在 RViz 顶部工具栏选择 `Publish Point`（一个小圆点图标）。
2. 在场景中点击目标点（会发布到 `/clicked_point`）。
3. 在弹窗先点 `Preview Clicked` 看效果。
4. 确认后点 `Run Clicked`。

### C) 直接坐标模式（最容易）

1. 看弹出的 `Piper Click-Run` 窗口。
2. 直接填写 `X Y Z`。
3. 先点 `Preview XYZ`。
4. 确认后点 `Run XYZ`。

如果你先点过 RViz 点，也可以点 `Fill From Clicked` 自动填入 XYZ。

说明：
- `Preview XYZ` / `Preview Clicked`：只预览，不下发真机。
- `Run XYZ` / `Run Clicked`：把规划轨迹下发到 `joint_ctrl_single`（真机控制话题），不是只做 fake execution。

可选：在 RViz 添加 `Marker` display，topic 设为 `/clicked_point_marker`，可看到点击点小球。

## 5. 快速自检

启动后应有实时关节数据：

```bash
rostopic hz /joint_states_single
rostopic hz /joint_states
```

如果没有数据，优先检查：

- `can_port` 是否和当前系统 CAN 名一致
- `ip -br link show type can` 是否为 `UP`
- `piper_ctrl_single_node` 是否报 `CAN socket ... does not exist`

如果你更新了代码但行为没变化，重新编译并重新 source：

```bash
cd /home/jinhe/Desktop/piper_master_slave_ws/piper_master_slave_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg piper_test
source devel/setup.bash
```
