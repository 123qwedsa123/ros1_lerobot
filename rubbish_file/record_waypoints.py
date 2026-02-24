#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piper 机械臂路点记录工具
功能：去使能后手动拖动机械臂，按键记录末端 6DOF 位姿(xyz+rpy)，退出时保存 CSV 并重新使能。
"""

import rospy
import yaml
import csv
import os
import sys
import select
import termios
import tty
import signal
import importlib
from datetime import datetime
from std_msgs.msg import Bool

# ===================== 全局状态 =====================
current_pose = [None] * 6          # 实时 [x, y, z, roll, pitch, yaw]
waypoints = []                      # 已记录的路点列表
original_term_settings = None       # 终端原始设置，用于退出时恢复
config = {}                         # 配置字典


def load_config(config_path):
    """从 YAML 加载配置，失败则抛出异常帮助 debug"""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        rospy.loginfo("[CONFIG] 配置加载成功: %s", config_path)
        return cfg
    except FileNotFoundError:
        rospy.logfatal("[CONFIG] 配置文件不存在: %s", config_path)
        raise
    except yaml.YAMLError as e:
        rospy.logfatal("[CONFIG] YAML 解析错误: %s", e)
        raise


def end_pose_callback(msg):
    """订阅 /end_pose_euler 的回调，更新实时 6DOF 位姿"""
    global current_pose
    try:
        current_pose = [msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw]
    except AttributeError as e:
        # 打印消息的所有字段，帮助排查
        fields = [s for s in dir(msg) if not s.startswith('_')]
        rospy.logerr_once("[CALLBACK] 字段错误: %s, 可用字段: %s", e, fields)


def set_enable(pub, enabled):
    """发布使能/去使能指令"""
    msg = Bool()
    msg.data = enabled
    for _ in range(5):
        pub.publish(msg)
        rospy.sleep(0.05)
    state = "使能" if enabled else "去使能"
    rospy.loginfo("[ENABLE] 机械臂已%s", state)


def get_key_nonblocking(timeout=0.1):
    """非阻塞读取单个按键，超时返回 None"""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def raw_print(text):
    """raw 模式下正确打印（\n 替换为 \r\n）"""
    sys.stdout.write(text.replace('\n', '\r\n'))
    sys.stdout.flush()


def clear_and_print_status():
    """清屏并打印实时 6DOF 位姿和操作提示"""
    # ANSI 清屏 + 光标归位
    sys.stdout.write("\033[2J\033[H")

    lines = []
    lines.append("=" * 55)
    lines.append("     Piper 机械臂路点记录工具 (6DOF)")
    lines.append("=" * 55)

    if current_pose[0] is not None:
        lines.append("")
        lines.append("  实时末端位姿:")
        lines.append("    X:     {:>10.6f} m".format(current_pose[0]))
        lines.append("    Y:     {:>10.6f} m".format(current_pose[1]))
        lines.append("    Z:     {:>10.6f} m".format(current_pose[2]))
        lines.append("    Roll:  {:>10.4f} rad".format(current_pose[3]))
        lines.append("    Pitch: {:>10.4f} rad".format(current_pose[4]))
        lines.append("    Yaw:   {:>10.4f} rad".format(current_pose[5]))
    else:
        lines.append("")
        lines.append("  等待末端位姿数据...")

    lines.append("")
    lines.append("  已记录路点数: {}".format(len(waypoints)))

    if waypoints:
        lines.append("  最近记录:")
        for wp in waypoints[-3:]:
            lines.append("    #{:>3d}  X:{:.3f} Y:{:.3f} Z:{:.3f} R:{:.2f} P:{:.2f} Y:{:.2f}".format(
                wp['id'], wp['x'], wp['y'], wp['z'],
                wp['roll'], wp['pitch'], wp['yaw']))

    lines.append("")
    lines.append("-" * 55)
    lines.append("  [{}] 记录  [{}] 撤销  [{}] 保存并退出".format(
        config['key_record'], config['key_undo'], config['key_quit']))
    lines.append("-" * 55)

    raw_print('\n'.join(lines))


def save_waypoints():
    """将路点保存为 CSV 文件"""
    if not waypoints:
        rospy.logwarn("[SAVE] 没有记录任何路点，跳过保存")
        return None

    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "waypoints_{}.csv".format(timestamp)
    filepath = os.path.join(save_dir, filename)

    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
            for wp in waypoints:
                writer.writerow([
                    wp['id'], wp['x'], wp['y'], wp['z'],
                    wp['roll'], wp['pitch'], wp['yaw']
                ])
        rospy.loginfo("[SAVE] 已保存 %d 个路点到: %s", len(waypoints), filepath)
        return filepath
    except IOError as e:
        rospy.logerr("[SAVE] 保存失败: %s", e)
        return None


def cleanup(enable_pub):
    """退出清理：保存数据、重新使能、恢复终端"""
    global original_term_settings
    # 先恢复终端
    if original_term_settings is not None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_term_settings)
        original_term_settings = None

    print("\n正在退出...")
    filepath = save_waypoints()
    if filepath:
        print("路点已保存到: {}".format(filepath))
    else:
        print("未记录任何路点。")

    set_enable(enable_pub, True)
    print("机械臂已重新使能，请注意安全！")


def main():
    global config, original_term_settings

    rospy.init_node('record_waypoints', anonymous=True)

    # 从 ROS 参数获取配置文件路径
    config_path = rospy.get_param('~config_path', 'record_waypoints_config.yaml')
    config = load_config(config_path)

    # 动态导入消息类型（从配置文件读取类名）
    msg_type_name = config.get('end_pose_msg_type', 'PiperEulerPose')
    try:
        piper_msgs_module = importlib.import_module('piper_msgs.msg')
        EndPoseMsg = getattr(piper_msgs_module, msg_type_name)
        rospy.loginfo("[IMPORT] 消息类型: piper_msgs/%s", msg_type_name)
    except (ImportError, AttributeError) as e:
        rospy.logfatal("[IMPORT] 无法导入 piper_msgs.msg.%s，"
                       "请确认已 source devel/setup.bash: %s", msg_type_name, e)
        sys.exit(1)

    # 创建 publisher 和 subscriber
    enable_pub = rospy.Publisher(
        config['enable_topic'], Bool, queue_size=1)
    rospy.Subscriber(
        config['end_pose_topic'], EndPoseMsg, end_pose_callback, queue_size=1)

    rospy.sleep(1.0)

    # 去使能，允许自由拖动
    set_enable(enable_pub, False)

    # 信号处理
    def signal_handler(sig, frame):
        cleanup(enable_pub)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # 设置终端为非阻塞原始模式
    original_term_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    try:
        while not rospy.is_shutdown():
            clear_and_print_status()
            key = get_key_nonblocking(1.0 / config['display_rate'])

            if key == config['key_record']:
                if current_pose[0] is not None:
                    wp = {
                        'id': len(waypoints) + 1,
                        'x': current_pose[0],
                        'y': current_pose[1],
                        'z': current_pose[2],
                        'roll': current_pose[3],
                        'pitch': current_pose[4],
                        'yaw': current_pose[5],
                    }
                    waypoints.append(wp)

            elif key == config['key_undo']:
                if waypoints:
                    waypoints.pop()

            elif key == config['key_quit']:
                break

    finally:
        cleanup(enable_pub)


if __name__ == '__main__':
    main()