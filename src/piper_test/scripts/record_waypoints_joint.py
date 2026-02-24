#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piper 机械臂路点记录工具 (关节角版本, OpenCV 可视化)
启动方式: roslaunch piper_test record_waypoints.launch
"""

import rospy
import cv2
import csv
import os
import sys
import time
import importlib
import numpy as np
from datetime import datetime
from std_msgs.msg import Bool

# ===================== 全局状态 =====================
current_joints = [None] * 7  # 7个关节
waypoints = []
flash_until = 0
cfg = {}
joint_topic_candidates = []
joint_topic_hint = ""
active_joint_topic = ""

def get_param(name):
    if not rospy.has_param("~" + name):
        rospy.logfatal("[CONFIG] 缺少必要参数: %s", name)
        sys.exit(1)
    return rospy.get_param("~" + name)

def load_all_params():
    keys = [
        "can_port", "camera_device", "camera_width", "camera_height",
        "joint_state_topic", "enable_topic", "joint_state_msg_type",
        "toggle_enable_during_recording",
        "joint_field_prefix", "num_joints",
        "crosshair_color", "crosshair_thickness", "center_radius",
        "save_dir", "key_record", "key_undo", "key_quit",
        "flash_color", "flash_duration_ms",
    ]
    for k in keys:
        cfg[k] = get_param(k)

def build_topic_candidates(can_port, topic_name):
    """返回候选话题: 先 /{can}/topic，再 /topic，并去重"""
    base = str(topic_name).strip().lstrip("/")
    can = str(can_port).strip().strip("/")
    topics = []
    if can:
        topics.append("/{}/{}".format(can, base))
    topics.append("/{}".format(base))

    uniq = []
    for t in topics:
        if t not in uniq:
            uniq.append(t)
    return uniq

def import_message_type(type_name):
    """支持 TypeName 或 package/TypeName 两种写法"""
    type_name = str(type_name).strip()
    if not type_name:
        raise ValueError("empty message type")

    if "/" in type_name:
        pkg, cls = type_name.split("/", 1)
        mod = importlib.import_module("{}.msg".format(pkg))
        return getattr(mod, cls)

    # 兼容旧配置: 先尝试 piper_msgs，再尝试 sensor_msgs
    for pkg in ("piper_msgs", "sensor_msgs"):
        try:
            mod = importlib.import_module("{}.msg".format(pkg))
            return getattr(mod, type_name)
        except (ImportError, AttributeError):
            pass
    raise AttributeError("message type '{}' not found".format(type_name))

def _extract_joints_from_msg(msg, n, prefix):
    """兼容 JointState(position[]) 与 joint1..jointN 两类消息"""
    if hasattr(msg, "position"):
        pos = list(getattr(msg, "position"))
        if len(pos) < n:
            rospy.logwarn_throttle(2.0, "[CALLBACK] JointState.position 长度不足: %d < %d", len(pos), n)
            return None
        return pos[:n]

    try:
        return [getattr(msg, "{}{}".format(prefix, i + 1)) for i in range(n)]
    except AttributeError:
        return None

def joint_state_callback(msg, topic_name=None):
    global current_joints, active_joint_topic
    n = cfg["num_joints"]
    prefix = cfg["joint_field_prefix"]
    joints = _extract_joints_from_msg(msg, n, prefix)
    if joints is not None:
        current_joints = joints
        if topic_name and active_joint_topic != topic_name:
            active_joint_topic = topic_name
            rospy.loginfo("[JOINT] 正在接收关节话题: %s", topic_name)
        return

    fields = [s for s in dir(msg) if not s.startswith('_')]
    rospy.logerr_once(
        "[CALLBACK] 无法从消息中提取关节，期望 position[] 或 %s1..%s%d，可用字段: %s",
        prefix, prefix, n, fields
    )

def set_enable(pubs, enabled):
    if not isinstance(pubs, (list, tuple)):
        pubs = [pubs]
    msg = Bool(data=enabled)
    for _ in range(5):
        for pub in pubs:
            pub.publish(msg)
        rospy.sleep(0.05)
    rospy.loginfo("[ENABLE] 机械臂已%s", "使能" if enabled else "去使能")

def open_camera():
    dev = cfg["camera_device"]
    cap = cv2.VideoCapture(dev if isinstance(dev, str) else int(dev))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera_height"])
        rospy.loginfo("[CAM] 相机已打开: %s", dev)
        return cap
    rospy.logwarn("[CAM] 无法打开相机: %s，将使用黑屏模式", dev)
    return None

def draw_crosshair(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = tuple(cfg["crosshair_color"])
    cv2.line(frame, (cx, 0), (cx, h), color, cfg["crosshair_thickness"])
    cv2.line(frame, (0, cy), (w, cy), color, cfg["crosshair_thickness"])
    cv2.circle(frame, (cx, cy), cfg["center_radius"], color, -1)

def draw_osd(frame):
    font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    n = cfg["num_joints"]
    y = 25

    def put(text):
        nonlocal y
        cv2.putText(frame, text, (10, y), font, scale, color, thick)
        y += 22

    if current_joints[0] is not None:
        row1 = " ".join(["J{}:{:.3f}".format(i+1, current_joints[i]) for i in range(min(4, n))])
        put(row1)
        if n > 4:
            row2 = " ".join(["J{}:{:.3f}".format(i+1, current_joints[i]) for i in range(4, n)])
            put(row2)
    else:
        topic = active_joint_topic if active_joint_topic else joint_topic_hint
        if not topic:
            topic = "/{}/{}".format(cfg["can_port"], str(cfg["joint_state_topic"]).lstrip("/"))
        put("Waiting for joint data from {}".format(topic))

    put("Waypoints: {}".format(len(waypoints)))
    if waypoints:
        wp = waypoints[-1]
        brief = " ".join(["J{}:{:.2f}".format(i+1, wp['joints'][i]) for i in range(min(3, n))])
        put("Last: #{} {}...".format(wp['id'], brief))

    h = frame.shape[0]
    keys = "[{}]Record [{}]Undo [{}]Quit".format(
        cfg["key_record"], cfg["key_undo"], cfg["key_quit"])
    cv2.putText(frame, keys, (10, h - 15), font, 0.5, (200, 200, 200), 1)

def draw_flash(frame):
    global flash_until
    if time.time() < flash_until:
        cv2.rectangle(frame, (0, 0),
                      (frame.shape[1] - 1, frame.shape[0] - 1),
                      tuple(cfg["flash_color"]), 4)

def save_image(frame, wp_id):
    img_dir = os.path.join(cfg["save_dir"], "images")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, "wp_{:04d}.png".format(wp_id))
    cv2.imwrite(path, frame)

def save_waypoints():
    if not waypoints:
        rospy.logwarn("[SAVE] 无路点，跳过保存")
        return
    n = cfg["num_joints"]
    os.makedirs(cfg["save_dir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cfg["save_dir"], "waypoints_{}.csv".format(ts))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['id'] + ['joint_{}'.format(i+1) for i in range(n)]
        w.writerow(header)
        for wp in waypoints:
            w.writerow([wp['id']] + list(wp['joints']))
    rospy.loginfo("[SAVE] %d 个路点已保存: %s", len(waypoints), path)

def main():
    global flash_until, joint_topic_candidates, joint_topic_hint

    rospy.init_node('record_waypoints', anonymous=True)
    load_all_params()

    can = cfg["can_port"]
    joint_topic_candidates = build_topic_candidates(can, cfg["joint_state_topic"])
    enable_topic_candidates = build_topic_candidates(can, cfg["enable_topic"])
    joint_topic_hint = joint_topic_candidates[0]
    rospy.loginfo("[INIT] CAN: %s | 关节话题候选: %s | 使能话题候选: %s",
                  can, joint_topic_candidates, enable_topic_candidates)

    # 动态导入消息类型
    try:
        JointMsg = import_message_type(cfg["joint_state_msg_type"])
    except Exception as e:
        rospy.logfatal("[IMPORT] 无法导入消息类型 %s: %s",
                       cfg["joint_state_msg_type"], e)
        sys.exit(1)

    enable_pubs = [
        rospy.Publisher(topic, Bool, queue_size=1)
        for topic in enable_topic_candidates
    ]
    for topic in joint_topic_candidates:
        rospy.Subscriber(topic, JointMsg, joint_state_callback, callback_args=topic, queue_size=1)
    rospy.sleep(1.0)
    if cfg["toggle_enable_during_recording"]:
        rospy.logwarn("[ENABLE] 已配置为录点时自动去使能，请注意机械臂下坠风险。")
        set_enable(enable_pubs, False)
    else:
        rospy.loginfo("[ENABLE] 已禁用自动切换使能，程序不会主动去使能机械臂。")

    cap = open_camera()
    win_name = "Piper Waypoint Recorder (Joint Mode)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cfg["camera_width"], cfg["camera_height"])

    w_cam, h_cam = cfg["camera_width"], cfg["camera_height"]
    key_map = {
        ord(cfg["key_record"]): "record",
        ord(cfg["key_undo"]): "undo",
        ord(cfg["key_quit"]): "quit",
    }

    try:
        while not rospy.is_shutdown():
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((h_cam, w_cam, 3), dtype=np.uint8)
            else:
                frame = np.zeros((h_cam, w_cam, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera - Waiting...",
                            (w_cam // 2 - 140, h_cam // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            draw_crosshair(frame)
            draw_osd(frame)
            draw_flash(frame)
            cv2.imshow(win_name, frame)

            key = cv2.waitKey(30) & 0xFF
            action = key_map.get(key)

            if action == "record" and current_joints[0] is not None:
                wp = {
                    'id': len(waypoints) + 1,
                    'joints': list(current_joints[:cfg["num_joints"]]),
                }
                waypoints.append(wp)
                save_image(frame, wp['id'])
                flash_until = time.time() + cfg["flash_duration_ms"] / 1000.0
                rospy.loginfo("[REC] 路点 #%d 已记录: %s", wp['id'], wp['joints'])

            elif action == "undo" and waypoints:
                removed = waypoints.pop()
                rospy.loginfo("[UNDO] 路点 #%d 已撤销", removed['id'])

            elif action == "quit":
                break

            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        save_waypoints()
        if cfg["toggle_enable_during_recording"]:
            set_enable(enable_pubs, True)
        else:
            rospy.loginfo("[ENABLE] 退出时保持当前使能状态（未发送 enable_flag）。")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        rospy.loginfo("[EXIT] 程序退出")

if __name__ == '__main__':
    main()
