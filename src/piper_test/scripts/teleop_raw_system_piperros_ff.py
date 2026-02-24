#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置驱动的遥操作原始数据录制系统（piper_ros 风格力反馈版）
- 从 YAML 读取所有参数，动态启动 master/slave/camera 节点
- 保存格式：rosbag（对齐 cheese-zj/catkin_ws 格式）
- 异步后台写 metadata.json
- 键盘: [SPACE]录制/停止  [Q]退出（OpenCV窗口聚焦时）
"""

import os, sys, time, queue, threading, yaml, csv, subprocess, glob, signal, re, json, socket
import importlib.util
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import cv2

import rospy
import roslaunch
from sensor_msgs.msg import JointState, Image
from piper_sdk import C_PiperInterface

WINDOW_NAME = 'Teleop Cameras'


def ensure_pinocchio_discovery():
    if importlib.util.find_spec('pinocchio') is not None:
        return True
    ros_distro = os.environ.get('ROS_DISTRO', 'noetic').strip() or 'noetic'
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        f"/opt/ros/{ros_distro}/lib/python{py_ver}/site-packages",
        f"/opt/ros/{ros_distro}/lib/python{py_ver}/dist-packages",
    ]
    candidates.extend(sorted(glob.glob("/opt/ros/*/lib/python*/site-packages")))
    candidates.extend(sorted(glob.glob("/opt/ros/*/lib/python*/dist-packages")))
    changed = False
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
            changed = True
    if changed:
        importlib.invalidate_caches()
    return importlib.util.find_spec('pinocchio') is not None


# ============================================================
# 工具
# ============================================================

def fix_len(x, n, fill=0.0):
    x = list(x) if x else []
    return (x + [fill] * n)[:n]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def as_float(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)

def as_bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in ('1', 'true', 'yes', 'on'):
            return True
        if lv in ('0', 'false', 'no', 'off'):
            return False
    return bool(default)

def as_list6(v, default):
    if isinstance(v, list):
        vals = v[:]
    elif isinstance(v, tuple):
        vals = list(v)
    else:
        try:
            scalar = float(v)
            vals = [scalar] * 6
        except (TypeError, ValueError):
            vals = list(default)
    if len(vals) == 0:
        vals = list(default)
    if len(vals) != 6:
        vals = (vals + [vals[0]] * 6)[:6]
    out = []
    for item in vals:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(float(default[0]))
    return out

def img_from_msg(msg):
    if msg.height <= 0 or msg.width <= 0:
        return None
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < msg.height * msg.step:
        return None
    return buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)

def resolve_rospack_file(package_name, relpath):
    try:
        pkg_path = (
            subprocess.check_output(["rospack", "find", package_name], stderr=subprocess.STDOUT)
            .strip().decode("utf-8")
        )
    except Exception as exc:
        return None, f"rospack find {package_name} failed: {exc}"
    abs_path = os.path.abspath(os.path.join(pkg_path, relpath))
    if not os.path.exists(abs_path):
        return None, f"file not found: {abs_path}"
    return abs_path, None


# ============================================================
# 主系统
# ============================================================

class TeleopRawSystem:

    def __init__(self):
        rospy.init_node('teleop_raw_system_piperros_ff', anonymous=True)

        config_file = rospy.get_param('~config_file')
        with open(config_file) as f:
            self.cfg = yaml.safe_load(f)

        g = self.cfg['global']
        self.rate_hz = g['rate_hz']

        # ---------- Rosbag 配置（对齐 cheese-zj）----------
        rb_cfg = self.cfg.get('rosbag', {})
        self.rosbag_lz4 = as_bool(rb_cfg.get('lz4', True), True)
        self.camera_transport = str(rb_cfg.get('camera_transport', 'raw')).strip()
        session_prefix = str(rb_cfg.get('session_prefix', 'act')).strip()
        session_name = str(rb_cfg.get('session_name', '')).strip()

        # session 目录结构（对齐 cheese-zj）
        data_dir = Path(g['data_dir']).expanduser().resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        if session_name:
            self.session_dir = data_dir / session_name
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = data_dir / f"{session_prefix}_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        save_cfg = g.get('save', {})
        self.save_queue_size = int(save_cfg.get('save_queue_size', 2))

        self.rec_cfg = self.cfg['recording']
        self.dim = self.rec_cfg['per_arm_dim']

        self.arm_pairs = self.cfg['arm_pairs']
        self.cameras = self.cfg['cameras']
        self.topics_tpl = self.cfg['topics']
        self.cam_topics_tpl = self.cfg['camera_topics']
        self.ctrl_cmd_cfg = self.cfg['ctrl_cmd']

        # ---------- 力反馈配置（不变）----------
        fb_cfg = self.cfg.get('force_feedback', {})
        self.force_fb_backend = str(fb_cfg.get('backend', 'piperros_node')).strip().lower()
        if self.force_fb_backend not in ('legacy_sdk', 'piperros_node'):
            self.force_fb_backend = 'piperros_node'

        self.force_fb = {
            'enabled': as_bool(fb_cfg.get('enabled', False), False),
            'pos_diff_threshold': max(0.001, as_float(fb_cfg.get('pos_diff_threshold', 0.02), 0.02)),
            'mit_kp_blocked': as_float(fb_cfg.get('mit_kp_blocked', 30.0), 30.0),
            'mit_kp_free': as_float(fb_cfg.get('mit_kp_free', 0.0), 0.0),
            'mit_kd': as_float(fb_cfg.get('mit_kd', 1.0), 1.0),
            'joint_ema_alpha': clamp(as_float(fb_cfg.get('joint_ema_alpha', 0.4), 0.4), 0.0, 1.0),
            'joint_gain': max(0.0, as_float(fb_cfg.get('joint_gain', 10.0), 10.0)),
            'gripper_contact_threshold': max(0.0, as_float(fb_cfg.get('gripper_contact_threshold', 0.05), 0.05)),
            'gripper_gain': max(0.0, as_float(fb_cfg.get('gripper_gain', 1.0), 1.0)),
            'gripper_base_effort': as_float(fb_cfg.get('gripper_base_effort', 0.5), 0.5),
            'gripper_min_effort': as_float(fb_cfg.get('gripper_min_effort', 0.5), 0.5),
            'gripper_max_effort': as_float(fb_cfg.get('gripper_max_effort', 2.0), 2.0),
            'gripper_ema_alpha': clamp(as_float(fb_cfg.get('gripper_ema_alpha', 0.2), 0.2), 0.0, 1.0),
            'csv_log': as_bool(fb_cfg.get('csv_log', True), True),
            'csv_path': str(fb_cfg.get('csv_path', '/tmp/force_feedback_log.csv')),
            'csv_flush_hz': max(0.1, as_float(fb_cfg.get('csv_flush_hz', 5.0), 5.0)),
        }
        fb = self.force_fb
        if fb['gripper_max_effort'] < fb['gripper_min_effort']:
            fb['gripper_max_effort'] = fb['gripper_min_effort']
        fb['gripper_base_effort'] = clamp(
            fb['gripper_base_effort'], fb['gripper_min_effort'], fb['gripper_max_effort']
        )

        self.ff_use_legacy = fb['enabled'] and self.force_fb_backend == 'legacy_sdk'
        self.ff_use_piperros_node = fb['enabled'] and self.force_fb_backend == 'piperros_node'

        ff_node_cfg = fb_cfg.get('node', {})
        mit_torque_scale_default = as_list6(
            ff_node_cfg.get('mit_torque_scale', [0.5, 0.15, 0.6, 0.5, 0.15, 0.6]),
            [0.5, 0.15, 0.6, 0.5, 0.15, 0.6],
        )
        mit_torque_feedback_sign_default = as_list6(
            ff_node_cfg.get('mit_torque_feedback_sign', [-1.0]*6), [-1.0]*6,
        )
        self.force_fb_node = {
            'auto_enable': as_bool(ff_node_cfg.get('auto_enable', True), True),
            'gripper_exist': as_bool(ff_node_cfg.get('gripper_exist', True), True),
            'gripper_val_multiple': as_float(ff_node_cfg.get('gripper_val_multiple', 1.0), 1.0),
            'enable_gripper': as_bool(ff_node_cfg.get('enable_gripper', True), True),
            'enable_gripper_haptic': as_bool(ff_node_cfg.get('enable_gripper_haptic', True), True),
            'gripper_range': as_float(ff_node_cfg.get('gripper_range', 0.08), 0.08),
            'gripper_reverse': as_bool(ff_node_cfg.get('gripper_reverse', False), False),
            'ctrl_mode': str(ff_node_cfg.get('ctrl_mode', 'mit')).strip().lower(),
            'p_speed': max(0, min(100, int(as_float(ff_node_cfg.get('p_speed', 30), 30)))),
            'mit_speed': max(0, min(100, int(as_float(ff_node_cfg.get('mit_speed', 50), 50)))),
            'mit_kp': as_list6(ff_node_cfg.get('mit_kp', [0.02]*6), [0.02]*6),
            'mit_kd': as_list6(ff_node_cfg.get('mit_kd', [0.02]*6), [0.02]*6),
            'mit_enable_pos': as_bool(ff_node_cfg.get('mit_enable_pos', True), True),
            'mit_enable_vel': as_bool(ff_node_cfg.get('mit_enable_vel', False), False),
            'mit_enable_tor': as_bool(ff_node_cfg.get('mit_enable_tor', True), True),
            'mit_enable_gravity': as_bool(ff_node_cfg.get('mit_enable_gravity', True), True),
            'mit_gravity_mix_mode': str(ff_node_cfg.get('mit_gravity_mix_mode', 'additive')).strip(),
            'mit_torque_scale': mit_torque_scale_default,
            'mit_torque_scale_left': as_list6(ff_node_cfg.get('mit_torque_scale_left', mit_torque_scale_default), mit_torque_scale_default),
            'mit_torque_scale_right': as_list6(ff_node_cfg.get('mit_torque_scale_right', mit_torque_scale_default), mit_torque_scale_default),
            'mit_torque_feedback_sign': mit_torque_feedback_sign_default,
            'mit_torque_feedback_sign_left': as_list6(ff_node_cfg.get('mit_torque_feedback_sign_left', mit_torque_feedback_sign_default), mit_torque_feedback_sign_default),
            'mit_torque_feedback_sign_right': as_list6(ff_node_cfg.get('mit_torque_feedback_sign_right', mit_torque_feedback_sign_default), mit_torque_feedback_sign_default),
            'mit_max_torque_abs': as_float(ff_node_cfg.get('mit_max_torque_abs', 18.0), 18.0),
            'enforce_joint_limits': as_bool(ff_node_cfg.get('enforce_joint_limits', True), True),
            'gripper_haptic_effort_sign': as_float(ff_node_cfg.get('gripper_haptic_effort_sign', 0.0), 0.0),
            'gripper_haptic_effort_deadband': as_float(ff_node_cfg.get('gripper_haptic_effort_deadband', 0.15), 0.15),
            'gripper_haptic_effort_bias': as_float(ff_node_cfg.get('gripper_haptic_effort_bias', 0.0), 0.0),
            'gripper_haptic_effort_max': as_float(ff_node_cfg.get('gripper_haptic_effort_max', 0.6), 0.6),
            'gripper_haptic_cmd_max': int(as_float(ff_node_cfg.get('gripper_haptic_cmd_max', 500), 500)),
            'gripper_haptic_cmd_enable_threshold': int(as_float(ff_node_cfg.get('gripper_haptic_cmd_enable_threshold', 80), 80)),
            'gripper_haptic_effort_alpha': as_float(ff_node_cfg.get('gripper_haptic_effort_alpha', 0.2), 0.2),
            'gripper_haptic_release_when_opening': as_bool(ff_node_cfg.get('gripper_haptic_release_when_opening', True), True),
            'gripper_haptic_opening_relief': as_float(ff_node_cfg.get('gripper_haptic_opening_relief', 0.15), 0.15),
            'gripper_haptic_opening_sign': as_float(ff_node_cfg.get('gripper_haptic_opening_sign', 0.0), 0.0),
            'gripper_haptic_opening_direction_eps': as_float(ff_node_cfg.get('gripper_haptic_opening_direction_eps', 0.0001), 0.0001),
            'publish_rate': as_float(ff_node_cfg.get('publish_rate', 100.0), 100.0),
            'control_rate': as_float(ff_node_cfg.get('control_rate', 50.0), 50.0),
            'subscribe_rate': as_float(ff_node_cfg.get('subscribe_rate', 50.0), 50.0),
            'filter_enable': as_bool(ff_node_cfg.get('filter_enable', True), True),
            'filter_alpha_position': as_float(ff_node_cfg.get('filter_alpha_position', 0.7), 0.7),
            'filter_alpha_velocity': as_float(ff_node_cfg.get('filter_alpha_velocity', 0.5), 0.5),
            'filter_alpha_effort': as_float(ff_node_cfg.get('filter_alpha_effort', 0.5), 0.5),
            'master_slave_enable': as_bool(ff_node_cfg.get('master_slave_enable', True), True),
            'master_slave_flag_topic': str(ff_node_cfg.get('master_slave_flag_topic', '/conrft_robot/slave_follow_flag')),
            'master_slave_kp_follow': as_list6(ff_node_cfg.get('master_slave_kp_follow', [7.0]*6), [7.0]*6),
            'shutdown_mode': str(ff_node_cfg.get('shutdown_mode', 'hold')).strip().lower(),
        }

        gcfg = fb_cfg.get('gravity_compensation', {})
        gravity_joint_scale_default = as_list6(gcfg.get('gravity_joint_scale', [1.0]*6), [1.0]*6)
        gravity_joint_sign_default = as_list6(gcfg.get('gravity_joint_sign', [1.0]*6), [1.0]*6)
        gravity_joint_pos_scale_default = as_list6(gcfg.get('gravity_joint_position_scale', [1.0]*6), [1.0]*6)
        gravity_joint_offset_default = as_list6(gcfg.get('gravity_joint_offset', [0.0]*6), [0.0]*6)
        self.gravity_comp = {
            'enabled': as_bool(gcfg.get('enabled', True), True),
            'urdf_package': str(gcfg.get('urdf_package', 'piper_x_description')).strip(),
            'urdf_relpath': str(gcfg.get('urdf_relpath', 'urdf/piper_x_description.urdf')).strip(),
            'gravity_joint_scale_left': as_list6(gcfg.get('gravity_joint_scale_left', gravity_joint_scale_default), gravity_joint_scale_default),
            'gravity_joint_scale_right': as_list6(gcfg.get('gravity_joint_scale_right', gravity_joint_scale_default), gravity_joint_scale_default),
            'gravity_joint_sign_left': as_list6(gcfg.get('gravity_joint_sign_left', gravity_joint_sign_default), gravity_joint_sign_default),
            'gravity_joint_sign_right': as_list6(gcfg.get('gravity_joint_sign_right', gravity_joint_sign_default), gravity_joint_sign_default),
            'gravity_joint_position_scale_left': as_list6(gcfg.get('gravity_joint_position_scale_left', gravity_joint_pos_scale_default), gravity_joint_pos_scale_default),
            'gravity_joint_position_scale_right': as_list6(gcfg.get('gravity_joint_position_scale_right', gravity_joint_pos_scale_default), gravity_joint_pos_scale_default),
            'gravity_joint_offset_left': as_list6(gcfg.get('gravity_joint_offset_left', gravity_joint_offset_default), gravity_joint_offset_default),
            'gravity_joint_offset_right': as_list6(gcfg.get('gravity_joint_offset_right', gravity_joint_offset_default), gravity_joint_offset_default),
            'clip_to_limits': as_bool(gcfg.get('clip_to_limits', True), True),
            'unwrap_joint_positions': as_bool(gcfg.get('unwrap_joint_positions', True), True),
            'max_joint_step': as_float(gcfg.get('max_joint_step', 0.08), 0.08),
            'max_torque_delta_warn': as_float(gcfg.get('max_torque_delta_warn', 2.0), 2.0),
        }
        self.gravity_urdf_path, self.gravity_urdf_error = resolve_rospack_file(
            self.gravity_comp['urdf_package'], self.gravity_comp['urdf_relpath']
        )

        rd_cfg = self.cfg.get('robot_description', {})
        self.robot_description = {
            'enabled': as_bool(rd_cfg.get('enabled', True), True),
            'package': str(rd_cfg.get('package', self.gravity_comp['urdf_package'])).strip(),
            'relpath': str(rd_cfg.get('relpath', self.gravity_comp['urdf_relpath'])).strip(),
            'param': str(rd_cfg.get('param', '/robot_description')).strip() or '/robot_description',
        }
        self.robot_description_loaded = False
        self.robot_description_error = None

        self.pinocchio_available = ensure_pinocchio_discovery()
        self.runtime_gravity_enabled = (
            self.ff_use_piperros_node
            and self.force_fb_node['mit_enable_gravity']
            and self.gravity_comp['enabled']
            and self.gravity_urdf_path is not None
            and self.pinocchio_available
            and len(self.arm_pairs) >= 2
        )

        self._master_pipers = {}
        if self.ff_use_legacy:
            for pair in self.arm_pairs:
                can_port = pair['master_can']
                piper = C_PiperInterface(can_name=can_port)
                piper.ConnectPort()
                self._master_pipers[pair['name']] = piper

        self._csv_file = None
        self._csv_writer = None
        self._csv_last_flush = 0.0
        if self.ff_use_legacy and fb['csv_log']:
            self._csv_file = open(fb['csv_path'], 'w', newline='')
            header = ['timestamp', 'pair']
            for i in range(6):
                header += [f'pos_diff_{i}', f'blend_{i}', f'kp_{i}']
            header += ['grip_eff', 'max_blend']
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(header)

        # ---------- 运行时状态 ----------
        self.arm_state = {}
        self.cam_state = {}
        self.is_recording = False
        self.frame_count = 0
        self.rec_start_time = None

        # 当前 episode 状态（对齐 cheese-zj）
        self.rosbag_proc = None
        self.current_ep_name = None
        self.current_ep_dir = None
        self.current_started_utc = None
        self.current_topics = []
        self.ep_count = self._next_ep_index()  # 1-indexed，对齐 cheese-zj

        # 异步保存（写 metadata.json，对齐 cheese-zj）
        self.save_q = queue.Queue(maxsize=self.save_queue_size)
        self.save_done = 0
        self.save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bag_save")
        threading.Thread(target=self._save_worker, daemon=True).start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        init_w = 480 * max(len(self.cameras), 1)
        cv2.resizeWindow(WINDOW_NAME, init_w, 410)

        self._load_robot_description_param()
        self._launch_child_nodes()
        self._setup_ros()
        self._print_banner()

    # ====================== 子节点启动（不变）======================

    def _load_robot_description_param(self):
        rd = self.robot_description
        if not rd['enabled']:
            return
        urdf_path, err = resolve_rospack_file(rd['package'], rd['relpath'])
        if urdf_path is None:
            self.robot_description_error = err
            rospy.logwarn("[启动] robot_description 加载失败: %s", err)
            return
        try:
            with open(urdf_path, 'r') as f:
                urdf_text = f.read()
            rospy.set_param('/robot_description', urdf_text)
            if rd['param'] != '/robot_description':
                rospy.set_param(rd['param'], urdf_text)
            self.robot_description_loaded = True
        except Exception as exc:
            self.robot_description_error = str(exc)
            rospy.logwarn("[启动] robot_description 写入失败: %s", str(exc))

    def _launch_child_nodes(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launcher = roslaunch.scriptapi.ROSLaunch()
        self.launcher.start()

        defaults = self.cfg.get('piper_ctrl_defaults', {})

        for pair_idx, pair in enumerate(self.arm_pairs):
            for role in ('master', 'slave'):
                ns = pair[role]
                prefix = f'/{ns}/piper_ctrl_node'
                if role == 'master' and self.ff_use_piperros_node:
                    ff = self.force_fb_node
                    gravity_for_master = ff['mit_enable_gravity'] and self.runtime_gravity_enabled
                    arm_side = 'left' if pair_idx == 0 else 'right'
                    master_joint_topic = self.topics_tpl['master_joint_states_tpl'].format(master=pair['master'])
                    slave_joint_topic = self.topics_tpl['slave_joint_states_tpl'].format(slave=pair['slave'])
                    slave_comp_tpl = self.topics_tpl.get('slave_joint_states_compensated_tpl', '/{slave}/joint_states_compensated')
                    slave_comp_topic = slave_comp_tpl.format(slave=pair['slave'])
                    torque_scale = ff['mit_torque_scale_left'] if arm_side == 'left' else ff['mit_torque_scale_right']
                    torque_feedback_sign = ff['mit_torque_feedback_sign_left'] if arm_side == 'left' else ff['mit_torque_feedback_sign_right']

                    params = {
                        'can_port': pair['master_can'], 'topic_prefix': f'/{pair["master"]}/',
                        'auto_enable': ff['auto_enable'], 'gripper_exist': ff['gripper_exist'],
                        'gripper_val_mutiple': ff['gripper_val_multiple'],
                        'enable_gripper': ff['enable_gripper'], 'enable_gripper_haptic': ff['enable_gripper_haptic'],
                        'gripper_range': ff['gripper_range'], 'gripper_reverse': ff['gripper_reverse'],
                        'ctrl_mode': ff['ctrl_mode'], 'p/speed': ff['p_speed'], 'mit/speed': ff['mit_speed'],
                        'mit/kp': ff['mit_kp'], 'mit/kd': ff['mit_kd'],
                        'mit/enable_pos': ff['mit_enable_pos'], 'mit/enable_vel': ff['mit_enable_vel'],
                        'mit/enable_tor': ff['mit_enable_tor'], 'mit/enable_gravity': gravity_for_master,
                        'mit/gravity_mix_mode': ff['mit_gravity_mix_mode'],
                        'mit/torque_scale': torque_scale, 'mit/torque_feedback_sign': torque_feedback_sign,
                        'mit/max_torque_abs': ff['mit_max_torque_abs'],
                        'enforce_joint_limits': ff['enforce_joint_limits'],
                        'gripper_haptic_effort_sign': ff['gripper_haptic_effort_sign'],
                        'gripper_haptic_effort_deadband': ff['gripper_haptic_effort_deadband'],
                        'gripper_haptic_effort_bias': ff['gripper_haptic_effort_bias'],
                        'gripper_haptic_effort_max': ff['gripper_haptic_effort_max'],
                        'gripper_haptic_cmd_max': ff['gripper_haptic_cmd_max'],
                        'gripper_haptic_cmd_enable_threshold': ff['gripper_haptic_cmd_enable_threshold'],
                        'gripper_haptic_effort_alpha': ff['gripper_haptic_effort_alpha'],
                        'gripper_haptic_release_when_opening': ff['gripper_haptic_release_when_opening'],
                        'gripper_haptic_opening_relief': ff['gripper_haptic_opening_relief'],
                        'gripper_haptic_opening_sign': ff['gripper_haptic_opening_sign'],
                        'gripper_haptic_opening_direction_eps': ff['gripper_haptic_opening_direction_eps'],
                        'publish_rate': ff['publish_rate'], 'control_rate': ff['control_rate'],
                        'subscribe_rate': ff['subscribe_rate'], 'filter/enable': ff['filter_enable'],
                        'filter/alpha_position': ff['filter_alpha_position'],
                        'filter/alpha_velocity': ff['filter_alpha_velocity'],
                        'filter/alpha_effort': ff['filter_alpha_effort'],
                        'master_slave/enable': ff['master_slave_enable'],
                        'master_slave/master_position_topic': slave_joint_topic,
                        'master_slave/master_flag_topic': ff['master_slave_flag_topic'],
                        'master_slave/kp_follow': ff['master_slave_kp_follow'],
                        'shutdown_mode': ff['shutdown_mode'],
                        'remap/joint_pos_cmd_to': master_joint_topic,
                        'remap/joint_tor_cmd_to': slave_joint_topic,
                        'remap/gripper_pos_cmd_to': master_joint_topic,
                        'remap/gripper_effort_cmd_to': slave_joint_topic,
                    }
                    if gravity_for_master:
                        params['remap/joint_states_compensated_to'] = slave_comp_topic
                    for k, v in params.items():
                        rospy.set_param(f'{prefix}/{k}', v)

                    node = roslaunch.core.Node('piper_test', 'teleop_force_feedback_node.py',
                                               name='piper_ctrl_node', namespace=ns, output='screen')
                    self.launcher.launch(node)
                    continue

                rospy.set_param(f'{prefix}/can_port', pair[f'{role}_can'])
                rospy.set_param(f'{prefix}/auto_enable', defaults.get('auto_enable', True))
                rospy.set_param(f'{prefix}/gripper_exist', defaults.get('gripper_exist', True))
                rospy.set_param(f'{prefix}/gripper_val_mutiple', defaults.get('gripper_val_multiple', 1))
                node = roslaunch.core.Node('piper', 'piper_ctrl_single_node.py',
                                           name='piper_ctrl_node', namespace=ns, output='screen')
                self.launcher.launch(node)

        # 重力补偿节点（不变）
        if self.ff_use_piperros_node and self.force_fb_node['mit_enable_gravity'] and self.gravity_comp['enabled']:
            if self.runtime_gravity_enabled and len(self.arm_pairs) >= 2:
                gc = self.gravity_comp
                gravity_prefix = '/piper_gravity_compensation_node'
                for k, v in {
                    'urdf_package': gc['urdf_package'], 'urdf_relpath': gc['urdf_relpath'],
                    'gravity_joint_scale_left': gc['gravity_joint_scale_left'],
                    'gravity_joint_scale_right': gc['gravity_joint_scale_right'],
                    'gravity_joint_sign_left': gc['gravity_joint_sign_left'],
                    'gravity_joint_sign_right': gc['gravity_joint_sign_right'],
                    'gravity_joint_position_scale_left': gc['gravity_joint_position_scale_left'],
                    'gravity_joint_position_scale_right': gc['gravity_joint_position_scale_right'],
                    'gravity_joint_offset_left': gc['gravity_joint_offset_left'],
                    'gravity_joint_offset_right': gc['gravity_joint_offset_right'],
                    'clip_to_limits': gc['clip_to_limits'], 'unwrap_joint_positions': gc['unwrap_joint_positions'],
                    'max_joint_step': gc['max_joint_step'], 'max_torque_delta_warn': gc['max_torque_delta_warn'],
                }.items():
                    rospy.set_param(f'{gravity_prefix}/{k}', v)

                left_pair, right_pair = self.arm_pairs[0], self.arm_pairs[1]
                left_slave_j = self.topics_tpl['slave_joint_states_tpl'].format(slave=left_pair['slave'])
                right_slave_j = self.topics_tpl['slave_joint_states_tpl'].format(slave=right_pair['slave'])
                comp_tpl = self.topics_tpl.get('slave_joint_states_compensated_tpl', '/{slave}/joint_states_compensated')
                gravity_node = roslaunch.core.Node(
                    'piper_test', 'teleop_gravity_compensation_node.py',
                    name='piper_gravity_compensation_node', output='screen',
                    remap_args=[
                        ('/robot/arm_left/joint_states_single', left_slave_j),
                        ('/robot/arm_right/joint_states_single', right_slave_j),
                        ('/robot/arm_left/joint_states_compensated', comp_tpl.format(slave=left_pair['slave'])),
                        ('/robot/arm_right/joint_states_compensated', comp_tpl.format(slave=right_pair['slave'])),
                    ],
                )
                self.launcher.launch(gravity_node)

        for cam in self.cameras:
            name = cam['name']
            node_name = f'realsense_pub_{name}'
            prefix = f'/{node_name}'
            s = cam['stream']
            for k, v in {
                'serial_no': str(cam['serial_no']), 'device_name': cam['device_name'],
                'color_topic': self.cam_topics_tpl['color_tpl'].format(name=name),
                'depth_topic': self.cam_topics_tpl['depth_tpl'].format(name=name),
                'enable_depth': s.get('enable_depth', False),
                'width': s['width'], 'height': s['height'], 'fps': s['fps'],
            }.items():
                rospy.set_param(f'{prefix}/{k}', v)
            self.launcher.launch(roslaunch.core.Node('piper_test', 'realsense_publisher2.py',
                                                     name=node_name, output='screen'))

    # ====================== ROS 话题 ======================

    def _setup_ros(self):
        for pair in self.arm_pairs:
            pn = pair['name']
            self.arm_state[pn] = {
                'master': None, 'slave': None,
                'fb_blend_filt': [0.0] * 6,
                'fb_eff_filt': self.force_fb['gripper_base_effort'],
                'fb_last_log_t': 0.0, 'mit_active': False,
            }
            mt = self.topics_tpl['master_joint_states_tpl'].format(master=pair['master'])
            rospy.Subscriber(mt, JointState, lambda msg, p=pn: self._master_cb(p, msg),
                             queue_size=1, tcp_nodelay=True)
            st = self.topics_tpl['slave_joint_states_tpl'].format(slave=pair['slave'])
            rospy.Subscriber(st, JointState, lambda msg, p=pn: self._slave_cb(p, msg),
                             queue_size=1, tcp_nodelay=True)
            ct = self.topics_tpl['slave_joint_ctrl_tpl'].format(slave=pair['slave'])
            self.arm_state[pn]['pub'] = rospy.Publisher(ct, JointState, queue_size=1, tcp_nodelay=True)

        for cam in self.cameras:
            cn = cam['name']
            self.cam_state[cn] = None
            color_topic = self.cam_topics_tpl['color_tpl'].format(name=cn)
            rospy.Subscriber(color_topic, Image, lambda msg, n=cn: self._cam_cb(n, msg),
                             queue_size=1, tcp_nodelay=True)

    # ====================== 回调（不变）======================

    def _master_cb(self, pair_name, msg):
        pos = fix_len(msg.position, self.dim)
        vel = fix_len(msg.velocity, self.dim)
        self.arm_state[pair_name]['master'] = {
            'pos': pos, 'vel': vel,
            'eff': fix_len(msg.effort, self.dim),
            'stamp': msg.header.stamp.to_sec()
        }
        self._send_ctrl(pair_name, pos, vel)
        if self.ff_use_legacy:
            self._send_master_feedback(pair_name)

    def _slave_cb(self, pair_name, msg):
        self.arm_state[pair_name]['slave'] = {
            'pos': fix_len(msg.position, self.dim),
            'vel': fix_len(msg.velocity, self.dim),
            'eff': fix_len(msg.effort, self.dim),
            'stamp': msg.header.stamp.to_sec()
        }

    def _send_ctrl(self, pair_name, target_pos, target_vel=None):
        cc = self.ctrl_cmd_cfg
        n_j = self.dim - 1
        cmd = JointState()
        cmd.header.stamp = rospy.Time.now()
        cmd.position = target_pos
        if target_vel is not None:
            cmd.velocity = list(target_vel[:n_j]) + [cc['gripper_velocity']]
        else:
            cmd.velocity = [cc['joint_velocity_default']] * n_j + [cc['gripper_velocity']]
        cmd.effort = [cc['joint_effort_default']] * n_j + [cc['gripper_effort']]
        self.arm_state[pair_name]['pub'].publish(cmd)

    def _send_master_feedback(self, pair_name):
        if not self.ff_use_legacy:
            return
        st = self.arm_state.get(pair_name)
        if st is None:
            return
        m = st.get('master')
        s = st.get('slave')
        piper = self._master_pipers.get(pair_name)
        if m is None or s is None or piper is None:
            return
        fb = self.force_fb
        blend = list(st['fb_blend_filt'])
        if not st['mit_active']:
            piper.MotionCtrl_2(0x01, 0x01, 50, 0xAD)
            st['mit_active'] = True
        for i in range(6):
            pos_diff = abs(m['pos'][i] - s['pos'][i])
            contact_i = max(0.0, pos_diff - fb['pos_diff_threshold'])
            target_blend = min(1.0, fb['joint_gain'] * contact_i)
            alpha_j = fb['joint_ema_alpha']
            blend[i] = alpha_j * target_blend + (1.0 - alpha_j) * blend[i]
            kp = fb['mit_kp_free'] * (1.0 - blend[i]) + fb['mit_kp_blocked'] * blend[i]
            piper.JointMitCtrl(i + 1, s['pos'][i], 0.0, kp, fb['mit_kd'], 0.0)
        st['fb_blend_filt'] = blend
        grip_pos_diff = abs(m['pos'][6] - s['pos'][6])
        grip_contact = max(0.0, grip_pos_diff - fb['pos_diff_threshold'])
        slave_eff_grip = abs(float(s['eff'][-1])) if s.get('eff') else 0.0
        eff_contact = max(0.0, slave_eff_grip - fb['gripper_contact_threshold'])
        contact_grip = max(grip_contact, eff_contact)
        target_eff = fb['gripper_base_effort'] + fb['gripper_gain'] * contact_grip
        target_eff = clamp(target_eff, fb['gripper_min_effort'], fb['gripper_max_effort'])
        filt_eff = fb['gripper_ema_alpha'] * target_eff + (1.0 - fb['gripper_ema_alpha']) * st.get('fb_eff_filt', fb['gripper_base_effort'])
        st['fb_eff_filt'] = filt_eff
        piper.GripperCtrl(round(abs(m['pos'][6]) * 1000000), round(clamp(filt_eff, 0.5, 3.0) * 1000), 0x01, 0)
        if self._csv_writer is not None:
            now = time.time()
            max_blend = max(blend) if blend else 0.0
            row = [f'{now:.4f}', pair_name]
            for i in range(6):
                pd = abs(m['pos'][i] - s['pos'][i])
                kp = fb['mit_kp_free'] * (1.0 - blend[i]) + fb['mit_kp_blocked'] * blend[i]
                row += [f'{pd:.5f}', f'{blend[i]:.4f}', f'{kp:.1f}']
            row += [f'{filt_eff:.4f}', f'{max_blend:.4f}']
            self._csv_writer.writerow(row)
            if now - self._csv_last_flush > 1.0 / fb['csv_flush_hz']:
                self._csv_file.flush()
                self._csv_last_flush = now

    def _cam_cb(self, cam_name, msg):
        img = img_from_msg(msg)
        if img is not None:
            if self.cam_state[cam_name] is None:
                self.cam_state[cam_name] = {}
            self.cam_state[cam_name]['color'] = img
            self.cam_state[cam_name]['stamp'] = msg.header.stamp.to_sec()

    # ====================== 预览（不变）======================

    def _camera_slave_tag(self, cam_name):
        name = str(cam_name).strip().lower()
        pair = None
        if 'left' in name and len(self.arm_pairs) >= 1:
            pair = self.arm_pairs[0]
        elif 'right' in name and len(self.arm_pairs) >= 2:
            pair = self.arm_pairs[1]
        if pair is None:
            return ""
        slave_can = str(pair.get('slave_can', '')).strip()
        slave_ns = str(pair.get('slave', '')).strip()
        if slave_can and slave_ns:
            return f"{slave_can} ({slave_ns})"
        return slave_can or slave_ns

    def _update_preview(self):
        pw, ph = 480, 360
        panels = []
        for cam in self.cameras:
            cn = cam['name']
            slave_tag = self._camera_slave_tag(cn)
            cs = self.cam_state.get(cn)
            if cs and 'color' in cs:
                panel = cv2.resize(cs['color'], (pw, ph))
            else:
                panel = np.zeros((ph, pw, 3), dtype=np.uint8)
                cv2.putText(panel, f'{cn}: waiting...', (pw // 6, ph // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            overlay = panel.copy()
            cv2.rectangle(overlay, (0, 0), (pw, 32), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, panel, 0.5, 0, panel)
            lbl_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cam_label = f'  {cn} ({cam["device_name"]})'
            if slave_tag:
                cam_label += f' -> {slave_tag}'
            cv2.putText(panel, cam_label, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lbl_color, 2)
            panels.append(panel)
        mosaic = np.hstack(panels) if panels else np.zeros((ph, pw, 3), dtype=np.uint8)
        total_w = mosaic.shape[1]
        bar_h = 48
        bar = np.zeros((bar_h, total_w, 3), dtype=np.uint8)
        if self.is_recording:
            elapsed = time.time() - self.rec_start_time if self.rec_start_time else 0
            m, s = int(elapsed) // 60, int(elapsed) % 60
            txt = f"  REC  |  frames: {self.frame_count}   time: {m:02d}:{s:02d}   ep={self.current_ep_name}  |  [SPACE]stop  [Q]uit"
            if int(elapsed * 2) % 2 == 0:
                cv2.circle(bar, (20, bar_h // 2), 8, (0, 0, 255), -1)
            cv2.putText(bar, txt, (36, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            txt = f"  IDLE  |  next=episode_{self.ep_count:03d}  saved={self.save_done}  |  [SPACE]record  [Q]uit"
            cv2.circle(bar, (20, bar_h // 2), 8, (0, 255, 0), -1)
            cv2.putText(bar, txt, (36, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        cv2.imshow(WINDOW_NAME, np.vstack([mosaic, bar]))

    # ====================== 就绪 ======================

    def _all_ready(self):
        for pn, st in self.arm_state.items():
            if st['master'] is None or st['slave'] is None:
                return False
        for cn, cs in self.cam_state.items():
            if cs is None or 'color' not in cs:
                return False
        return True

    # ====================== Rosbag 录制（对齐 cheese-zj）======================

    def _build_record_topics(self):
        """动态生成录制话题列表，对齐 cheese-zj 的 profile_record_topics"""
        topics = []
        for pair in self.arm_pairs:
            topics.append(self.topics_tpl['master_joint_states_tpl'].format(master=pair['master']))
            topics.append(self.topics_tpl['slave_joint_states_tpl'].format(slave=pair['slave']))
        for cam in self.cameras:
            cn = cam['name']
            base = self.cam_topics_tpl['color_tpl'].format(name=cn)
            if self.camera_transport == 'compressed':
                topics.append(f"{base}/compressed")
            else:
                topics.append(base)
            if cam['stream'].get('enable_depth') and cam['record'].get('save_depth'):
                topics.append(self.cam_topics_tpl['depth_tpl'].format(name=cn))
        # 去重保序
        seen, out = set(), []
        for t in topics:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _next_ep_index(self):
        """扫描现有 episode_XXX 目录，返回下一个可用索引（1-indexed，对齐 cheese-zj）"""
        used = set()
        for d in self.session_dir.iterdir():
            if d.is_dir():
                m = re.match(r'^episode_(\d+)$', d.name)
                if m:
                    used.add(int(m.group(1)))
        idx = 1
        while idx in used:
            idx += 1
        return idx

    def _stop_process_group(self, proc):
        if proc.poll() is not None:
            return True
        try:
            pgid = os.getpgid(proc.pid)
        except Exception:
            return proc.poll() is not None
        for sig, wait_sec in [(signal.SIGINT, 5.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)]:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return True
            except Exception:
                pass
            deadline = time.time() + wait_sec
            while time.time() < deadline:
                if proc.poll() is not None:
                    return True
                time.sleep(0.1)
        return proc.poll() is not None

    def _start_episode(self):
        """启动 rosbag record，对齐 cheese-zj 的 _start_episode"""
        if self.is_recording:
            return

        ep_name = f"episode_{self.ep_count:03d}"
        ep_dir = self.session_dir / ep_name
        ep_dir.mkdir(parents=True, exist_ok=True)

        topics = self._build_record_topics()
        bag_prefix = ep_dir / "episode"  # rosbag 会生成 episode.bag

        cmd = ["rosbag", "record", "-O", str(bag_prefix)]
        if self.rosbag_lz4:
            cmd.append("--lz4")
        cmd.extend(topics)

        rosbag_log = ep_dir / "rosbag_record.log"
        log_fh = open(rosbag_log, 'w')
        self.rosbag_proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT,
                                             preexec_fn=os.setsid, text=True)
        self._rosbag_log_fh = log_fh
        self._rosbag_log_path = rosbag_log

        self.current_ep_name = ep_name
        self.current_ep_dir = ep_dir
        self.current_started_utc = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
        self.current_topics = topics
        self.rec_start_time = time.time()
        self.frame_count = 0
        self.is_recording = True

        rospy.loginfo("[record] START %s  topics=%d  bag=%s",
                      ep_name, len(topics), bag_prefix)
        rospy.loginfo("[record] topics: %s", topics)

    def _stop_episode(self, stop_reason="user_stop"):
        """停止 rosbag，异步提交 metadata 写入，对齐 cheese-zj 的 _stop_episode"""
        if not self.is_recording:
            return

        ep_name = self.current_ep_name
        ep_dir = self.current_ep_dir
        started_utc = self.current_started_utc
        topics = list(self.current_topics)
        duration = time.time() - self.rec_start_time if self.rec_start_time else None

        rosbag_ok = False
        if self.rosbag_proc is not None:
            rosbag_ok = self._stop_process_group(self.rosbag_proc)
            self.rosbag_proc = None
        if hasattr(self, '_rosbag_log_fh') and self._rosbag_log_fh:
            self._rosbag_log_fh.flush()
            self._rosbag_log_fh.close()
            self._rosbag_log_fh = None

        ended_utc = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
        self.is_recording = False
        self.ep_count = self._next_ep_index()

        rospy.loginfo("[record] STOP %s  reason=%s  duration=%.2fs",
                      ep_name, stop_reason, duration or 0)

        # 异步写 metadata.json（对齐 cheese-zj）
        job = {
            'ep_name': ep_name,
            'ep_dir': ep_dir,
            'started_utc': started_utc,
            'ended_utc': ended_utc,
            'duration_sec': duration,
            'stop_reason': stop_reason,
            'topics': topics,
            'rosbag_ok': rosbag_ok,
            'camera_transport': self.camera_transport,
            'lz4': self.rosbag_lz4,
            'rosbag_log_path': str(getattr(self, '_rosbag_log_path', '')),
        }
        try:
            self.save_q.put_nowait(job)
        except queue.Full:
            rospy.logwarn("[save] 队列满，跳过 metadata 写入")

    # ====================== 异步保存（写 metadata.json，对齐 cheese-zj）======================

    def _save_worker(self):
        while not rospy.is_shutdown():
            try:
                job = self.save_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._finalize_episode(job)
                self.save_done += 1
            except Exception as e:
                rospy.logerr("[save] metadata 写入失败: %s", e)
            finally:
                self.save_q.task_done()

    def _collect_bag_stats(self, ep_dir: Path):
        """收集 bag 文件统计信息（对齐 cheese-zj 的 _aggregate_bag_stats）"""
        bag_files = sorted(ep_dir.glob("episode*.bag*"))
        bag_files = [f for f in bag_files if f.is_file()]

        total_size = 0
        total_messages = 0
        per_topic = {}

        for bf in bag_files:
            try:
                total_size += bf.stat().st_size
            except Exception:
                pass
            try:
                res = subprocess.run(["rosbag", "info", "--yaml", str(bf)],
                                     capture_output=True, text=True, timeout=20)
                if res.returncode == 0:
                    info = yaml.safe_load(res.stdout) or {}
                    total_messages += int(info.get('messages', 0))
                    for t in info.get('topics', []) or []:
                        tn = t.get('topic')
                        if not tn:
                            continue
                        entry = per_topic.setdefault(tn, {'messages': 0, 'type': t.get('type')})
                        entry['messages'] += int(t.get('messages', 0))
            except Exception:
                pass

        return {
            'bag_files': [str(f) for f in bag_files],
            'bag_count': len(bag_files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 ** 2), 2),
            'total_messages': total_messages,
            'per_topic': per_topic,
        }

    def _finalize_episode(self, job: dict):
        """写 metadata.json（对齐 cheese-zj 的 metadata 格式）"""
        ep_dir = Path(job['ep_dir'])
        bag_stats = self._collect_bag_stats(ep_dir)

        warnings = []
        if not job.get('rosbag_ok'):
            warnings.append("rosbag process did not terminate cleanly")
        if bag_stats['bag_count'] == 0:
            warnings.append("No bag files found in episode directory")
        for topic in job.get('topics', []):
            info = bag_stats['per_topic'].get(topic)
            if not info or int(info.get('messages', 0)) == 0:
                warnings.append(f"Topic has zero messages: {topic}")

        metadata = {
            "schema_version": 1,
            "session_name": self.session_dir.name,
            "episode_name": job['ep_name'],
            "episode_index": int(job['ep_name'].split('_')[-1]),
            "episode_dir": str(ep_dir),
            "started_at_utc": job.get('started_utc'),
            "ended_at_utc": job.get('ended_utc'),
            "duration_sec": job.get('duration_sec'),
            "stop_reason": job.get('stop_reason'),
            "recording": {
                "compression": "lz4" if job.get('lz4') else "none",
                "camera_transport": job.get('camera_transport'),
                "arm_pairs": [p['name'] for p in self.arm_pairs],
                "cameras": [c['name'] for c in self.cameras],
            },
            "topics": {
                "recorded": job.get('topics', []),
                "per_topic_stats": bag_stats['per_topic'],
            },
            "bag_stats": bag_stats,
            "environment": {
                "host": socket.gethostname(),
                "user": os.environ.get("USER"),
                "ros_master_uri": os.environ.get("ROS_MASTER_URI"),
            },
            "rosbag_process_terminated_cleanly": job.get('rosbag_ok', False),
            "warnings": warnings,
        }

        metadata_path = ep_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

        sz = bag_stats['total_size_mb']
        status = "WARN" if warnings else "OK"
        rospy.loginfo("[save] %s  bags=%d  size=%.1fMB  msgs=%d  [%s]",
                      job['ep_name'], bag_stats['bag_count'], sz,
                      bag_stats['total_messages'], status)
        if warnings:
            for w in warnings:
                rospy.logwarn("[save] warning: %s", w)

    # ====================== 主循环 ======================

    def _print_banner(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("Teleop Raw System PiperROS-FF  →  Rosbag 格式")
        rospy.loginfo("  session_dir = %s", self.session_dir)
        rospy.loginfo("  rate        = %d Hz", self.rate_hz)
        rospy.loginfo("  arm_pairs   = %s", [p['name'] for p in self.arm_pairs])
        rospy.loginfo("  cameras     = %s", [c['name'] for c in self.cameras])
        rospy.loginfo("  transport   = %s  lz4=%s", self.camera_transport, self.rosbag_lz4)
        rospy.loginfo("  topics      = %s", self._build_record_topics())
        rospy.loginfo("[SPACE]录制/停止  [Q]退出")
        rospy.loginfo("=" * 60)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("等待所有数据源就绪...")
        while not rospy.is_shutdown():
            self._update_preview()
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                return
            if self._all_ready():
                rospy.loginfo("所有数据源就绪！")
                break
            rate.sleep()

        while not rospy.is_shutdown():
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if self.is_recording:
                    self._stop_episode("user_stop")
                else:
                    self._start_episode()

            elif key == ord('q'):
                rospy.loginfo("退出...")
                if self.is_recording:
                    self._stop_episode("user_stop")
                break

            if self.is_recording:
                self.frame_count += 1

            self._update_preview()
            rate.sleep()

        cv2.destroyAllWindows()
        self.save_q.join()


def main():
    try:
        TeleopRawSystem().run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
