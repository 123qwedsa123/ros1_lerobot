#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置驱动的遥操作原始数据录制系统（piper_ros 风格力反馈版）
- 从 YAML 读取所有参数，动态启动 master/slave/camera 节点
- master 端支持 piper_ctrl_node 风格 MIT 力反馈（来自 slave 实测力矩）
- 保留原始相机预览和 HDF5 保存流程
- 键盘: [SPACE]录制  [S]保存  [D]丢弃  [Q]退出（OpenCV窗口聚焦时）
"""

import os, sys, time, queue, threading, yaml, csv, subprocess, glob
import importlib.util
from datetime import datetime

import numpy as np
import h5py
import cv2

import rospy
import roslaunch
from sensor_msgs.msg import JointState, Image
from piper_sdk import C_PiperInterface

WINDOW_NAME = 'Teleop Cameras'


def ensure_pinocchio_discovery():
    """Make ROS-installed pinocchio discoverable when setup env misses site-packages."""
    if importlib.util.find_spec('pinocchio') is not None:
        return True

    ros_distro = os.environ.get('ROS_DISTRO', 'noetic').strip() or 'noetic'
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        f"/opt/ros/{ros_distro}/lib/python{py_ver}/site-packages",
        f"/opt/ros/{ros_distro}/lib/python{py_ver}/dist-packages",
    ]

    # Fallback scan across /opt/ros for mixed distro/environment setups.
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


def depth_from_msg(msg):
    if msg.height <= 0 or msg.width <= 0:
        return None
    buf = np.frombuffer(msg.data, dtype=np.uint16)
    row_u16 = msg.step // 2
    if buf.size < msg.height * row_u16:
        return None
    return buf.reshape(msg.height, row_u16)[:, :msg.width].reshape(msg.height, msg.width)


def resolve_rospack_file(package_name, relpath):
    try:
        pkg_path = (
            subprocess.check_output(["rospack", "find", package_name], stderr=subprocess.STDOUT)
            .strip()
            .decode("utf-8")
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

        # ---------- 加载 YAML ----------
        config_file = rospy.get_param('~config_file')
        with open(config_file) as f:
            self.cfg = yaml.safe_load(f)

        g = self.cfg['global']
        self.rate_hz = g['rate_hz']
        self.data_dir = g['data_dir']
        os.makedirs(self.data_dir, exist_ok=True)

        save_cfg = g['save']
        self.img_comp = save_cfg['image_compression']
        self.gzip_level = save_cfg['gzip_level']

        self.rec = self.cfg['recording']
        self.dim = self.rec['per_arm_dim']

        self.arm_pairs = self.cfg['arm_pairs']
        self.cameras = self.cfg['cameras']
        self.topics_tpl = self.cfg['topics']
        self.cam_topics_tpl = self.cfg['camera_topics']
        self.hdf5_keys = self.cfg['hdf5_keys']
        self.ctrl_cmd_cfg = self.cfg['ctrl_cmd']

        fb_cfg = self.cfg.get('force_feedback', {})
        # legacy_sdk: 旧版（脚本里直接用 SDK 发 MIT 指令）
        # piperros_node: 新版（用 piper_ctrl_node 风格力反馈节点，推荐）
        self.force_fb_backend = str(fb_cfg.get('backend', 'piperros_node')).strip().lower()
        if self.force_fb_backend not in ('legacy_sdk', 'piperros_node'):
            rospy.logwarn("Unknown force_feedback.backend=%s, fallback to piperros_node", self.force_fb_backend)
            self.force_fb_backend = 'piperros_node'

        self.force_fb = {
            'enabled': as_bool(fb_cfg.get('enabled', False), False),
            # legacy_sdk 模式参数（保留兼容）
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
            ff_node_cfg.get('mit_torque_feedback_sign', [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
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
            'mit_kp': as_list6(ff_node_cfg.get('mit_kp', [0.02, 0.1, 0.15, 0.1, 0.1, 0.1]),
                               [0.02, 0.1, 0.15, 0.1, 0.1, 0.1]),
            'mit_kd': as_list6(ff_node_cfg.get('mit_kd', [0.02, 0.03, 0.01, 0.02, 0.02, 0.02]),
                               [0.02, 0.03, 0.01, 0.02, 0.02, 0.02]),
            'mit_enable_pos': as_bool(ff_node_cfg.get('mit_enable_pos', True), True),
            'mit_enable_vel': as_bool(ff_node_cfg.get('mit_enable_vel', False), False),
            'mit_enable_tor': as_bool(ff_node_cfg.get('mit_enable_tor', True), True),
            'mit_enable_gravity': as_bool(ff_node_cfg.get('mit_enable_gravity', True), True),
            'mit_gravity_mix_mode': str(ff_node_cfg.get('mit_gravity_mix_mode', 'additive')).strip(),
            'mit_torque_scale': mit_torque_scale_default,
            'mit_torque_scale_left': as_list6(
                ff_node_cfg.get('mit_torque_scale_left', mit_torque_scale_default),
                mit_torque_scale_default,
            ),
            'mit_torque_scale_right': as_list6(
                ff_node_cfg.get('mit_torque_scale_right', mit_torque_scale_default),
                mit_torque_scale_default,
            ),
            'mit_torque_feedback_sign': mit_torque_feedback_sign_default,
            'mit_torque_feedback_sign_left': as_list6(
                ff_node_cfg.get('mit_torque_feedback_sign_left', mit_torque_feedback_sign_default),
                mit_torque_feedback_sign_default,
            ),
            'mit_torque_feedback_sign_right': as_list6(
                ff_node_cfg.get('mit_torque_feedback_sign_right', mit_torque_feedback_sign_default),
                mit_torque_feedback_sign_default,
            ),
            'mit_max_torque_abs': as_float(ff_node_cfg.get('mit_max_torque_abs', 18.0), 18.0),
            'enforce_joint_limits': as_bool(ff_node_cfg.get('enforce_joint_limits', True), True),
            'gripper_haptic_effort_sign': as_float(ff_node_cfg.get('gripper_haptic_effort_sign', 0.0), 0.0),
            'gripper_haptic_effort_deadband': as_float(ff_node_cfg.get('gripper_haptic_effort_deadband', 0.15), 0.15),
            'gripper_haptic_effort_bias': as_float(ff_node_cfg.get('gripper_haptic_effort_bias', 0.0), 0.0),
            'gripper_haptic_effort_max': as_float(ff_node_cfg.get('gripper_haptic_effort_max', 0.6), 0.6),
            'gripper_haptic_cmd_max': int(as_float(ff_node_cfg.get('gripper_haptic_cmd_max', 500), 500)),
            'gripper_haptic_cmd_enable_threshold': int(
                as_float(ff_node_cfg.get('gripper_haptic_cmd_enable_threshold', 80), 80)
            ),
            'gripper_haptic_effort_alpha': as_float(ff_node_cfg.get('gripper_haptic_effort_alpha', 0.2), 0.2),
            'gripper_haptic_release_when_opening': as_bool(
                ff_node_cfg.get('gripper_haptic_release_when_opening', True), True
            ),
            'gripper_haptic_opening_relief': as_float(ff_node_cfg.get('gripper_haptic_opening_relief', 0.15), 0.15),
            'gripper_haptic_opening_sign': as_float(ff_node_cfg.get('gripper_haptic_opening_sign', 0.0), 0.0),
            'gripper_haptic_opening_direction_eps': as_float(
                ff_node_cfg.get('gripper_haptic_opening_direction_eps', 0.0001), 0.0001
            ),
            'publish_rate': as_float(ff_node_cfg.get('publish_rate', 100.0), 100.0),
            'control_rate': as_float(ff_node_cfg.get('control_rate', 50.0), 50.0),
            'subscribe_rate': as_float(ff_node_cfg.get('subscribe_rate', 50.0), 50.0),
            'filter_enable': as_bool(ff_node_cfg.get('filter_enable', True), True),
            'filter_alpha_position': as_float(ff_node_cfg.get('filter_alpha_position', 0.7), 0.7),
            'filter_alpha_velocity': as_float(ff_node_cfg.get('filter_alpha_velocity', 0.5), 0.5),
            'filter_alpha_effort': as_float(ff_node_cfg.get('filter_alpha_effort', 0.5), 0.5),
            'master_slave_enable': as_bool(ff_node_cfg.get('master_slave_enable', True), True),
            'master_slave_flag_topic': str(ff_node_cfg.get('master_slave_flag_topic', '/conrft_robot/slave_follow_flag')),
            'master_slave_kp_follow': as_list6(
                ff_node_cfg.get('master_slave_kp_follow', [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]),
                [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            ),
            'shutdown_mode': str(ff_node_cfg.get('shutdown_mode', 'hold')).strip().lower(),
        }

        gcfg = fb_cfg.get('gravity_compensation', {})
        gravity_joint_scale_default = as_list6(
            gcfg.get('gravity_joint_scale', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        gravity_joint_sign_default = as_list6(
            gcfg.get('gravity_joint_sign', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        gravity_joint_pos_scale_default = as_list6(
            gcfg.get('gravity_joint_position_scale', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        gravity_joint_offset_default = as_list6(
            gcfg.get('gravity_joint_offset', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.gravity_comp = {
            'enabled': as_bool(gcfg.get('enabled', True), True),
            'urdf_package': str(gcfg.get('urdf_package', 'piper_x_description')).strip(),
            'urdf_relpath': str(gcfg.get('urdf_relpath', 'urdf/piper_x_description.urdf')).strip(),
            'gravity_joint_scale_left': as_list6(
                gcfg.get('gravity_joint_scale_left', gravity_joint_scale_default),
                gravity_joint_scale_default,
            ),
            'gravity_joint_scale_right': as_list6(
                gcfg.get('gravity_joint_scale_right', gravity_joint_scale_default),
                gravity_joint_scale_default,
            ),
            'gravity_joint_sign_left': as_list6(
                gcfg.get('gravity_joint_sign_left', gravity_joint_sign_default),
                gravity_joint_sign_default,
            ),
            'gravity_joint_sign_right': as_list6(
                gcfg.get('gravity_joint_sign_right', gravity_joint_sign_default),
                gravity_joint_sign_default,
            ),
            'gravity_joint_position_scale_left': as_list6(
                gcfg.get('gravity_joint_position_scale_left', gravity_joint_pos_scale_default),
                gravity_joint_pos_scale_default,
            ),
            'gravity_joint_position_scale_right': as_list6(
                gcfg.get('gravity_joint_position_scale_right', gravity_joint_pos_scale_default),
                gravity_joint_pos_scale_default,
            ),
            'gravity_joint_offset_left': as_list6(
                gcfg.get('gravity_joint_offset_left', gravity_joint_offset_default),
                gravity_joint_offset_default,
            ),
            'gravity_joint_offset_right': as_list6(
                gcfg.get('gravity_joint_offset_right', gravity_joint_offset_default),
                gravity_joint_offset_default,
            ),
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

        # legacy_sdk 模式：脚本直接连 SDK 给 master 发力反馈
        self._master_pipers = {}
        if self.ff_use_legacy:
            for pair in self.arm_pairs:
                can_port = pair['master_can']
                rospy.loginfo("[力反馈 legacy_sdk] connect master sdk: %s can=%s", pair['master'], can_port)
                piper = C_PiperInterface(can_name=can_port)
                piper.ConnectPort()
                self._master_pipers[pair['name']] = piper
                rospy.loginfo("[力反馈 legacy_sdk] connected: %s", pair['master'])

        # legacy_sdk 模式下保留 CSV 力反馈日志
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
            rospy.loginfo("[力反馈 legacy_sdk] csv log: %s", fb['csv_path'])

        # ---------- 运行时状态 ----------
        self.arm_state = {}
        self.cam_state = {}
        self.is_recording = False
        self.ep_data = None
        self.ep_count = self._next_ep_num()
        self.frame_count = 0
        self.rec_start_time = None

        # 异步保存
        self.save_q = queue.Queue(maxsize=save_cfg['save_queue_size'])
        self.save_done = 0
        threading.Thread(target=self._save_worker, daemon=True).start()

        # ---------- OpenCV 窗口：WINDOW_NORMAL 允许自由拖拽缩放 ----------
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        init_w = 480 * max(len(self.cameras), 1)
        init_h = 360 + 50
        cv2.resizeWindow(WINDOW_NAME, init_w, init_h)

        # ---------- 先写 robot_description，避免关节限位回退到默认 ----------
        self._load_robot_description_param()

        # ---------- 启动子节点 ----------
        self._launch_child_nodes()

        # ---------- ROS 订阅/发布 ----------
        self._setup_ros()

        self._print_banner()

    # ====================== 子节点启动 ======================

    def _load_robot_description_param(self):
        rd = self.robot_description
        if not rd['enabled']:
            rospy.loginfo("[启动] robot_description 自动加载已禁用")
            return

        urdf_path, err = resolve_rospack_file(rd['package'], rd['relpath'])
        if urdf_path is None:
            self.robot_description_error = err
            rospy.logwarn(
                "[启动] robot_description 加载失败: %s/%s (%s)",
                rd['package'],
                rd['relpath'],
                err,
            )
            return

        try:
            with open(urdf_path, 'r') as f:
                urdf_text = f.read()
            rospy.set_param('/robot_description', urdf_text)
            if rd['param'] != '/robot_description':
                rospy.set_param(rd['param'], urdf_text)
            self.robot_description_loaded = True
            rospy.loginfo(
                "[启动] robot_description 已加载: %s/%s",
                rd['package'],
                rd['relpath'],
            )
        except Exception as exc:
            self.robot_description_error = str(exc)
            rospy.logwarn("[启动] robot_description 写入参数失败: %s", str(exc))

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
                    slave_comp_tpl = self.topics_tpl.get(
                        'slave_joint_states_compensated_tpl', '/{slave}/joint_states_compensated'
                    )
                    slave_comp_topic = slave_comp_tpl.format(slave=pair['slave'])
                    torque_scale = ff['mit_torque_scale_left'] if arm_side == 'left' else ff['mit_torque_scale_right']
                    torque_feedback_sign = (
                        ff['mit_torque_feedback_sign_left']
                        if arm_side == 'left'
                        else ff['mit_torque_feedback_sign_right']
                    )

                    rospy.set_param(f'{prefix}/can_port', pair['master_can'])
                    rospy.set_param(f'{prefix}/topic_prefix', f'/{pair["master"]}/')
                    rospy.set_param(f'{prefix}/auto_enable', ff['auto_enable'])
                    rospy.set_param(f'{prefix}/gripper_exist', ff['gripper_exist'])
                    rospy.set_param(f'{prefix}/gripper_val_mutiple', ff['gripper_val_multiple'])
                    rospy.set_param(f'{prefix}/enable_gripper', ff['enable_gripper'])
                    rospy.set_param(f'{prefix}/enable_gripper_haptic', ff['enable_gripper_haptic'])
                    rospy.set_param(f'{prefix}/gripper_range', ff['gripper_range'])
                    rospy.set_param(f'{prefix}/gripper_reverse', ff['gripper_reverse'])
                    rospy.set_param(f'{prefix}/ctrl_mode', ff['ctrl_mode'])
                    rospy.set_param(f'{prefix}/p/speed', ff['p_speed'])
                    rospy.set_param(f'{prefix}/mit/speed', ff['mit_speed'])
                    rospy.set_param(f'{prefix}/mit/kp', ff['mit_kp'])
                    rospy.set_param(f'{prefix}/mit/kd', ff['mit_kd'])
                    rospy.set_param(f'{prefix}/mit/enable_pos', ff['mit_enable_pos'])
                    rospy.set_param(f'{prefix}/mit/enable_vel', ff['mit_enable_vel'])
                    rospy.set_param(f'{prefix}/mit/enable_tor', ff['mit_enable_tor'])
                    rospy.set_param(f'{prefix}/mit/enable_gravity', gravity_for_master)
                    rospy.set_param(f'{prefix}/mit/gravity_mix_mode', ff['mit_gravity_mix_mode'])
                    rospy.set_param(f'{prefix}/mit/torque_scale', torque_scale)
                    rospy.set_param(f'{prefix}/mit/torque_feedback_sign', torque_feedback_sign)
                    rospy.set_param(f'{prefix}/mit/max_torque_abs', ff['mit_max_torque_abs'])
                    rospy.set_param(f'{prefix}/enforce_joint_limits', ff['enforce_joint_limits'])
                    rospy.set_param(f'{prefix}/gripper_haptic_effort_sign', ff['gripper_haptic_effort_sign'])
                    rospy.set_param(f'{prefix}/gripper_haptic_effort_deadband', ff['gripper_haptic_effort_deadband'])
                    rospy.set_param(f'{prefix}/gripper_haptic_effort_bias', ff['gripper_haptic_effort_bias'])
                    rospy.set_param(f'{prefix}/gripper_haptic_effort_max', ff['gripper_haptic_effort_max'])
                    rospy.set_param(f'{prefix}/gripper_haptic_cmd_max', ff['gripper_haptic_cmd_max'])
                    rospy.set_param(
                        f'{prefix}/gripper_haptic_cmd_enable_threshold',
                        ff['gripper_haptic_cmd_enable_threshold'],
                    )
                    rospy.set_param(f'{prefix}/gripper_haptic_effort_alpha', ff['gripper_haptic_effort_alpha'])
                    rospy.set_param(
                        f'{prefix}/gripper_haptic_release_when_opening',
                        ff['gripper_haptic_release_when_opening'],
                    )
                    rospy.set_param(f'{prefix}/gripper_haptic_opening_relief', ff['gripper_haptic_opening_relief'])
                    rospy.set_param(f'{prefix}/gripper_haptic_opening_sign', ff['gripper_haptic_opening_sign'])
                    rospy.set_param(
                        f'{prefix}/gripper_haptic_opening_direction_eps',
                        ff['gripper_haptic_opening_direction_eps'],
                    )
                    rospy.set_param(f'{prefix}/publish_rate', ff['publish_rate'])
                    rospy.set_param(f'{prefix}/control_rate', ff['control_rate'])
                    rospy.set_param(f'{prefix}/subscribe_rate', ff['subscribe_rate'])
                    rospy.set_param(f'{prefix}/filter/enable', ff['filter_enable'])
                    rospy.set_param(f'{prefix}/filter/alpha_position', ff['filter_alpha_position'])
                    rospy.set_param(f'{prefix}/filter/alpha_velocity', ff['filter_alpha_velocity'])
                    rospy.set_param(f'{prefix}/filter/alpha_effort', ff['filter_alpha_effort'])
                    rospy.set_param(f'{prefix}/master_slave/enable', ff['master_slave_enable'])
                    rospy.set_param(f'{prefix}/master_slave/master_position_topic', slave_joint_topic)
                    rospy.set_param(f'{prefix}/master_slave/master_flag_topic', ff['master_slave_flag_topic'])
                    rospy.set_param(f'{prefix}/master_slave/kp_follow', ff['master_slave_kp_follow'])
                    rospy.set_param(f'{prefix}/shutdown_mode', ff['shutdown_mode'])
                    rospy.set_param(f'{prefix}/remap/joint_pos_cmd_to', master_joint_topic)
                    rospy.set_param(f'{prefix}/remap/joint_tor_cmd_to', slave_joint_topic)
                    rospy.set_param(f'{prefix}/remap/gripper_pos_cmd_to', master_joint_topic)
                    rospy.set_param(f'{prefix}/remap/gripper_effort_cmd_to', slave_joint_topic)
                    if gravity_for_master:
                        rospy.set_param(f'{prefix}/remap/joint_states_compensated_to', slave_comp_topic)

                    node = roslaunch.core.Node(
                        'piper_test',
                        'teleop_force_feedback_node.py',
                        name='piper_ctrl_node',
                        namespace=ns,
                        output='screen',
                    )
                    self.launcher.launch(node)
                    rospy.loginfo(
                        "[启动] %s/piper_ctrl_node (piperros_node) can=%s torque_from=%s",
                        ns,
                        pair['master_can'],
                        pair['slave'],
                    )
                    continue

                rospy.set_param(f'{prefix}/can_port', pair[f'{role}_can'])
                rospy.set_param(f'{prefix}/auto_enable', defaults.get('auto_enable', True))
                rospy.set_param(f'{prefix}/gripper_exist', defaults.get('gripper_exist', True))
                rospy.set_param(f'{prefix}/gripper_val_mutiple', defaults.get('gripper_val_multiple', 1))

                node = roslaunch.core.Node(
                    'piper',
                    'piper_ctrl_single_node.py',
                    name='piper_ctrl_node',
                    namespace=ns,
                    output='screen',
                )
                self.launcher.launch(node)
                rospy.loginfo("[启动] %s/piper_ctrl_node (single_node) can=%s", ns, pair[f'{role}_can'])

        if self.ff_use_piperros_node and self.force_fb_node['mit_enable_gravity'] and self.gravity_comp['enabled']:
            if self.gravity_urdf_path is None:
                rospy.logwarn(
                    "[力反馈 piperros_node] 重力模型不可用: %s/%s (%s)",
                    self.gravity_comp['urdf_package'],
                    self.gravity_comp['urdf_relpath'],
                    self.gravity_urdf_error,
                )
                rospy.logwarn(
                    "[力反馈 piperros_node] 跳过重力补偿节点并自动关闭 mit_enable_gravity。"
                )
            elif not self.pinocchio_available:
                rospy.logwarn(
                    "[力反馈 piperros_node] pinocchio 未安装，跳过重力补偿节点并自动关闭 mit_enable_gravity。"
                )
                rospy.logwarn(
                    "[力反馈 piperros_node] 可安装后再开启: pip install pin 或 apt 安装 python3-pinocchio。"
                )
            elif len(self.arm_pairs) < 2:
                rospy.logwarn(
                    "[力反馈 piperros_node] gravity compensation needs 2 pairs, got %d, skip",
                    len(self.arm_pairs),
                )
            else:
                gc = self.gravity_comp
                left_pair = self.arm_pairs[0]
                right_pair = self.arm_pairs[1]
                gravity_prefix = '/piper_gravity_compensation_node'
                rospy.set_param(f'{gravity_prefix}/urdf_package', gc['urdf_package'])
                rospy.set_param(f'{gravity_prefix}/urdf_relpath', gc['urdf_relpath'])
                rospy.set_param(f'{gravity_prefix}/gravity_joint_scale_left', gc['gravity_joint_scale_left'])
                rospy.set_param(f'{gravity_prefix}/gravity_joint_scale_right', gc['gravity_joint_scale_right'])
                rospy.set_param(f'{gravity_prefix}/gravity_joint_sign_left', gc['gravity_joint_sign_left'])
                rospy.set_param(f'{gravity_prefix}/gravity_joint_sign_right', gc['gravity_joint_sign_right'])
                rospy.set_param(
                    f'{gravity_prefix}/gravity_joint_position_scale_left',
                    gc['gravity_joint_position_scale_left'],
                )
                rospy.set_param(
                    f'{gravity_prefix}/gravity_joint_position_scale_right',
                    gc['gravity_joint_position_scale_right'],
                )
                rospy.set_param(f'{gravity_prefix}/gravity_joint_offset_left', gc['gravity_joint_offset_left'])
                rospy.set_param(f'{gravity_prefix}/gravity_joint_offset_right', gc['gravity_joint_offset_right'])
                rospy.set_param(f'{gravity_prefix}/clip_to_limits', gc['clip_to_limits'])
                rospy.set_param(f'{gravity_prefix}/unwrap_joint_positions', gc['unwrap_joint_positions'])
                rospy.set_param(f'{gravity_prefix}/max_joint_step', gc['max_joint_step'])
                rospy.set_param(f'{gravity_prefix}/max_torque_delta_warn', gc['max_torque_delta_warn'])

                left_slave_joint = self.topics_tpl['slave_joint_states_tpl'].format(slave=left_pair['slave'])
                right_slave_joint = self.topics_tpl['slave_joint_states_tpl'].format(slave=right_pair['slave'])
                left_slave_comp = self.topics_tpl.get(
                    'slave_joint_states_compensated_tpl', '/{slave}/joint_states_compensated'
                ).format(slave=left_pair['slave'])
                right_slave_comp = self.topics_tpl.get(
                    'slave_joint_states_compensated_tpl', '/{slave}/joint_states_compensated'
                ).format(slave=right_pair['slave'])

                gravity_node = roslaunch.core.Node(
                    'piper_test',
                    'teleop_gravity_compensation_node.py',
                    name='piper_gravity_compensation_node',
                    output='screen',
                    remap_args=[
                        ('/robot/arm_left/joint_states_single', left_slave_joint),
                        ('/robot/arm_right/joint_states_single', right_slave_joint),
                        ('/robot/arm_left/joint_states_compensated', left_slave_comp),
                        ('/robot/arm_right/joint_states_compensated', right_slave_comp),
                    ],
                )
                self.launcher.launch(gravity_node)
                rospy.loginfo(
                    "[启动] piper_gravity_compensation_node left=%s right=%s",
                    left_pair['slave'],
                    right_pair['slave'],
                )

        for cam in self.cameras:
            name = cam['name']
            node_name = f'realsense_pub_{name}'
            prefix = f'/{node_name}'
            color_topic = self.cam_topics_tpl['color_tpl'].format(name=name)
            depth_topic = self.cam_topics_tpl['depth_tpl'].format(name=name)
            s = cam['stream']

            rospy.set_param(f'{prefix}/serial_no', str(cam['serial_no']))
            rospy.set_param(f'{prefix}/device_name', cam['device_name'])
            rospy.set_param(f'{prefix}/color_topic', color_topic)
            rospy.set_param(f'{prefix}/depth_topic', depth_topic)
            rospy.set_param(f'{prefix}/enable_depth', s.get('enable_depth', False))
            rospy.set_param(f'{prefix}/width', s['width'])
            rospy.set_param(f'{prefix}/height', s['height'])
            rospy.set_param(f'{prefix}/fps', s['fps'])

            node = roslaunch.core.Node('piper_test', 'realsense_publisher2.py',
                                       name=node_name, output='screen')
            self.launcher.launch(node)
            rospy.loginfo(f"[启动] {node_name}  serial={cam['serial_no']}")

    # ====================== ROS 话题 ======================

    def _setup_ros(self):
        for pair in self.arm_pairs:
            pn = pair['name']
            self.arm_state[pn] = {
                'master': None,
                'slave': None,
                'fb_blend_filt': [0.0] * 6,   # 6个手臂关节的钳制系数
                'fb_eff_filt': self.force_fb['gripper_base_effort'],
                'fb_last_log_t': 0.0,
                'mit_active': False,           # MIT 模式是否已激活
            }

            mt = self.topics_tpl['master_joint_states_tpl'].format(master=pair['master'])
            rospy.Subscriber(mt, JointState,
                             lambda msg, p=pn: self._master_cb(p, msg),
                             queue_size=1, tcp_nodelay=True)

            st = self.topics_tpl['slave_joint_states_tpl'].format(slave=pair['slave'])
            rospy.Subscriber(st, JointState,
                             lambda msg, p=pn: self._slave_cb(p, msg),
                             queue_size=1, tcp_nodelay=True)

            ct = self.topics_tpl['slave_joint_ctrl_tpl'].format(slave=pair['slave'])
            self.arm_state[pn]['pub'] = rospy.Publisher(ct, JointState, queue_size=1, tcp_nodelay=True)

        for cam in self.cameras:
            cn = cam['name']
            self.cam_state[cn] = None
            color_topic = self.cam_topics_tpl['color_tpl'].format(name=cn)
            rospy.Subscriber(color_topic, Image,
                             lambda msg, n=cn: self._cam_cb(n, msg),
                             queue_size=1, tcp_nodelay=True)

            if cam['record'].get('save_depth', False):
                depth_topic = self.cam_topics_tpl['depth_tpl'].format(name=cn)
                rospy.Subscriber(depth_topic, Image,
                                 lambda msg, n=cn: self._depth_cb(n, msg),
                                 queue_size=1, tcp_nodelay=True)

    # ====================== 回调 ======================

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

        # ---- 切换 master 到 MIT 模式 ----
        if not st['mit_active']:
            piper.MotionCtrl_2(0x01, 0x01, 50, 0xAD)
            st['mit_active'] = True
            rospy.loginfo(f"[力反馈] {pair_name} 已启用 MIT 模式")

        # ---- 手臂关节 0..5: MIT 力矩反馈 ----
        # 位置差 → blend → kp 动态调节
        # kp 大 → 电机产生强阻力，你能明显感受到
        for i in range(6):
            pos_diff = abs(m['pos'][i] - s['pos'][i])
            contact_i = max(0.0, pos_diff - fb['pos_diff_threshold'])
            target_blend = min(1.0, fb['joint_gain'] * contact_i)

            alpha_j = fb['joint_ema_alpha']
            blend[i] = alpha_j * target_blend + (1.0 - alpha_j) * blend[i]

            # MIT: kp 决定阻力强度, pos_ref=slave位置(受阻位置)
            kp = fb['mit_kp_free'] * (1.0 - blend[i]) + fb['mit_kp_blocked'] * blend[i]
            kd = fb['mit_kd']

            # 当有接触: motor 被拉向 slave 位置，产生回弹阻力
            # 当无接触: kp=0, motor 自由（人可以随意移动）
            piper.JointMitCtrl(
                i + 1,              # motor_num: 1-6
                s['pos'][i],        # pos_ref: slave 的位置（受阻位置）
                0.0,                # vel_ref
                kp,                 # kp: 位置刚度（受阻时变大）
                kd,                 # kd: 阻尼
                0.0                 # t_ref: 前馈力矩
            )

        st['fb_blend_filt'] = blend

        # ---- 夹爪: effort 反馈（通过 GripperCtrl）----
        grip_pos_diff = abs(m['pos'][6] - s['pos'][6])
        grip_contact = max(0.0, grip_pos_diff - fb['pos_diff_threshold'])
        slave_eff_grip = abs(float(s['eff'][-1])) if s.get('eff') else 0.0
        eff_contact = max(0.0, slave_eff_grip - fb['gripper_contact_threshold'])
        contact_grip = max(grip_contact, eff_contact)

        target_eff = fb['gripper_base_effort'] + fb['gripper_gain'] * contact_grip
        target_eff = clamp(target_eff, fb['gripper_min_effort'], fb['gripper_max_effort'])
        alpha_g = fb['gripper_ema_alpha']
        prev_eff = st.get('fb_eff_filt', fb['gripper_base_effort'])
        filt_eff = alpha_g * target_eff + (1.0 - alpha_g) * prev_eff
        st['fb_eff_filt'] = filt_eff

        gripper_angle = round(abs(m['pos'][6]) * 1000000)
        gripper_effort = round(clamp(filt_eff, 0.5, 3.0) * 1000)
        piper.GripperCtrl(gripper_angle, gripper_effort, 0x01, 0)

        # ---- CSV 日志 ----
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

    def _depth_cb(self, cam_name, msg):
        dep = depth_from_msg(msg)
        if dep is not None and self.cam_state[cam_name] is not None:
            self.cam_state[cam_name]['depth'] = dep

    def _camera_slave_tag(self, cam_name):
        """Best-effort mapping for GUI labels: left->pair1 slave, right->pair2 slave."""
        name = str(cam_name).strip().lower()
        if not name:
            return ""

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

    # ====================== 预览 ======================

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
                waiting_label = f'{cn}: waiting...' if not slave_tag else f'{cn}/{slave_tag}: waiting...'
                cv2.putText(panel, waiting_label,
                            (pw // 6, ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 半透明标签背景
            overlay = panel.copy()
            cv2.rectangle(overlay, (0, 0), (pw, 32), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, panel, 0.5, 0, panel)
            lbl_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cam_label = f'  {cn} ({cam["device_name"]})'
            if slave_tag:
                cam_label += f' -> {slave_tag}'
            cv2.putText(panel, cam_label,
                        (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lbl_color, 2)
            panels.append(panel)

        mosaic = np.hstack(panels) if panels else np.zeros((ph, pw, 3), dtype=np.uint8)
        total_w = mosaic.shape[1]

        # 状态栏
        bar_h = 48
        bar = np.zeros((bar_h, total_w, 3), dtype=np.uint8)
        if self.is_recording:
            elapsed = time.time() - self.rec_start_time if self.rec_start_time else 0
            m, s = int(elapsed) // 60, int(elapsed) % 60
            txt = f"  REC  |  frames: {self.frame_count}   time: {m:02d}:{s:02d}   |  [S]ave  [D]iscard  [SPACE]stop"
            if int(elapsed * 2) % 2 == 0:
                cv2.circle(bar, (20, bar_h // 2), 8, (0, 0, 255), -1)
            cv2.putText(bar, txt, (36, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            txt = f"  IDLE  |  next_ep={self.ep_count}  saved={self.save_done}  |  [SPACE]record  [Q]uit"
            cv2.circle(bar, (20, bar_h // 2), 8, (0, 255, 0), -1)
            cv2.putText(bar, txt, (36, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        display = np.vstack([mosaic, bar])
        cv2.imshow(WINDOW_NAME, display)

    # ====================== 就绪 ======================

    def _all_ready(self):
        for pn, st in self.arm_state.items():
            if st['master'] is None or st['slave'] is None:
                return False
        for cn, cs in self.cam_state.items():
            if cs is None or 'color' not in cs:
                return False
        return True

    # ====================== 录制 ======================

    def _new_episode(self):
        ep = {'timestamps': [], 'masters': {}, 'slaves': {}, 'cameras': {}}
        for pair in self.arm_pairs:
            pn = pair['name']
            ep['masters'][pn] = {'positions': [], 'velocities': [], 'efforts': [], 'stamps': []}
            ep['slaves'][pn] = {'positions': [], 'velocities': [], 'efforts': [], 'stamps': []}
        for cam in self.cameras:
            cn = cam['name']
            ep['cameras'][cn] = {'color': [], 'stamps': []}
            if cam['record'].get('save_depth', False):
                ep['cameras'][cn]['depth'] = []
        return ep

    def _record_frame(self):
        if not self._all_ready():
            return False

        t = rospy.Time.now().to_sec()
        ep = self.ep_data
        ep['timestamps'].append(t)

        rec_m, rec_s = self.rec['master'], self.rec['slave']

        for pair in self.arm_pairs:
            pn = pair['name']
            m = self.arm_state[pn]['master']
            s = self.arm_state[pn]['slave']

            if rec_m['save_positions']:
                ep['masters'][pn]['positions'].append(m['pos'][:])
            if rec_m.get('save_velocities'):
                ep['masters'][pn]['velocities'].append(m['vel'][:])
            if rec_m.get('save_efforts'):
                ep['masters'][pn]['efforts'].append(m['eff'][:])
            if rec_m.get('save_stamp'):
                ep['masters'][pn]['stamps'].append(m['stamp'])

            if rec_s['save_positions']:
                ep['slaves'][pn]['positions'].append(s['pos'][:])
            if rec_s.get('save_velocities'):
                ep['slaves'][pn]['velocities'].append(s['vel'][:])
            if rec_s.get('save_efforts'):
                ep['slaves'][pn]['efforts'].append(s['eff'][:])
            if rec_s.get('save_stamp'):
                ep['slaves'][pn]['stamps'].append(s['stamp'])

        for cam in self.cameras:
            cn = cam['name']
            cs = self.cam_state[cn]
            rec = cam['record']
            if rec.get('save_color', True):
                img = cs['color']
                sw, sh = rec.get('save_width'), rec.get('save_height')
                if sw and sh and (sw != img.shape[1] or sh != img.shape[0]):
                    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                ep['cameras'][cn]['color'].append(img.copy())
            if rec.get('save_stamp'):
                ep['cameras'][cn]['stamps'].append(cs.get('stamp', t))
            if rec.get('save_depth') and 'depth' in cs:
                dep = cs['depth']
                sw, sh = rec.get('save_width'), rec.get('save_height')
                if sw and sh and (sw != dep.shape[1] or sh != dep.shape[0]):
                    dep = cv2.resize(dep, (sw, sh), interpolation=cv2.INTER_NEAREST)
                ep['cameras'][cn]['depth'].append(dep.copy())

        self.frame_count += 1
        return True

    # ====================== HDF5 ======================

    def _next_ep_num(self):
        files = [f for f in os.listdir(self.data_dir)
                 if f.startswith('episode_') and f.endswith('.hdf5')]
        if not files:
            return 0
        return max(int(f.split('_')[1].split('.')[0]) for f in files) + 1

    def _h5_comp(self):
        c = (self.img_comp or '').lower()
        if c in ('none', 'off', ''):
            return {}
        if c == 'lzf':
            return {'compression': 'lzf', 'shuffle': True}
        return {'compression': 'gzip', 'compression_opts': self.gzip_level, 'shuffle': True}

    def _submit_save(self):
        n = len(self.ep_data['timestamps'])
        if n == 0:
            rospy.logwarn("没有数据可保存")
            return False
        payload = {
            'ep_id': self.ep_count,
            'filename': os.path.join(self.data_dir, f'episode_{self.ep_count}.hdf5'),
            'data': self.ep_data,
            'rate_hz': self.rate_hz,
            'created_at': datetime.now().isoformat(),
        }
        try:
            self.save_q.put_nowait(payload)
        except queue.Full:
            rospy.logwarn("保存队列满，跳过")
            return False
        rospy.loginfo(f"提交保存: episode_{self.ep_count} ({n} 帧)")
        self.ep_count += 1
        self.ep_data = None
        return True

    def _save_worker(self):
        while not rospy.is_shutdown():
            try:
                payload = self.save_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._write_hdf5(payload)
                self.save_done += 1
            except Exception as e:
                rospy.logerr(f"保存失败: {e}")
            finally:
                self.save_q.task_done()

    def _write_hdf5(self, payload):
        fn = payload['filename']
        data = payload['data']
        comp = self._h5_comp()
        keys = self.hdf5_keys

        with h5py.File(fn, 'w') as f:
            f.create_dataset(keys['timestamps'],
                             data=np.array(data['timestamps'], dtype=np.float64),
                             chunks=True, **comp)

            mg = f.create_group(keys['masters_group'])
            for pn, md in data['masters'].items():
                pg = mg.create_group(pn)
                for field in ('positions', 'velocities', 'efforts'):
                    if md[field]:
                        pg.create_dataset(field, data=np.array(md[field], dtype=np.float32),
                                          chunks=True, **comp)
                if md['stamps']:
                    pg.create_dataset('stamp', data=np.array(md['stamps'], dtype=np.float64),
                                      chunks=True, **comp)

            sg = f.create_group(keys['slaves_group'])
            for pn, sd in data['slaves'].items():
                pg = sg.create_group(pn)
                for field in ('positions', 'velocities', 'efforts'):
                    if sd[field]:
                        pg.create_dataset(field, data=np.array(sd[field], dtype=np.float32),
                                          chunks=True, **comp)
                if sd['stamps']:
                    pg.create_dataset('stamp', data=np.array(sd['stamps'], dtype=np.float64),
                                      chunks=True, **comp)

            cg = f.create_group(keys['cameras_group'])
            for cn, cd in data['cameras'].items():
                ccg = cg.create_group(cn)
                if cd['color']:
                    imgs = cd['color']
                    h, w, c = imgs[0].shape
                    ds = ccg.create_dataset('color', shape=(len(imgs), h, w, c),
                                            dtype=np.uint8, chunks=(1, h, w, c), **comp)
                    for i, im in enumerate(imgs):
                        ds[i] = im
                if cd['stamps']:
                    ccg.create_dataset('stamp', data=np.array(cd['stamps'], dtype=np.float64),
                                       chunks=True, **comp)
                if 'depth' in cd and cd['depth']:
                    deps = cd['depth']
                    dh, dw = deps[0].shape
                    ds = ccg.create_dataset('depth', shape=(len(deps), dh, dw),
                                            dtype=np.uint16, chunks=(1, dh, dw), **comp)
                    for i, d in enumerate(deps):
                        ds[i] = d

            f.attrs['episode_id'] = payload['ep_id']
            f.attrs['num_frames'] = len(data['timestamps'])
            f.attrs['frequency'] = payload['rate_hz']
            f.attrs['created_at'] = payload['created_at']
            f.attrs['schema_version'] = self.cfg.get('schema_version', 1)

        sz = os.path.getsize(fn) / (1024 * 1024)
        rospy.loginfo(f"保存完成: {fn} | {len(data['timestamps'])} 帧 | {sz:.1f} MB")

    # ====================== 主循环 ======================

    def _print_banner(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("Teleop Raw System PiperROS-FF (YAML 配置驱动)")
        rospy.loginfo(f"  data_dir  = {self.data_dir}")
        rospy.loginfo(f"  rate      = {self.rate_hz} Hz")
        rospy.loginfo(f"  arm_pairs = {[p['name'] for p in self.arm_pairs]}")
        rospy.loginfo(f"  cameras   = {[c['name'] for c in self.cameras]}")
        rospy.loginfo(f"  compress  = {self.img_comp} (gzip_level={self.gzip_level})")
        fb = self.force_fb
        rospy.loginfo("  force_fb  = %s (backend=%s)", fb['enabled'], self.force_fb_backend)
        if self.ff_use_piperros_node:
            ff = self.force_fb_node
            rospy.loginfo(
                "    piperros_node: ctrl=%s mit(pos/tor/gravity)=%s/%s/%s",
                ff['ctrl_mode'],
                ff['mit_enable_pos'],
                ff['mit_enable_tor'],
                self.runtime_gravity_enabled,
            )
            rospy.loginfo(
                "    mit params: kp=%s kd=%s torque_scale=%s",
                [round(v, 3) for v in ff['mit_kp']],
                [round(v, 3) for v in ff['mit_kd']],
                [round(v, 3) for v in ff['mit_torque_scale']],
            )
            rospy.loginfo(
                "    torque_scale L/R=%s | %s",
                [round(v, 3) for v in ff['mit_torque_scale_left']],
                [round(v, 3) for v in ff['mit_torque_scale_right']],
            )
            if self.runtime_gravity_enabled:
                gc = self.gravity_comp
                rospy.loginfo(
                    "    gravity: urdf=%s/%s max_joint_step=%.3f max_torque_delta_warn=%.3f",
                    gc['urdf_package'],
                    gc['urdf_relpath'],
                    gc['max_joint_step'],
                    gc['max_torque_delta_warn'],
                )
                rospy.loginfo(
                    "    gravity_joint_scale L/R=%s | %s",
                    [round(v, 3) for v in gc['gravity_joint_scale_left']],
                    [round(v, 3) for v in gc['gravity_joint_scale_right']],
                )
            elif ff['mit_enable_gravity'] and self.gravity_comp['enabled']:
                if self.gravity_urdf_path is None:
                    rospy.logwarn(
                        "    gravity disabled at runtime: missing model %s/%s (%s)",
                        self.gravity_comp['urdf_package'],
                        self.gravity_comp['urdf_relpath'],
                        self.gravity_urdf_error,
                    )
                elif not self.pinocchio_available:
                    rospy.logwarn("    gravity disabled at runtime: missing python module 'pinocchio'")
                elif len(self.arm_pairs) < 2:
                    rospy.logwarn("    gravity disabled at runtime: arm_pairs < 2")

        if self.robot_description['enabled'] and not self.robot_description_loaded:
            rospy.logwarn(
                "    robot_description not loaded: %s/%s (%s)",
                self.robot_description['package'],
                self.robot_description['relpath'],
                self.robot_description_error,
            )
        if self.ff_use_legacy:
            rospy.loginfo(
                "    legacy_sdk: kp_free=%.2f kp_blocked=%.2f kd=%.2f",
                fb['mit_kp_free'],
                fb['mit_kp_blocked'],
                fb['mit_kd'],
            )
            rospy.loginfo(
                "    detect: pos_diff_th=%.4f gain=%.2f gripper_th=%.4f",
                fb['pos_diff_threshold'],
                fb['joint_gain'],
                fb['gripper_contact_threshold'],
            )
            if fb['csv_log']:
                rospy.loginfo("    csv_log: %s", fb['csv_path'])
        rospy.loginfo("HDF5: /timestamps  /masters/{pair}/  /slaves/{pair}/  /cameras/{cam}/")
        rospy.loginfo("预览窗口可自由拖拽缩放，按键在窗口聚焦时生效")
        rospy.loginfo("[SPACE]录制  [S]保存  [D]丢弃  [Q]退出")
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
                self.is_recording = not self.is_recording
                if self.is_recording:
                    self.ep_data = self._new_episode()
                    self.frame_count = 0
                    self.rec_start_time = time.time()
                    rospy.loginfo("开始录制")
                else:
                    rospy.loginfo(f"暂停录制 ({self.frame_count} 帧)")

            elif key == ord('s'):
                self.is_recording = False
                if self.ep_data:
                    self._submit_save()

            elif key == ord('d'):
                self.is_recording = False
                n = len(self.ep_data['timestamps']) if self.ep_data else 0
                self.ep_data = None
                rospy.loginfo(f"已丢弃 {n} 帧")

            elif key == ord('q'):
                rospy.loginfo("退出...")
                break

            if self.is_recording and self.ep_data is not None:
                self._record_frame()

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
