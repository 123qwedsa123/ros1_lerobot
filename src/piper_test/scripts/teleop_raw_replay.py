#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teleop raw HDF5 replay with safe slow reset + OpenCV GUI.

GUI features:
  - Episode selector: browse all episodes in data_dir, pick one to replay
  - Camera viewer:    shows recorded camera images frame-by-frame during replay
  - Playback controls via keyboard (window must have focus):
      SPACE   pause / resume
      L       toggle loop
      + / =   speed up  (×1.5)
      -       slow down (÷1.5)
      Q / ESC stop and quit
  - Episode selector navigation:
      W / ↑   move selection up
      S / ↓   move selection down
      ENTER   start replay
      Q / ESC quit

Usage:
  roslaunch piper_test teleop_raw_replay.launch
  roslaunch piper_test teleop_raw_replay.launch config:=/path/to/teleop_raw_replay.yaml
"""

import math
import os
import re
import time

import cv2
import h5py
import numpy as np
import roslaunch
import rospy
import yaml
from sensor_msgs.msg import JointState


# ─────────────────────────── helpers ───────────────────────────

def fix_len(values, n, fill=0.0):
    items = list(values) if values is not None else []
    return (items + [fill] * n)[:n]


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def as_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    return bool(default)


# ─────────────────────────── main class ───────────────────────────

class TeleopRawReplay:

    # ── init ──────────────────────────────────────────────────────
    def __init__(self):
        rospy.init_node("teleop_raw_replay", anonymous=True)

        config_file = rospy.get_param("~config_file")
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)

        # ---------- global ----------
        g = self.cfg.get("global", {})
        self.data_dir = str(g.get("data_dir", "/home/jinhe/Desktop/piper_master_slave_ws/piper_master_slave_ws/data"))
        self.fallback_rate_hz = max(as_float(g.get("fallback_rate_hz", 30.0), 30.0), 1e-6)
        self.ctrl_startup_wait_sec = max(as_float(g.get("ctrl_startup_wait_sec", 2.0), 2.0), 0.0)
        self.ctrl_ready_timeout_sec = max(as_float(g.get("ctrl_ready_timeout_sec", 8.0), 8.0), 0.0)
        self.min_dt_sec = max(as_float(g.get("min_dt_sec", 0.001), 0.001), 0.0)
        self.max_dt_sec = max(as_float(g.get("max_dt_sec", 0.2), 0.2), self.min_dt_sec)

        # ---------- replay ----------
        r = self.cfg.get("replay", {})
        self.episode_file = str(r.get("episode_file", "")).strip()
        self.episode_id = as_int(r.get("episode_id", -1), -1)
        self.speed_scale = max(as_float(r.get("speed_scale", 1.0), 1.0), 1e-6)
        self.loop = as_bool(r.get("loop", False), False)
        self.hold_last_sec = max(as_float(r.get("hold_last_sec", 1.0), 1.0), 0.0)
        self.hold_hz = max(as_float(r.get("hold_hz", 10.0), 10.0), 1e-6)

        # ---------- safety reset ----------
        sr = self.cfg.get("safety_reset", {})
        self.reset_enabled = as_bool(sr.get("enabled", True), True)
        self.reset_max_joint_speed_rad_s = max(
            as_float(sr.get("max_joint_speed_rad_s", 0.10), 0.10), 1e-6
        )
        self.reset_min_steps = max(as_int(sr.get("min_steps", 30), 30), 1)
        self.reset_rate_hz = max(as_float(sr.get("reset_rate_hz", 30.0), 30.0), 1e-6)
        self.reset_settle_hold_sec = max(as_float(sr.get("settle_hold_sec", 0.3), 0.3), 0.0)
        self.reset_err_thresh_rad = max(as_float(sr.get("err_thresh_rad", 0.03), 0.03), 0.0)
        self.reset_timeout_sec = max(as_float(sr.get("timeout_sec", 12.0), 12.0), 0.1)
        self.reset_fail_policy = str(sr.get("fail_policy", "abort_replay")).strip()
        if self.reset_fail_policy != "abort_replay":
            rospy.logwarn(
                "[CONFIG] unsupported safety_reset.fail_policy=%s, force to abort_replay",
                self.reset_fail_policy,
            )
            self.reset_fail_policy = "abort_replay"

        # ---------- topics / ctrl ----------
        topics = self.cfg.get("topics", {})
        self.slave_joint_ctrl_tpl = str(
            topics.get("slave_joint_ctrl_tpl", "/{slave}/joint_ctrl_single")
        )
        self.slave_joint_states_tpl = str(
            topics.get("slave_joint_states_tpl", "/{slave}/joint_states_single")
        )

        defaults = self.cfg.get("piper_ctrl_defaults", {})
        self.ctrl_auto_enable = as_bool(defaults.get("auto_enable", True), True)
        self.ctrl_gripper_exist = as_bool(defaults.get("gripper_exist", True), True)
        self.ctrl_can_judge_flag = as_bool(defaults.get("can_judge_flag", False), False)
        self.ctrl_gripper_val_multiple = as_int(
            defaults.get("gripper_val_multiple", defaults.get("gripper_val_mutiple", 1)),
            1,
        )

        cc = self.cfg.get("ctrl_cmd", {})
        self.joint_velocity_default = as_float(cc.get("joint_velocity_default", 0.0), 0.0)
        self.joint_effort_default = as_float(cc.get("joint_effort_default", 0.0), 0.0)
        self.gripper_velocity = as_float(cc.get("gripper_velocity", 100.0), 100.0)
        self.gripper_effort = as_float(cc.get("gripper_effort", 1.0), 1.0)

        keys = self.cfg.get("hdf5_keys", {})
        self.key_timestamps = str(keys.get("timestamps", "timestamps"))
        self.key_slaves_group = str(keys.get("slaves_group", "slaves"))
        self.key_cameras_group = str(keys.get("cameras_group", "cameras"))

        self.arm_pairs = self.cfg.get("arm_pairs", [])
        if not self.arm_pairs:
            raise RuntimeError("[CONFIG] arm_pairs is empty")
        for idx, pair in enumerate(self.arm_pairs):
            for required in ("name", "slave", "slave_can"):
                if required not in pair:
                    raise RuntimeError("[CONFIG] arm_pairs[{}] missing '{}'".format(idx, required))

        os.makedirs(self.data_dir, exist_ok=True)

        # ---------- GUI config ----------
        gui_cfg = self.cfg.get("gui", {})
        self.gui_enabled = as_bool(gui_cfg.get("enabled", True), True)
        self.gui_window_name = str(gui_cfg.get("window_name", "Teleop Raw Replay"))
        self.gui_show_selector = as_bool(gui_cfg.get("show_selector", True), True)
        # list of camera names to display; empty = show all
        self.gui_show_cameras = gui_cfg.get("show_cameras", []) or []
        # thumbnail size per camera in the replay view
        self.gui_thumb_w = as_int(gui_cfg.get("thumb_width", 320), 320)
        self.gui_thumb_h = as_int(gui_cfg.get("thumb_height", 240), 240)

        # ---------- runtime ----------
        self.launcher = None
        self.ctrl_pubs = {}
        self.slave_states = {}
        self.state_topics = {}

        self.selected_episode_path = ""
        self.selected_episode_id = -1
        self.frame_count = 0
        self.timestamps = None
        self.replay_pos = {}

        # GUI runtime state
        self.camera_images = {}   # cam_name -> np.ndarray (N, H, W, 3) BGR
        self.paused = False
        self.stop_requested = False

        self._launch_child_nodes()
        self._setup_ros()
        self._print_banner()

    # ── child nodes ───────────────────────────────────────────────
    def _preflight_can_interfaces(self):
        missing = []
        not_up = []
        for pair in self.arm_pairs:
            can_port = str(pair["slave_can"])
            operstate_path = "/sys/class/net/{}/operstate".format(can_port)
            if not os.path.exists(operstate_path):
                missing.append(can_port)
                continue
            try:
                with open(operstate_path, "r") as f:
                    state = f.read().strip().lower()
            except OSError as exc:
                rospy.logwarn("[CAN] failed reading %s: %s", operstate_path, exc)
                continue
            if state != "up":
                not_up.append((can_port, state))

        if missing:
            can_list = ", ".join(sorted(set(missing)))
            msg = (
                "missing CAN interfaces in current namespace: {}. "
                "If using Docker/Rocker, start with --network=host --privileged."
            ).format(can_list)
            rospy.logfatal("[CAN] %s", msg)
            raise RuntimeError(msg)

        for can_port, state in not_up:
            rospy.logwarn("[CAN] %s state=%s (expected up)", can_port, state)

    def _launch_child_nodes(self):
        self._preflight_can_interfaces()
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launcher = roslaunch.scriptapi.ROSLaunch()
        self.launcher.start()

        for pair in self.arm_pairs:
            slave = pair["slave"]
            can_port = pair["slave_can"]
            prefix = "/{}/piper_ctrl_node".format(slave)

            rospy.set_param(prefix + "/can_port", can_port)
            rospy.set_param(prefix + "/auto_enable", self.ctrl_auto_enable)
            rospy.set_param(prefix + "/gripper_exist", self.ctrl_gripper_exist)
            rospy.set_param(prefix + "/girpper_exist", self.ctrl_gripper_exist)
            rospy.set_param(prefix + "/can_judge_flag", self.ctrl_can_judge_flag)
            rospy.set_param(prefix + "/gripper_val_mutiple", self.ctrl_gripper_val_multiple)

            node = roslaunch.core.Node(
                package="piper",
                node_type="piper_ctrl_single_node.py",
                name="piper_ctrl_node",
                namespace=slave,
                output="screen",
            )
            self.launcher.launch(node)
            rospy.loginfo("[启动] %s/piper_ctrl_node can=%s", slave, can_port)

        if self.ctrl_startup_wait_sec > 0:
            rospy.sleep(self.ctrl_startup_wait_sec)

    # ── ROS setup ─────────────────────────────────────────────────
    def _setup_ros(self):
        for pair in self.arm_pairs:
            pair_name = pair["name"]
            slave = pair["slave"]
            ctrl_topic = self.slave_joint_ctrl_tpl.format(slave=slave)
            state_topic = self.slave_joint_states_tpl.format(slave=slave)

            self.ctrl_pubs[pair_name] = rospy.Publisher(
                ctrl_topic, JointState, queue_size=1, tcp_nodelay=True
            )
            self.slave_states[pair_name] = None
            self.state_topics[pair_name] = state_topic

            rospy.Subscriber(
                state_topic,
                JointState,
                lambda msg, p=pair_name: self._slave_state_cb(p, msg),
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.loginfo("[ROS] pair=%s ctrl=%s state=%s", pair_name, ctrl_topic, state_topic)

    def _slave_state_cb(self, pair_name, msg):
        self.slave_states[pair_name] = {
            "pos": fix_len(msg.position, 7, 0.0),
            "stamp": msg.header.stamp.to_sec(),
        }

    def _print_banner(self):
        rospy.loginfo("=" * 70)
        rospy.loginfo("Teleop Raw Replay (safe slow reset + OpenCV GUI)")
        rospy.loginfo("  data_dir          = %s", self.data_dir)
        rospy.loginfo("  episode_file      = %s", self.episode_file if self.episode_file else "(auto)")
        rospy.loginfo("  episode_id        = %d", self.episode_id)
        rospy.loginfo("  speed_scale       = %.3f", self.speed_scale)
        rospy.loginfo("  loop              = %s", self.loop)
        rospy.loginfo("  hold_last         = %.2fs @ %.2fHz", self.hold_last_sec, self.hold_hz)
        rospy.loginfo(
            "  safety_reset      = %s (max_speed=%.3f rad/s, min_steps=%d, hz=%.2f, "
            "err=%.4f, timeout=%.2f)",
            self.reset_enabled,
            self.reset_max_joint_speed_rad_s,
            self.reset_min_steps,
            self.reset_rate_hz,
            self.reset_err_thresh_rad,
            self.reset_timeout_sec,
        )
        rospy.loginfo("  arm_pairs         = %s", [p["name"] for p in self.arm_pairs])
        rospy.loginfo("  gui.enabled       = %s", self.gui_enabled)
        rospy.loginfo("  gui.show_selector = %s", self.gui_show_selector)
        rospy.loginfo("=" * 70)

    # ── wait for slave states ──────────────────────────────────────
    def _wait_slave_states_ready(self):
        """Wait for all slave JointState topics. Shows a GUI screen if gui_enabled."""
        if self.gui_enabled:
            return self._wait_slave_states_gui()
        return self._wait_slave_states_headless()

    def _wait_slave_states_headless(self):
        start_time = time.time()
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            missing = [pn for pn, st in self.slave_states.items() if st is None]
            if not missing:
                rospy.loginfo("[READY] all slave states are ready")
                return True
            if self.ctrl_ready_timeout_sec > 0 and (time.time() - start_time) > self.ctrl_ready_timeout_sec:
                rospy.logfatal("[READY] timeout waiting state topics, missing=%s", missing)
                return False
            rate.sleep()
        return False

    def _wait_slave_states_gui(self):
        W, H = 640, 220
        cv2.namedWindow(self.gui_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.gui_window_name, W, H)

        start_time = time.time()
        dot_count = 0
        while not rospy.is_shutdown():
            missing = [pn for pn, st in self.slave_states.items() if st is None]
            if not missing:
                rospy.loginfo("[READY] all slave states are ready")
                return True

            elapsed = time.time() - start_time
            if self.ctrl_ready_timeout_sec > 0 and elapsed > self.ctrl_ready_timeout_sec:
                rospy.logfatal("[READY] timeout, missing=%s", missing)
                return False

            canvas = np.full((H, W, 3), 30, dtype=np.uint8)
            cv2.rectangle(canvas, (0, 0), (W, 50), (40, 40, 60), -1)
            cv2.putText(canvas, self.gui_window_name,
                        (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 255), 1)

            dots = "." * (dot_count % 4)
            msg = "Waiting for arm controllers" + dots
            cv2.putText(canvas, msg, (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 100), 1)
            cv2.putText(canvas, "Missing: " + ", ".join(missing),
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 100, 100), 1)
            remain = max(0, int(self.ctrl_ready_timeout_sec - elapsed))
            cv2.putText(canvas, "Timeout in {}s  |  [Q] Quit".format(remain),
                        (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)

            cv2.imshow(self.gui_window_name, canvas)
            key = cv2.waitKey(200) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                return False
            dot_count += 1
        return False

    # ── episode scanning ──────────────────────────────────────────
    def _scan_all_episodes(self):
        """Return sorted list of (episode_id, path) from data_dir."""
        files = []
        try:
            for name in os.listdir(self.data_dir):
                m = re.match(r"^episode_(\d+)\.hdf5$", name)
                if m:
                    files.append((int(m.group(1)), os.path.join(self.data_dir, name)))
        except OSError:
            pass
        return sorted(files, key=lambda x: x[0])

    def _get_episode_meta(self, path):
        """Quick-read: return (n_frames, cam_names) from an HDF5 file."""
        try:
            with h5py.File(path, "r") as f:
                n = int(f[self.key_timestamps].shape[0]) if self.key_timestamps in f else 0
                cams = list(f[self.key_cameras_group].keys()) \
                    if self.key_cameras_group in f else []
                return n, cams
        except Exception:
            return 0, []

    # ── episode selector GUI ──────────────────────────────────────
    def _show_episode_selector(self):
        """
        Show a full-window episode list. Returns the selected episode dict
        {"eid": int, "path": str, "n_frames": int, "cams": list} or None.
        """
        all_eps = self._scan_all_episodes()
        if not all_eps:
            rospy.logerr("[GUI] no episode_*.hdf5 found in %s", self.data_dir)
            return None

        # Build metadata list, newest episode first
        ep_list = []
        for eid, path in reversed(all_eps):
            n_frames, cam_names = self._get_episode_meta(path)
            ep_list.append({"eid": eid, "path": path, "n_frames": n_frames, "cams": cam_names})

        W, H = 760, 520
        cv2.namedWindow(self.gui_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.gui_window_name, W, H)

        sel = 0          # currently highlighted index
        ITEM_H = 54
        LIST_Y = 60
        FOOTER_H = 60

        while not rospy.is_shutdown():
            canvas = np.full((H, W, 3), 28, dtype=np.uint8)

            # ── title bar ──
            cv2.rectangle(canvas, (0, 0), (W, LIST_Y - 4), (40, 42, 65), -1)
            cv2.putText(canvas, "SELECT EPISODE TO REPLAY",
                        (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 255), 1)
            cv2.line(canvas, (0, LIST_Y - 4), (W, LIST_Y - 4), (70, 70, 100), 1)

            # ── episode list ──
            n_visible = (H - LIST_Y - FOOTER_H) // ITEM_H
            scroll = max(0, sel - n_visible // 2)
            scroll = max(0, min(scroll, max(0, len(ep_list) - n_visible)))

            for disp_i, ep_i in enumerate(range(scroll, min(scroll + n_visible, len(ep_list)))):
                ep = ep_list[ep_i]
                y = LIST_Y + disp_i * ITEM_H
                is_sel = (ep_i == sel)

                bg_col = (42, 72, 118) if is_sel else (38, 38, 38)
                border_col = (100, 160, 245) if is_sel else (52, 52, 52)
                cv2.rectangle(canvas, (8, y + 2), (W - 8, y + ITEM_H - 2), bg_col, -1)
                cv2.rectangle(canvas, (8, y + 2), (W - 8, y + ITEM_H - 2), border_col, 1)

                arrow = "> " if is_sel else "  "
                title_col = (255, 255, 255) if is_sel else (175, 175, 175)
                cv2.putText(canvas,
                            "{}episode_{}.hdf5".format(arrow, ep["eid"]),
                            (22, y + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, title_col, 1)

                cam_str = ", ".join(ep["cams"]) if ep["cams"] else "no cameras"
                info = "{} frames  |  cameras: {}".format(ep["n_frames"], cam_str)
                info_col = (110, 200, 120) if is_sel else (90, 130, 90)
                cv2.putText(canvas, info,
                            (35, y + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, info_col, 1)

            # ── scroll indicator ──
            if len(ep_list) > n_visible:
                track_h = H - LIST_Y - FOOTER_H
                thumb_h = max(20, track_h * n_visible // len(ep_list))
                thumb_y = LIST_Y + track_h * scroll // len(ep_list)
                cv2.rectangle(canvas, (W - 6, LIST_Y), (W - 2, LIST_Y + track_h), (50, 50, 50), -1)
                cv2.rectangle(canvas, (W - 6, thumb_y), (W - 2, thumb_y + thumb_h), (120, 120, 180), -1)

            # ── footer ──
            cv2.rectangle(canvas, (0, H - FOOTER_H), (W, H), (38, 38, 38), -1)
            cv2.line(canvas, (0, H - FOOTER_H), (W, H - FOOTER_H), (65, 65, 65), 1)
            cv2.putText(canvas,
                        "[W/S] Navigate    [ENTER] Start Replay    [Q/ESC] Quit",
                        (16, H - FOOTER_H + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (155, 155, 155), 1)
            cv2.putText(canvas,
                        "{} episode(s)  in  {}".format(len(ep_list), self.data_dir),
                        (16, H - FOOTER_H + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)

            cv2.imshow(self.gui_window_name, canvas)
            key = cv2.waitKey(50)
            if key == -1:
                continue
            kc = key & 0xFF

            if kc in (ord('q'), ord('Q'), 27):       # Q or ESC → quit
                return None
            elif kc in (ord('w'), ord('W')) or key == 82:   # W or ↑
                sel = max(0, sel - 1)
            elif kc in (ord('s'), ord('S')) or key == 84:   # S or ↓
                sel = min(len(ep_list) - 1, sel + 1)
            elif kc == 13:                            # ENTER → confirm
                return ep_list[sel]

        return None

    # ── episode loading ───────────────────────────────────────────
    def _resolve_episode(self):
        if self.episode_file:
            candidate = os.path.expanduser(self.episode_file)
            if not os.path.isabs(candidate):
                candidate = os.path.join(self.data_dir, candidate)
            candidate = os.path.abspath(candidate)
            if not os.path.isfile(candidate):
                raise RuntimeError("episode_file not found: {}".format(candidate))
            self.selected_episode_path = candidate
            match = re.match(r"^episode_(\d+)\.hdf5$", os.path.basename(candidate))
            self.selected_episode_id = int(match.group(1)) if match else -1
            return

        files = []
        for name in os.listdir(self.data_dir):
            m = re.match(r"^episode_(\d+)\.hdf5$", name)
            if m:
                files.append((int(m.group(1)), os.path.join(self.data_dir, name)))

        if not files:
            raise RuntimeError("no episode_*.hdf5 found in {}".format(self.data_dir))

        files.sort(key=lambda x: x[0])
        available_ids = [eid for eid, _ in files]

        if self.episode_id < 0:
            self.selected_episode_id, self.selected_episode_path = files[-1]
        else:
            hit = [item for item in files if item[0] == self.episode_id]
            if not hit:
                raise RuntimeError(
                    "episode_{}.hdf5 not found in {} (available ids: {})".format(
                        self.episode_id, self.data_dir, available_ids
                    )
                )
            self.selected_episode_id, self.selected_episode_path = hit[0]

    def _load_episode(self):
        self._resolve_episode()
        rospy.loginfo(
            "[LOAD] episode id=%s  path=%s",
            self.selected_episode_id if self.selected_episode_id >= 0 else "unknown",
            self.selected_episode_path,
        )

        with h5py.File(self.selected_episode_path, "r") as f:
            if self.key_timestamps not in f:
                raise RuntimeError("missing key '/{}' in {}".format(
                    self.key_timestamps, self.selected_episode_path))
            if self.key_slaves_group not in f:
                raise RuntimeError("missing key '/{}' in {}".format(
                    self.key_slaves_group, self.selected_episode_path))

            timestamps = np.asarray(f[self.key_timestamps], dtype=np.float64).reshape(-1)
            if timestamps.size == 0:
                raise RuntimeError("timestamps is empty in {}".format(self.selected_episode_path))

            slaves_group = f[self.key_slaves_group]
            replay_pos = {}
            lengths = [int(timestamps.shape[0])]

            for pair in self.arm_pairs:
                pair_name = pair["name"]
                if pair_name not in slaves_group:
                    raise RuntimeError(
                        "missing group '/{}/{}' in {}".format(
                            self.key_slaves_group, pair_name, self.selected_episode_path))
                pair_group = slaves_group[pair_name]
                if "positions" not in pair_group:
                    raise RuntimeError(
                        "missing dataset '/{}/{}/positions' in {}".format(
                            self.key_slaves_group, pair_name, self.selected_episode_path))

                pos = np.asarray(pair_group["positions"], dtype=np.float32)
                if pos.ndim == 1:
                    pos = pos.reshape(1, -1)
                if pos.ndim != 2:
                    raise RuntimeError(
                        "invalid shape for '/{}/{}/positions': {}".format(
                            self.key_slaves_group, pair_name, pos.shape))
                replay_pos[pair_name] = pos
                lengths.append(int(pos.shape[0]))

        common_len = min(lengths)
        if common_len <= 0:
            raise RuntimeError("no usable frames in {}".format(self.selected_episode_path))

        if len(set(lengths)) != 1:
            rospy.logwarn(
                "[LOAD] length mismatch timestamps/positions=%s, trim to common_len=%d",
                lengths, common_len)

        self.timestamps = timestamps[:common_len]
        self.replay_pos = {}
        for pair_name, pos in replay_pos.items():
            trimmed = pos[:common_len]
            normalized = np.zeros((common_len, 7), dtype=np.float32)
            for i in range(common_len):
                normalized[i] = np.asarray(fix_len(trimmed[i], 7, 0.0), dtype=np.float32)
            self.replay_pos[pair_name] = normalized

        self.frame_count = common_len
        rospy.loginfo("[LOAD] frames=%d  pairs=%s", self.frame_count, list(self.replay_pos.keys()))

    # ── camera images ──────────────────────────────────────────────
    def _load_camera_images(self):
        """Load color images from HDF5 into self.camera_images (BGR, uint8)."""
        self.camera_images = {}
        try:
            with h5py.File(self.selected_episode_path, "r") as f:
                if self.key_cameras_group not in f:
                    rospy.loginfo("[GUI] no '%s' group in HDF5, camera view skipped",
                                  self.key_cameras_group)
                    return
                cams_group = f[self.key_cameras_group]
                for cam_name in sorted(cams_group.keys()):
                    if self.gui_show_cameras and cam_name not in self.gui_show_cameras:
                        continue
                    cam_grp = cams_group[cam_name]
                    if "color" not in cam_grp:
                        continue
                    imgs = np.asarray(cam_grp["color"])          # (N, H, W, 3) BGR uint8
                    n = min(imgs.shape[0], self.frame_count)
                    self.camera_images[cam_name] = imgs[:n]
                    rospy.loginfo("[GUI] camera '%s': %s", cam_name,
                                  self.camera_images[cam_name].shape)
        except Exception as exc:
            rospy.logwarn("[GUI] failed to load camera images: %s", exc)

    # ── GUI frame builders ────────────────────────────────────────
    def _build_replay_frame(self, frame_idx, paused):
        """Build the composite OpenCV frame shown during replay."""
        TW, TH = self.gui_thumb_w, self.gui_thumb_h

        cam_names = sorted(self.camera_images.keys())
        n_cams = len(cam_names)

        if n_cams > 0:
            thumbs = []
            for name in cam_names:
                imgs = self.camera_images[name]
                if frame_idx < imgs.shape[0]:
                    img = imgs[frame_idx]          # already BGR
                    thumb = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_LINEAR)
                else:
                    thumb = np.zeros((TH, TW, 3), dtype=np.uint8)
                # camera name label
                cv2.putText(thumb, name, (6, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 60), 1,
                            cv2.LINE_AA)
                thumbs.append(thumb)

            # arrange in rows of up to 3 columns
            COLS = min(n_cams, 3)
            while len(thumbs) % COLS != 0:
                thumbs.append(np.zeros((TH, TW, 3), dtype=np.uint8))
            rows = [np.hstack(thumbs[i * COLS:(i + 1) * COLS])
                    for i in range(len(thumbs) // COLS)]
            cam_panel = np.vstack(rows)
        else:
            # placeholder when no cameras
            cam_panel = np.full((TH, TW * 2, 3), 28, dtype=np.uint8)
            cv2.putText(cam_panel, "No camera images in this episode",
                        (18, TH // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

        panel_w = cam_panel.shape[1]

        # ── status bar ──
        SB_H = 110
        sb = np.full((SB_H, panel_w, 3), 33, dtype=np.uint8)

        # progress bar
        M = 14
        BAR_H = 15
        bar_w = panel_w - 2 * M
        progress = (frame_idx + 1) / max(self.frame_count, 1)
        filled = int(bar_w * progress)
        cv2.rectangle(sb, (M, M), (M + bar_w, M + BAR_H), (52, 52, 52), -1)
        cv2.rectangle(sb, (M, M), (M + filled, M + BAR_H), (50, 180, 70), -1)
        cv2.rectangle(sb, (M, M), (M + bar_w, M + BAR_H), (105, 105, 105), 1)

        # text lines
        ep_bn = os.path.basename(self.selected_episode_path)
        state_lbl = "PAUSED" if paused else "PLAYING"
        loop_lbl = "LOOP:ON" if self.loop else "LOOP:OFF"
        line1 = "{}   Frame {}/{}   Speed {:.1f}x   [{}]   {}".format(
            ep_bn, frame_idx + 1, self.frame_count,
            self.speed_scale, state_lbl, loop_lbl)
        line2 = "[SPACE] Pause/Resume   [L] Loop   [+/-] Speed   [Q/ESC] Quit"

        cv2.putText(sb, line1, (M, M + BAR_H + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 215, 215), 1)
        cv2.putText(sb, line2, (M, M + BAR_H + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (125, 125, 125), 1)

        # arm pair info
        pair_names = "  ".join(p["name"] for p in self.arm_pairs)
        cv2.putText(sb, "Arms: " + pair_names, (M, M + BAR_H + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 140, 190), 1)

        return np.vstack([cam_panel, sb])

    def _build_reset_frame(self):
        """Build an overlay shown while the arm is doing its slow reset."""
        if self.frame_count > 0:
            base = self._build_replay_frame(0, False)
        else:
            base = np.full((300, 640, 3), 33, dtype=np.uint8)

        overlay = base.copy()
        cv2.rectangle(overlay, (0, 0), (base.shape[1], base.shape[0]),
                      (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, base, 0.45, 0, base)

        cx, cy = base.shape[1] // 2, base.shape[0] // 2
        cv2.putText(base, "RESETTING TO START POSITION...",
                    (cx - 220, cy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 200, 50), 2, cv2.LINE_AA)
        cv2.putText(base, "Slowly moving arm to initial pose",
                    (cx - 175, cy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (175, 175, 175), 1)
        return base

    # ── keyboard handler ──────────────────────────────────────────
    def _handle_key(self, key):
        if key == -1:
            return
        kc = key & 0xFF
        if kc in (ord('q'), ord('Q'), 27):
            self.stop_requested = True
            rospy.loginfo("[GUI] stop requested by user")
        elif kc == ord(' '):
            self.paused = not self.paused
            rospy.loginfo("[GUI] paused=%s", self.paused)
        elif kc in (ord('l'), ord('L')):
            self.loop = not self.loop
            rospy.loginfo("[GUI] loop=%s", self.loop)
        elif kc in (ord('+'), ord('=')):
            self.speed_scale = min(self.speed_scale * 1.5, 10.0)
            rospy.loginfo("[GUI] speed_scale=%.2f", self.speed_scale)
        elif kc == ord('-'):
            self.speed_scale = max(self.speed_scale / 1.5, 0.1)
            rospy.loginfo("[GUI] speed_scale=%.2f", self.speed_scale)

    # ── core replay logic (unchanged) ────────────────────────────
    def _build_joint_msg(self, position7):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = fix_len(position7, 7, 0.0)
        msg.velocity = [self.joint_velocity_default] * 6 + [self.gripper_velocity]
        msg.effort = [self.joint_effort_default] * 6 + [self.gripper_effort]
        return msg

    def _publish_positions(self, pos_by_pair):
        for pair_name, pos in pos_by_pair.items():
            self.ctrl_pubs[pair_name].publish(self._build_joint_msg(pos))

    def _get_current_positions(self):
        current = {}
        for pair in self.arm_pairs:
            pair_name = pair["name"]
            st = self.slave_states.get(pair_name)
            if st is None:
                return None
            current[pair_name] = np.asarray(fix_len(st.get("pos", []), 7, 0.0), dtype=np.float32)
        return current

    def _max_error_to_target(self, target_by_pair):
        current = self._get_current_positions()
        if current is None:
            return float("inf")
        err = 0.0
        for pair_name, target in target_by_pair.items():
            err = max(err, float(np.max(np.abs(current[pair_name] - target))))
        return err

    def _run_slow_reset(self, target_by_pair):
        current = self._get_current_positions()
        if current is None:
            rospy.logerr("[RESET] current slave states unavailable")
            return False

        max_abs_delta = 0.0
        for pair_name, target in target_by_pair.items():
            delta = float(np.max(np.abs(target - current[pair_name])))
            max_abs_delta = max(max_abs_delta, delta)

        max_step = max(self.reset_max_joint_speed_rad_s / self.reset_rate_hz, 1e-6)
        steps = max(self.reset_min_steps, int(math.ceil(max_abs_delta / max_step)))
        timeout = max(
            self.reset_timeout_sec,
            (float(steps) / self.reset_rate_hz) + self.reset_settle_hold_sec + 1.0,
        )
        start_ts = time.time()
        rate = rospy.Rate(self.reset_rate_hz)

        rospy.loginfo(
            "[RESET] slow reset start delta=%.6f max_step=%.6f steps=%d timeout=%.2f",
            max_abs_delta, max_step, steps, timeout,
        )

        for i in range(1, steps + 1):
            if rospy.is_shutdown():
                return False
            alpha = float(i) / float(steps)
            cmd = {}
            for pair_name, target in target_by_pair.items():
                start = current[pair_name]
                cmd[pair_name] = start + alpha * (target - start)
            self._publish_positions(cmd)
            if (time.time() - start_ts) > timeout:
                rospy.logerr("[RESET] timeout during interpolation")
                return False
            rate.sleep()
            if self.gui_enabled:
                cv2.waitKey(1)   # keep window responsive

        settle_end = time.time() + self.reset_settle_hold_sec
        while not rospy.is_shutdown() and time.time() < settle_end:
            self._publish_positions(target_by_pair)
            if (time.time() - start_ts) > timeout:
                rospy.logerr("[RESET] timeout during settle hold")
                return False
            rate.sleep()
            if self.gui_enabled:
                cv2.waitKey(1)

        while not rospy.is_shutdown():
            self._publish_positions(target_by_pair)
            err = self._max_error_to_target(target_by_pair)
            if err < self.reset_err_thresh_rad:
                rospy.loginfo("[RESET] done err=%.6f", err)
                return True
            if (time.time() - start_ts) > timeout:
                rospy.logerr("[RESET] timeout waiting convergence err=%.6f", err)
                return False
            rate.sleep()
            if self.gui_enabled:
                cv2.waitKey(1)

        return False

    def _compute_dt(self, idx):
        if idx >= (self.frame_count - 1):
            return None
        next_t = float(self.timestamps[idx + 1])
        cur_t = float(self.timestamps[idx])
        raw = next_t - cur_t
        if not np.isfinite(raw) or raw <= 0.0:
            raw = 1.0 / self.fallback_rate_hz
        scaled = raw / self.speed_scale
        return clamp(scaled, self.min_dt_sec, self.max_dt_sec)

    # ── headless replay (no GUI) ──────────────────────────────────
    def _play_once(self):
        rospy.loginfo("[REPLAY] start frames=%d speed_scale=%.3f", self.frame_count, self.speed_scale)
        last_log = time.time()
        for idx in range(self.frame_count):
            if rospy.is_shutdown():
                return False
            cmd = {p["name"]: self.replay_pos[p["name"]][idx] for p in self.arm_pairs}
            self._publish_positions(cmd)

            now = time.time()
            if now - last_log >= 1.0:
                rospy.loginfo("[REPLAY] frame %d/%d", idx + 1, self.frame_count)
                last_log = now

            dt = self._compute_dt(idx)
            if dt is not None and dt > 0.0:
                rospy.sleep(dt)

        rospy.loginfo("[REPLAY] done")
        return True

    # ── GUI replay ────────────────────────────────────────────────
    def _play_once_with_gui(self):
        rospy.loginfo("[REPLAY] start frames=%d speed_scale=%.3f", self.frame_count, self.speed_scale)
        last_log = time.time()

        for idx in range(self.frame_count):
            if rospy.is_shutdown() or self.stop_requested:
                return False

            # ── pause loop ──
            while self.paused and not rospy.is_shutdown() and not self.stop_requested:
                cv2.imshow(self.gui_window_name, self._build_replay_frame(idx, True))
                self._handle_key(cv2.waitKey(80))
            if rospy.is_shutdown() or self.stop_requested:
                return False

            # ── publish joint positions ──
            cmd = {p["name"]: self.replay_pos[p["name"]][idx] for p in self.arm_pairs}
            self._publish_positions(cmd)

            # ── show camera + status frame ──
            cv2.imshow(self.gui_window_name, self._build_replay_frame(idx, False))

            # ── inter-frame sleep (GUI-friendly) ──
            dt = self._compute_dt(idx)
            if dt is not None and dt > 0.0:
                deadline = time.time() + dt
                while not rospy.is_shutdown() and not self.stop_requested:
                    remaining = deadline - time.time()
                    if remaining <= 0.0:
                        break
                    wait_ms = max(1, min(int(remaining * 1000), 50))
                    self._handle_key(cv2.waitKey(wait_ms))
                    # handle pause triggered during the sleep
                    if self.paused:
                        while self.paused and not rospy.is_shutdown() and not self.stop_requested:
                            cv2.imshow(self.gui_window_name, self._build_replay_frame(idx, True))
                            self._handle_key(cv2.waitKey(80))
                        deadline = time.time()   # resume immediately
            else:
                self._handle_key(cv2.waitKey(1))

            now = time.time()
            if now - last_log >= 1.0:
                rospy.loginfo("[REPLAY] frame %d/%d", idx + 1, self.frame_count)
                last_log = now

        rospy.loginfo("[REPLAY] done")
        return True

    # ── hold last ─────────────────────────────────────────────────
    def _hold_last(self):
        if self.hold_last_sec <= 0.0:
            return
        cmd = {p["name"]: self.replay_pos[p["name"]][-1] for p in self.arm_pairs}
        end_ts = time.time() + self.hold_last_sec
        rate = rospy.Rate(self.hold_hz)
        rospy.loginfo("[HOLD] keep last frame for %.2fs @ %.2fHz", self.hold_last_sec, self.hold_hz)
        while not rospy.is_shutdown() and time.time() < end_ts:
            if self.gui_enabled and self.stop_requested:
                break
            self._publish_positions(cmd)
            rate.sleep()
            if self.gui_enabled:
                cv2.waitKey(1)

    # ── main run ──────────────────────────────────────────────────
    def run(self):
        if not self._wait_slave_states_ready():
            return

        # ── GUI episode selector ──
        if self.gui_enabled and self.gui_show_selector:
            ep = self._show_episode_selector()
            if ep is None:
                rospy.loginfo("[GUI] user cancelled episode selection, exiting")
                cv2.destroyAllWindows()
                return
            # Override config selection with GUI choice
            self.episode_file = ep["path"]
            self.episode_id = ep["eid"]

        self._load_episode()

        if self.gui_enabled:
            rospy.loginfo("[GUI] loading camera images into memory...")
            self._load_camera_images()

        cycle = 0
        while not rospy.is_shutdown() and not self.stop_requested:
            cycle += 1
            rospy.loginfo("[RUN] cycle=%d", cycle)

            if self.reset_enabled:
                if self.gui_enabled:
                    cv2.imshow(self.gui_window_name, self._build_reset_frame())
                    cv2.waitKey(1)
                target = {p["name"]: self.replay_pos[p["name"]][0] for p in self.arm_pairs}
                ok = self._run_slow_reset(target)
                if not ok and self.reset_fail_policy == "abort_replay":
                    rospy.logfatal("[RUN] slow reset failed, abort replay")
                    break

            if self.gui_enabled:
                if not self._play_once_with_gui():
                    break
            else:
                if not self._play_once():
                    break

            self._hold_last()
            if not self.loop:
                break

        # ── show completion frame ──
        if self.gui_enabled and not rospy.is_shutdown() and self.frame_count > 0:
            done_frame = self._build_replay_frame(self.frame_count - 1, False)
            h, w = done_frame.shape[:2]
            cv2.rectangle(done_frame, (0, h // 2 - 30), (w, h // 2 + 30), (20, 20, 20), -1)
            cv2.putText(done_frame, "REPLAY COMPLETE — press any key to exit",
                        (w // 2 - 230, h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (80, 230, 100), 2, cv2.LINE_AA)
            cv2.imshow(self.gui_window_name, done_frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── cleanup ───────────────────────────────────────────────────
    def close(self):
        if self.launcher is not None:
            try:
                self.launcher.stop()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ─────────────────────────── entry point ───────────────────────────

def main():
    node = None
    try:
        node = TeleopRawReplay()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("[FATAL] %s", exc)
        raise
    finally:
        if node is not None:
            node.close()


if __name__ == "__main__":
    main()
