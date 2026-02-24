#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teleop raw BAG replay with safe slow reset + OpenCV GUI.

GUI features:
  - Episode selector: browse episode_XXX/episode*.bag in data_dir
  - Camera viewer:    shows recorded camera images frame-by-frame
  - Playback controls via keyboard (window must have focus):
      SPACE   pause / resume
      L       toggle loop
      + / =   speed up  (x1.5)
      -       slow down (x1/1.5)
      Q / ESC stop and quit
  - Episode selector navigation:
      W / Up   move selection up
      S / Down move selection down
      ENTER    start replay
      Q / ESC  quit

Usage:
  roslaunch piper_test teleop_raw_replay.launch
  roslaunch piper_test teleop_raw_replay.launch config:=/path/to/teleop_raw_replay.yaml
"""

import json
import math
import os
import re
import time

import cv2
import numpy as np
import rosbag
import roslaunch
import rospy
import yaml
from sensor_msgs.msg import JointState


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def unique_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def msg_stamp(msg, bag_t):
    header = getattr(msg, "header", None)
    if header is not None:
        st = getattr(header, "stamp", None)
        if st is not None:
            sec = float(st.to_sec())
            if sec > 0.0:
                return sec
    return float(bag_t.to_sec())


def infer_cam_key(topic):
    if "/color/image_raw" in topic:
        prefix = topic.split("/color/image_raw")[0]
        key = prefix.strip("/").split("/")[-1]
    else:
        key = topic.strip("/").split("/")[-1]
    for pfx in ("realsense_", "camera_", "cam_"):
        if key.startswith(pfx):
            key = key[len(pfx):]
            break
    return key


def decode_color_image(msg):
    # CompressedImage
    if hasattr(msg, "format") and hasattr(msg, "data"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # sensor_msgs/Image
    h = int(getattr(msg, "height", 0))
    w = int(getattr(msg, "width", 0))
    step = int(getattr(msg, "step", 0))
    if h <= 0 or w <= 0 or step <= 0:
        return None
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < h * step:
        return None
    img = data.reshape(h, step)
    enc = str(getattr(msg, "encoding", "")).lower()

    if enc == "bgr8":
        if img.shape[1] < w * 3:
            return None
        return img[:, : w * 3].reshape(h, w, 3)
    if enc == "rgb8":
        if img.shape[1] < w * 3:
            return None
        rgb = img[:, : w * 3].reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        if img.shape[1] < w * 4:
            return None
        bgra = img[:, : w * 4].reshape(h, w, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        if img.shape[1] < w * 4:
            return None
        rgba = img[:, : w * 4].reshape(h, w, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        if img.shape[1] < w:
            return None
        gray = img[:, :w].reshape(h, w)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if img.shape[1] >= w * 3:
        return img[:, : w * 3].reshape(h, w, 3)
    return None


class TeleopRawReplay:

    def __init__(self):
        rospy.init_node("teleop_raw_replay", anonymous=True)

        config_file = rospy.get_param("~config_file")
        with open(config_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        g = self.cfg.get("global", {})
        self.data_dir = str(g.get("data_dir", "/workspace/piper_master_slave_ws/data"))
        self.fallback_rate_hz = max(as_float(g.get("fallback_rate_hz", 30.0), 30.0), 1e-6)
        self.ctrl_startup_wait_sec = max(as_float(g.get("ctrl_startup_wait_sec", 2.0), 2.0), 0.0)
        self.ctrl_ready_timeout_sec = max(as_float(g.get("ctrl_ready_timeout_sec", 8.0), 8.0), 0.0)
        self.min_dt_sec = max(as_float(g.get("min_dt_sec", 0.001), 0.001), 0.0)
        self.max_dt_sec = max(as_float(g.get("max_dt_sec", 0.2), 0.2), self.min_dt_sec)

        r = self.cfg.get("replay", {})
        self.episode_file = str(r.get("episode_file", "")).strip()
        self.episode_id = as_int(r.get("episode_id", -1), -1)
        self.replay_session_name = str(r.get("session_name", "")).strip()
        self.speed_scale = max(as_float(r.get("speed_scale", 1.0), 1.0), 1e-6)
        self.loop = as_bool(r.get("loop", False), False)
        self.hold_last_sec = max(as_float(r.get("hold_last_sec", 1.0), 1.0), 0.0)
        self.hold_hz = max(as_float(r.get("hold_hz", 10.0), 10.0), 1e-6)

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

        bag_cfg = self.cfg.get("bag", {})
        self.bag_use_metadata_topics = as_bool(
            bag_cfg.get("use_metadata_topics", True), True
        )
        self.bag_slave_joint_state_topics = bag_cfg.get("slave_joint_state_topics", []) or []
        if not isinstance(self.bag_slave_joint_state_topics, list):
            self.bag_slave_joint_state_topics = []
        self.bag_pair_slave_joint_topic_map = (
            bag_cfg.get("pair_slave_joint_topic_map", {}) or {}
        )
        if not isinstance(self.bag_pair_slave_joint_topic_map, dict):
            self.bag_pair_slave_joint_topic_map = {}
        self.bag_camera_topics = bag_cfg.get("camera_topics", {}) or {}
        if not isinstance(self.bag_camera_topics, dict):
            self.bag_camera_topics = {}

        self.arm_pairs = self.cfg.get("arm_pairs", [])
        if not self.arm_pairs:
            raise RuntimeError("[CONFIG] arm_pairs is empty")
        for idx, pair in enumerate(self.arm_pairs):
            for required in ("name", "slave", "slave_can"):
                if required not in pair:
                    raise RuntimeError("[CONFIG] arm_pairs[{}] missing '{}'".format(idx, required))

        os.makedirs(self.data_dir, exist_ok=True)

        gui_cfg = self.cfg.get("gui", {})
        self.gui_enabled = as_bool(gui_cfg.get("enabled", True), True)
        self.gui_window_name = str(gui_cfg.get("window_name", "Teleop Raw Replay"))
        self.gui_show_selector = as_bool(gui_cfg.get("show_selector", True), True)
        self.gui_show_cameras = gui_cfg.get("show_cameras", []) or []
        self.gui_thumb_w = as_int(gui_cfg.get("thumb_width", 320), 320)
        self.gui_thumb_h = as_int(gui_cfg.get("thumb_height", 240), 240)

        self.launcher = None
        self.ctrl_pubs = {}
        self.slave_states = {}
        self.state_topics = {}

        self.selected_episode_path = ""
        self.selected_episode_id = -1
        self.selected_episode_label = ""
        self.selected_episode_metadata = {}
        self.frame_count = 0
        self.timestamps = None
        self.replay_pos = {}
        self.replay_joint_topic_map = {}
        self.camera_topic_map = {}

        self.camera_images = {}
        self.paused = False
        self.stop_requested = False

        self._launch_child_nodes()
        self._setup_ros()
        self._print_banner()

    # -----------------------------------------------------------------------
    # child nodes
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # ROS setup
    # -----------------------------------------------------------------------
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
        rospy.loginfo("Teleop Raw Replay (BAG only)")
        rospy.loginfo("  data_dir          = %s", self.data_dir)
        rospy.loginfo("  episode_file      = %s", self.episode_file if self.episode_file else "(auto)")
        rospy.loginfo("  episode_id        = %d", self.episode_id)
        rospy.loginfo("  session_name      = %s", self.replay_session_name if self.replay_session_name else "(all)")
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

    # -----------------------------------------------------------------------
    # wait slave states
    # -----------------------------------------------------------------------
    def _wait_slave_states_ready(self):
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
        w, h = 640, 220
        cv2.namedWindow(self.gui_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.gui_window_name, w, h)

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

            canvas = np.full((h, w, 3), 30, dtype=np.uint8)
            cv2.rectangle(canvas, (0, 0), (w, 50), (40, 40, 60), -1)
            cv2.putText(
                canvas, self.gui_window_name, (15, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 255), 1
            )

            dots = "." * (dot_count % 4)
            msg = "Waiting for arm controllers" + dots
            cv2.putText(
                canvas, msg, (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 100), 1
            )
            cv2.putText(
                canvas, "Missing: " + ", ".join(missing), (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 100, 100), 1
            )
            remain = max(0, int(self.ctrl_ready_timeout_sec - elapsed))
            cv2.putText(
                canvas, "Timeout in {}s  |  [Q] Quit".format(remain), (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1
            )

            cv2.imshow(self.gui_window_name, canvas)
            key = cv2.waitKey(200) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return False
            dot_count += 1
        return False

    # -----------------------------------------------------------------------
    # episode discovery + metadata
    # -----------------------------------------------------------------------
    def _find_bag_in_episode_dir(self, ep_dir):
        if not os.path.isdir(ep_dir):
            return None
        bag_names = []
        try:
            for name in os.listdir(ep_dir):
                if re.match(r"^episode.*\.bag$", name):
                    bag_names.append(name)
        except OSError:
            return None
        if not bag_names:
            return None
        bag_names.sort()
        return os.path.join(ep_dir, bag_names[0])

    def _load_episode_metadata(self, bag_path):
        meta_path = os.path.join(os.path.dirname(os.path.abspath(bag_path)), "metadata.json")
        if not os.path.isfile(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    return obj
        except Exception as exc:
            rospy.logwarn("[BAG] failed to read metadata %s: %s", meta_path, exc)
        return {}

    def _topic_message_count(self, topic_info, topic):
        info = topic_info.get(topic)
        if info is None:
            return 0
        msg_count = getattr(info, "message_count", None)
        if msg_count is None and isinstance(info, tuple) and len(info) >= 2:
            msg_count = info[1]
        if msg_count is None:
            return 0
        try:
            return int(msg_count)
        except Exception:
            return 0

    def _scan_all_episodes(self):
        episodes = []
        data_dir_abs = os.path.abspath(self.data_dir)
        try:
            for root, dirs, _files in os.walk(self.data_dir):
                base = os.path.basename(root)
                m = re.match(r"^episode_(\d+)$", base)
                if not m:
                    continue

                parent = os.path.abspath(os.path.dirname(root))
                session_name = "" if parent == data_dir_abs else os.path.basename(parent)
                if self.replay_session_name and session_name != self.replay_session_name:
                    dirs[:] = []
                    continue

                bag_path = self._find_bag_in_episode_dir(root)
                if not bag_path:
                    dirs[:] = []
                    continue

                try:
                    mtime = float(os.path.getmtime(bag_path))
                except OSError:
                    mtime = 0.0

                if session_name:
                    display = "{}/{}".format(session_name, base)
                else:
                    display = base

                episodes.append(
                    {
                        "eid": int(m.group(1)),
                        "path": bag_path,
                        "session": session_name,
                        "display": display,
                        "mtime": mtime,
                    }
                )
                dirs[:] = []
        except OSError:
            pass

        episodes.sort(key=lambda x: (x["mtime"], x["eid"], x["display"]))
        return episodes

    # -----------------------------------------------------------------------
    # topic mapping
    # -----------------------------------------------------------------------
    def _infer_slave_joint_topics(self, bag_topics, metadata):
        bag_set = set(bag_topics)
        candidates = []

        for item in self.bag_slave_joint_state_topics:
            topic = str(item).strip()
            if topic:
                candidates.append(topic)

        if self.bag_use_metadata_topics and metadata:
            rec = (((metadata.get("topics") or {}).get("recorded")) or [])
            for topic in rec:
                t = str(topic).strip()
                if not t.endswith("/joint_states_single"):
                    continue
                if "/teleop/" in t.lower():
                    continue
                candidates.append(t)

        for topic in bag_topics:
            if topic.endswith("/joint_states_single") and "/teleop/" not in topic.lower():
                candidates.append(topic)

        if len(candidates) < len(self.arm_pairs):
            for topic in bag_topics:
                if topic.endswith("/joint_states_single"):
                    candidates.append(topic)

        merged = unique_keep_order(candidates)
        return [t for t in merged if t in bag_set]

    def _resolve_pair_source_topics(self, bag_topics, metadata, strict=True):
        bag_set = set(bag_topics)
        pair_names = [p["name"] for p in self.arm_pairs]

        per_pair_explicit = {}
        all_per_pair_provided = True
        for p in self.arm_pairs:
            topic = str(
                p.get("replay_slave_joint_topic", p.get("replay_topic", ""))
            ).strip()
            if not topic:
                all_per_pair_provided = False
                break
            per_pair_explicit[p["name"]] = topic
        if all_per_pair_provided:
            missing = [t for t in per_pair_explicit.values() if t not in bag_set]
            if missing and strict:
                raise RuntimeError(
                    "configured arm_pairs[*].replay_slave_joint_topic missing in bag: {}".format(
                        missing
                    )
                )
            if not missing:
                return per_pair_explicit
            return {}

        if self.bag_pair_slave_joint_topic_map:
            mapping = {}
            missing_cfg = []
            missing_bag = []
            for pn in pair_names:
                topic = str(self.bag_pair_slave_joint_topic_map.get(pn, "")).strip()
                if not topic:
                    missing_cfg.append(pn)
                    continue
                mapping[pn] = topic
                if topic not in bag_set:
                    missing_bag.append(topic)
            if missing_cfg and strict:
                raise RuntimeError(
                    "bag.pair_slave_joint_topic_map missing pair keys: {}".format(missing_cfg)
                )
            if missing_bag and strict:
                raise RuntimeError(
                    "bag.pair_slave_joint_topic_map topics missing in bag: {}".format(missing_bag)
                )
            if not missing_cfg and not missing_bag:
                return mapping
            if strict:
                return {}

        inferred = self._infer_slave_joint_topics(bag_topics, metadata)
        if len(inferred) < len(pair_names):
            if strict:
                raise RuntimeError(
                    "not enough slave joint topics in bag. need={} got={} candidates={}".format(
                        len(pair_names), len(inferred), inferred
                    )
                )
            return {}

        mapping = {}
        for idx, pn in enumerate(pair_names):
            mapping[pn] = inferred[idx]
        return mapping

    def _resolve_camera_topics(self, bag_topics, metadata, strict=False):
        bag_set = set(bag_topics)
        show_only = [str(x).strip() for x in self.gui_show_cameras if str(x).strip()]

        if self.bag_camera_topics:
            out = {}
            missing = []
            for cam_name, topic in self.bag_camera_topics.items():
                name = str(cam_name).strip()
                t = str(topic).strip()
                if not name or not t:
                    continue
                if show_only and name not in show_only:
                    continue
                if t not in bag_set:
                    missing.append("{}:{}".format(name, t))
                    continue
                out[name] = t
            if missing and strict:
                raise RuntimeError("bag.camera_topics missing in bag: {}".format(missing))
            if out:
                return out
            if strict and show_only:
                raise RuntimeError("bag.camera_topics has no usable camera topics")

        candidates = []
        if self.bag_use_metadata_topics and metadata:
            rec = (((metadata.get("topics") or {}).get("recorded")) or [])
            for topic in rec:
                t = str(topic).strip()
                if "/color/image_raw" not in t:
                    continue
                if t.endswith("/compressed") or t.endswith("/image_raw"):
                    candidates.append(t)
        for topic in bag_topics:
            if "/color/image_raw" in topic and (
                topic.endswith("/compressed") or topic.endswith("/image_raw")
            ):
                candidates.append(topic)

        candidates = [t for t in unique_keep_order(candidates) if t in bag_set]
        if not candidates:
            return {}

        by_key = {}
        for t in sorted(candidates):
            key = infer_cam_key(t)
            old = by_key.get(key)
            if old is None:
                by_key[key] = t
            else:
                old_comp = old.endswith("/compressed")
                new_comp = t.endswith("/compressed")
                if new_comp and not old_comp:
                    by_key[key] = t

        preferred = list(show_only)
        if not preferred and metadata:
            cams = (((metadata.get("recording") or {}).get("cameras")) or [])
            preferred = [str(c).strip() for c in cams if str(c).strip()]

        if preferred:
            out = {}
            used = set()
            for name in preferred:
                lname = name.lower()
                hit = None
                for key in sorted(by_key.keys()):
                    lk = key.lower()
                    if lk == lname or lk.endswith("_" + lname):
                        hit = key
                        break
                if hit is not None:
                    used.add(hit)
                    out[name] = by_key[hit]
            if out:
                return out

        return {k: by_key[k] for k in sorted(by_key.keys())}

    # -----------------------------------------------------------------------
    # selector GUI
    # -----------------------------------------------------------------------
    def _get_episode_meta(self, bag_path):
        metadata = self._load_episode_metadata(bag_path)
        try:
            with rosbag.Bag(bag_path, "r") as bag:
                topic_info = bag.get_type_and_topic_info().topics
                bag_topics = sorted(topic_info.keys())
        except Exception:
            return 0, []

        pair_map = self._resolve_pair_source_topics(bag_topics, metadata, strict=False)
        n_frames = 0
        if pair_map:
            anchor_pair = self.arm_pairs[0]["name"]
            anchor_topic = pair_map.get(anchor_pair)
            if anchor_topic:
                n_frames = self._topic_message_count(topic_info, anchor_topic)

        cam_map = self._resolve_camera_topics(bag_topics, metadata, strict=False)
        return n_frames, sorted(cam_map.keys())

    def _show_episode_selector(self):
        all_eps = self._scan_all_episodes()
        if not all_eps:
            rospy.logerr("[GUI] no episode_*/episode*.bag found in %s", self.data_dir)
            return None

        ep_list = []
        for ep in reversed(all_eps):
            n_frames, cam_names = self._get_episode_meta(ep["path"])
            item = dict(ep)
            item["n_frames"] = n_frames
            item["cams"] = cam_names
            ep_list.append(item)

        w, h = 820, 560
        cv2.namedWindow(self.gui_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.gui_window_name, w, h)

        sel = 0
        item_h = 60
        list_y = 60
        footer_h = 64

        while not rospy.is_shutdown():
            canvas = np.full((h, w, 3), 28, dtype=np.uint8)

            cv2.rectangle(canvas, (0, 0), (w, list_y - 4), (40, 42, 65), -1)
            cv2.putText(
                canvas, "SELECT BAG EPISODE TO REPLAY", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 255), 1
            )
            cv2.line(canvas, (0, list_y - 4), (w, list_y - 4), (70, 70, 100), 1)

            n_visible = (h - list_y - footer_h) // item_h
            scroll = max(0, sel - n_visible // 2)
            scroll = max(0, min(scroll, max(0, len(ep_list) - n_visible)))

            for disp_i, ep_i in enumerate(range(scroll, min(scroll + n_visible, len(ep_list)))):
                ep = ep_list[ep_i]
                y = list_y + disp_i * item_h
                is_sel = ep_i == sel

                bg_col = (42, 72, 118) if is_sel else (38, 38, 38)
                border_col = (100, 160, 245) if is_sel else (52, 52, 52)
                cv2.rectangle(canvas, (8, y + 2), (w - 8, y + item_h - 2), bg_col, -1)
                cv2.rectangle(canvas, (8, y + 2), (w - 8, y + item_h - 2), border_col, 1)

                arrow = "> " if is_sel else "  "
                title_col = (255, 255, 255) if is_sel else (175, 175, 175)
                title = "{}{}  (id={:03d})".format(arrow, ep["display"], ep["eid"])
                cv2.putText(
                    canvas, title, (22, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, title_col, 1
                )

                cam_str = ", ".join(ep["cams"]) if ep["cams"] else "no cameras"
                info = "{} frames  |  cameras: {}".format(ep["n_frames"], cam_str)
                info_col = (110, 200, 120) if is_sel else (90, 130, 90)
                cv2.putText(
                    canvas, info, (35, y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, info_col, 1
                )

            if len(ep_list) > n_visible:
                track_h = h - list_y - footer_h
                thumb_h = max(20, track_h * n_visible // len(ep_list))
                thumb_y = list_y + track_h * scroll // len(ep_list)
                cv2.rectangle(canvas, (w - 6, list_y), (w - 2, list_y + track_h), (50, 50, 50), -1)
                cv2.rectangle(canvas, (w - 6, thumb_y), (w - 2, thumb_y + thumb_h), (120, 120, 180), -1)

            cv2.rectangle(canvas, (0, h - footer_h), (w, h), (38, 38, 38), -1)
            cv2.line(canvas, (0, h - footer_h), (w, h - footer_h), (65, 65, 65), 1)
            cv2.putText(
                canvas, "[W/S] Navigate    [ENTER] Start Replay    [Q/ESC] Quit",
                (16, h - footer_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (155, 155, 155), 1
            )
            cv2.putText(
                canvas, "{} episode(s)  in  {}".format(len(ep_list), self.data_dir),
                (16, h - footer_h + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1
            )

            cv2.imshow(self.gui_window_name, canvas)
            key = cv2.waitKey(50)
            if key == -1:
                continue
            kc = key & 0xFF

            if kc in (ord("q"), ord("Q"), 27):
                return None
            if kc in (ord("w"), ord("W")) or key == 82:
                sel = max(0, sel - 1)
                continue
            if kc in (ord("s"), ord("S")) or key == 84:
                sel = min(len(ep_list) - 1, sel + 1)
                continue
            if kc == 13:
                return ep_list[sel]

        return None

    # -----------------------------------------------------------------------
    # load episode from bag
    # -----------------------------------------------------------------------
    def _resolve_episode(self):
        if self.episode_file:
            candidate = os.path.expanduser(self.episode_file)
            if not os.path.isabs(candidate):
                candidate = os.path.join(self.data_dir, candidate)
            candidate = os.path.abspath(candidate)

            if os.path.isdir(candidate):
                bag_path = self._find_bag_in_episode_dir(candidate)
                if not bag_path:
                    raise RuntimeError(
                        "episode_file dir has no episode*.bag: {}".format(candidate)
                    )
                candidate = bag_path
            elif not os.path.isfile(candidate):
                raise RuntimeError("episode_file not found: {}".format(candidate))

            if not candidate.endswith(".bag"):
                raise RuntimeError("episode_file must be .bag file or episode dir: {}".format(candidate))

            ep_dir = os.path.basename(os.path.dirname(candidate))
            ep_match = re.match(r"^episode_(\d+)$", ep_dir)
            ep_id = int(ep_match.group(1)) if ep_match else -1
            parent = os.path.abspath(os.path.dirname(os.path.dirname(candidate)))
            session_name = ""
            if parent == os.path.abspath(self.data_dir):
                session_name = ""
            else:
                session_name = os.path.basename(os.path.dirname(candidate))
            display = ep_dir if not session_name else "{}/{}".format(session_name, ep_dir)

            self.selected_episode_path = candidate
            self.selected_episode_id = ep_id
            self.selected_episode_label = display
            return

        all_eps = self._scan_all_episodes()
        if not all_eps:
            raise RuntimeError("no episode_*/episode*.bag found in {}".format(self.data_dir))

        if self.episode_id < 0:
            sel = all_eps[-1]
        else:
            hits = [ep for ep in all_eps if ep["eid"] == self.episode_id]
            if not hits:
                ids = sorted(set(ep["eid"] for ep in all_eps))
                raise RuntimeError(
                    "episode id {} not found in {} (available ids: {})".format(
                        self.episode_id, self.data_dir, ids
                    )
                )
            sel = hits[-1]

        self.selected_episode_path = sel["path"]
        self.selected_episode_id = sel["eid"]
        self.selected_episode_label = sel["display"]

    def _load_episode(self):
        self._resolve_episode()
        rospy.loginfo(
            "[LOAD] episode id=%s path=%s",
            self.selected_episode_id if self.selected_episode_id >= 0 else "unknown",
            self.selected_episode_path,
        )

        self.selected_episode_metadata = self._load_episode_metadata(self.selected_episode_path)

        with rosbag.Bag(self.selected_episode_path, "r") as bag:
            topic_info = bag.get_type_and_topic_info().topics
            bag_topics = sorted(topic_info.keys())
            pair_topic_map = self._resolve_pair_source_topics(
                bag_topics, self.selected_episode_metadata, strict=True
            )
            topic_to_pair = {topic: name for name, topic in pair_topic_map.items()}
            needed_topics = sorted(topic_to_pair.keys())

            rospy.loginfo("[LOAD] replay topics: %s", pair_topic_map)
            for topic in needed_topics:
                cnt = self._topic_message_count(topic_info, topic)
                rospy.loginfo("[LOAD]   %s  msgs=%d", topic, cnt)

            pair_ts = {p["name"]: [] for p in self.arm_pairs}
            pair_pos = {p["name"]: [] for p in self.arm_pairs}

            for topic, msg, bag_t in bag.read_messages(topics=needed_topics):
                pair_name = topic_to_pair.get(topic)
                if pair_name is None:
                    continue
                pair_ts[pair_name].append(msg_stamp(msg, bag_t))
                pair_pos[pair_name].append(
                    np.asarray(
                        fix_len(getattr(msg, "position", []), 7, 0.0),
                        dtype=np.float32,
                    )
                )

        for p in self.arm_pairs:
            name = p["name"]
            if not pair_ts[name]:
                raise RuntimeError(
                    "bag has no messages for replay pair {} (topic {})".format(
                        name, pair_topic_map.get(name)
                    )
                )

        anchor_pair = self.arm_pairs[0]["name"]
        anchor_ts = np.asarray(pair_ts[anchor_pair], dtype=np.float64).reshape(-1)
        if anchor_ts.size <= 0:
            raise RuntimeError("anchor pair '{}' has zero frames".format(anchor_pair))

        replay_pos = {}
        for p in self.arm_pairs:
            name = p["name"]
            ts_arr = np.asarray(pair_ts[name], dtype=np.float64).reshape(-1)
            pos_arr = np.asarray(pair_pos[name], dtype=np.float32)
            if pos_arr.ndim == 1:
                pos_arr = pos_arr.reshape(1, -1)

            j = 0
            out = np.zeros((anchor_ts.shape[0], 7), dtype=np.float32)
            for i, ft in enumerate(anchor_ts):
                while (j + 1) < ts_arr.shape[0] and ts_arr[j + 1] <= ft:
                    j += 1
                out[i] = np.asarray(fix_len(pos_arr[j], 7, 0.0), dtype=np.float32)
            replay_pos[name] = out

        self.timestamps = anchor_ts
        self.replay_pos = replay_pos
        self.replay_joint_topic_map = pair_topic_map
        self.frame_count = int(anchor_ts.shape[0])
        rospy.loginfo("[LOAD] frames=%d pairs=%s", self.frame_count, list(self.replay_pos.keys()))

    # -----------------------------------------------------------------------
    # load camera frames from bag
    # -----------------------------------------------------------------------
    def _load_camera_images(self):
        self.camera_images = {}
        self.camera_topic_map = {}

        try:
            with rosbag.Bag(self.selected_episode_path, "r") as bag:
                bag_topics = sorted(bag.get_type_and_topic_info().topics.keys())
                cam_map = self._resolve_camera_topics(
                    bag_topics, self.selected_episode_metadata, strict=False
                )
                if not cam_map:
                    rospy.loginfo("[GUI] no camera topics found in bag, camera view skipped")
                    return

                topic_to_cam = {topic: name for name, topic in cam_map.items()}
                cam_topics = sorted(topic_to_cam.keys())
                self.camera_topic_map = cam_map
                rospy.loginfo("[GUI] camera topics: %s", cam_map)

                cam_ts = {name: [] for name in cam_map.keys()}
                for topic, msg, bag_t in bag.read_messages(topics=cam_topics):
                    cam_name = topic_to_cam.get(topic)
                    if cam_name is None:
                        continue
                    cam_ts[cam_name].append(msg_stamp(msg, bag_t))

            valid_cams = {}
            for cam_name, ts_list in cam_ts.items():
                if ts_list:
                    valid_cams[cam_name] = ts_list
                else:
                    rospy.logwarn("[GUI] camera '%s' has zero messages, skip", cam_name)

            if not valid_cams:
                rospy.loginfo("[GUI] no usable camera streams")
                return

            cam_need_idx_to_frames = {}
            for cam_name, ts_list in valid_cams.items():
                j = 0
                need = {}
                for fi, ft in enumerate(self.timestamps):
                    while (j + 1) < len(ts_list) and ts_list[j + 1] <= ft:
                        j += 1
                    need.setdefault(j, []).append(fi)
                cam_need_idx_to_frames[cam_name] = need

            cam_frames = {name: [None] * self.frame_count for name in valid_cams.keys()}
            cam_seen_idx = {name: 0 for name in valid_cams.keys()}
            decode_fail = {name: 0 for name in valid_cams.keys()}
            cam_topics = sorted(
                [topic for topic, name in topic_to_cam.items() if name in valid_cams]
            )

            with rosbag.Bag(self.selected_episode_path, "r") as bag:
                for topic, msg, _bag_t in bag.read_messages(topics=cam_topics):
                    cam_name = topic_to_cam.get(topic)
                    if cam_name not in valid_cams:
                        continue
                    src_idx = cam_seen_idx[cam_name]
                    cam_seen_idx[cam_name] += 1
                    frame_ids = cam_need_idx_to_frames[cam_name].get(src_idx)
                    if not frame_ids:
                        continue

                    img = decode_color_image(msg)
                    if img is None:
                        decode_fail[cam_name] += 1
                        continue
                    for fi in frame_ids:
                        cam_frames[cam_name][fi] = img.copy()

            for cam_name, fail_n in decode_fail.items():
                if fail_n > 0:
                    rospy.logwarn("[GUI] camera '%s' decode failures=%d", cam_name, fail_n)

            for cam_name, frames in cam_frames.items():
                first_valid = None
                for img in frames:
                    if img is not None:
                        first_valid = img
                        break
                if first_valid is None:
                    rospy.logwarn("[GUI] camera '%s' has no decodable frames", cam_name)
                    continue

                if first_valid.ndim == 2:
                    first_valid = cv2.cvtColor(first_valid, cv2.COLOR_GRAY2BGR)
                if first_valid.ndim == 3 and first_valid.shape[2] == 4:
                    first_valid = cv2.cvtColor(first_valid, cv2.COLOR_BGRA2BGR)

                h0, w0 = first_valid.shape[:2]
                last = first_valid
                for i in range(self.frame_count):
                    img = frames[i]
                    if img is None:
                        frames[i] = last.copy()
                        continue
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.ndim == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    if img.shape[0] != h0 or img.shape[1] != w0:
                        img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    frames[i] = img
                    last = img

                arr = np.stack(frames, axis=0).astype(np.uint8)
                self.camera_images[cam_name] = arr
                rospy.loginfo("[GUI] camera '%s': %s", cam_name, arr.shape)
        except Exception as exc:
            rospy.logwarn("[GUI] failed to load camera images from bag: %s", exc)

    # -----------------------------------------------------------------------
    # GUI frame builders
    # -----------------------------------------------------------------------
    def _build_replay_frame(self, frame_idx, paused):
        tw, th = self.gui_thumb_w, self.gui_thumb_h

        cam_names = sorted(self.camera_images.keys())
        n_cams = len(cam_names)

        if n_cams > 0:
            thumbs = []
            for name in cam_names:
                imgs = self.camera_images[name]
                if frame_idx < imgs.shape[0]:
                    img = imgs[frame_idx]
                    thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                else:
                    thumb = np.zeros((th, tw, 3), dtype=np.uint8)
                cv2.putText(
                    thumb, name, (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 60), 1, cv2.LINE_AA
                )
                thumbs.append(thumb)

            cols = min(n_cams, 3)
            while len(thumbs) % cols != 0:
                thumbs.append(np.zeros((th, tw, 3), dtype=np.uint8))
            rows = [np.hstack(thumbs[i * cols:(i + 1) * cols]) for i in range(len(thumbs) // cols)]
            cam_panel = np.vstack(rows)
        else:
            cam_panel = np.full((th, tw * 2, 3), 28, dtype=np.uint8)
            cv2.putText(
                cam_panel, "No camera images in this episode", (18, th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1
            )

        panel_w = cam_panel.shape[1]

        sb_h = 110
        sb = np.full((sb_h, panel_w, 3), 33, dtype=np.uint8)

        m = 14
        bar_h = 15
        bar_w = panel_w - 2 * m
        progress = (frame_idx + 1) / max(self.frame_count, 1)
        filled = int(bar_w * progress)
        cv2.rectangle(sb, (m, m), (m + bar_w, m + bar_h), (52, 52, 52), -1)
        cv2.rectangle(sb, (m, m), (m + filled, m + bar_h), (50, 180, 70), -1)
        cv2.rectangle(sb, (m, m), (m + bar_w, m + bar_h), (105, 105, 105), 1)

        ep_label = self.selected_episode_label or os.path.basename(self.selected_episode_path)
        state_lbl = "PAUSED" if paused else "PLAYING"
        loop_lbl = "LOOP:ON" if self.loop else "LOOP:OFF"
        line1 = "{}   Frame {}/{}   Speed {:.1f}x   [{}]   {}".format(
            ep_label, frame_idx + 1, self.frame_count, self.speed_scale, state_lbl, loop_lbl
        )
        line2 = "[SPACE] Pause/Resume   [L] Loop   [+/-] Speed   [Q/ESC] Quit"

        cv2.putText(
            sb, line1, (m, m + bar_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 215, 215), 1
        )
        cv2.putText(
            sb, line2, (m, m + bar_h + 46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (125, 125, 125), 1
        )

        pair_names = "  ".join(p["name"] for p in self.arm_pairs)
        cv2.putText(
            sb, "Arms: " + pair_names, (m, m + bar_h + 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 140, 190), 1
        )

        return np.vstack([cam_panel, sb])

    def _build_reset_frame(self):
        if self.frame_count > 0:
            base = self._build_replay_frame(0, False)
        else:
            base = np.full((300, 640, 3), 33, dtype=np.uint8)

        overlay = base.copy()
        cv2.rectangle(overlay, (0, 0), (base.shape[1], base.shape[0]), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, base, 0.45, 0, base)

        cx, cy = base.shape[1] // 2, base.shape[0] // 2
        cv2.putText(
            base, "RESETTING TO START POSITION...", (cx - 220, cy - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 200, 50), 2, cv2.LINE_AA
        )
        cv2.putText(
            base, "Slowly moving arm to initial pose", (cx - 175, cy + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (175, 175, 175), 1
        )
        return base

    # -----------------------------------------------------------------------
    # keyboard + replay core
    # -----------------------------------------------------------------------
    def _handle_key(self, key):
        if key == -1:
            return
        kc = key & 0xFF
        if kc in (ord("q"), ord("Q"), 27):
            self.stop_requested = True
            rospy.loginfo("[GUI] stop requested by user")
        elif kc == ord(" "):
            self.paused = not self.paused
            rospy.loginfo("[GUI] paused=%s", self.paused)
        elif kc in (ord("l"), ord("L")):
            self.loop = not self.loop
            rospy.loginfo("[GUI] loop=%s", self.loop)
        elif kc in (ord("+"), ord("=")):
            self.speed_scale = min(self.speed_scale * 1.5, 10.0)
            rospy.loginfo("[GUI] speed_scale=%.2f", self.speed_scale)
        elif kc == ord("-"):
            self.speed_scale = max(self.speed_scale / 1.5, 0.1)
            rospy.loginfo("[GUI] speed_scale=%.2f", self.speed_scale)

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
                cv2.waitKey(1)

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

    def _play_once_with_gui(self):
        rospy.loginfo("[REPLAY] start frames=%d speed_scale=%.3f", self.frame_count, self.speed_scale)
        last_log = time.time()

        for idx in range(self.frame_count):
            if rospy.is_shutdown() or self.stop_requested:
                return False

            while self.paused and not rospy.is_shutdown() and not self.stop_requested:
                cv2.imshow(self.gui_window_name, self._build_replay_frame(idx, True))
                self._handle_key(cv2.waitKey(80))
            if rospy.is_shutdown() or self.stop_requested:
                return False

            cmd = {p["name"]: self.replay_pos[p["name"]][idx] for p in self.arm_pairs}
            self._publish_positions(cmd)

            cv2.imshow(self.gui_window_name, self._build_replay_frame(idx, False))

            dt = self._compute_dt(idx)
            if dt is not None and dt > 0.0:
                deadline = time.time() + dt
                while not rospy.is_shutdown() and not self.stop_requested:
                    remaining = deadline - time.time()
                    if remaining <= 0.0:
                        break
                    wait_ms = max(1, min(int(remaining * 1000), 50))
                    self._handle_key(cv2.waitKey(wait_ms))
                    if self.paused:
                        while self.paused and not rospy.is_shutdown() and not self.stop_requested:
                            cv2.imshow(self.gui_window_name, self._build_replay_frame(idx, True))
                            self._handle_key(cv2.waitKey(80))
                        deadline = time.time()
            else:
                self._handle_key(cv2.waitKey(1))

            now = time.time()
            if now - last_log >= 1.0:
                rospy.loginfo("[REPLAY] frame %d/%d", idx + 1, self.frame_count)
                last_log = now

        rospy.loginfo("[REPLAY] done")
        return True

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

    # -----------------------------------------------------------------------
    # main
    # -----------------------------------------------------------------------
    def run(self):
        if not self._wait_slave_states_ready():
            return

        if self.gui_enabled and self.gui_show_selector:
            ep = self._show_episode_selector()
            if ep is None:
                rospy.loginfo("[GUI] user cancelled episode selection, exiting")
                cv2.destroyAllWindows()
                return
            self.episode_file = ep["path"]
            self.episode_id = ep["eid"]
            self.selected_episode_label = ep["display"]

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

        if self.gui_enabled and not rospy.is_shutdown() and self.frame_count > 0:
            done_frame = self._build_replay_frame(self.frame_count - 1, False)
            h, w = done_frame.shape[:2]
            cv2.rectangle(done_frame, (0, h // 2 - 30), (w, h // 2 + 30), (20, 20, 20), -1)
            cv2.putText(
                done_frame, "REPLAY COMPLETE - press any key to exit",
                (w // 2 - 230, h // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (80, 230, 100), 2, cv2.LINE_AA
            )
            cv2.imshow(self.gui_window_name, done_frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

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
