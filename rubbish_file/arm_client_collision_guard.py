#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arm Client + Collision Guard (Docker ROS1)
- ZeroMQ 接收 gaze server 指令
- 对 slave3 目标位姿做与 slave1/slave2 的末端距离防碰撞检查
- 碰撞风险时阻止 move，并通过 PUSH 回传 blocked_collision
"""

import importlib
import json
import math
import sys
import time

import rospy
import yaml
import zmq
from std_msgs.msg import Bool, String

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def import_msg(msg_type_name):
    try:
        mod = importlib.import_module("piper_msgs.msg")
        cls = getattr(mod, msg_type_name)
        rospy.loginfo("[IMPORT] piper_msgs/%s", msg_type_name)
        return cls
    except (ImportError, AttributeError) as e:
        rospy.logfatal("[IMPORT] 无法导入 piper_msgs.msg.%s: %s", msg_type_name, e)
        sys.exit(1)

def pose_to_list(msg):
    try:
        return [msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw]
    except AttributeError:
        return None

class ArmClientCollisionGuard:

    def __init__(self):
        rospy.init_node("arm_client_collision_guard", anonymous=True)

        config_path = rospy.get_param("~config_path", "arm_client_slave3_collision.yaml")
        self.cfg = load_config(config_path)

        acfg = self.cfg["arm"]
        zcfg = self.cfg["zmq"]
        pcfg = self.cfg["pos_cmd"]
        ccfg = self.cfg.get("collision", {})

        PosCmdMsg = import_msg(acfg["pos_cmd_msg_type"])
        EndPoseMsg = import_msg(acfg["end_pose_msg_type"])
        self.PosCmdMsg = PosCmdMsg

        self.cmd_mode1 = pcfg.get("mode1", 1)
        self.cmd_mode2 = pcfg.get("mode2", 0)
        self.cmd_gripper = pcfg.get("gripper", 0.0)

        self.toggle_enable_on_estop = bool(self.cfg.get("toggle_enable_on_estop", True))
        self.heartbeat_timeout_action = self._normalize_timeout_action(
            self.cfg.get("heartbeat_timeout_action", "hold_no_disable")
        )
        self.estop_action = self._normalize_estop_action(
            self.cfg.get("estop_action", "hold_no_disable")
        )
        self.timeout_log_throttle_sec = float(self.cfg.get("timeout_log_throttle_sec", 1.0))
        self._last_timeout_log_time = 0.0
        self._timeout_hold_active = False
        self.auto_resume_on_move = bool(self.cfg.get("auto_resume_on_move", False))
        self.stream_target_enabled = bool(self.cfg.get("stream_target_enabled", False))
        self.stream_target_hz = float(self.cfg.get("stream_target_hz", 20.0))
        if self.stream_target_hz <= 0.0:
            rospy.logwarn(
                "[CFG] invalid stream_target_hz=%.3f, disable stream_target",
                self.stream_target_hz,
            )
            self.stream_target_enabled = False
            self._stream_target_period_sec = 0.0
        else:
            self._stream_target_period_sec = 1.0 / self.stream_target_hz

        self.move_burst_count = max(1, int(self.cfg.get("move_burst_count", 1)))
        self.move_burst_interval_sec = max(0.0, float(self.cfg.get("move_burst_interval_sec", 0.0)))
        self.move_drop_log_throttle_sec = max(
            0.0, float(self.cfg.get("move_drop_log_throttle_sec", 1.0))
        )
        self._last_move_drop_log_time = {}
        self._last_stream_publish_time = 0.0
        self._arrived_latched = False

        self.enable_pub = rospy.Publisher(acfg["enable_topic"], Bool, queue_size=1)
        self.ctrl_pub = rospy.Publisher(acfg["pos_cmd_topic"], PosCmdMsg, queue_size=1)

        self.current_pose = None
        rospy.Subscriber(acfg["end_pose_topic"], EndPoseMsg, self._pose_self_cb, queue_size=1)

        self.collision_enabled = bool(ccfg.get("enabled", True))
        self.min_distance_m = float(ccfg.get("min_distance_m", 0.12))
        self.warn_throttle_sec = float(ccfg.get("warn_throttle_sec", 1.0))
        self.compare_poses = {}
        self.compare_topics = {}
        self._last_warn_time = 0.0

        self.blocked_topic = ccfg.get("blocked_topic", "/slave3/collision_blocked")
        self.blocked_pub = rospy.Publisher(self.blocked_topic, String, queue_size=10)

        check_items = ccfg.get("check_against_slaves", [])
        for item in check_items:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                topic = str(item.get("end_pose_topic", "")).strip()
            else:
                name = str(item).strip()
                topic = "/{}/end_pose_euler".format(name)
            if not name or not topic:
                continue
            self.compare_poses[name] = None
            self.compare_topics[name] = topic
            rospy.Subscriber(topic, EndPoseMsg, self._pose_other_cb, callback_args=name, queue_size=1)

        self.ctx = zmq.Context()

        self.sub_sock = self.ctx.socket(zmq.SUB)
        self.sub_sock.connect("tcp://{}:{}".format(zcfg["server_ip"], zcfg["sub_port"]))
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_sock.setsockopt(zmq.RCVTIMEO, 0)
        rospy.loginfo("[ZMQ] SUB -> %s:%s", zcfg["server_ip"], zcfg["sub_port"])

        self.push_sock = self.ctx.socket(zmq.PUSH)
        self.push_sock.connect("tcp://{}:{}".format(zcfg["server_ip"], zcfg["push_port"]))
        rospy.loginfo("[ZMQ] PUSH -> %s:%s", zcfg["server_ip"], zcfg["push_port"])

        self.target_pose = None
        self.target_id = -1
        self.last_hb_time = time.time()
        self.estop = False

        self.tol_pos = float(self.cfg.get("arrive_tolerance_pos", 0.01))
        self.tol_ori = float(self.cfg.get("arrive_tolerance_ori", 0.05))
        self.hb_timeout = float(self.cfg.get("heartbeat_timeout_sec", 3.0))
        self.loop_rate_hz = float(self.cfg.get("loop_rate_hz", 20.0))

        self.safe_pose = self.cfg.get("safe_pose", {})
        self.reset_topic = str(self.cfg.get("reset_topic", "/slave3/reset_to_home")).strip()
        self.reset_respects_collision = bool(self.cfg.get("reset_respects_collision", True))
        self._reset_requested = False
        if self.reset_topic:
            rospy.Subscriber(self.reset_topic, Bool, self._reset_cb, queue_size=1)
            rospy.loginfo(
                "[RESET] topic=%s respects_collision=%s",
                self.reset_topic,
                self.reset_respects_collision,
            )

        rospy.sleep(1.0)
        rospy.loginfo("[READY] arm_client_collision_guard 启动完成")
        rospy.loginfo(
            "[COLLISION] enabled=%s min_distance=%.3fm check=%s",
            self.collision_enabled,
            self.min_distance_m,
            list(self.compare_topics.keys()),
        )
        rospy.loginfo(
            "[SAFETY] heartbeat_timeout_action=%s estop_action=%s toggle_enable_on_estop=%s",
            self.heartbeat_timeout_action,
            self.estop_action,
            self.toggle_enable_on_estop,
        )
        rospy.loginfo(
            "[CTRL] auto_resume_on_move=%s stream_target=%s stream_hz=%.1f "
            "move_burst=%d interval=%.3fs",
            self.auto_resume_on_move,
            self.stream_target_enabled,
            self.stream_target_hz,
            self.move_burst_count,
            self.move_burst_interval_sec,
        )

    def _normalize_timeout_action(self, action):
        s = str(action or "hold_no_disable").strip().lower()
        aliases = {
            "hold": "hold_no_disable",
            "hold_no_estop": "hold_no_disable",
            "estop": "estop_no_disable",
        }
        s = aliases.get(s, s)
        allowed = {"hold_no_disable", "estop_no_disable", "disable"}
        if s not in allowed:
            rospy.logwarn(
                "[CFG] invalid heartbeat_timeout_action=%s, fallback hold_no_disable",
                action,
            )
            return "hold_no_disable"
        return s

    def _normalize_estop_action(self, action):
        s = str(action or "hold_no_disable").strip().lower()
        aliases = {
            "hold": "hold_no_disable",
            "estop_no_disable": "hold_no_disable",
        }
        s = aliases.get(s, s)
        allowed = {"hold_no_disable", "disable"}
        if s not in allowed:
            rospy.logwarn("[CFG] invalid estop_action=%s, fallback hold_no_disable", action)
            return "hold_no_disable"
        return s

    def _pose_self_cb(self, msg):
        self.current_pose = pose_to_list(msg)

    def _pose_other_cb(self, msg, name):
        p = pose_to_list(msg)
        if p is not None:
            self.compare_poses[name] = p

    def _reset_cb(self, msg):
        if bool(msg.data):
            self._reset_requested = True

    def _publish_pos_cmd(self, x, y, z, roll, pitch, yaw):
        msg = self.PosCmdMsg()
        msg.x = x
        msg.y = y
        msg.z = z
        msg.roll = roll
        msg.pitch = pitch
        msg.yaw = yaw
        msg.gripper = self.cmd_gripper
        msg.mode1 = self.cmd_mode1
        msg.mode2 = self.cmd_mode2
        self.ctrl_pub.publish(msg)

    def _send_feedback(self, status, **extra):
        data = {"status": status, "target_id": self.target_id, "ts": time.time()}
        data.update(extra)
        try:
            self.push_sock.send_json(data, zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _set_enable(self, enabled):
        if not self.toggle_enable_on_estop:
            return
        msg = Bool(data=enabled)
        for _ in range(5):
            self.enable_pub.publish(msg)
            rospy.sleep(0.05)
        rospy.loginfo("[ARM] %s", "使能" if enabled else "去使能")

    def _timeout_warn(self, text):
        now = time.time()
        if now - self._last_timeout_log_time >= self.timeout_log_throttle_sec:
            self._last_timeout_log_time = now
            rospy.logwarn(text)

    def _warn_move_drop(self, reason, text):
        now = time.time()
        last = self._last_move_drop_log_time.get(reason, 0.0)
        if now - last >= self.move_drop_log_throttle_sec:
            self._last_move_drop_log_time[reason] = now
            rospy.logwarn(text)

    def _clear_target(self):
        self.target_pose = None
        self._arrived_latched = False

    def _publish_target_burst(self, target):
        self._publish_pos_cmd(*target)
        self._last_stream_publish_time = time.time()
        for _ in range(1, self.move_burst_count):
            if self.move_burst_interval_sec > 0.0:
                rospy.sleep(self.move_burst_interval_sec)
            self._publish_pos_cmd(*target)
            self._last_stream_publish_time = time.time()

    def _maybe_stream_target(self, now):
        if not self.stream_target_enabled:
            return
        if self.estop or self.target_pose is None:
            return
        if (now - self._last_stream_publish_time) < self._stream_target_period_sec:
            return
        self._publish_pos_cmd(*self.target_pose)
        self._last_stream_publish_time = now

    def _handle_timeout_action(self):
        self._clear_target()

        if self.heartbeat_timeout_action == "disable":
            self._timeout_warn("[TIMEOUT] 心跳超时, 进入急停并去使能")
            self._set_enable(False)
            self.estop = True
            if not self._timeout_hold_active:
                self._send_feedback("timeout_disable")
                self._timeout_hold_active = True
            return

        if self.heartbeat_timeout_action == "estop_no_disable":
            self._timeout_warn("[TIMEOUT] 心跳超时, 进入急停保持(不去使能)")
            self.estop = True
            if not self._timeout_hold_active:
                self._send_feedback("timeout_estop_hold")
                self._timeout_hold_active = True
            return

        self._timeout_warn("[TIMEOUT] 心跳超时, 仅保持当前位置(不去使能)")
        if not self._timeout_hold_active:
            self._send_feedback("timeout_hold")
            self._timeout_hold_active = True

    def _safe_pose_target(self):
        if not self.safe_pose:
            return None
        return [
            float(self.safe_pose.get("x", 0.25)),
            float(self.safe_pose.get("y", 0.0)),
            float(self.safe_pose.get("z", 0.35)),
            float(self.safe_pose.get("roll", 0.0)),
            float(self.safe_pose.get("pitch", 0.0)),
            float(self.safe_pose.get("yaw", 0.0)),
        ]

    def _handle_reset_request(self):
        self._reset_requested = False
        self.last_hb_time = time.time()
        self._timeout_hold_active = False

        if self.estop:
            rospy.logwarn("[RESET] 当前处于 estop，忽略复位请求")
            self._send_feedback("reset_blocked_estop")
            return

        target = self._safe_pose_target()
        if target is None:
            rospy.logwarn("[RESET] 未配置 safe_pose，无法复位")
            self._send_feedback("reset_no_safe_pose")
            return

        if self.reset_respects_collision:
            blocked, hit, dmin = self._calc_collision(target[:3])
            if blocked:
                rospy.logwarn(
                    "[RESET] 复位被防碰撞阻止: against=%s dmin=%.3f < %.3f",
                    hit,
                    dmin,
                    self.min_distance_m,
                )
                self._send_feedback(
                    "blocked_collision_reset",
                    blocked_by=hit,
                    distance=dmin,
                    threshold=self.min_distance_m,
                )
                return

        self.target_pose = target
        self._arrived_latched = False
        self._publish_target_burst(target)
        self._send_feedback("reset_sent")
        rospy.loginfo(
            "[RESET] 已发送复位位姿: (%.3f, %.3f, %.3f)",
            target[0],
            target[1],
            target[2],
        )

    def _calc_collision(self, target_xyz):
        if not self.collision_enabled:
            return False, None, None

        tx, ty, tz = target_xyz
        dmin = None
        hit = None

        for name, pose in self.compare_poses.items():
            if pose is None:
                continue
            dx = tx - float(pose[0])
            dy = ty - float(pose[1])
            dz = tz - float(pose[2])
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dmin is None or d < dmin:
                dmin = d
                hit = name

        if dmin is None:
            return False, None, None

        return dmin < self.min_distance_m, hit, dmin

    def _publish_blocked_event(self, payload):
        try:
            self.blocked_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        except Exception:
            pass

    def _warn_collision(self, text):
        now = time.time()
        if now - self._last_warn_time >= self.warn_throttle_sec:
            self._last_warn_time = now
            rospy.logwarn(text)

    def _handle_cmd(self, cmd):
        c = cmd.get("cmd")

        if c == "heartbeat":
            self.last_hb_time = time.time()
            self._timeout_hold_active = False
            return

        if c == "hold":
            self.last_hb_time = time.time()
            self._timeout_hold_active = False
            self._clear_target()
            return

        if c == "move":
            self.last_hb_time = time.time()
            self._timeout_hold_active = False
            try:
                new_id = int(cmd.get("id", -1))
                target = [
                    float(cmd["x"]),
                    float(cmd["y"]),
                    float(cmd["z"]),
                    float(cmd["roll"]),
                    float(cmd["pitch"]),
                    float(cmd["yaw"]),
                ]
            except (KeyError, TypeError, ValueError) as e:
                self._warn_move_drop("invalid_cmd", "[MOVE] 非法 payload, 忽略: {} cmd={}".format(e, cmd))
                self._send_feedback("invalid_cmd", reason="invalid_move_payload")
                return

            if self.estop:
                self.target_id = new_id
                if self.auto_resume_on_move:
                    self.estop = False
                    self._timeout_hold_active = False
                    if self.estop_action == "disable":
                        self._set_enable(True)
                    rospy.logwarn("[MOVE] estop中收到move，自动恢复执行 id=%d", new_id)
                    self._send_feedback("auto_resumed_on_move")
                else:
                    self._warn_move_drop(
                        "estop",
                        "[MOVE] estop中忽略 move id={} target=({:.3f},{:.3f},{:.3f})".format(
                            new_id, target[0], target[1], target[2]
                        ),
                    )
                    self._send_feedback("move_ignored_estop", reason="estop")
                    return

            blocked, hit, dmin = self._calc_collision(target[:3])
            if blocked:
                self.target_id = new_id
                self._clear_target()
                msg = (
                    "[COLLISION] blocked move id={} target=({:.3f},{:.3f},{:.3f}) "
                    "against={} dmin={:.3f}m < {:.3f}m"
                ).format(new_id, target[0], target[1], target[2], hit, dmin, self.min_distance_m)
                self._warn_collision(msg)

                event = {
                    "status": "blocked_collision",
                    "target_id": new_id,
                    "blocked_by": hit,
                    "distance": dmin,
                    "threshold": self.min_distance_m,
                    "target": {
                        "x": target[0],
                        "y": target[1],
                        "z": target[2],
                        "roll": target[3],
                        "pitch": target[4],
                        "yaw": target[5],
                    },
                    "ts": time.time(),
                }
                self._publish_blocked_event(event)
                self._send_feedback(
                    "blocked_collision",
                    blocked_by=hit,
                    distance=dmin,
                    threshold=self.min_distance_m,
                )
                return

            self.target_id = new_id
            self.target_pose = target
            self._arrived_latched = False
            self._publish_target_burst(self.target_pose)
            rospy.loginfo("[MOVE] -> 点%d (%.3f,%.3f,%.3f)", self.target_id, target[0], target[1], target[2])
            return

        if c == "estop":
            if not self.estop:
                self.estop = True
                self._clear_target()
                if self.estop_action == "disable":
                    self._set_enable(False)
                    rospy.logwarn("[ESTOP] 急停! 已去使能")
                    self._send_feedback("estop_disable")
                else:
                    rospy.logwarn("[ESTOP] 急停保持! 不去使能")
                    self._send_feedback("estop_hold")
            return

        if c == "resume":
            if self.estop:
                self.estop = False
                self._timeout_hold_active = False
                if self.estop_action == "disable":
                    self._set_enable(True)
                if self.safe_pose:
                    self._publish_pos_cmd(
                        float(self.safe_pose.get("x", 0.25)),
                        float(self.safe_pose.get("y", 0.0)),
                        float(self.safe_pose.get("z", 0.35)),
                        float(self.safe_pose.get("roll", 0.0)),
                        float(self.safe_pose.get("pitch", 0.0)),
                        float(self.safe_pose.get("yaw", 0.0)),
                    )
                rospy.loginfo("[RESUME] 已恢复")
            return

        if c == "shutdown":
            rospy.loginfo("[SHUTDOWN] 收到关闭指令")
            rospy.signal_shutdown("Server shutdown")
            return

        self._warn_move_drop("invalid_cmd", "[CMD] 未知指令, 忽略: {}".format(cmd))
        self._send_feedback("invalid_cmd", reason="unknown_cmd")

    def _check_arrived(self):
        if self.current_pose is None or self.target_pose is None:
            return False

        pos_err = sum((c - t) ** 2 for c, t in zip(self.current_pose[:3], self.target_pose[:3])) ** 0.5
        ori_err = max(abs(c - t) for c, t in zip(self.current_pose[3:], self.target_pose[3:]))
        return pos_err < self.tol_pos and ori_err < self.tol_ori

    def run(self):
        rate = rospy.Rate(self.loop_rate_hz)

        while not rospy.is_shutdown():
            now = time.time()

            while True:
                try:
                    cmd = self.sub_sock.recv_json(zmq.NOBLOCK)
                    self._handle_cmd(cmd)
                except zmq.Again:
                    break
                except (json.JSONDecodeError, Exception) as e:
                    rospy.logwarn("[ZMQ] 解析错误: %s", e)
                    break

            if self._reset_requested:
                self._handle_reset_request()

            if (now - self.last_hb_time) > self.hb_timeout and not self.estop:
                self._handle_timeout_action()

            if self.target_pose and not self.estop:
                self._maybe_stream_target(now)
                arrived = self._check_arrived()
                if arrived and not self._arrived_latched:
                    self._send_feedback("arrived")
                    self._arrived_latched = True
                elif not arrived and self._arrived_latched:
                    self._arrived_latched = False
            else:
                self._arrived_latched = False

            rate.sleep()

        self._cleanup()

    def _cleanup(self):
        rospy.loginfo("[CLEANUP] 关闭中...")
        try:
            self.sub_sock.close()
            self.push_sock.close()
            self.ctx.term()
        except Exception:
            pass
        rospy.loginfo("[CLEANUP] 完成")

def main():
    try:
        node = ArmClientCollisionGuard()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal("[FATAL] %s", e)
        import traceback

        traceback.print_exc()

if __name__ == "__main__":
    main()
