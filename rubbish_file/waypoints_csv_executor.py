#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
离线 CSV 路点执行器（无 ZMQ / 无 server）

功能:
1) 读取 waypoints.csv (支持带/不带表头，支持可选 id 列)
2) 依次发布 PosCmd 到目标机械臂
3) 按末端位姿反馈判断到位后切下一点
4) 全程只发送 enable=true，不发送 enable=false
5) 执行结束或超时中止后，常驻重发末点并保活 enable
"""

import csv
import importlib
import math
import os
import sys
import time

import rospy
from std_msgs.msg import Bool


REQUIRED_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


def _parse_csv_poses(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV file not found: {}".format(csv_path))

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            clean = [c.strip() for c in row]
            if any(clean):
                rows.append(clean)

    if not rows:
        raise ValueError("CSV is empty: {}".format(csv_path))

    header = [c.lower() for c in rows[0]]
    has_header = all(k in header for k in REQUIRED_FIELDS)

    poses = []
    if has_header:
        idx = {k: header.index(k) for k in REQUIRED_FIELDS}
        data_rows = rows[1:]
        if not data_rows:
            raise ValueError("CSV has header but no data rows: {}".format(csv_path))
        for row_no, row in enumerate(data_rows, start=2):
            try:
                pose = [float(row[idx[k]]) for k in REQUIRED_FIELDS]
            except Exception as e:
                raise ValueError("Failed to parse row {}: {}".format(row_no, e))
            poses.append(pose)
    else:
        for row_no, row in enumerate(rows, start=1):
            if len(row) == 6:
                fields = row
            elif len(row) == 7:
                fields = row[1:]
            else:
                raise ValueError(
                    "Invalid column count in row {}: expected 6 or 7, got {}".format(
                        row_no, len(row)
                    )
                )
            try:
                pose = [float(v) for v in fields]
            except Exception as e:
                raise ValueError("Non-numeric value in row {}: {}".format(row_no, e))
            poses.append(pose)

    if not poses:
        raise ValueError("No valid poses in CSV: {}".format(csv_path))

    return poses


def _import_msg(msg_type_name):
    try:
        mod = importlib.import_module("piper_msgs.msg")
        return getattr(mod, msg_type_name)
    except Exception as e:
        raise RuntimeError("Cannot import piper_msgs.msg.{}: {}".format(msg_type_name, e))


def _pose_to_list(msg):
    try:
        return [msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw]
    except Exception:
        return None


class WaypointsCsvExecutor(object):
    def __init__(self):
        rospy.init_node("waypoints_csv_executor", anonymous=False)

        self.csv_path = str(rospy.get_param("~csv_path", "/workspace/piper_master_slave_ws/waypoints.csv"))
        self.arm_ns = str(rospy.get_param("~arm_ns", "slave3")).strip().strip("/")

        self.pos_cmd_topic = str(
            rospy.get_param("~pos_cmd_topic", "/{}/pos_cmd".format(self.arm_ns))
        )
        self.end_pose_topic = str(
            rospy.get_param("~end_pose_topic", "/{}/end_pose_euler".format(self.arm_ns))
        )
        self.enable_topic = str(
            rospy.get_param("~enable_topic", "/{}/enable_flag".format(self.arm_ns))
        )

        self.pos_cmd_msg_type = str(rospy.get_param("~pos_cmd_msg_type", "PosCmd"))
        self.end_pose_msg_type = str(rospy.get_param("~end_pose_msg_type", "PiperEulerPose"))

        self.mode1 = int(rospy.get_param("~mode1", 1))
        self.mode2 = int(rospy.get_param("~mode2", 0))
        self.gripper = float(rospy.get_param("~gripper", 0.0))

        self.move_burst_count = max(1, int(rospy.get_param("~move_burst_count", 3)))
        self.move_burst_interval_sec = max(0.0, float(rospy.get_param("~move_burst_interval_sec", 0.02)))
        self.arrive_tolerance_pos_m = max(0.0, float(rospy.get_param("~arrive_tolerance_pos_m", 0.01)))
        self.arrive_tolerance_ori_rad = max(0.0, float(rospy.get_param("~arrive_tolerance_ori_rad", 0.05)))
        self.arrive_timeout_sec = max(0.1, float(rospy.get_param("~arrive_timeout_sec", 25.0)))
        self.dwell_sec = max(0.0, float(rospy.get_param("~dwell_sec", 0.10)))
        self.loop_rate_hz = max(1.0, float(rospy.get_param("~loop_rate_hz", 30.0)))

        self.enable_burst_count = max(1, int(rospy.get_param("~enable_burst_count", 5)))
        self.enable_burst_interval_sec = max(0.0, float(rospy.get_param("~enable_burst_interval_sec", 0.05)))
        self.enable_keepalive_interval_sec = max(
            0.0, float(rospy.get_param("~enable_keepalive_interval_sec", 2.0))
        )
        self.hold_republish_hz = max(0.0, float(rospy.get_param("~hold_republish_hz", 5.0)))
        self.stop_on_timeout = bool(rospy.get_param("~stop_on_timeout", True))

        self.first_pose_wait_timeout_sec = max(
            0.1, float(rospy.get_param("~first_pose_wait_timeout_sec", 5.0))
        )

        PosCmdMsg = _import_msg(self.pos_cmd_msg_type)
        EndPoseMsg = _import_msg(self.end_pose_msg_type)
        self.PosCmdMsg = PosCmdMsg

        self.enable_pub = rospy.Publisher(self.enable_topic, Bool, queue_size=1)
        self.pos_cmd_pub = rospy.Publisher(self.pos_cmd_topic, PosCmdMsg, queue_size=1)
        self.current_pose = None
        rospy.Subscriber(self.end_pose_topic, EndPoseMsg, self._pose_cb, queue_size=1)

        self.last_enable_keepalive_time = 0.0

        self.poses = _parse_csv_poses(self.csv_path)

        rospy.loginfo("[INIT] csv_path=%s", self.csv_path)
        rospy.loginfo("[INIT] loaded poses=%d", len(self.poses))
        if len(self.poses) != 16:
            rospy.logwarn("[INIT] expected 16 poses, got %d", len(self.poses))
        rospy.loginfo("[INIT] pos_cmd_topic=%s", self.pos_cmd_topic)
        rospy.loginfo("[INIT] end_pose_topic=%s", self.end_pose_topic)
        rospy.loginfo("[INIT] enable_topic=%s", self.enable_topic)
        rospy.loginfo("[INIT] keep enabled only; never publish enable=false")

    def _pose_cb(self, msg):
        p = _pose_to_list(msg)
        if p is not None:
            self.current_pose = p

    def _publish_enable_true_once(self):
        self.enable_pub.publish(Bool(data=True))
        self.last_enable_keepalive_time = time.time()

    def _publish_enable_true_burst(self):
        for i in range(self.enable_burst_count):
            self._publish_enable_true_once()
            if i < (self.enable_burst_count - 1) and self.enable_burst_interval_sec > 0.0:
                rospy.sleep(self.enable_burst_interval_sec)
        rospy.loginfo("[ENABLE] sent true burst x%d", self.enable_burst_count)

    def _maybe_enable_keepalive(self, now):
        if self.enable_keepalive_interval_sec <= 0.0:
            return
        if (now - self.last_enable_keepalive_time) >= self.enable_keepalive_interval_sec:
            self._publish_enable_true_once()

    def _publish_pos_cmd_once(self, pose):
        msg = self.PosCmdMsg()
        msg.x = float(pose[0])
        msg.y = float(pose[1])
        msg.z = float(pose[2])
        msg.roll = float(pose[3])
        msg.pitch = float(pose[4])
        msg.yaw = float(pose[5])
        msg.gripper = self.gripper
        msg.mode1 = self.mode1
        msg.mode2 = self.mode2
        self.pos_cmd_pub.publish(msg)

    def _publish_pos_cmd_burst(self, pose):
        for i in range(self.move_burst_count):
            self._publish_pos_cmd_once(pose)
            if i < (self.move_burst_count - 1) and self.move_burst_interval_sec > 0.0:
                rospy.sleep(self.move_burst_interval_sec)

    def _calc_err(self, target):
        if self.current_pose is None:
            return None, None
        pos_err = math.sqrt(
            (self.current_pose[0] - target[0]) ** 2
            + (self.current_pose[1] - target[1]) ** 2
            + (self.current_pose[2] - target[2]) ** 2
        )
        ori_err = max(
            abs(self.current_pose[3] - target[3]),
            abs(self.current_pose[4] - target[4]),
            abs(self.current_pose[5] - target[5]),
        )
        return pos_err, ori_err

    def _wait_first_pose(self):
        deadline = time.time() + self.first_pose_wait_timeout_sec
        while not rospy.is_shutdown():
            if self.current_pose is not None:
                rospy.loginfo("[POSE] first end_pose received")
                return True
            if time.time() >= deadline:
                rospy.logwarn(
                    "[POSE] no end_pose within %.2fs, continue anyway",
                    self.first_pose_wait_timeout_sec,
                )
                return False
            rospy.sleep(0.02)
        return False

    def _execute_sequence(self):
        last_target = None
        total = len(self.poses)
        rate = rospy.Rate(self.loop_rate_hz)

        for idx, pose in enumerate(self.poses, start=1):
            if rospy.is_shutdown():
                break

            last_target = pose
            self._publish_pos_cmd_burst(pose)
            rospy.loginfo(
                "[SEND] pose=%d/%d x=%.4f y=%.4f z=%.4f r=%.3f p=%.3f yw=%.3f burst=%d",
                idx,
                total,
                pose[0],
                pose[1],
                pose[2],
                pose[3],
                pose[4],
                pose[5],
                self.move_burst_count,
            )

            t0 = time.time()
            arrived = False
            while not rospy.is_shutdown():
                now = time.time()
                self._maybe_enable_keepalive(now)

                pos_err, ori_err = self._calc_err(pose)
                if pos_err is not None and ori_err is not None:
                    if pos_err < self.arrive_tolerance_pos_m and ori_err < self.arrive_tolerance_ori_rad:
                        arrived = True
                        rospy.loginfo(
                            "[ARRIVED] pose=%d/%d elapsed=%.2fs pos_err=%.5f ori_err=%.5f",
                            idx,
                            total,
                            (now - t0),
                            pos_err,
                            ori_err,
                        )
                        break

                if (now - t0) >= self.arrive_timeout_sec:
                    rospy.logwarn(
                        "[TIMEOUT] pose=%d/%d after %.2fs (stop_on_timeout=%s)",
                        idx,
                        total,
                        self.arrive_timeout_sec,
                        self.stop_on_timeout,
                    )
                    if self.stop_on_timeout:
                        return last_target, False
                    break

                rate.sleep()

            if arrived and self.dwell_sec > 0.0 and not rospy.is_shutdown():
                rospy.sleep(self.dwell_sec)

        return last_target, True

    def _hold_target_forever(self, target, finished_ok):
        if target is None:
            rospy.logwarn("[HOLD] no valid target to hold; keep enable=true only")
            hold_rate = rospy.Rate(max(1.0, self.loop_rate_hz))
            while not rospy.is_shutdown():
                self._maybe_enable_keepalive(time.time())
                hold_rate.sleep()
            return

        rospy.loginfo(
            "[HOLD] %s, keep target x=%.4f y=%.4f z=%.4f (republish_hz=%.2f)",
            "sequence completed" if finished_ok else "stopped by timeout",
            target[0],
            target[1],
            target[2],
            self.hold_republish_hz,
        )

        if self.hold_republish_hz > 0.0:
            hold_rate = rospy.Rate(self.hold_republish_hz)
            while not rospy.is_shutdown():
                self._publish_pos_cmd_once(target)
                self._maybe_enable_keepalive(time.time())
                hold_rate.sleep()
        else:
            hold_rate = rospy.Rate(max(1.0, self.loop_rate_hz))
            while not rospy.is_shutdown():
                self._maybe_enable_keepalive(time.time())
                hold_rate.sleep()

    def run(self):
        rospy.sleep(0.8)
        self._publish_enable_true_burst()
        self._wait_first_pose()
        last_target, finished_ok = self._execute_sequence()
        self._hold_target_forever(last_target, finished_ok)


def main():
    try:
        node = WaypointsCsvExecutor()
        node.run()
    except Exception as e:
        rospy.logfatal("[FATAL] %s", e)
        raise


if __name__ == "__main__":
    main()
