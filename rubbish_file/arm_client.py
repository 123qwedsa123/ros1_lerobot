#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arm Client (Docker ROS1 端)
ZeroMQ 接收指令 → 通过 /pos_cmd (PosCmd) 控制 Piper 机械臂
"""
import importlib
import json
import sys
import time

import rospy
import yaml
import zmq
from std_msgs.msg import Bool


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def import_msg(msg_type_name):
    """动态导入 piper_msgs 中的消息类型"""
    try:
        mod = importlib.import_module('piper_msgs.msg')
        cls = getattr(mod, msg_type_name)
        rospy.loginfo("[IMPORT] piper_msgs/%s", msg_type_name)
        return cls
    except (ImportError, AttributeError) as e:
        rospy.logfatal("[IMPORT] 无法导入 piper_msgs.msg.%s: %s", msg_type_name, e)
        sys.exit(1)


def pose_to_list(msg):
    """PiperEulerPose → [x,y,z,r,p,y]"""
    try:
        return [msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw]
    except AttributeError:
        return None


class ArmClient:
    def __init__(self):
        rospy.init_node('arm_gaze_client', anonymous=True)

        # 配置
        config_path = rospy.get_param('~config_path', 'arm_client_config.yaml')
        self.cfg = load_config(config_path)
        acfg = self.cfg['arm']
        zcfg = self.cfg['zmq']
        pcfg = self.cfg['pos_cmd']

        # 导入消息类型
        PosCmdMsg = import_msg(acfg['pos_cmd_msg_type'])       # PosCmd (控制)
        EndPoseMsg = import_msg(acfg['end_pose_msg_type'])     # PiperEulerPose (反馈)
        self.PosCmdMsg = PosCmdMsg

        # PosCmd 固定参数
        self.cmd_mode1 = pcfg.get('mode1', 1)
        self.cmd_mode2 = pcfg.get('mode2', 0)
        self.cmd_gripper = pcfg.get('gripper', 0.0)

        # ROS pub/sub
        self.enable_pub = rospy.Publisher(acfg['enable_topic'], Bool, queue_size=1)
        self.ctrl_pub = rospy.Publisher(acfg['pos_cmd_topic'], PosCmdMsg, queue_size=1)
        self.current_pose = None
        rospy.Subscriber(acfg['end_pose_topic'], EndPoseMsg, self._pose_cb, queue_size=1)

        # ZeroMQ
        self.ctx = zmq.Context()
        self.sub_sock = self.ctx.socket(zmq.SUB)
        self.sub_sock.connect(f"tcp://{zcfg['server_ip']}:{zcfg['sub_port']}")
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_sock.setsockopt(zmq.RCVTIMEO, 0)
        rospy.loginfo("[ZMQ] SUB → %s:%s", zcfg['server_ip'], zcfg['sub_port'])

        self.push_sock = self.ctx.socket(zmq.PUSH)
        self.push_sock.connect(f"tcp://{zcfg['server_ip']}:{zcfg['push_port']}")
        rospy.loginfo("[ZMQ] PUSH → %s:%s", zcfg['server_ip'], zcfg['push_port'])

        # 状态
        self.target_pose = None
        self.target_id = -1
        self.last_hb_time = time.time()
        self.estop = False
        self.tol_pos = self.cfg.get('arrive_tolerance_pos', 0.01)
        self.tol_ori = self.cfg.get('arrive_tolerance_ori', 0.05)
        self.hb_timeout = self.cfg.get('heartbeat_timeout_sec', 3.0)

        rospy.sleep(1.0)
        rospy.loginfo("[READY] arm_gaze_client 启动完成")

    def _pose_cb(self, msg):
        self.current_pose = pose_to_list(msg)

    def _publish_pos_cmd(self, x, y, z, roll, pitch, yaw):
        """构造 PosCmd 并发布到 /pos_cmd"""
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
        msg = Bool(data=enabled)
        for _ in range(5):
            self.enable_pub.publish(msg)
            rospy.sleep(0.05)
        rospy.loginfo("[ARM] %s", "使能" if enabled else "去使能")

    def _handle_cmd(self, cmd):
        c = cmd.get("cmd")

        if c == "heartbeat":
            self.last_hb_time = time.time()

        elif c == "move" and not self.estop:
            self.last_hb_time = time.time()
            new_id = cmd.get("id", -1)
            self.target_id = new_id
            self.target_pose = [cmd["x"], cmd["y"], cmd["z"],
                                cmd["roll"], cmd["pitch"], cmd["yaw"]]
            # 每收到 move 都发一次 PosCmd
            self._publish_pos_cmd(*self.target_pose)
            rospy.loginfo("[MOVE] → 点%d (%.3f,%.3f,%.3f)",
                          self.target_id, cmd["x"], cmd["y"], cmd["z"])

        elif c == "hold":
            self.last_hb_time = time.time()
            # hold = 不发新指令, 机械臂维持当前位置

        elif c == "estop":
            if not self.estop:
                self.estop = True
                self._set_enable(False)
                rospy.logwarn("[ESTOP] 急停! 已去使能")

        elif c == "resume":
            if self.estop:
                self.estop = False
                self._set_enable(True)
                sp = self.cfg.get('safe_pose', {})
                if sp:
                    self._publish_pos_cmd(
                        sp['x'], sp['y'], sp['z'],
                        sp['roll'], sp['pitch'], sp['yaw'])
                rospy.loginfo("[RESUME] 已恢复")

        elif c == "shutdown":
            rospy.loginfo("[SHUTDOWN] 收到关闭指令")
            rospy.signal_shutdown("Server shutdown")

    def _check_arrived(self):
        if self.current_pose is None or self.target_pose is None:
            return False
        pos_err = sum((c - t) ** 2 for c, t in
                      zip(self.current_pose[:3], self.target_pose[:3])) ** 0.5
        ori_err = max(abs(c - t) for c, t in
                      zip(self.current_pose[3:], self.target_pose[3:]))
        return pos_err < self.tol_pos and ori_err < self.tol_ori

    def run(self):
        rate = rospy.Rate(self.cfg.get('loop_rate_hz', 20))

        while not rospy.is_shutdown():
            now = time.time()

            # 接收所有指令
            while True:
                try:
                    cmd = self.sub_sock.recv_json(zmq.NOBLOCK)
                    self._handle_cmd(cmd)
                except zmq.Again:
                    break
                except (json.JSONDecodeError, Exception) as e:
                    rospy.logwarn("[ZMQ] 解析错误: %s", e)
                    break

            # 心跳超时 → 停止
            if (now - self.last_hb_time) > self.hb_timeout and not self.estop:
                rospy.logwarn("[TIMEOUT] 心跳超时, 停止机械臂")
                self._set_enable(False)
                self.estop = True

            # 到位反馈
            if self.target_pose and not self.estop:
                if self._check_arrived():
                    self._send_feedback("arrived")

            rate.sleep()

        self._cleanup()

    def _cleanup(self):
        rospy.loginfo("[CLEANUP] 关闭中...")
        self.sub_sock.close()
        self.push_sock.close()
        self.ctx.term()
        rospy.loginfo("[CLEANUP] 完成")


if __name__ == '__main__':
    try:
        node = ArmClient()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logfatal("[FATAL] %s", e)
        import traceback
        traceback.print_exc()
