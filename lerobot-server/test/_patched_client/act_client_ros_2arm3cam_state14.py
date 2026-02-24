#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT Client ROS - 三相机（state 配置化）
特征名: image_left, image_right, image_middle

默认 state:
- positions + efforts, 每臂 7 维 -> 总 28 维
- reset payload 仍为两臂 positions，共 14 维
"""

import json
import time

import cv2
import numpy as np
import rospy
import zmq
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

VALID_STATE_FIELDS = {"positions", "velocities", "efforts"}


def fix_len(x, n=7):
    x = list(x) if x is not None else []
    if len(x) >= n:
        return x[:n]
    return x + [0.0] * (n - len(x))


def parse_state_fields(raw):
    if isinstance(raw, (list, tuple)):
        items = [str(v).strip().lower() for v in raw]
    else:
        text = str(raw).strip()
        items = [v.strip().lower() for v in text.split(",")] if text else []

    out = []
    for k in items:
        if not k:
            continue
        if k not in VALID_STATE_FIELDS:
            rospy.logwarn(f"忽略不支持的 state field: {k}, 支持: {sorted(VALID_STATE_FIELDS)}")
            continue
        if k not in out:
            out.append(k)
    return out if out else ["positions", "efforts"]


class ACTClient3Cam:
    def __init__(self):
        rospy.init_node("act_client_ros_3cam_statecfg", anonymous=True)

        self.server_ip = rospy.get_param("~server_ip", "127.0.0.1")
        self.server_port = int(rospy.get_param("~server_port", 5560))
        self.hz = float(rospy.get_param("~control_hz", 30.0))
        self.img_size = int(rospy.get_param("~image_size", 256))
        self.jpeg_q = int(rospy.get_param("~jpeg_quality", 80))
        self.err_th = float(rospy.get_param("~reset_err_thresh", 0.03))
        self.reset_timeout = float(rospy.get_param("~reset_timeout", 8.0))
        self.do_reset = bool(rospy.get_param("~do_reset", True))

        self.per_arm_dim = int(rospy.get_param("~per_arm_dim", 7))
        self.state_fields = parse_state_fields(rospy.get_param("~state_fields", "positions,efforts"))
        self.state_dim = int(
            rospy.get_param("~state_dim", self.per_arm_dim * 2 * len(self.state_fields))
        )
        self.reset_dim = int(rospy.get_param("~reset_dim", self.per_arm_dim * 2))
        self.action_dim = int(rospy.get_param("~action_dim", self.per_arm_dim * 2))

        # 三个相机话题
        self.image_topic1 = rospy.get_param("~image_topic1", "/cam1/color/image_raw")
        self.image_topic2 = rospy.get_param("~image_topic2", "/cam2/color/image_raw")
        self.image_topic3 = rospy.get_param("~image_topic3", "/cam3/color/image_raw")

        self.s1_js_topic = rospy.get_param("~slave1_js_topic", "/slave1/joint_states_single")
        self.s2_js_topic = rospy.get_param("~slave2_js_topic", "/slave2/joint_states_single")
        self.s1_cmd_topic = rospy.get_param("~slave1_cmd_topic", "/slave1/joint_states")
        self.s2_cmd_topic = rospy.get_param("~slave2_cmd_topic", "/slave2/joint_states")

        self.bridge = CvBridge()
        self.last_img1 = None
        self.last_img2 = None
        self.last_img3 = None

        zeros = [0.0] * self.per_arm_dim
        self.s1 = {"pos": list(zeros), "vel": list(zeros), "eff": list(zeros)}
        self.s2 = {"pos": list(zeros), "vel": list(zeros), "eff": list(zeros)}

        self.got_s1 = False
        self.got_s2 = False
        self.got_img1 = False
        self.got_img2 = False
        self.got_img3 = False

        # 订阅三个相机
        rospy.Subscriber(self.image_topic1, Image, self.img1_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.image_topic2, Image, self.img2_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.image_topic3, Image, self.img3_cb, queue_size=1, tcp_nodelay=True)

        rospy.Subscriber(self.s1_js_topic, JointState, self.s1_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.s2_js_topic, JointState, self.s2_cb, queue_size=1, tcp_nodelay=True)

        self.pub1 = rospy.Publisher(self.s1_cmd_topic, JointState, queue_size=1, tcp_nodelay=True)
        self.pub2 = rospy.Publisher(self.s2_cmd_topic, JointState, queue_size=1, tcp_nodelay=True)

        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, 1500)
        self.sock.setsockopt(zmq.SNDTIMEO, 1500)
        self.sock.connect(f"tcp://{self.server_ip}:{self.server_port}")

        rospy.loginfo("=" * 60)
        rospy.loginfo("ACT Client ROS - 三相机（state 配置化）")
        rospy.loginfo(f"server: {self.server_ip}:{self.server_port}")
        rospy.loginfo(f"cam1(left): {self.image_topic1}")
        rospy.loginfo(f"cam2(right): {self.image_topic2}")
        rospy.loginfo(f"cam3(middle): {self.image_topic3}")
        rospy.loginfo(f"state_fields={self.state_fields} state_dim={self.state_dim}")
        rospy.loginfo(f"reset_dim={self.reset_dim} action_dim={self.action_dim}")
        rospy.loginfo(f"do_reset: {self.do_reset}")
        rospy.loginfo("=" * 60)

    def img1_cb(self, msg):
        try:
            self.last_img1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.got_img1 = True
        except Exception as e:
            rospy.logwarn(f"img1_cb error: {e}")

    def img2_cb(self, msg):
        try:
            self.last_img2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.got_img2 = True
        except Exception as e:
            rospy.logwarn(f"img2_cb error: {e}")

    def img3_cb(self, msg):
        try:
            self.last_img3 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.got_img3 = True
        except Exception as e:
            rospy.logwarn(f"img3_cb error: {e}")

    def s1_cb(self, msg):
        self.s1["pos"] = fix_len(msg.position, self.per_arm_dim)
        self.s1["vel"] = fix_len(msg.velocity, self.per_arm_dim)
        self.s1["eff"] = fix_len(msg.effort, self.per_arm_dim)
        self.got_s1 = True

    def s2_cb(self, msg):
        self.s2["pos"] = fix_len(msg.position, self.per_arm_dim)
        self.s2["vel"] = fix_len(msg.velocity, self.per_arm_dim)
        self.s2["eff"] = fix_len(msg.effort, self.per_arm_dim)
        self.got_s2 = True

    def publish_setpoint(self, q1, q2):
        q1 = fix_len(q1, self.per_arm_dim)
        q2 = fix_len(q2, self.per_arm_dim)
        now = rospy.Time.now()

        m1 = JointState()
        m1.header.stamp = now
        m1.position = list(map(float, q1))
        m1.velocity = [0.0] * max(self.per_arm_dim - 1, 0) + [100.0]
        m1.effort = [0.0] * max(self.per_arm_dim - 1, 0) + [1.0]

        m2 = JointState()
        m2.header.stamp = now
        m2.position = list(map(float, q2))
        m2.velocity = [0.0] * max(self.per_arm_dim - 1, 0) + [100.0]
        m2.effort = [0.0] * max(self.per_arm_dim - 1, 0) + [1.0]

        self.pub1.publish(m1)
        self.pub2.publish(m2)

    def _field_from_arm(self, arm_state, field_key):
        if field_key == "positions":
            return arm_state["pos"]
        if field_key == "velocities":
            return arm_state["vel"]
        if field_key == "efforts":
            return arm_state["eff"]
        return [0.0] * self.per_arm_dim

    def get_state_vec(self):
        out = []
        for arm in (self.s1, self.s2):
            for field_key in self.state_fields:
                out.extend(self._field_from_arm(arm, field_key))
        state_vec = np.asarray(out, dtype=np.float32)
        if state_vec.size != self.state_dim:
            rospy.logwarn_throttle(
                2.0,
                f"state size mismatch: got {state_vec.size}, expected {self.state_dim}, fields={self.state_fields}",
            )
        return state_vec

    def encode_img(self, img):
        if img is None:
            return None
        resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
        return buf.tobytes() if ok else None

    def zmq_req(self, cmd, jpg_bytes1=b"", jpg_bytes2=b"", jpg_bytes3=b"", state_vec=None):
        meta = {"cmd": cmd, "ts": time.time()}
        state_bytes = b"" if state_vec is None else np.asarray(state_vec, np.float32).tobytes()

        self.sock.send_multipart(
            [
                json.dumps(meta).encode("utf-8"),
                jpg_bytes1,
                jpg_bytes2,
                jpg_bytes3,
                state_bytes,
            ]
        )

        meta_rsp_b, payload = self.sock.recv_multipart()
        return json.loads(meta_rsp_b.decode("utf-8")), payload

    def wait_ready(self):
        t0 = time.time()
        r = rospy.Rate(50)
        rospy.loginfo("等待传感器就绪...")

        while not rospy.is_shutdown():
            if self.got_img1 and self.got_img2 and self.got_img3 and self.got_s1 and self.got_s2:
                rospy.loginfo("✅ 所有传感器就绪！")
                return True

            if time.time() - t0 > 5.0:
                rospy.logwarn(
                    f"等待超时: img1={self.got_img1} img2={self.got_img2} img3={self.got_img3} "
                    f"s1={self.got_s1} s2={self.got_s2}"
                )
                t0 = time.time()

            r.sleep()
        return False

    def run_reset(self):
        rospy.loginfo("请求 reset...")
        meta, payload = self.zmq_req("get_reset")

        if not meta.get("ok", False):
            raise RuntimeError(f"get_reset failed: {meta}")

        pos = np.frombuffer(payload, dtype=np.float32)
        if pos.size != self.reset_dim:
            raise RuntimeError(f"bad reset payload size: {pos.size}, expected {self.reset_dim}")
        if self.reset_dim < self.per_arm_dim * 2:
            raise RuntimeError(
                f"reset_dim={self.reset_dim} is too small for 2 arms per_arm_dim={self.per_arm_dim}"
            )

        q1 = pos[: self.per_arm_dim]
        q2 = pos[self.per_arm_dim : self.per_arm_dim * 2]
        t0 = time.time()
        r = rospy.Rate(self.hz)

        while not rospy.is_shutdown():
            self.publish_setpoint(q1, q2)

            err1 = float(np.max(np.abs(np.asarray(self.s1["pos"], np.float32) - q1)))
            err2 = float(np.max(np.abs(np.asarray(self.s2["pos"], np.float32) - q2)))

            if max(err1, err2) < self.err_th:
                rospy.loginfo("✅ Reset 完成")
                return

            if time.time() - t0 > self.reset_timeout:
                rospy.logwarn(f"reset timeout: err1={err1:.4f} err2={err2:.4f}")
                return

            r.sleep()

    def run_stream(self):
        rospy.loginfo("开始推理循环...")
        r = rospy.Rate(self.hz)

        while not rospy.is_shutdown():
            if self.last_img1 is None or self.last_img2 is None or self.last_img3 is None:
                r.sleep()
                continue

            state_vec = self.get_state_vec()
            if state_vec.size != self.state_dim:
                r.sleep()
                continue

            jpg1 = self.encode_img(self.last_img1)
            jpg2 = self.encode_img(self.last_img2)
            jpg3 = self.encode_img(self.last_img3)

            if jpg1 is None or jpg2 is None or jpg3 is None:
                r.sleep()
                continue

            try:
                meta, payload = self.zmq_req("infer", jpg1, jpg2, jpg3, state_vec)
                if meta.get("ok", False):
                    action = np.frombuffer(payload, dtype=np.float32)
                    if action.size == self.action_dim:
                        q1 = action[: self.per_arm_dim]
                        q2 = action[self.per_arm_dim : self.per_arm_dim * 2]
                        self.publish_setpoint(q1, q2)
                    else:
                        rospy.logwarn_throttle(
                            2.0,
                            f"bad action size={action.size}, expected {self.action_dim}",
                        )
                else:
                    rospy.logwarn(f"infer failed: {meta}")
            except Exception as e:
                rospy.logwarn(f"zmq error: {e}")

            r.sleep()


def main():
    node = ACTClient3Cam()
    node.wait_ready()
    if node.do_reset:
        node.run_reset()
    node.run_stream()


if __name__ == "__main__":
    main()
