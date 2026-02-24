#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT Client ROS - 双相机 (state14 版)

发送:
- jpg1 (cam1)
- jpg2 (cam2)
- state14 = [s1_pos7, s2_pos7]  # 只用绝对位置

ZMQ:
- get_reset: 发送 [meta, jpg1, jpg2, empty] 也行（server 不读）
- infer    : 发送 [meta, jpg1, jpg2, state14_bytes]
"""

import json
import time
import numpy as np
import zmq
import cv2
import rospy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

def fix_len(x, n=7):
    x = list(x) if x is not None else []
    if len(x) >= n:
        return x[:n]
    return x + [0.0] * (n - len(x))

class ACTClientDualCam:
    def __init__(self):
        rospy.init_node("act_client_ros_dual_cam_state14", anonymous=True)

        self.server_ip = rospy.get_param("~server_ip", "127.0.0.1")
        self.server_port = int(rospy.get_param("~server_port", 5560))
        self.hz = float(rospy.get_param("~control_hz", 30.0))
        self.img_size = int(rospy.get_param("~image_size", 256))
        self.jpeg_q = int(rospy.get_param("~jpeg_quality", 80))
        self.err_th = float(rospy.get_param("~reset_err_thresh", 0.03))
        self.reset_timeout = float(rospy.get_param("~reset_timeout", 8.0))
        self.do_reset = bool(rospy.get_param("~do_reset", True))

        self.image_topic1 = rospy.get_param("~image_topic1", "/cam1/color/image_raw")
        self.image_topic2 = rospy.get_param("~image_topic2", "/cam2/color/image_raw")

        self.s1_js_topic = rospy.get_param("~slave1_js_topic", "/slave1/joint_states_single")
        self.s2_js_topic = rospy.get_param("~slave2_js_topic", "/slave2/joint_states_single")
        self.s1_cmd_topic = rospy.get_param("~slave1_cmd_topic", "/slave1/joint_states")
        self.s2_cmd_topic = rospy.get_param("~slave2_cmd_topic", "/slave2/joint_states")

        self.bridge = CvBridge()
        self.last_img1 = None
        self.last_img2 = None

        self.s1 = {"pos": [0]*7}
        self.s2 = {"pos": [0]*7}

        self.got_s1 = False
        self.got_s2 = False
        self.got_img1 = False
        self.got_img2 = False

        rospy.Subscriber(self.image_topic1, Image, self.img1_cb, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(self.image_topic2, Image, self.img2_cb, queue_size=1, tcp_nodelay=True)
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
        rospy.loginfo("ACT Client ROS - 双相机 (state14)")
        rospy.loginfo(f"server: {self.server_ip}:{self.server_port}")
        rospy.loginfo(f"cam1: {self.image_topic1}")
        rospy.loginfo(f"cam2: {self.image_topic2}")
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

    def s1_cb(self, msg):
        self.s1["pos"] = fix_len(msg.position, 7)
        self.got_s1 = True

    def s2_cb(self, msg):
        self.s2["pos"] = fix_len(msg.position, 7)
        self.got_s2 = True

    def publish_setpoint(self, q1, q2):
        now = rospy.Time.now()

        m1 = JointState()
        m1.header.stamp = now
        m1.position = list(map(float, q1))
        m1.velocity = [0.0] * 6 + [100.0]
        m1.effort = [0.0] * 6 + [1.0]

        m2 = JointState()
        m2.header.stamp = now
        m2.position = list(map(float, q2))
        m2.velocity = [0.0] * 6 + [100.0]
        m2.effort = [0.0] * 6 + [1.0]

        self.pub1.publish(m1)
        self.pub2.publish(m2)

    def get_state14(self):
        """state14 = [s1_pos7, s2_pos7]"""
        s = self.s1["pos"] + self.s2["pos"]
        return np.asarray(s, dtype=np.float32)

    def encode_img(self, img):
        if img is None:
            return None
        resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
        return buf.tobytes() if ok else None

    def zmq_req(self, cmd, jpg_bytes1=b"", jpg_bytes2=b"", state14=None):
        meta = {"cmd": cmd, "ts": time.time()}
        state_bytes = b"" if state14 is None else np.asarray(state14, np.float32).tobytes()

        self.sock.send_multipart([
            json.dumps(meta).encode("utf-8"),
            jpg_bytes1,
            jpg_bytes2,
            state_bytes,
        ])

        meta_rsp_b, payload = self.sock.recv_multipart()
        return json.loads(meta_rsp_b.decode("utf-8")), payload

    def wait_ready(self):
        t0 = time.time()
        r = rospy.Rate(50)
        rospy.loginfo("等待传感器就绪...")

        while not rospy.is_shutdown():
            if self.got_img1 and self.got_img2 and self.got_s1 and self.got_s2:
                rospy.loginfo("✅ 所有传感器就绪！")
                return True

            if time.time() - t0 > 5.0:
                rospy.logwarn(f"等待超时: img1={self.got_img1} img2={self.got_img2} s1={self.got_s1} s2={self.got_s2}")
                t0 = time.time()

            r.sleep()
        return False

    def run_reset(self):
        rospy.loginfo("请求 reset...")
        meta, payload = self.zmq_req("get_reset")

        if not meta.get("ok", False):
            raise RuntimeError(f"get_reset failed: {meta}")

        pos14 = np.frombuffer(payload, dtype=np.float32)
        if pos14.size != 14:
            raise RuntimeError(f"bad reset pos14 size: {pos14.size}")

        q1, q2 = pos14[:7], pos14[7:]
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
            if self.last_img1 is None or self.last_img2 is None:
                r.sleep()
                continue

            state14 = self.get_state14()
            jpg1 = self.encode_img(self.last_img1)
            jpg2 = self.encode_img(self.last_img2)
            if jpg1 is None or jpg2 is None:
                r.sleep()
                continue

            try:
                meta, payload = self.zmq_req("infer", jpg1, jpg2, state14)
                if meta.get("ok", False):
                    a14 = np.frombuffer(payload, dtype=np.float32)
                    if a14.size == 14:
                        self.publish_setpoint(a14[:7], a14[7:])
                else:
                    rospy.logwarn(f"infer failed: {meta}")
            except Exception as e:
                rospy.logwarn(f"zmq error: {e}")

            r.sleep()

def main():
    node = ACTClientDualCam()
    node.wait_ready()
    if node.do_reset:
        node.run_reset()
    node.run_stream()

if __name__ == "__main__":
    main()
