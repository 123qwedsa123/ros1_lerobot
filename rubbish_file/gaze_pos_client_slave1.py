#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import zmq
import rospy
from sensor_msgs.msg import JointState

class GazePosClientSlave1:
    def __init__(self):
        rospy.init_node("gaze_pos_client_slave1", anonymous=True)

        self.server_ip = rospy.get_param("~server_ip", "127.0.0.1")
        self.server_port = int(rospy.get_param("~server_port", 5588))
        self.hz = float(rospy.get_param("~control_hz", 20.0))
        self.cmd_topic = rospy.get_param("~slave1_cmd_topic", "/slave1/joint_states")

        self.pub = rospy.Publisher(self.cmd_topic, JointState, queue_size=1, tcp_nodelay=True)

        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, 800)
        self.sock.setsockopt(zmq.SNDTIMEO, 800)
        self.sock.connect(f"tcp://{self.server_ip}:{self.server_port}")

        rospy.loginfo(f"[Client] server={self.server_ip}:{self.server_port}, topic={self.cmd_topic}, hz={self.hz}")

    def publish_pos7(self, q):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = [float(x) for x in q]
        msg.velocity = [0.0]*6 + [100.0]
        msg.effort = [0.0]*6 + [1.0]
        self.pub.publish(msg)

    def run(self):
        r = rospy.Rate(self.hz)
        while not rospy.is_shutdown():
            try:
                self.sock.send_multipart([b"get"])
                meta_b, payload = self.sock.recv_multipart()
                meta = meta_b.decode("utf-8", errors="ignore")
                if payload and ("ok=1" in meta):
                    q = np.frombuffer(payload, dtype=np.float32)
                    if q.size == 7:
                        self.publish_pos7(q)
                else:
                    rospy.logwarn_throttle(2.0, f"server not ready: {meta}")
            except Exception as e:
                rospy.logwarn_throttle(2.0, f"zmq error: {e}")
            r.sleep()

def main():
    node = GazePosClientSlave1()
    node.run()

if __name__ == "__main__":
    main()
