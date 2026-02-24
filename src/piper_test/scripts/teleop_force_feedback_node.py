#!/usr/bin/env python3
# -*-coding:utf8-*-
from typing import Optional
import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
import time
import threading
import math
import numpy as np
from piper_sdk import *
from piper_sdk import C_PiperInterface
from std_srvs.srv import Trigger, TriggerResponse
from piper_msgs.msg import PiperStatusMsg, PiperEulerPose
from piper_msgs.srv import Enable, EnableResponse
from piper_msgs.srv import Gripper, GripperResponse
from piper_msgs.srv import GoZero, GoZeroResponse
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
import xml.etree.ElementTree as ET


def check_ros_master():
    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")


class C_PiperCtrlNode:
    """Unified robot arm control node supporting position and MIT control modes"""

    def __init__(self) -> None:
        check_ros_master()
        if not rospy.get_node_uri():
            rospy.init_node("piper_ctrl_node", anonymous=False)

        # Basic parameters
        self.can_port = rospy.get_param("~can_port", "can0")
        rospy.loginfo("CAN port: %s", self.can_port)

        self.topic_prefix = rospy.get_param("~topic_prefix", "/")
        if not self.topic_prefix.endswith("/"):
            self.topic_prefix += "/"
        rospy.loginfo("Topic prefix: %s", self.topic_prefix)

        self.auto_enable = rospy.get_param("~auto_enable", False)
        rospy.loginfo("Auto enable: %s", self.auto_enable)
        legacy_disable_on_shutdown = rospy.get_param("~disable_on_shutdown", None)
        self.shutdown_mode = str(rospy.get_param("~shutdown_mode", "hold")).strip().lower()
        if legacy_disable_on_shutdown is not None and not rospy.has_param("~shutdown_mode"):
            if isinstance(legacy_disable_on_shutdown, str):
                legacy_value = legacy_disable_on_shutdown.strip().lower() in ("1", "true", "yes", "on")
            else:
                legacy_value = bool(legacy_disable_on_shutdown)
            self.shutdown_mode = "disable" if legacy_value else "none"
        if self.shutdown_mode not in ("hold", "disable", "none"):
            rospy.logwarn("Invalid shutdown_mode=%s, fallback to 'hold'", self.shutdown_mode)
            self.shutdown_mode = "hold"
        self.shutdown_hold_retries = max(1, int(rospy.get_param("~shutdown_hold_retries", 5)))
        self.shutdown_hold_interval_sec = max(
            0.0, float(rospy.get_param("~shutdown_hold_interval_sec", 0.05))
        )
        self.shutdown_disable_retries = max(1, int(rospy.get_param("~shutdown_disable_retries", 3)))
        self.shutdown_disable_interval_sec = max(
            0.0, float(rospy.get_param("~shutdown_disable_interval_sec", 0.05))
        )
        rospy.loginfo(
            "Shutdown safety: mode=%s hold_retries=%d hold_interval=%.3fs disable_retries=%d disable_interval=%.3fs",
            self.shutdown_mode,
            self.shutdown_hold_retries,
            self.shutdown_hold_interval_sec,
            self.shutdown_disable_retries,
            self.shutdown_disable_interval_sec,
        )

        self.gripper_exist = rospy.get_param("~gripper_exist", True)
        rospy.loginfo("Gripper exist: %s", self.gripper_exist)

        self.gripper_val_mutiple = rospy.get_param("~gripper_val_mutiple", 1.0)
        if self.gripper_val_mutiple <= 0:
            rospy.logwarn("Invalid gripper_val_mutiple value: must be positive. Using default value of 1.")
            self.gripper_val_mutiple = 1.0
        rospy.loginfo("Gripper value multiple: %.2f", self.gripper_val_mutiple)

        self.enable_gripper = rospy.get_param("~enable_gripper", True)
        rospy.loginfo("Enable gripper: %s", self.enable_gripper)

        self.enable_gripper_haptic = rospy.get_param("~enable_gripper_haptic", False)
        rospy.loginfo("Enable gripper haptic: %s", self.enable_gripper_haptic)

        self.gripper_range = rospy.get_param("~gripper_range", 0.1)
        rospy.loginfo("Gripper range: %.2f", self.gripper_range)

        self.gripper_reverse = rospy.get_param("~gripper_reverse", False)
        rospy.loginfo("Gripper reverse: %s", self.gripper_reverse)

        # Gripper haptic shaping parameters. These suppress idle feedback when
        # measured effort has baseline bias/noise and only react to contact.
        haptic_sign_param = float(rospy.get_param("~gripper_haptic_effort_sign", 0.0))
        if abs(haptic_sign_param) < 1e-6:
            self.gripper_haptic_effort_sign = -1.0 if not self.gripper_reverse else 1.0
        else:
            self.gripper_haptic_effort_sign = 1.0 if haptic_sign_param > 0.0 else -1.0
        self.gripper_haptic_effort_deadband = max(
            0.0, float(rospy.get_param("~gripper_haptic_effort_deadband", 0.15))
        )
        self.gripper_haptic_effort_max = float(rospy.get_param("~gripper_haptic_effort_max", 0.6))
        if self.gripper_haptic_effort_max <= self.gripper_haptic_effort_deadband:
            self.gripper_haptic_effort_max = self.gripper_haptic_effort_deadband + 1e-6
        self.gripper_haptic_effort_bias = float(rospy.get_param("~gripper_haptic_effort_bias", 0.0))
        self.gripper_haptic_cmd_max = max(0, int(rospy.get_param("~gripper_haptic_cmd_max", 500)))
        self.gripper_haptic_cmd_enable_threshold = int(
            rospy.get_param("~gripper_haptic_cmd_enable_threshold", 80)
        )
        self.gripper_haptic_cmd_enable_threshold = max(
            0, min(self.gripper_haptic_cmd_enable_threshold, self.gripper_haptic_cmd_max)
        )
        self.gripper_haptic_effort_alpha = float(rospy.get_param("~gripper_haptic_effort_alpha", 0.2))
        self.gripper_haptic_effort_alpha = max(0.0, min(1.0, self.gripper_haptic_effort_alpha))
        self.gripper_haptic_release_when_opening = bool(
            rospy.get_param("~gripper_haptic_release_when_opening", True)
        )
        self.gripper_haptic_opening_relief = float(rospy.get_param("~gripper_haptic_opening_relief", 0.15))
        self.gripper_haptic_opening_relief = max(0.0, min(1.0, self.gripper_haptic_opening_relief))
        opening_sign_param = float(rospy.get_param("~gripper_haptic_opening_sign", 0.0))
        if abs(opening_sign_param) < 1e-6:
            self.gripper_haptic_opening_sign = 1.0 if not self.gripper_reverse else -1.0
        else:
            self.gripper_haptic_opening_sign = 1.0 if opening_sign_param > 0.0 else -1.0
        self.gripper_haptic_opening_direction_eps = float(
            rospy.get_param("~gripper_haptic_opening_direction_eps", 1e-4)
        )
        rospy.loginfo(
            "Gripper haptic shaping: sign=%.0f deadband=%.3f max=%.3f bias=%.3f cmd_max=%d threshold=%d alpha=%.2f open_relief=%s/%.2f open_sign=%.0f",
            self.gripper_haptic_effort_sign,
            self.gripper_haptic_effort_deadband,
            self.gripper_haptic_effort_max,
            self.gripper_haptic_effort_bias,
            self.gripper_haptic_cmd_max,
            self.gripper_haptic_cmd_enable_threshold,
            self.gripper_haptic_effort_alpha,
            self.gripper_haptic_release_when_opening,
            self.gripper_haptic_opening_relief,
            self.gripper_haptic_opening_sign,
        )

        # Control mode parameters
        self.ctrl_mode = rospy.get_param("~ctrl_mode", "p").lower()
        if self.ctrl_mode not in ["p", "mit"]:
            rospy.logwarn("Invalid ctrl_mode: %s. Using default 'p'", self.ctrl_mode)
            self.ctrl_mode = "p"
        rospy.loginfo("Control mode: %s", self.ctrl_mode)

        # Position control parameters
        self.p_speed = rospy.get_param("~p/speed", 50)
        self.p_speed = max(0, min(100, self.p_speed))
        rospy.loginfo("Position control speed: %d", self.p_speed)

        # MIT control parameters
        self.mit_speed = rospy.get_param("~mit/speed", 50)
        self.mit_speed = max(0, min(100, self.mit_speed))

        # kp parameter: can be a single value or a list of 6 values
        kp_param = rospy.get_param("~mit/kp", 10.0)
        if isinstance(kp_param, list):
            if len(kp_param) == 6:
                self.mit_kp = list(kp_param)
            else:
                rospy.logwarn("kp list length should be 6. Using first value for all joints.")
                self.mit_kp = [kp_param[0] if len(kp_param) > 0 else 10.0] * 6
        else:
            self.mit_kp = [float(kp_param)] * 6

        # kd parameter: can be a single value or a list of 6 values
        kd_param = rospy.get_param("~mit/kd", 0.8)
        if isinstance(kd_param, list):
            if len(kd_param) == 6:
                self.mit_kd = list(kd_param)
            else:
                rospy.logwarn("kd list length should be 6. Using first value for all joints.")
                self.mit_kd = [kd_param[0] if len(kd_param) > 0 else 0.8] * 6
        else:
            self.mit_kd = [float(kd_param)] * 6

        self.mit_enable_pos = rospy.get_param("~mit/enable_pos", True)
        self.mit_enable_vel = rospy.get_param("~mit/enable_vel", False)
        self.mit_enable_tor = rospy.get_param("~mit/enable_tor", False)
        self.mit_enable_gravity = rospy.get_param("~mit/enable_gravity", False)
        self.mit_gravity_mix_mode = rospy.get_param("~mit/gravity_mix_mode", "additive")
        if self.mit_gravity_mix_mode not in ["additive", "legacy_diff", "gravity_only"]:
            rospy.logwarn(
                "Invalid ~mit/gravity_mix_mode=%s, fallback to 'additive'",
                self.mit_gravity_mix_mode,
            )
            self.mit_gravity_mix_mode = "additive"
        torque_scale_param = rospy.get_param("~mit/torque_scale", 1.0)
        if isinstance(torque_scale_param, list):
            if len(torque_scale_param) == 6:
                self.mit_torque_scale = list(torque_scale_param)
            else:
                rospy.logwarn("torque_scale list length should be 6. Using first value for all joints.")
                self.mit_torque_scale = [torque_scale_param[0] if len(torque_scale_param) > 0 else 1.0] * 6
        else:
            self.mit_torque_scale = [float(torque_scale_param)] * 6
        torque_feedback_sign_param = rospy.get_param("~mit/torque_feedback_sign", -1.0)
        if isinstance(torque_feedback_sign_param, list):
            if len(torque_feedback_sign_param) == 6:
                self.mit_torque_feedback_sign = list(torque_feedback_sign_param)
            else:
                rospy.logwarn(
                    "torque_feedback_sign list length should be 6. Using first value for all joints."
                )
                self.mit_torque_feedback_sign = [
                    torque_feedback_sign_param[0] if len(torque_feedback_sign_param) > 0 else -1.0
                ] * 6
        else:
            self.mit_torque_feedback_sign = [float(torque_feedback_sign_param)] * 6
        self.mit_max_torque_abs = float(rospy.get_param("~mit/max_torque_abs", 18.0))
        if self.mit_max_torque_abs <= 0.0:
            rospy.logwarn("~mit/max_torque_abs must be > 0. Fallback to 18.0")
            self.mit_max_torque_abs = 18.0

        # Enforce position references inside URDF joint limits to prevent runaways.
        self.enforce_joint_limits = bool(rospy.get_param("~enforce_joint_limits", True))
        self.joint_limit_log_epsilon = float(rospy.get_param("~joint_limit_log_epsilon", 1e-6))
        (
            self.joint_pos_lower_limits,
            self.joint_pos_upper_limits,
            self.joint_limits_source,
        ) = self._load_joint_position_limits()
        rospy.loginfo(
            "Joint position limit enforcement: %s (source=%s)",
            self.enforce_joint_limits,
            self.joint_limits_source,
        )
        rospy.loginfo("Joint lower limits (rad): %s", [round(v, 4) for v in self.joint_pos_lower_limits])
        rospy.loginfo("Joint upper limits (rad): %s", [round(v, 4) for v in self.joint_pos_upper_limits])
        
        #Qiu:主从切换参数 
        # Master-slave switch parameters (for slave arm to follow master arm)
        self.enable_master_slave_switch = rospy.get_param("~master_slave/enable", False)
        self.master_position_topic = rospy.get_param("~master_slave/master_position_topic", "/conrft_robot/arm_left/joint_states_single")
        self.master_flag_topic = rospy.get_param("~master_slave/master_flag_topic", "/conrft_robot/arm_left/slave_follow_flag")
        
        # kp values when in slave follow mode (higher kp to follow master)
        kp_follow_param = rospy.get_param("~master_slave/kp_follow", 1.0)
        if isinstance(kp_follow_param, list):
            if len(kp_follow_param) == 6:
                self.mit_kp_follow = list(kp_follow_param)
            else:
                self.mit_kp_follow = [kp_follow_param[0] if len(kp_follow_param) > 0 else 1.0] * 6
        else:
            self.mit_kp_follow = [float(kp_follow_param)] * 6
        
        # Store original kp for teleop mode
        self.mit_kp_teleop = self.mit_kp[:]
        
        # Current mode flag (False = teleop mode, True = slave follow mode)
        self.slave_follow_mode = False
        self.slave_follow_mode_lock = threading.Lock()
        
        # Master arm position storage
        self.master_positions_lock = threading.Lock()
        self.master_positions = [0.0] * 7
        
        if self.enable_master_slave_switch:
            rospy.loginfo("Master-slave switch enabled")
            rospy.loginfo("  Master position topic: %s", self.master_position_topic)
            rospy.loginfo("  Master flag topic: %s", self.master_flag_topic)
            rospy.loginfo("  kp for follow mode: %s", self.mit_kp_follow)

        rospy.loginfo("MIT control parameters: speed=%d", self.mit_speed)
        rospy.loginfo("MIT kp: %s", self.mit_kp)
        rospy.loginfo("MIT kd: %s", self.mit_kd)
        rospy.loginfo(
            "MIT enable flags: pos=%s, vel=%s, tor=%s, gravity=%s",
            self.mit_enable_pos,
            self.mit_enable_vel,
            self.mit_enable_tor,
            self.mit_enable_gravity,
        )
        rospy.loginfo("MIT torque scale: %s", self.mit_torque_scale)
        rospy.loginfo("MIT torque feedback sign: %s", self.mit_torque_feedback_sign)
        rospy.loginfo("MIT gravity mix mode: %s", self.mit_gravity_mix_mode)
        rospy.loginfo("MIT max torque abs: %.2f", self.mit_max_torque_abs)

        # Topic name parameters
        def ensure_topic_prefix(param_name, default_suffix):
            """Ensure topic name uses topic_prefix as prefix"""
            if rospy.has_param(param_name):
                topic = rospy.get_param(param_name)
                topic = topic.lstrip("/")
                return self.topic_prefix + topic
            else:
                return self.topic_prefix + default_suffix

        def get_remap_topic(remap_param_name, default_topic):
            """Get remapped topic name from parameter, or use default if not set.
            
            If remap parameter is set, use it directly as the full topic path.
            Otherwise, use the default topic (typically topic_prefix + suffix).
            """
            if rospy.has_param(remap_param_name):
                return rospy.get_param(remap_param_name)
            return default_topic

        # Build default topic names (topic_prefix + suffix)
        default_p_joint_pos_cmd_topic = ensure_topic_prefix("~p/joint_pos_cmd_topic", "joint_pos_cmd")
        default_mit_joint_pos_cmd_topic = ensure_topic_prefix("~mit/joint_pos_cmd_topic", "joint_pos_cmd")
        default_mit_joint_vel_cmd_topic = ensure_topic_prefix("~mit/joint_vel_cmd_topic", "joint_vel_cmd")
        default_mit_joint_tor_cmd_topic = ensure_topic_prefix("~mit/joint_tor_cmd_topic", "joint_tor_cmd")
        default_gripper_pos_cmd_topic = ensure_topic_prefix("~gripper_pos_cmd_topic", "gripper_pos_cmd")
        default_gripper_effort_cmd_topic = ensure_topic_prefix("~gripper_effort_cmd_topic", "gripper_effort_cmd")

        # Apply remap if configured (remap/xxx_to parameters override default topics)
        self.p_joint_pos_cmd_topic = get_remap_topic("~remap/joint_pos_cmd_to", default_p_joint_pos_cmd_topic)
        self.mit_joint_pos_cmd_topic = get_remap_topic("~remap/joint_pos_cmd_to", default_mit_joint_pos_cmd_topic)
        self.mit_joint_vel_cmd_topic = get_remap_topic("~remap/joint_vel_cmd_to", default_mit_joint_vel_cmd_topic)
        self.mit_joint_tor_cmd_topic = get_remap_topic("~remap/joint_tor_cmd_to", default_mit_joint_tor_cmd_topic)
        self.gripper_pos_cmd_topic = get_remap_topic("~remap/gripper_pos_cmd_to", default_gripper_pos_cmd_topic)
        self.gripper_effort_cmd_topic = get_remap_topic("~remap/gripper_effort_cmd_to", default_gripper_effort_cmd_topic)

        if self.ctrl_mode == "p":
            rospy.loginfo("Position control command topic: %s", self.p_joint_pos_cmd_topic)
        else:
            rospy.loginfo("MIT control command topics:")
            if self.mit_enable_pos:
                rospy.loginfo("  Position: %s", self.mit_joint_pos_cmd_topic)
            if self.mit_enable_vel:
                rospy.loginfo("  Velocity: %s", self.mit_joint_vel_cmd_topic)
            if self.mit_enable_tor:
                rospy.loginfo("  Torque: %s", self.mit_joint_tor_cmd_topic)

        if self.mit_enable_gravity:
            default_joint_states_compensated_topic = ensure_topic_prefix(
                "~mit/joint_states_compensated_topic", "joint_states_compensated"
            )
            self.mit_joint_states_compensated_topic = get_remap_topic(
                "~remap/joint_states_compensated_to", default_joint_states_compensated_topic
            )
            rospy.loginfo("Gravity compensation topic: %s", self.mit_joint_states_compensated_topic)

        # Log gripper topics
        if self.enable_gripper:
            rospy.loginfo("Gripper position command topic: %s", self.gripper_pos_cmd_topic)
            rospy.loginfo("Gripper effort command topic: %s", self.gripper_effort_cmd_topic)

        # Thread rate parameters
        self.publish_rate = rospy.get_param("~publish_rate", 200.0)
        self.control_rate = rospy.get_param("~control_rate", 200.0)
        self.subscribe_rate = rospy.get_param("~subscribe_rate", 100.0)
        rospy.loginfo(
            "Thread rates: publish=%.1f Hz, control=%.1f Hz, subscribe=%.1f Hz",
            self.publish_rate,
            self.control_rate,
            self.subscribe_rate,
        )

        # Filter parameters
        self.filter_enable = rospy.get_param("~filter/enable", True)
        self.filter_alpha_pos = rospy.get_param("~filter/alpha_position", 0.7)
        self.filter_alpha_vel = rospy.get_param("~filter/alpha_velocity", 0.5)
        self.filter_alpha_effort = rospy.get_param("~filter/alpha_effort", 0.5)
        # Clamp alpha values to [0, 1]
        self.filter_alpha_pos = max(0.0, min(1.0, self.filter_alpha_pos))
        self.filter_alpha_vel = max(0.0, min(1.0, self.filter_alpha_vel))
        self.filter_alpha_effort = max(0.0, min(1.0, self.filter_alpha_effort))
        rospy.loginfo(
            "Filter enabled: %s, alpha: pos=%.2f, vel=%.2f, effort=%.2f",
            self.filter_enable,
            self.filter_alpha_pos,
            self.filter_alpha_vel,
            self.filter_alpha_effort,
        )

        # Create piper interface
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort()

        if self.ctrl_mode == "mit":
            self.piper.MotionCtrl_2(0x01, 0x04, self.mit_speed, 0xAD)
        else:
            self.piper.MotionCtrl_2(0x01, 0x01, self.p_speed, 0)

        self.block_ctrl_flag = False
        self.__enable_flag = False

        # Publishers
        self.joint_pub = rospy.Publisher(self.topic_prefix + "joint_states_single", JointState, queue_size=1)
        self.arm_status_pub = rospy.Publisher(self.topic_prefix + "arm_status", PiperStatusMsg, queue_size=1)
        self.end_pose_pub = rospy.Publisher(self.topic_prefix + "end_pose", PoseStamped, queue_size=1)
        self.end_pose_euler_pub = rospy.Publisher(self.topic_prefix + "end_pose_euler", PiperEulerPose, queue_size=1)       

        # Services
        self.enable_service = rospy.Service(self.topic_prefix + "enable_srv", Enable, self.handle_enable_service)
        self.gripper_service = rospy.Service(self.topic_prefix + "gripper_srv", Gripper, self.handle_gripper_service)
        self.stop_service = rospy.Service(self.topic_prefix + "stop_srv", Trigger, self.handle_stop_service)
        self.reset_service = rospy.Service(self.topic_prefix + "reset_srv", Trigger, self.handle_reset_service)
        self.go_zero_service = rospy.Service(self.topic_prefix + "go_zero_srv", GoZero, self.handle_go_zero_service)
        self.block_arm_service = rospy.Service(self.topic_prefix + "block_arm", SetBool, self.handle_block_arm_service)

        # Joint state message
        self.joint_states = JointState()
        self.joint_states.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        self.joint_states.position = [0.0] * 7
        self.joint_states.velocity = [0.0] * 7
        self.joint_states.effort = [0.0] * 7

        # Control data storage (thread-safe)
        self.control_data_lock = threading.Lock()
        self.joint_positions_cmd = [0.0] * 7
        self.joint_velocities_cmd = [0.0] * 6
        self.joint_torques_cmd = [0.0] * 6
        self.current_joint_positions = [0.0] * 6
        self.current_joint_velocities = [0.0] * 6
        self.current_joint_efforts = [0.0] * 6

        self.gravity_torques_lock = threading.Lock()
        self.gravity_torques = [0.0] * 6

        # Gripper command data storage (thread-safe)
        self.gripper_cmd_lock = threading.Lock()
        self.gripper_cmd_position = 0.0
        self.gripper_cmd_effort = 0.0
        self.gripper_haptic_effort_filtered = 0.0
        self.gripper_cmd_position_prev = 0.0

        # Filter state storage (thread-safe)
        self.filter_lock = threading.Lock()
        self.filtered_positions = [0.0] * 7
        self.filtered_velocities = [0.0] * 7
        self.filtered_efforts = [0.0] * 7
        self.filter_initialized = False

        # Subscribers
        if self.ctrl_mode == "p":
            rospy.Subscriber(
                self.p_joint_pos_cmd_topic, JointState, self.joint_pos_cmd_callback, queue_size=1, tcp_nodelay=True
            )
            rospy.loginfo("Subscribing to position command: %s", self.p_joint_pos_cmd_topic)
        else:
            if self.mit_enable_pos:
                rospy.Subscriber(
                    self.mit_joint_pos_cmd_topic,
                    JointState,
                    self.joint_pos_cmd_callback,
                    queue_size=1,
                    tcp_nodelay=True,
                )
                rospy.loginfo("Subscribing to position command: %s", self.mit_joint_pos_cmd_topic)

            if self.mit_enable_vel:
                rospy.Subscriber(
                    self.mit_joint_vel_cmd_topic,
                    JointState,
                    self.joint_vel_cmd_callback,
                    queue_size=1,
                    tcp_nodelay=True,
                )
                rospy.loginfo("Subscribing to velocity command: %s", self.mit_joint_vel_cmd_topic)

            if self.mit_enable_tor:
                rospy.Subscriber(
                    self.mit_joint_tor_cmd_topic,
                    JointState,
                    self.joint_tor_cmd_callback,
                    queue_size=1,
                    tcp_nodelay=True,
                )
                rospy.loginfo("Subscribing to torque command: %s", self.mit_joint_tor_cmd_topic)

            if self.mit_enable_gravity:
                rospy.Subscriber(
                    self.mit_joint_states_compensated_topic,
                    JointState,
                    self.gravity_torque_callback,
                    queue_size=1,
                    tcp_nodelay=True,
                )
                rospy.loginfo("Subscribing to gravity compensation: %s", self.mit_joint_states_compensated_topic)

        rospy.Subscriber(self.topic_prefix + "enable_flag", Bool, self.enable_callback, queue_size=1, tcp_nodelay=True)

        # Gripper command subscriber
        if self.enable_gripper:
            rospy.Subscriber(
                self.gripper_pos_cmd_topic, JointState, self.gripper_pos_cmd_callback, queue_size=1, tcp_nodelay=True
            )
            rospy.loginfo("Subscribing to gripper position command: %s", self.gripper_pos_cmd_topic)

            rospy.Subscriber(
                self.gripper_effort_cmd_topic,
                JointState,
                self.gripper_effort_cmd_callback,
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.loginfo("Subscribing to gripper effort command: %s", self.gripper_effort_cmd_topic)

        # Master-slave switch subscribers
        if self.enable_master_slave_switch:
            rospy.Subscriber(
                self.master_position_topic,
                JointState,
                self.master_position_callback,
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.loginfo("Subscribing to master position: %s", self.master_position_topic)
            
            rospy.Subscriber(
                self.master_flag_topic,
                Bool,
                self.master_flag_callback,
                queue_size=1,
                tcp_nodelay=True,
            )
            rospy.loginfo("Subscribing to master flag: %s", self.master_flag_topic)

        # Thread control flags
        self.publish_thread_running = False
        self.control_thread_running = False
        self.publish_thread = None
        self.control_thread = None
        self._shutdown_disable_done = False
        self._shutdown_lock = threading.Lock()
        rospy.on_shutdown(self._on_shutdown)

    def GetEnableFlag(self):
        return self.__enable_flag

    def _load_joint_position_limits(self):
        """Load 6-DOF joint limits from params, then robot_description, else fallback."""
        fallback_lower = [-2.618, 0.0, -2.9671, -1.57, -1.57, -3.14]
        fallback_upper = [2.618, 3.14, 0.0, 1.57, 1.57, 3.14]

        def parse_limit_list(raw_value, param_name):
            if raw_value is None:
                return None
            if not isinstance(raw_value, list) or len(raw_value) != 6:
                rospy.logwarn("%s must be a list of 6 values. Got: %s", param_name, str(raw_value))
                return None
            try:
                return [float(v) for v in raw_value]
            except Exception:
                rospy.logwarn("%s contains non-numeric values: %s", param_name, str(raw_value))
                return None

        lower_from_param = parse_limit_list(rospy.get_param("~joint_pos_lower_limits", None), "~joint_pos_lower_limits")
        upper_from_param = parse_limit_list(rospy.get_param("~joint_pos_upper_limits", None), "~joint_pos_upper_limits")
        if lower_from_param is not None and upper_from_param is not None:
            return lower_from_param, upper_from_param, "node_params"
        if lower_from_param is not None or upper_from_param is not None:
            rospy.logwarn("Ignoring partial joint limit params. Both lower+upper must be provided.")

        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        urdf_xml = None
        if rospy.has_param("/robot_description"):
            urdf_xml = rospy.get_param("/robot_description")
        elif rospy.has_param("robot_description"):
            urdf_xml = rospy.get_param("robot_description")

        if urdf_xml:
            try:
                root = ET.fromstring(urdf_xml)
                joint_map = {j.get("name"): j for j in root.findall("joint")}
                lower = []
                upper = []
                for joint_name in joint_names:
                    joint_elem = joint_map.get(joint_name)
                    if joint_elem is None:
                        raise ValueError(f"joint '{joint_name}' not found in robot_description")
                    limit_elem = joint_elem.find("limit")
                    if limit_elem is None:
                        raise ValueError(f"joint '{joint_name}' has no <limit>")
                    lo = limit_elem.get("lower")
                    hi = limit_elem.get("upper")
                    if lo is None or hi is None:
                        raise ValueError(f"joint '{joint_name}' missing lower/upper limits")
                    lower.append(float(lo))
                    upper.append(float(hi))
                return lower, upper, "robot_description"
            except Exception as exc:
                rospy.logwarn("Failed to parse joint limits from robot_description: %s", str(exc))

        return fallback_lower, fallback_upper, "fallback_defaults"

    def _clamp_joint_positions(self, positions, mode_label):
        """Clamp 6-DOF position refs to configured joint limits."""
        if not self.enforce_joint_limits:
            return positions

        clamped = list(positions)
        clipped_indices = []
        for i in range(6):
            raw = float(clamped[i])
            lo = float(self.joint_pos_lower_limits[i])
            hi = float(self.joint_pos_upper_limits[i])
            clipped = max(lo, min(hi, raw))
            clamped[i] = clipped
            if abs(raw - clipped) > self.joint_limit_log_epsilon:
                clipped_indices.append(i)

        if clipped_indices:
            max_idx = max(clipped_indices, key=lambda j: abs(float(positions[j]) - float(clamped[j])))
            rospy.logwarn_throttle(
                0.5,
                "Joint pos ref clamped (%s) on joint%d: raw=%.3f clipped=%.3f lim=[%.3f, %.3f]",
                mode_label,
                max_idx + 1,
                float(positions[max_idx]),
                float(clamped[max_idx]),
                float(self.joint_pos_lower_limits[max_idx]),
                float(self.joint_pos_upper_limits[max_idx]),
            )
        return clamped

    # Callback functions
    def joint_pos_cmd_callback(self, msg: JointState):
        """Position command callback"""
        if not self.block_ctrl_flag and len(msg.position) >= 6:
            with self.control_data_lock:
                self.joint_positions_cmd[:6] = list(msg.position[:6])
                if len(msg.position) >= 7:
                    self.joint_positions_cmd[6] = msg.position[6]

    def joint_vel_cmd_callback(self, msg: JointState):
        """Velocity command callback"""
        if not self.block_ctrl_flag and len(msg.velocity) >= 6:
            with self.control_data_lock:
                self.joint_velocities_cmd = list(msg.velocity[:6])

    def joint_tor_cmd_callback(self, msg: JointState):
        """Torque command callback"""
        if not self.block_ctrl_flag and len(msg.effort) >= 6:
            with self.control_data_lock:
                self.joint_torques_cmd = list(msg.effort[:6])

    def gravity_torque_callback(self, msg: JointState):
        """Gravity compensation torque callback"""
        if len(msg.effort) >= 6:
            with self.gravity_torques_lock:
                self.gravity_torques = list(msg.effort[:6])

    def gripper_pos_cmd_callback(self, msg: JointState):
        """Gripper command callback"""
        if not self.block_ctrl_flag:
            with self.gripper_cmd_lock:
                if len(msg.position) >= 7:
                    self.gripper_cmd_position = msg.position[6]

    def gripper_effort_cmd_callback(self, msg: JointState):
        """Gripper effort command callback"""
        if not self.block_ctrl_flag:
            with self.gripper_cmd_lock:
                if len(msg.effort) >= 7:
                    self.gripper_cmd_effort = msg.effort[6]

    def enable_callback(self, enable_flag: Bool):
        """Enable callback"""
        rospy.loginfo("Received enable flag: %s", enable_flag.data)
        if enable_flag.data:
            self.__enable_flag = True
            self.piper.EnableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.__enable_flag = False
            self.piper.DisableArm(7)
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x00, 0)

    def master_position_callback(self, msg: JointState):
        """Callback for master arm position (used in slave follow mode)"""
        if len(msg.position) >= 6:
            with self.master_positions_lock:
                self.master_positions[:6] = list(msg.position[:6])
                if len(msg.position) >= 7:
                    self.master_positions[6] = msg.position[6]

    def master_flag_callback(self, msg: Bool):
        """Callback for master-slave switch flag
        
        When flag is True (1): Switch to slave follow mode
            - Use master arm position as target
            - Use higher kp to follow master
        When flag is False (0): Switch to teleop mode
            - Use self position (gravity compensation mode)
            - Use lower kp for free movement
        """
        with self.slave_follow_mode_lock:
            if msg.data != self.slave_follow_mode:
                self.slave_follow_mode = msg.data
                if self.slave_follow_mode:
                    # Switch to slave follow mode - use higher kp
                    self.mit_kp = self.mit_kp_follow[:]
                    rospy.loginfo("Switched to SLAVE FOLLOW mode, kp=%s", self.mit_kp)
                else:
                    # Switch to teleop mode - use original low kp
                    self.mit_kp = self.mit_kp_teleop[:]
                    rospy.loginfo("Switched to TELEOP mode, kp=%s", self.mit_kp)

    def _compute_gripper_haptic_effort_cmd(self):
        """Map signed measured gripper effort to a positive haptic command."""
        with self.gripper_cmd_lock:
            raw_eff = float(self.gripper_cmd_effort)
            pos_now = float(self.gripper_cmd_position)
            pos_prev = float(self.gripper_cmd_position_prev)
            self.gripper_cmd_position_prev = pos_now

        signed_eff = raw_eff * self.gripper_haptic_effort_sign
        active_eff = max(0.0, signed_eff - self.gripper_haptic_effort_bias)

        alpha = self.gripper_haptic_effort_alpha
        if alpha > 0.0:
            self.gripper_haptic_effort_filtered = (
                (1.0 - alpha) * self.gripper_haptic_effort_filtered + alpha * active_eff
            )
            active_eff = self.gripper_haptic_effort_filtered
        else:
            self.gripper_haptic_effort_filtered = active_eff

        if active_eff <= self.gripper_haptic_effort_deadband:
            cmd = 0
        else:
            span = max(1e-6, self.gripper_haptic_effort_max - self.gripper_haptic_effort_deadband)
            normalized = min(1.0, (active_eff - self.gripper_haptic_effort_deadband) / span)
            cmd = int(round(normalized * self.gripper_haptic_cmd_max))

        opening = False
        pos_delta = pos_now - pos_prev
        opening_metric = pos_delta * self.gripper_haptic_opening_sign
        if opening_metric > self.gripper_haptic_opening_direction_eps:
            opening = True
        if opening and self.gripper_haptic_release_when_opening:
            cmd = int(round(cmd * self.gripper_haptic_opening_relief))

        rospy.loginfo_throttle(
            1.0,
            "Gripper haptic raw=%.3f signed=%.3f active=%.3f dpos=%.5f open_metric=%.5f opening=%s cmd=%d",
            raw_eff,
            signed_eff,
            active_eff,
            pos_delta,
            opening_metric,
            opening,
            cmd,
        )
        return cmd

    # Control thread
    def ControlThread(self):
        """Control thread"""
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown() and self.control_thread_running:
            if not self.__enable_flag:
                rate.sleep()
                continue

            try:
                if self.ctrl_mode == "p":
                    self._position_control()
                else:
                    self._mit_control()
            except Exception as e:
                rospy.logerr("Error in control thread: %s", str(e))

            rate.sleep()

    def _position_control(self):
        """Position control implementation"""
        with self.control_data_lock:
            positions = self.joint_positions_cmd[:]
            has_gripper = len(self.joint_positions_cmd) >= 7
            gripper_pos = self.joint_positions_cmd[6] if has_gripper else 0.0

        positions = self._clamp_joint_positions(positions, "p")

        factor = 1000 * 180 / np.pi
        joint_0 = round(positions[0] * factor)
        joint_1 = round(positions[1] * factor)
        joint_2 = round(positions[2] * factor)
        joint_3 = round(positions[3] * factor)
        joint_4 = round(positions[4] * factor)
        joint_5 = round(positions[5] * factor)

        # self.piper.MotionCtrl_2(0x01, 0x01, self.p_speed, 0)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

        if self.enable_gripper and self.gripper_exist:
            if has_gripper:
                joint_6 = round(gripper_pos * 1000 * 1000 * self.gripper_val_mutiple)
            else:
                # Use gripper command if available
                with self.gripper_cmd_lock:
                    joint_6 = round(self.gripper_cmd_position * 1000 * 1000 * self.gripper_val_mutiple)

            joint_6 = max(0, min(80000, joint_6))
            if abs(joint_6) < 200:
                joint_6 = 0

            # Get effort from gripper command if haptic is enabled
            if self.enable_gripper_haptic:
                gripper_effort = self._compute_gripper_haptic_effort_cmd()
                gripper_code = 0x00 if gripper_effort < self.gripper_haptic_cmd_enable_threshold else 0x01
                if gripper_code == 0x00:
                    gripper_effort = 0
            else:
                gripper_effort = round(1000 * 1.0)
                gripper_code = 0x01

            self.piper.GripperCtrl(abs(joint_6), gripper_effort, gripper_code, 0)

    def _mit_control(self):
        """MIT control implementation"""
        # Check if in slave follow mode (master arm leads, slave arm follows)
        # Also get current kp values under lock for thread safety
        with self.slave_follow_mode_lock:
            is_slave_follow = self.slave_follow_mode and self.enable_master_slave_switch
            current_kp = self.mit_kp[:]  # Copy kp values under lock
        
        # Get master gripper position if in slave follow mode
        master_gripper_pos = 0.0

        position_ref_source = "current"
        with self.control_data_lock:
            if is_slave_follow:
                # In slave follow mode: use master arm position as target
                with self.master_positions_lock:
                    master_gripper_pos = self.master_positions[6] if len(self.master_positions) > 6 else 0.0
                    positions = self.master_positions[:6]
                    position_ref_source = "master"
                    
            elif self.mit_enable_pos:
                positions = self.joint_positions_cmd[:]
                position_ref_source = "cmd"
            else:
                positions = self.current_joint_positions[:]
                position_ref_source = "current"

            if self.mit_enable_vel:
                velocities = self.joint_velocities_cmd[:]
            else:
                velocities = [0.0] * 6

            if self.mit_enable_tor:
                torques = self.joint_torques_cmd[:]
            else:
                torques = [0.0] * 6

        gravity_torques = [0.0] * 6
        if self.mit_enable_gravity:
            with self.gravity_torques_lock:
                gravity_torques = self.gravity_torques[:]

            for i in range(6):
                torque_cmd = torques[i] * self.mit_torque_feedback_sign[i]
                if self.mit_gravity_mix_mode == "legacy_diff":
                    torques[i] = -(torque_cmd - gravity_torques[i]) * self.mit_torque_scale[i] + gravity_torques[i]
                elif self.mit_gravity_mix_mode == "gravity_only":
                    torques[i] = gravity_torques[i]
                else:
                    torques[i] = torque_cmd * self.mit_torque_scale[i] + gravity_torques[i]
        else:
            for i in range(6):
                torque_cmd = torques[i] * self.mit_torque_feedback_sign[i]
                torques[i] = torque_cmd * self.mit_torque_scale[i]

        torques_pre_clamp = torques[:]
        # Only clamp external references. When using current joint state as ref
        # (mit_enable_pos:=false), clamping can create artificial position error
        # near model limits and feel like unexpected stiffness.
        if position_ref_source != "current":
            positions = self._clamp_joint_positions(positions, "mit/" + position_ref_source)
        for i in range(6):
            velocities[i] = max(-45.0, min(45.0, velocities[i]))
            torques[i] = max(-self.mit_max_torque_abs, min(self.mit_max_torque_abs, torques[i]))

        clipped_indices = [i for i in range(6) if abs(torques_pre_clamp[i] - torques[i]) > 1e-6]
        if clipped_indices:
            max_idx = max(clipped_indices, key=lambda j: abs(torques_pre_clamp[j] - torques[j]))
            rospy.logwarn_throttle(
                0.5,
                "MIT torque clamped on joint%d: raw=%.3f clamped=%.3f gravity=%.3f pos_ref=%.3f",
                max_idx + 1,
                float(torques_pre_clamp[max_idx]),
                float(torques[max_idx]),
                float(gravity_torques[max_idx]),
                float(positions[max_idx]),
            )

        try:
            
            for motor_num in range(1, 7):
                joint_idx = motor_num - 1
                self.piper.JointMitCtrl(
                    motor_num=motor_num,
                    pos_ref=positions[joint_idx],
                    vel_ref=velocities[joint_idx],
                    kp=current_kp[joint_idx],
                    kd=self.mit_kd[joint_idx],
                    t_ref=torques[joint_idx],
                )
            
        except Exception as e:
            rospy.logerr("Error in JointMitCtrl: %s", str(e))

        # Gripper control for MIT mode
        if self.enable_gripper and self.gripper_exist:
            # Get gripper position: use master gripper in slave follow mode, otherwise use command
            if is_slave_follow:
                gripper_pos = master_gripper_pos
            else:
                with self.gripper_cmd_lock:
                    gripper_pos = self.gripper_cmd_position

            joint_6 = round(gripper_pos * 1000 * 1000 * self.gripper_val_mutiple)
            joint_6 = max(0, min(80000, joint_6))
            if abs(joint_6) < 200:
                joint_6 = 0

            # Get effort from gripper command if haptic is enabled
            if self.enable_gripper_haptic:
                gripper_effort = self._compute_gripper_haptic_effort_cmd()
                gripper_code = 0x00 if gripper_effort < self.gripper_haptic_cmd_enable_threshold else 0x01
                if gripper_code == 0x00:
                    gripper_effort = 0
            else:
                gripper_effort = round(3 * 1000)
                gripper_code = 0x01

            self.piper.GripperCtrl(abs(joint_6), gripper_effort, gripper_code, 0)

    # Publish thread
    def PublishThread(self):
        """Publish thread"""
        rate = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown() and self.publish_thread_running:
            self.PublishArmState()
            self.PublishArmEndPose()
            self.PublishArmJointAndGripper()
            rate.sleep()

    def PublishArmState(self):
        """Publish arm status"""
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        )
        arm_status.communication_status_joint_2 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        )
        arm_status.communication_status_joint_3 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        )
        arm_status.communication_status_joint_4 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        )
        arm_status.communication_status_joint_5 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        )
        arm_status.communication_status_joint_6 = (
            self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        )
        self.arm_status_pub.publish(arm_status)

    def PublishArmJointAndGripper(self):
        """Publish joint states and gripper state"""
        joint_0: float = (self.piper.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        vel_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000
        vel_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000
        vel_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000
        vel_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000
        vel_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000
        vel_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000
        effort_0: float = self.piper.GetArmHighSpdInfoMsgs().motor_1.effort / 1000
        effort_1: float = self.piper.GetArmHighSpdInfoMsgs().motor_2.effort / 1000
        effort_2: float = self.piper.GetArmHighSpdInfoMsgs().motor_3.effort / 1000
        effort_3: float = self.piper.GetArmHighSpdInfoMsgs().motor_4.effort / 1000
        effort_4: float = self.piper.GetArmHighSpdInfoMsgs().motor_5.effort / 1000
        effort_5: float = self.piper.GetArmHighSpdInfoMsgs().motor_6.effort / 1000

        with self.control_data_lock:
            self.current_joint_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
            self.current_joint_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5]
            self.current_joint_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5]

        # Get gripper data
        if self.enable_gripper and self.gripper_exist:
            joint_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
            if self.gripper_reverse:
                joint_6 = self.gripper_range - joint_6
            effort_6: float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort / 1000
            raw_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
            raw_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]
            raw_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5, effort_6]
        else:
            raw_positions = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
            raw_velocities = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5]
            raw_efforts = [effort_0, effort_1, effort_2, effort_3, effort_4, effort_5]

        # Apply filter if enabled
        if self.filter_enable:
            with self.filter_lock:
                if not self.filter_initialized:
                    # Initialize filter with current values
                    self.filtered_positions = raw_positions[:]
                    self.filtered_velocities = raw_velocities[:]
                    self.filtered_efforts = raw_efforts[:]
                    self.filter_initialized = True
                else:
                    # Apply low-pass filter (exponential moving average)
                    # filtered = alpha * new + (1 - alpha) * previous
                    for i in range(len(raw_positions)):
                        self.filtered_positions[i] = (
                            self.filter_alpha_pos * raw_positions[i]
                            + (1.0 - self.filter_alpha_pos) * self.filtered_positions[i]
                        )
                    for i in range(len(raw_velocities)):
                        self.filtered_velocities[i] = (
                            self.filter_alpha_vel * raw_velocities[i]
                            + (1.0 - self.filter_alpha_vel) * self.filtered_velocities[i]
                        )
                    for i in range(len(raw_efforts)):
                        self.filtered_efforts[i] = (
                            self.filter_alpha_effort * raw_efforts[i]
                            + (1.0 - self.filter_alpha_effort) * self.filtered_efforts[i]
                        )
                # Use filtered values
                filtered_positions = self.filtered_positions[:]
                filtered_velocities = self.filtered_velocities[:]
                filtered_efforts = self.filtered_efforts[:]
        else:
            # Use raw values without filtering
            filtered_positions = raw_positions[:]
            filtered_velocities = raw_velocities[:]
            filtered_efforts = raw_efforts[:]

        self.joint_states.header.stamp = rospy.Time.now()
        self.joint_states.position = filtered_positions
        self.joint_states.velocity = filtered_velocities
        self.joint_states.effort = filtered_efforts

        self.joint_pub.publish(self.joint_states)

    def PublishArmEndPose(self):
        """Publish end effector pose"""
        endpos = PoseStamped()
        endpos.pose.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        endpos.pose.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        endpos.pose.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis / 1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis / 1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis / 1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        endpos.header.stamp = rospy.Time.now()
        self.end_pose_pub.publish(endpos)

        end_pose_euler = PiperEulerPose()
        end_pose_euler.header.stamp = rospy.Time.now()
        end_pose_euler.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis / 1000000
        end_pose_euler.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis / 1000000
        end_pose_euler.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis / 1000000
        end_pose_euler.roll = roll
        end_pose_euler.pitch = pitch
        end_pose_euler.yaw = yaw
        self.end_pose_euler_pub.publish(end_pose_euler)


        #   # 计算 delta
        # if self.prev_end_pose is not None:
        #     delta = PoseStamped()
        #     delta.header.stamp = rospy.Time.now()
        #     delta.header.frame_id = ""  # 可选，保持和 end_pose 一致

        #     # 位置 delta
        #     delta.pose.position.x = endpos.pose.position.x - self.prev_end_pose.pose.position.x
        #     delta.pose.position.y = endpos.pose.position.y - self.prev_end_pose.pose.position.y
        #     delta.pose.position.z = endpos.pose.position.z - self.prev_end_pose.pose.position.z

        #     # 四元数 delta (近似差)
        #     # 可以用简单相减（小角度假设），或者更精确的 quaternion 差法
        #     delta.pose.orientation.x = endpos.pose.orientation.x - self.prev_end_pose.pose.orientation.x
        #     delta.pose.orientation.y = endpos.pose.orientation.y - self.prev_end_pose.pose.orientation.y
        #     delta.pose.orientation.z = endpos.pose.orientation.z - self.prev_end_pose.pose.orientation.z
        #     delta.pose.orientation.w = endpos.pose.orientation.w - self.prev_end_pose.pose.orientation.w

        #     # 发布 delta
        #     self.end_pose_delta_pub.publish(delta)
        #         # 保存当前 end_pose 作为下一次 delta 基准
        # self.prev_end_pose = endpos



        #     # ------------------- 计算欧拉增量 -------------------
        # if self.prev_end_pose_euler is not None:
        #     delta = PiperEulerPose()
        #     delta.header.stamp = rospy.Time.now()
        #     delta.x = end_pose_euler.x - self.prev_end_pose_euler.x
        #     delta.y = end_pose_euler.y - self.prev_end_pose_euler.y
        #     delta.z = end_pose_euler.z - self.prev_end_pose_euler.z
        #     delta.roll = end_pose_euler.roll - self.prev_end_pose_euler.roll
        #     delta.pitch = end_pose_euler.pitch - self.prev_end_pose_euler.pitch
        #     delta.yaw = end_pose_euler.yaw - self.prev_end_pose_euler.yaw

        #     self.end_pose_euler_delta_pub.publish(delta)

        # # 更新上一帧
        # self.prev_end_pose_euler = end_pose_euler


    # Service handlers
    def handle_gripper_service(self, req):
        response = GripperResponse()
        response.code = 15999
        response.status = False
        if self.gripper_exist:
            rospy.loginfo("-----------------------Gripper---------------------------")
            rospy.loginfo("Received request:")
            rospy.loginfo("PS: Piper should be enable. Please ensure piper is enable")
            rospy.loginfo("gripper_angle:%f, range is [0m, 0.07m]", req.gripper_angle)
            rospy.loginfo("gripper_effort:%f, range is [0.5N/m, 2N/m]", req.gripper_effort)
            rospy.loginfo(
                "gripper_code:%d, range is [0, 1, 2, 3]\n \
                0x00: Disable\n \
                0x01: Enable\n \
                0x03/0x02: Enable and clear error / Disable and clear error",
                req.gripper_code,
            )
            rospy.loginfo(
                "set_zero:%d, range is [0, 0xAE] \n \
                0x00: Invalid value \n \
                0xAE: Set zero point",
                req.set_zero,
            )
            rospy.loginfo("-----------------------Gripper---------------------------")
            gripper_angle = round(max(0, min(req.gripper_angle, 0.07)) * 1e6)
            gripper_effort = round(max(0.5, min(req.gripper_effort, 2)) * 1e3)
            if req.gripper_code not in [0x00, 0x01, 0x02, 0x03]:
                rospy.logwarn("gripper_code should be in [0, 1, 2, 3], default val is 1")
                gripper_code = 1
                response.code = 15901
            else:
                gripper_code = req.gripper_code
            if req.set_zero not in [0x00, 0xAE]:
                rospy.logwarn("set_zero should be in [0, 0xAE], default val is 0")
                set_zero = 0
                response.code = 15902
            else:
                set_zero = req.set_zero
            response.code = 15900
            self.piper.GripperCtrl(abs(gripper_angle), gripper_effort, gripper_code, set_zero)
            response.status = True
        else:
            rospy.logwarn("gripper_exist param is False.")
            response.code = 15903
            response.status = False
        rospy.loginfo("Returning GripperResponse: %d, %s", response.code, response.status)
        return response

    def handle_enable_service(self, req):
        rospy.loginfo("Received request: %s", req.enable_request)
        enable_flag = False
        loop_flag = False
        timeout = 5
        start_time = time.time()
        elapsed_time_flag = False
        while not loop_flag:
            elapsed_time = time.time() - start_time
            rospy.loginfo("--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if req.enable_request:
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
            else:
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x02, 0)
            rospy.loginfo("Enable status: %s", enable_flag)
            self.__enable_flag = enable_flag
            rospy.loginfo("--------------------")
            if enable_flag == req.enable_request:
                loop_flag = True
                enable_flag = True
            else:
                loop_flag = False
                enable_flag = False
            if elapsed_time > timeout:
                rospy.logwarn("Timeout...")
                elapsed_time_flag = True
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        response = enable_flag
        rospy.loginfo("Returning response: %s", response)
        return EnableResponse(response)

    def handle_stop_service(self, req):
        response = TriggerResponse()
        response.success = False
        response.message = "stop piper failed"
        rospy.loginfo("-----------------------STOP---------------------------")
        rospy.loginfo("Stop piper.")
        rospy.loginfo("-----------------------STOP---------------------------")
        self.piper.MotionCtrl_1(0x01, 0, 0)
        response.success = True
        response.message = "stop piper success"
        rospy.loginfo("Returning StopResponse: %s, %s", response.success, response.message)
        return response

    def handle_reset_service(self, req):
        response = TriggerResponse()
        response.success = False
        response.message = "reset piper failed"
        rospy.loginfo("-----------------------RESET---------------------------")
        rospy.loginfo("reset piper.")
        rospy.loginfo("-----------------------RESET---------------------------")
        self.piper.MotionCtrl_1(0x02, 0, 0)
        response.success = True
        response.message = "reset piper success"
        rospy.loginfo("Returning resetResponse: %s, %s", response.success, response.message)
        return response

    def handle_go_zero_service(self, req):
        response = GoZeroResponse()
        response.status = False
        response.code = 151000
        rospy.loginfo("-----------------------GOZERO---------------------------")
        rospy.loginfo("piper go zero.")
        rospy.loginfo("-----------------------GOZERO---------------------------")
        if req.is_mit_mode:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0xAD)
        else:
            self.piper.MotionCtrl_2(0x01, 0x01, 50, 0)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        response.status = True
        response.code = 151001
        rospy.loginfo("Returning GoZeroResponse: %s, %d", response.status, response.code)
        return response

    def handle_block_arm_service(self, req):
        response = SetBoolResponse()
        rospy.loginfo("-----------------------BLOCK_ARM---------------------------")
        if req.data:
            response.success = req.data
            response.message = "You will block arm ctrl msg send"
        else:
            response.success = req.data
            response.message = "You will unblock arm ctrl msg send"
        self.block_ctrl_flag = req.data
        rospy.loginfo("piper block arm.")
        rospy.loginfo("Returning BlockArmResponse: %s, %s", response.success, response.message)
        rospy.loginfo("-----------------------BLOCK_ARM---------------------------")
        return response

    # Enable function
    def EnableArm(self):
        """Enable arm"""
        enable_flag = False
        timeout = 5
        start_time = time.time()

        while not enable_flag:
            elapsed_time = time.time() - start_time
            rospy.loginfo("--------------------[Arm Enable]--------------------")
            enable_flag = (
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
                and self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            )
            rospy.loginfo("Enable status: %s", enable_flag)
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            if enable_flag:
                self.__enable_flag = True
            rospy.loginfo("--------------------[Arm Enable]--------------------")

            if elapsed_time > timeout:
                rospy.logwarn("Enable timeout")
                break
            time.sleep(1)

    # Run function
    def Run(self):
        """Run node"""
        if self.auto_enable:
            enable_thread = threading.Thread(target=self.EnableArm, daemon=True)
            enable_thread.start()

        self.publish_thread_running = True
        self.publish_thread = threading.Thread(target=self.PublishThread, daemon=True)
        self.publish_thread.start()
        rospy.loginfo("Publish thread started")

        self.control_thread_running = True
        self.control_thread = threading.Thread(target=self.ControlThread, daemon=True)
        self.control_thread.start()
        rospy.loginfo("Control thread started")

        rospy.spin()

    def _disable_arm_on_shutdown(self):
        """Best-effort arm disable for safety during node shutdown."""
        for idx in range(self.shutdown_disable_retries):
            try:
                self.piper.DisableArm(7)
                if self.gripper_exist:
                    self.piper.GripperCtrl(0, 1000, 0x02, 0)
            except Exception as exc:
                rospy.logwarn(
                    "Shutdown disable attempt %d/%d failed: %s",
                    idx + 1,
                    self.shutdown_disable_retries,
                    str(exc),
                )
            if idx + 1 < self.shutdown_disable_retries:
                time.sleep(self.shutdown_disable_interval_sec)
        self.__enable_flag = False

    def _hold_position_on_shutdown(self):
        """Best-effort hold current joint position during node shutdown."""
        with self.control_data_lock:
            hold_positions = self.current_joint_positions[:]
        if len(hold_positions) < 6:
            hold_positions = [0.0] * 6
        hold_positions = self._clamp_joint_positions(hold_positions, "shutdown_hold")

        factor = 1000 * 180 / np.pi
        joint_targets = [round(hold_positions[i] * factor) for i in range(6)]

        joint_6 = None
        if self.enable_gripper and self.gripper_exist:
            with self.gripper_cmd_lock:
                gripper_pos = self.gripper_cmd_position
            gripper_pos = max(0.0, min(self.gripper_range, gripper_pos))
            joint_6 = round(gripper_pos * 1000 * 1000 * self.gripper_val_mutiple)
            joint_6 = max(0, min(80000, joint_6))

        for idx in range(self.shutdown_hold_retries):
            try:
                self.piper.MotionCtrl_2(0x01, 0x01, self.p_speed, 0)
                self.piper.JointCtrl(
                    joint_targets[0],
                    joint_targets[1],
                    joint_targets[2],
                    joint_targets[3],
                    joint_targets[4],
                    joint_targets[5],
                )
                if joint_6 is not None:
                    self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
            except Exception as exc:
                rospy.logwarn(
                    "Shutdown hold attempt %d/%d failed: %s",
                    idx + 1,
                    self.shutdown_hold_retries,
                    str(exc),
                )
            if idx + 1 < self.shutdown_hold_retries:
                time.sleep(self.shutdown_hold_interval_sec)

    def _on_shutdown(self):
        """ROS shutdown callback."""
        with self._shutdown_lock:
            if self._shutdown_disable_done:
                return
            self._shutdown_disable_done = True
        self.publish_thread_running = False
        self.control_thread_running = False
        if self.shutdown_mode == "none":
            rospy.loginfo("Shutdown: mode=none, skip shutdown safety action.")
            return
        if self.shutdown_mode == "hold":
            rospy.logwarn("Shutdown: mode=hold, locking current arm position.")
            self._hold_position_on_shutdown()
            return
        rospy.logwarn("Shutdown: mode=disable, auto disabling arm for safety.")
        self._disable_arm_on_shutdown()


if __name__ == "__main__":
    try:
        check_ros_master()
        node = C_PiperCtrlNode()
        node.Run()
    except rospy.ROSInterruptException:
        pass
