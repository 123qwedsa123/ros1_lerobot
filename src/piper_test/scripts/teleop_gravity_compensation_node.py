#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gravity compensation torque calculation node for dual-arm robot.
Subscribes to joint angles from both arms, computes and publishes gravity compensation torques.
Gripper is included in calculation but output contains only first 6 joints' torques.
"""

import os
import sys
import glob
import subprocess
import rospy
import numpy as np
from sensor_msgs.msg import JointState


def _ensure_pinocchio_on_syspath():
    if 'pinocchio' in sys.modules:
        return
    ros_distro = os.environ.get("ROS_DISTRO", "noetic").strip() or "noetic"
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        f"/opt/ros/{ros_distro}/lib/python{py_ver}/site-packages",
        f"/opt/ros/{ros_distro}/lib/python{py_ver}/dist-packages",
    ]
    candidates.extend(sorted(glob.glob("/opt/ros/*/lib/python*/site-packages")))
    candidates.extend(sorted(glob.glob("/opt/ros/*/lib/python*/dist-packages")))
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_ensure_pinocchio_on_syspath()
import pinocchio as pin


class GravityCompensationArm:
    """Gravity compensation node for a single arm"""

    def __init__(self, arm_side: str = "left"):
        """
        Initialize gravity compensation node

        Args:
            arm_side: Arm side, 'left' or 'right'
        """
        if arm_side not in ["left", "right"]:
            raise ValueError("arm_side must be 'left' or 'right'")

        self.arm_side = arm_side
        self.urdf_package = rospy.get_param("~urdf_package", "piper_x_description")
        self.urdf_relpath = rospy.get_param("~urdf_relpath", "urdf/piper_x_description.urdf")
        self.clip_to_limits = bool(rospy.get_param("~clip_to_limits", True))
        self.limit_warn_tol = float(rospy.get_param("~limit_warn_tol", 1e-4))
        self.limit_push_warn_margin = float(rospy.get_param("~limit_push_warn_margin", 0.15))
        self.unwrap_joint_positions = bool(rospy.get_param("~unwrap_joint_positions", True))
        self.max_joint_step = float(rospy.get_param("~max_joint_step", 0.0))
        self.max_torque_delta_warn = float(rospy.get_param("~max_torque_delta_warn", 4.0))
        self.last_model_joint_positions = None
        self.last_compensated_torques = None

        sign_param_name = f"~gravity_joint_sign_{arm_side}"
        raw_sign = rospy.get_param(sign_param_name, None)
        if raw_sign is None:
            sign_param_name = "~gravity_joint_sign"
            raw_sign = rospy.get_param(sign_param_name, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        if not isinstance(raw_sign, list) or len(raw_sign) != 6:
            rospy.logwarn(
                "[%s] Invalid %s=%s, expected list of 6. Using [1,1,1,1,1,1].",
                self.arm_side,
                sign_param_name,
                str(raw_sign),
            )
            raw_sign = [1.0] * 6
        self.gravity_joint_sign = np.array([float(v) for v in raw_sign], dtype=float)
        self.gravity_sign_param_name = sign_param_name

        pos_scale_param_name = f"~gravity_joint_position_scale_{arm_side}"
        raw_pos_scale = rospy.get_param(pos_scale_param_name, None)
        if raw_pos_scale is None:
            pos_scale_param_name = "~gravity_joint_position_scale"
            raw_pos_scale = rospy.get_param(pos_scale_param_name, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        if not isinstance(raw_pos_scale, list) or len(raw_pos_scale) != 6:
            rospy.logwarn(
                "[%s] Invalid %s=%s, expected list of 6. Using [1,1,1,1,1,1].",
                self.arm_side,
                pos_scale_param_name,
                str(raw_pos_scale),
            )
            raw_pos_scale = [1.0] * 6
        self.gravity_joint_position_scale = np.array([float(v) for v in raw_pos_scale], dtype=float)
        self.gravity_pos_scale_param_name = pos_scale_param_name

        offset_param_name = f"~gravity_joint_offset_{arm_side}"
        raw_offset = rospy.get_param(offset_param_name, None)
        if raw_offset is None:
            offset_param_name = "~gravity_joint_offset"
            raw_offset = rospy.get_param(offset_param_name, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not isinstance(raw_offset, list) or len(raw_offset) != 6:
            rospy.logwarn(
                "[%s] Invalid %s=%s, expected list of 6. Using [0,0,0,0,0,0].",
                self.arm_side,
                offset_param_name,
                str(raw_offset),
            )
            raw_offset = [0.0] * 6
        self.gravity_joint_offset = np.array([float(v) for v in raw_offset], dtype=float)
        self.gravity_offset_param_name = offset_param_name

        scale_param_name = f"~gravity_joint_scale_{arm_side}"
        raw_scale = rospy.get_param(scale_param_name, None)
        if raw_scale is None:
            scale_param_name = "~gravity_joint_scale"
            raw_scale = rospy.get_param(scale_param_name, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        if not isinstance(raw_scale, list) or len(raw_scale) != 6:
            rospy.logwarn(
                "[%s] Invalid %s=%s, expected list of 6. Using [1,1,1,1,1,1].",
                self.arm_side,
                scale_param_name,
                str(raw_scale),
            )
            raw_scale = [1.0] * 6
        self.gravity_joint_scale = np.array([float(v) for v in raw_scale], dtype=float)
        self.gravity_scale_param_name = scale_param_name

        # Get URDF file path
        try:
            package_path = (
                subprocess.check_output(["rospack", "find", self.urdf_package], stderr=subprocess.STDOUT)
                .strip()
                .decode("utf-8")
            )
            urdf_path = os.path.join(package_path, self.urdf_relpath)
            urdf_path = os.path.abspath(urdf_path)
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"URDF not found: {urdf_path}")

            rospy.loginfo(
                "[%s] URDF package/path: %s/%s", self.arm_side, self.urdf_package, self.urdf_relpath
            )
            rospy.loginfo("[%s] URDF absolute path: %s", self.arm_side, urdf_path)
            processed_urdf_path = urdf_path

        except Exception as e:
            rospy.logerr("[%s] Failed to process URDF file: %s", self.arm_side, str(e))
            raise

        # Load robot model
        try:
            # Load dynamics model only. This avoids visual mesh dependency and keeps
            # gravity compensation robust even when mesh packages are not in ROS_PACKAGE_PATH.
            self.model = pin.buildModelFromUrdf(processed_urdf_path)
            rospy.loginfo("[%s] Robot model loaded successfully", self.arm_side)
            rospy.loginfo("[%s] Number of joints: %d", self.arm_side, self.model.nq)
            self.lower_limits = np.array(self.model.lowerPositionLimit, dtype=float)
            self.upper_limits = np.array(self.model.upperPositionLimit, dtype=float)
        except Exception as e:
            rospy.logerr("[%s] Failed to load robot model: %s", self.arm_side, str(e))
            raise

        # Use full model (including gripper) for calculation
        # Gripper gravity affects first 6 joints' torques, but output contains only first 6 joints
        rospy.loginfo("[%s] Using full robot model (including gripper) for gravity compensation", self.arm_side)

        # Create data object
        self.data = self.model.createData()

        # Subscribe to joint states
        self.joint_state_sub = rospy.Subscriber(
            f"/robot/arm_{arm_side}/joint_states_single", JointState, self.joint_state_callback, queue_size=1
        )

        # Publish gravity compensation torques
        self.torque_pub = rospy.Publisher(f"/robot/arm_{arm_side}/joint_states_compensated", JointState, queue_size=1)

        # Joint names (6 joints + gripper)
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

        rospy.loginfo("[%s] Gravity compensation node initialized", self.arm_side)
        rospy.loginfo("[%s] Subscribing to: /robot/arm_%s/joint_states_single", self.arm_side, arm_side)
        rospy.loginfo("[%s] Publishing to: /robot/arm_%s/joint_states_compensated", self.arm_side, arm_side)
        rospy.loginfo(
            "[%s] Gravity joint scale from %s: %s",
            self.arm_side,
            self.gravity_scale_param_name,
            self.gravity_joint_scale.tolist(),
        )
        rospy.loginfo(
            "[%s] Gravity joint sign from %s: %s",
            self.arm_side,
            self.gravity_sign_param_name,
            self.gravity_joint_sign.tolist(),
        )
        rospy.loginfo(
            "[%s] Gravity joint position scale from %s: %s",
            self.arm_side,
            self.gravity_pos_scale_param_name,
            self.gravity_joint_position_scale.tolist(),
        )
        rospy.loginfo(
            "[%s] Gravity joint offset from %s: %s",
            self.arm_side,
            self.gravity_offset_param_name,
            self.gravity_joint_offset.tolist(),
        )
        rospy.loginfo("[%s] Clip to URDF limits: %s", self.arm_side, self.clip_to_limits)
        rospy.loginfo("[%s] Limit push warn margin (rad): %.4f", self.arm_side, self.limit_push_warn_margin)
        rospy.loginfo("[%s] Unwrap joint positions: %s", self.arm_side, self.unwrap_joint_positions)
        rospy.loginfo("[%s] Max joint step (rad/cycle): %.4f", self.arm_side, self.max_joint_step)
        rospy.loginfo("[%s] Torque jump warn threshold (Nm): %.4f", self.arm_side, self.max_torque_delta_warn)

    def joint_state_callback(self, msg):
        """Joint state callback: compute and publish gravity compensation torques"""
        try:
            if len(msg.position) < 6:
                rospy.logwarn("[%s] Received joint state with less than 6 joints", self.arm_side)
                return

            # Get all joint angles (including gripper)
            if len(msg.position) >= 7:
                joint_positions = np.array(msg.position[:7])
            else:
                joint_positions = np.array(list(msg.position[:6]) + [0.0])

            # Map encoder-space joint values into URDF joint-space
            mapped_joint_positions = joint_positions.copy()
            mapped_joint_positions[:6] = (
                self.gravity_joint_sign * (self.gravity_joint_position_scale * mapped_joint_positions[:6])
                + self.gravity_joint_offset
            )
            # Keep angle branch continuous across +/-pi to avoid sudden gravity jumps
            if self.unwrap_joint_positions and self.last_model_joint_positions is not None:
                prev_q = self.last_model_joint_positions
                wrapped = prev_q + (mapped_joint_positions[:6] - prev_q + np.pi) % (2.0 * np.pi) - np.pi
                if np.any(np.abs(wrapped - mapped_joint_positions[:6]) > 1e-6):
                    rospy.logwarn_throttle(
                        1.0,
                        "[%s] Applied angle unwrap. raw6=%s mapped6_before=%s mapped6_after=%s",
                        self.arm_side,
                        np.array2string(np.array(msg.position[:6]), precision=3, suppress_small=True),
                        np.array2string(np.array(mapped_joint_positions[:6]), precision=3, suppress_small=True),
                        np.array2string(np.array(wrapped), precision=3, suppress_small=True),
                    )
                mapped_joint_positions[:6] = wrapped

            # Optional per-cycle step clamp to suppress sensor spikes/runaway near wraps
            if self.max_joint_step > 0.0 and self.last_model_joint_positions is not None:
                prev_q = self.last_model_joint_positions
                step = mapped_joint_positions[:6] - prev_q
                step_clamped = np.clip(step, -self.max_joint_step, self.max_joint_step)
                if np.any(np.abs(step - step_clamped) > 1e-9):
                    max_idx = int(np.argmax(np.abs(step - step_clamped)))
                    rospy.logwarn_throttle(
                        1.0,
                        "[%s] Joint step clamped on joint%d. step=%.3f -> %.3f rad",
                        self.arm_side,
                        max_idx + 1,
                        float(step[max_idx]),
                        float(step_clamped[max_idx]),
                    )
                mapped_joint_positions[:6] = prev_q + step_clamped

            joint_positions = mapped_joint_positions

            # Pad or truncate to match model joint count
            nq_model = self.model.nq
            if len(joint_positions) < nq_model:
                joint_positions = np.append(joint_positions, [0.0] * (nq_model - len(joint_positions)))
            elif len(joint_positions) > nq_model:
                joint_positions = joint_positions[:nq_model]

            # Clamp to URDF limits to avoid runaway when raw joints exceed model domain
            if self.clip_to_limits:
                unclipped = joint_positions.copy()
                joint_positions = np.clip(joint_positions, self.lower_limits, self.upper_limits)
                clipped_idx = np.where(np.abs(unclipped - joint_positions) > self.limit_warn_tol)[0]
                clipped_arm_idx = [int(i) for i in clipped_idx if int(i) < 6]
                if clipped_arm_idx:
                    rospy.logwarn_throttle(
                        1.0,
                        "[%s] Clipped ARM joints to URDF limits idx=%s. raw6=%s mapped6=%s clipped6=%s",
                        self.arm_side,
                        str([i + 1 for i in clipped_arm_idx]),
                        np.array2string(np.array(msg.position[:6]), precision=3, suppress_small=True),
                        np.array2string(np.array(unclipped[:6]), precision=3, suppress_small=True),
                        np.array2string(np.array(joint_positions[:6]), precision=3, suppress_small=True),
                    )
                elif clipped_idx.size > 0:
                    rospy.logdebug_throttle(
                        2.0,
                        "[%s] Only non-arm joints clipped idx=%s",
                        self.arm_side,
                        str([int(i) + 1 for i in clipped_idx]),
                    )

            # Compute gravity compensation torques
            gravity_torques = pin.computeGeneralizedGravity(self.model, self.data, joint_positions)

            # Extract first 6 joints' torques
            gravity_torques_6dof = gravity_torques[:6]

            # Per-joint scaling for fast on-robot tuning
            compensated_torques = gravity_torques_6dof * self.gravity_joint_scale

            # Diagnostic: detect torque that pushes further into nearby joint limits.
            if self.limit_push_warn_margin > 0.0:
                for i in range(6):
                    lo = float(self.lower_limits[i])
                    hi = float(self.upper_limits[i])
                    q_i = float(joint_positions[i])
                    tau_i = float(compensated_torques[i])
                    near_upper = q_i > (hi - self.limit_push_warn_margin)
                    near_lower = q_i < (lo + self.limit_push_warn_margin)
                    pushing_upper = tau_i > 0.0
                    pushing_lower = tau_i < 0.0
                    if (near_upper and pushing_upper) or (near_lower and pushing_lower):
                        target = "upper" if near_upper else "lower"
                        rospy.logwarn_throttle(
                            0.5,
                            "[%s] Gravity torque pushes toward %s limit on joint%d: q=%.3f lim=[%.3f, %.3f] tau=%.3f",
                            self.arm_side,
                            target,
                            i + 1,
                            q_i,
                            lo,
                            hi,
                            tau_i,
                        )

            if self.max_torque_delta_warn > 0.0 and self.last_compensated_torques is not None:
                torque_delta = compensated_torques - self.last_compensated_torques
                max_idx = int(np.argmax(np.abs(torque_delta)))
                if abs(float(torque_delta[max_idx])) > self.max_torque_delta_warn:
                    rospy.logwarn_throttle(
                        0.5,
                        "[%s] Torque jump on joint%d: d_tau=%.3f prev=%.3f now=%.3f raw_q=%.3f mapped_q=%.3f used_q=%.3f",
                        self.arm_side,
                        max_idx + 1,
                        float(torque_delta[max_idx]),
                        float(self.last_compensated_torques[max_idx]),
                        float(compensated_torques[max_idx]),
                        float(msg.position[max_idx]),
                        float(mapped_joint_positions[max_idx]),
                        float(joint_positions[max_idx]),
                    )

            self.last_model_joint_positions = np.array(joint_positions[:6], dtype=float)
            self.last_compensated_torques = np.array(compensated_torques, dtype=float)

            # Create output message
            output_msg = JointState()
            output_msg.header.stamp = rospy.Time.now()
            output_msg.header.frame_id = msg.header.frame_id if msg.header.frame_id else ""

            # Set joint names
            if len(msg.name) >= 7:
                output_msg.name = list(msg.name[:7])
            else:
                output_msg.name = self.joint_names.copy()

            # Set positions
            if len(msg.position) >= 7:
                output_msg.position = list(msg.position[:7])
            else:
                output_msg.position = list(msg.position[:6]) + [0.0]

            # Set velocities
            if len(msg.velocity) >= 7:
                output_msg.velocity = list(msg.velocity[:7])
            elif len(msg.velocity) >= 6:
                output_msg.velocity = list(msg.velocity[:6]) + [0.0]
            else:
                output_msg.velocity = [0.0] * 7

            # Set torques (first 6 joints compensated, gripper = 0)
            output_msg.effort = list(compensated_torques) + [0.0]

            # Publish message
            self.torque_pub.publish(output_msg)

        except Exception as e:
            rospy.logerr("[%s] Error in joint_state_callback: %s", self.arm_side, str(e))


def check_ros_master():
    """Check if ROS master is running"""
    import rosnode

    try:
        rosnode.rosnode_ping("rosout", max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")


def main():
    """Main function: create gravity compensation nodes for both arms"""
    try:
        check_ros_master()

        rospy.init_node("piper_gravity_compensation_node", anonymous=False)

        rospy.loginfo("Creating gravity compensation nodes for both arms...")
        arm_left = GravityCompensationArm(arm_side="left")
        arm_right = GravityCompensationArm(arm_side="right")

        rospy.loginfo("Gravity compensation nodes initialized successfully")
        rospy.loginfo("Gripper gravity included in calculation")
        rospy.loginfo("Gravity joint scale default: [1,1,1,1,1,1] (override with ~gravity_joint_scale)")

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Node failed: %s", str(e))


if __name__ == "__main__":
    main()
