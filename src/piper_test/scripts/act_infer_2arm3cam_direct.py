#!/usr/bin/env python3
"""Run ACT checkpoint online inference and publish two-arm joint commands.

This node is publish-only by design:
- It does NOT call mux services.
- It does NOT change teleop mode.
- It does NOT publish slave_follow_flag.

Launch usage remains unchanged:
  roslaunch piper_test act_infer_2arm3cam_direct.launch
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, JointState

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("opencv-python is required in this environment.") from exc

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("torch is required in this environment.") from exc

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
    try:
        from lerobot.processor.pipeline import ProcessorMigrationError
    except Exception:  # pragma: no cover
        class ProcessorMigrationError(RuntimeError):
            pass
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "lerobot is required in this environment. Please use the inference venv in ROS container."
    ) from exc


DEFAULT_JOINT_NAMES = "joint1,joint2,joint3,joint4,joint5,joint6,gripper"
DEFAULT_FEATURES = {
    "state": "observation.state",
    "image_left": "observation.image_left",
    "image_right": "observation.image_right",
    "image_top": "observation.image_top",
}


@dataclass(frozen=True)
class GuardProfile:
    ema_alpha: float
    max_joint_step: float
    max_gripper_step: float
    gripper_min: float
    gripper_max: float


GUARD_PRESETS: Dict[str, GuardProfile] = {
    "conservative": GuardProfile(
        ema_alpha=0.25,
        max_joint_step=0.03,
        max_gripper_step=0.003,
        gripper_min=0.0,
        gripper_max=0.08,
    ),
    "medium": GuardProfile(
        ema_alpha=0.40,
        max_joint_step=0.05,
        max_gripper_step=0.005,
        gripper_min=0.0,
        gripper_max=0.08,
    ),
    "aggressive": GuardProfile(
        ema_alpha=0.60,
        max_joint_step=0.08,
        max_gripper_step=0.01,
        gripper_min=0.0,
        gripper_max=0.08,
    ),
}


@dataclass
class Snapshot:
    left_pos: np.ndarray
    left_effort: np.ndarray
    right_pos: np.ndarray
    right_effort: np.ndarray
    top_image: np.ndarray
    left_image: np.ndarray
    right_image: np.ndarray
    stamps: Tuple[float, float, float, float, float]


class PolicyPublisherNode:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        self.left_pos: Optional[np.ndarray] = None
        self.left_effort: Optional[np.ndarray] = None
        self.left_stamp: Optional[float] = None
        self.right_pos: Optional[np.ndarray] = None
        self.right_effort: Optional[np.ndarray] = None
        self.right_stamp: Optional[float] = None

        self.top_image: Optional[np.ndarray] = None
        self.top_stamp: Optional[float] = None
        self.left_image: Optional[np.ndarray] = None
        self.left_img_stamp: Optional[float] = None
        self.right_image: Optional[np.ndarray] = None
        self.right_img_stamp: Optional[float] = None

        self.prev_left_cmd: Optional[np.ndarray] = None
        self.prev_right_cmd: Optional[np.ndarray] = None
        self.publish_count = 0

        self.cb_count = {
            "left_joint": 0,
            "right_joint": 0,
            "top_image": 0,
            "left_wrist_image": 0,
            "right_wrist_image": 0,
        }
        self.cb_raw_count = {
            "left_joint": 0,
            "right_joint": 0,
            "top_image": 0,
            "left_wrist_image": 0,
            "right_wrist_image": 0,
        }
        self.cb_drop_count = {
            "left_joint_short": 0,
            "right_joint_short": 0,
            "left_effort_missing": 0,
            "right_effort_missing": 0,
            "top_decode_fail": 0,
            "left_decode_fail": 0,
            "right_decode_fail": 0,
        }

        self.csv_fp: Optional[TextIO] = None
        self.csv_writer: Optional[csv.writer] = None
        self.csv_last_flush_sec = 0.0
        self.csv_step = 0

        self._load_runtime_config()
        self.policy, self.preprocessor, self.postprocessor = self._load_policy_and_processors()
        self._open_output_csv()

        self.pub_left = rospy.Publisher(self.out_left_topic, JointState, queue_size=1)
        self.pub_right = rospy.Publisher(self.out_right_topic, JointState, queue_size=1)

        self.subscribers: List[Any] = [
            rospy.Subscriber(
                self.left_state_topic, JointState, self._cb_left_joint, queue_size=1, tcp_nodelay=True
            ),
            rospy.Subscriber(
                self.right_state_topic, JointState, self._cb_right_joint, queue_size=1, tcp_nodelay=True
            ),
        ]
        self._subscribe_image_stream("top", self.top_image_topic, self._cb_top_image_raw, self._cb_top_image_compressed)
        self._subscribe_image_stream("left", self.left_image_topic, self._cb_left_image_raw, self._cb_left_image_compressed)
        self._subscribe_image_stream("right", self.right_image_topic, self._cb_right_image_raw, self._cb_right_image_compressed)

    def _load_runtime_config(self) -> None:
        model_cfg = dict(self.cfg.get("model", {}))
        runtime_cfg = dict(self.cfg.get("runtime", {}))
        topics_cfg = dict(self.cfg.get("topics", {}))
        ctrl_cfg = dict(self.cfg.get("control_msg", {}))
        feat_cfg = dict(self.cfg.get("features", {}))

        self.checkpoint_dir = str(model_cfg.get("checkpoint_dir", "")).strip()
        self.device = self._resolve_device(str(model_cfg.get("device", "cuda")))
        self.temporal_ensemble_coeff = float(model_cfg.get("temporal_ensemble_coeff", 0.01))
        self.action_dim = int(model_cfg.get("action_dim", 14))
        self.auto_migrate_processors = bool(model_cfg.get("auto_migrate_processors", True))
        self.migration_timeout_sec = float(model_cfg.get("migration_timeout_sec", 180.0))

        self.rate_hz = float(runtime_cfg.get("rate_hz", 20.0))
        self.image_width = int(runtime_cfg.get("image_width", 256))
        self.image_height = int(runtime_cfg.get("image_height", 256))
        self.max_input_staleness_sec = float(runtime_cfg.get("max_input_staleness_sec", 0.30))
        self.max_sync_delta_sec = float(runtime_cfg.get("max_sync_delta_sec", 0.08))
        self.startup_grace_sec = float(runtime_cfg.get("startup_grace_sec", 1.0))
        self.debug_streams = bool(runtime_cfg.get("debug_streams", False))
        self.enable_gripper = bool(runtime_cfg.get("enable_gripper", True))
        self.output_csv_path = str(
            runtime_cfg.get("output_csv_path", runtime_cfg.get("model_output_csv_path", ""))
        ).strip()
        self.output_csv_flush_hz = float(
            runtime_cfg.get("output_csv_flush_hz", runtime_cfg.get("model_output_csv_flush_hz", 5.0))
        )

        guard_name = str(runtime_cfg.get("guard_profile", "medium")).strip().lower()
        if guard_name not in GUARD_PRESETS:
            raise RuntimeError(f"unsupported guard_profile={guard_name}, valid={sorted(GUARD_PRESETS.keys())}")
        self.guard = GUARD_PRESETS[guard_name]

        self.left_state_topic = str(topics_cfg.get("left_state", "/slave1/joint_states_single"))
        self.right_state_topic = str(topics_cfg.get("right_state", "/slave2/joint_states_single"))

        left_raw = str(topics_cfg.get("left_image", "/cam_left/color/image_raw"))
        right_raw = str(topics_cfg.get("right_image", "/cam_right/color/image_raw"))
        top_raw = str(topics_cfg.get("top_image", topics_cfg.get("middle_image", "/cam_top/color/image_raw")))

        left_cmp = str(topics_cfg.get("left_image_compressed", "")).strip()
        right_cmp = str(topics_cfg.get("right_image_compressed", "")).strip()
        top_cmp = str(topics_cfg.get("top_image_compressed", "")).strip()

        self.left_image_topic = left_cmp if left_cmp else left_raw
        self.right_image_topic = right_cmp if right_cmp else right_raw
        self.top_image_topic = top_cmp if top_cmp else top_raw

        self.out_left_topic = str(topics_cfg.get("left_cmd", "/slave1/joint_states"))
        self.out_right_topic = str(topics_cfg.get("right_cmd", "/slave2/joint_states"))

        names_raw = ctrl_cfg.get("joint_names", DEFAULT_JOINT_NAMES)
        if isinstance(names_raw, (list, tuple)):
            self.joint_names = [str(x).strip() for x in names_raw if str(x).strip()]
        else:
            self.joint_names = [x.strip() for x in str(names_raw).split(",") if x.strip()]
        if len(self.joint_names) != 7:
            raise RuntimeError(f"joint_names must contain exactly 7 names, got {len(self.joint_names)}")

        self.velocity_default = float(ctrl_cfg.get("joint_velocity_default", 0.0))
        self.effort_default = float(ctrl_cfg.get("joint_effort_default", 0.0))
        self.gripper_velocity = float(ctrl_cfg.get("gripper_velocity", 100.0))
        self.gripper_effort = float(ctrl_cfg.get("gripper_effort", 1.0))

        self.features = dict(DEFAULT_FEATURES)
        for key in DEFAULT_FEATURES:
            if key in feat_cfg and str(feat_cfg[key]).strip():
                self.features[key] = str(feat_cfg[key]).strip()
        if (not str(feat_cfg.get("image_top", "")).strip()) and str(feat_cfg.get("image_middle", "")).strip():
            self.features["image_top"] = str(feat_cfg["image_middle"]).strip()

        if not self.checkpoint_dir:
            raise RuntimeError("model.checkpoint_dir must be set")
        if self.rate_hz <= 0:
            raise RuntimeError("runtime.rate_hz must be > 0")
        if self.output_csv_flush_hz <= 0:
            raise RuntimeError("runtime.output_csv_flush_hz must be > 0")
        if self.migration_timeout_sec <= 0:
            raise RuntimeError("model.migration_timeout_sec must be > 0")

    def _resolve_device(self, requested: str) -> str:
        req = requested.strip().lower()
        if req.startswith("cuda") and not torch.cuda.is_available():
            rospy.logwarn("Requested device=%s but CUDA is unavailable. Falling back to cpu.", requested)
            return "cpu"
        return requested

    def _load_policy_and_processors(self):
        ckpt = Path(self.checkpoint_dir).expanduser().resolve()
        if not ckpt.is_dir():
            raise RuntimeError(f"Checkpoint dir does not exist: {ckpt}")
        if not (ckpt / "config.json").is_file() or not (ckpt / "model.safetensors").is_file():
            raise RuntimeError(f"Checkpoint dir must contain config.json and model.safetensors: {ckpt}")

        attempted_auto_migration = False
        if self.auto_migrate_processors and not self._has_processor_configs(ckpt):
            self._auto_migrate_processors(ckpt, "missing processor config files")
            attempted_auto_migration = True

        cli_overrides = [
            "--n_action_steps", "1",
            "--temporal_ensemble_coeff", str(self.temporal_ensemble_coeff),
            "--device", self.device,
        ]

        rospy.loginfo("Loading ACT policy from: %s", ckpt)
        policy = ACTPolicy.from_pretrained(str(ckpt), local_files_only=True, cli_overrides=cli_overrides)
        try:
            preprocessor, postprocessor = self._build_processors(policy, ckpt)
        except Exception as exc:
            can_retry = (
                self.auto_migrate_processors
                and not attempted_auto_migration
                and self._is_missing_processor_error(exc)
            )
            if not can_retry:
                raise
            self._auto_migrate_processors(ckpt, f"processor load failed: {type(exc).__name__}: {exc}")
            preprocessor, postprocessor = self._build_processors(policy, ckpt)

        rospy.loginfo(
            "Policy loaded. device=%s temporal_ensemble_coeff=%.4f n_action_steps=%d",
            policy.config.device,
            policy.config.temporal_ensemble_coeff or 0.0,
            policy.config.n_action_steps,
        )
        rospy.loginfo("Policy input features: %s", list(policy.config.input_features.keys()))
        return policy, preprocessor, postprocessor

    def _build_processors(self, policy: ACTPolicy, ckpt: Path):
        return make_pre_post_processors(
            policy.config,
            pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def _has_processor_configs(self, ckpt: Path) -> bool:
        cfg_paths = [
            ckpt / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            ckpt / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        ]
        for cfg_path in cfg_paths:
            if not cfg_path.is_file():
                return False
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict):
                    return False
                steps = cfg.get("steps", [])
                if not isinstance(steps, list):
                    return False
                for step in steps:
                    if not isinstance(step, dict):
                        return False
                    state_file = step.get("state_file")
                    if state_file and not (ckpt / str(state_file)).is_file():
                        return False
            except Exception:
                return False
        return True

    def _is_missing_processor_error(self, exc: Exception) -> bool:
        if isinstance(exc, ProcessorMigrationError):
            return True
        msg = str(exc)
        return (
            f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json" in msg
            or f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json" in msg
        )

    def _auto_migrate_processors(self, ckpt: Path, reason: str) -> None:
        if self._has_processor_configs(ckpt):
            return

        cmd = [
            sys.executable,
            "-m",
            "lerobot.processor.migrate_policy_normalization",
            "--pretrained-path",
            str(ckpt),
            "--output-dir",
            str(ckpt),
        ]
        rospy.logwarn("Checkpoint missing deploy processors (%s). Running auto migration: %s", reason, ckpt)
        rospy.loginfo("Migration command: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.migration_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Auto migration timeout after {self.migration_timeout_sec:.1f}s: {ckpt}"
            ) from exc

        merged_output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            tail = "\n".join(merged_output.splitlines()[-40:])
            raise RuntimeError(f"Auto migration failed (exit={proc.returncode}) for {ckpt}\n{tail}")

        if not self._has_processor_configs(ckpt):
            tail = "\n".join(merged_output.splitlines()[-40:])
            raise RuntimeError(
                f"Auto migration finished but processor files still missing in {ckpt}\n{tail}"
            )
        rospy.loginfo("Auto migration done. Processor configs generated at: %s", ckpt)

    def _open_output_csv(self) -> None:
        if not self.output_csv_path:
            return

        out_path = Path(self.output_csv_path).expanduser()
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = out_path.is_file() and out_path.stat().st_size > 0
        self.csv_fp = open(out_path, "a", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.csv_fp)
        if not file_exists:
            header = (
                [
                    "step",
                    "stamp_sec",
                    "left_state_stamp_sec",
                    "right_state_stamp_sec",
                    "top_img_stamp_sec",
                    "left_img_stamp_sec",
                    "right_img_stamp_sec",
                ]
                + [f"action_{i}" for i in range(self.action_dim)]
                + [f"left_cmd_{i}" for i in range(7)]
                + [f"right_cmd_{i}" for i in range(7)]
            )
            self.csv_writer.writerow(header)
            self.csv_fp.flush()
        self.csv_last_flush_sec = rospy.Time.now().to_sec()
        rospy.loginfo("Model output CSV enabled: %s", str(out_path))

    def _close_output_csv(self) -> None:
        if self.csv_fp is None:
            return
        try:
            self.csv_fp.flush()
            self.csv_fp.close()
        finally:
            self.csv_fp = None
            self.csv_writer = None

    def _write_output_csv(
        self,
        stamp_sec: float,
        snap: Snapshot,
        action: np.ndarray,
        left_cmd: np.ndarray,
        right_cmd: np.ndarray,
    ) -> None:
        if self.csv_writer is None:
            return
        try:
            row = [
                int(self.csv_step),
                float(stamp_sec),
                float(snap.stamps[0]),
                float(snap.stamps[1]),
                float(snap.stamps[2]),
                float(snap.stamps[3]),
                float(snap.stamps[4]),
            ]
            row.extend(float(v) for v in action.tolist())
            row.extend(float(v) for v in left_cmd.tolist())
            row.extend(float(v) for v in right_cmd.tolist())
            self.csv_writer.writerow(row)
            self.csv_step += 1
            if (stamp_sec - self.csv_last_flush_sec) >= (1.0 / self.output_csv_flush_hz):
                self.csv_fp.flush()
                self.csv_last_flush_sec = stamp_sec
        except Exception as exc:
            rospy.logerr_throttle(2.0, "CSV write failed: %s", str(exc))

    def _is_compressed_topic(self, topic: str) -> bool:
        return str(topic).strip().endswith("/compressed")

    def _subscribe_image_stream(self, name: str, topic: str, raw_cb, compressed_cb) -> None:
        if self._is_compressed_topic(topic):
            self.subscribers.append(
                rospy.Subscriber(topic, CompressedImage, compressed_cb, queue_size=1, tcp_nodelay=True)
            )
            rospy.loginfo("Subscribe %s image topic=%s type=CompressedImage", name, topic)
        else:
            self.subscribers.append(
                rospy.Subscriber(topic, Image, raw_cb, queue_size=1, tcp_nodelay=True)
            )
            rospy.loginfo("Subscribe %s image topic=%s type=Image", name, topic)

    def _stamp_to_sec(self, stamp: rospy.Time) -> float:
        if stamp is not None and hasattr(stamp, "to_sec"):
            sec = float(stamp.to_sec())
            if sec > 0.0:
                return sec
        return rospy.Time.now().to_sec()

    def _decode_bgr_to_chw(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None:
            raise RuntimeError("image is None")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise RuntimeError(f"unsupported image shape: {img_bgr.shape}")

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != self.image_width or rgb.shape[0] != self.image_height:
            rgb = cv2.resize(rgb, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        return np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0

    def _decode_compressed_to_chw(self, msg: CompressedImage) -> Optional[np.ndarray]:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if arr.size == 0:
            return None
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"unsupported decoded compressed image shape: {img.shape}")

        if rgb.shape[1] != self.image_width or rgb.shape[0] != self.image_height:
            rgb = cv2.resize(rgb, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        return np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0

    def _cb_left_joint(self, msg: JointState) -> None:
        with self.lock:
            self.cb_raw_count["left_joint"] += 1
        if len(msg.position) < 7:
            with self.lock:
                self.cb_drop_count["left_joint_short"] += 1
            rospy.logwarn_throttle(2.0, "Left joint_states_single short dims: pos=%d", len(msg.position))
            return

        if len(msg.effort) >= 7:
            effort = np.asarray(msg.effort[:7], dtype=np.float32)
        else:
            effort = np.zeros(7, dtype=np.float32)
            with self.lock:
                self.cb_drop_count["left_effort_missing"] += 1
            rospy.logwarn_throttle(5.0, "Left joint effort missing (<7); filling zeros.")

        with self.lock:
            self.left_pos = np.asarray(msg.position[:7], dtype=np.float32)
            self.left_effort = effort
            self.left_stamp = self._stamp_to_sec(msg.header.stamp)
            self.cb_count["left_joint"] += 1

    def _cb_right_joint(self, msg: JointState) -> None:
        with self.lock:
            self.cb_raw_count["right_joint"] += 1
        if len(msg.position) < 7:
            with self.lock:
                self.cb_drop_count["right_joint_short"] += 1
            rospy.logwarn_throttle(2.0, "Right joint_states_single short dims: pos=%d", len(msg.position))
            return

        if len(msg.effort) >= 7:
            effort = np.asarray(msg.effort[:7], dtype=np.float32)
        else:
            effort = np.zeros(7, dtype=np.float32)
            with self.lock:
                self.cb_drop_count["right_effort_missing"] += 1
            rospy.logwarn_throttle(5.0, "Right joint effort missing (<7); filling zeros.")

        with self.lock:
            self.right_pos = np.asarray(msg.position[:7], dtype=np.float32)
            self.right_effort = effort
            self.right_stamp = self._stamp_to_sec(msg.header.stamp)
            self.cb_count["right_joint"] += 1

    def _cb_top_image_raw(self, msg: Image) -> None:
        try:
            with self.lock:
                self.cb_raw_count["top_image"] += 1
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            chw = self._decode_bgr_to_chw(bgr)
            with self.lock:
                self.top_image = chw
                self.top_stamp = self._stamp_to_sec(msg.header.stamp)
                self.cb_count["top_image"] += 1
        except Exception as exc:
            with self.lock:
                self.cb_drop_count["top_decode_fail"] += 1
            rospy.logerr_throttle(2.0, "Top image callback exception: %s", str(exc))

    def _cb_left_image_raw(self, msg: Image) -> None:
        try:
            with self.lock:
                self.cb_raw_count["left_wrist_image"] += 1
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            chw = self._decode_bgr_to_chw(bgr)
            with self.lock:
                self.left_image = chw
                self.left_img_stamp = self._stamp_to_sec(msg.header.stamp)
                self.cb_count["left_wrist_image"] += 1
        except Exception as exc:
            with self.lock:
                self.cb_drop_count["left_decode_fail"] += 1
            rospy.logerr_throttle(2.0, "Left image callback exception: %s", str(exc))

    def _cb_right_image_raw(self, msg: Image) -> None:
        try:
            with self.lock:
                self.cb_raw_count["right_wrist_image"] += 1
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            chw = self._decode_bgr_to_chw(bgr)
            with self.lock:
                self.right_image = chw
                self.right_img_stamp = self._stamp_to_sec(msg.header.stamp)
                self.cb_count["right_wrist_image"] += 1
        except Exception as exc:
            with self.lock:
                self.cb_drop_count["right_decode_fail"] += 1
            rospy.logerr_throttle(2.0, "Right image callback exception: %s", str(exc))

    def _cb_top_image_compressed(self, msg: CompressedImage) -> None:
        try:
            with self.lock:
                self.cb_raw_count["top_image"] += 1
            chw = self._decode_compressed_to_chw(msg)
            if chw is None:
                with self.lock:
                    self.cb_drop_count["top_decode_fail"] += 1
                rospy.logwarn_throttle(2.0, "Failed to decode top compressed image.")
                return
            with self.lock:
                self.top_image = chw
                self.top_stamp = self._stamp_to_sec(msg.header.stamp)
                self.cb_count["top_image"] += 1
        except Exception as exc:
            with self.lock:
                self.cb_drop_count["top_decode_fail"] += 1
            rospy.logerr_throttle(2.0, "Top compressed image callback exception: %s", str(exc))

    def _cb_left_image_compressed(self, msg: CompressedImage) -> None:
        try:
            with self.lock:
                self.cb_raw_count["left_wrist_image"] += 1
            chw = self._decode_compressed_to_chw(msg)
            if chw is None:
                with self.lock:
                    self.cb_drop_count["left_decode_fail"] += 1
                rospy.logwarn_throttle(2.0, "Failed to decode left compressed image.")
                return
            with self.lock:
                self.left_image = chw
                self.left_img_stamp = self._stamp_to_sec(msg.header.stamp)
                self.cb_count["left_wrist_image"] += 1
        except Exception as exc:
            with self.lock:
                self.cb_drop_count["left_decode_fail"] += 1
            rospy.logerr_throttle(2.0, "Left compressed image callback exception: %s", str(exc))

    def _cb_right_image_compressed(self, msg: CompressedImage) -> None:
        try:
            with self.lock:
                self.cb_raw_count["right_wrist_image"] += 1
            chw = self._decode_compressed_to_chw(msg)
            if chw is None:
                with self.lock:
                    self.cb_drop_count["right_decode_fail"] += 1
                rospy.logwarn_throttle(2.0, "Failed to decode right compressed image.")
                return
            with self.lock:
                self.right_image = chw
                self.right_img_stamp = self._stamp_to_sec(msg.header.stamp)
                self.cb_count["right_wrist_image"] += 1
        except Exception as exc:
            with self.lock:
                self.cb_drop_count["right_decode_fail"] += 1
            rospy.logerr_throttle(2.0, "Right compressed image callback exception: %s", str(exc))

    def _missing_streams_unlocked(self) -> List[str]:
        missing: List[str] = []
        if self.left_pos is None or self.left_effort is None or self.left_stamp is None:
            missing.append("left_joint")
        if self.right_pos is None or self.right_effort is None or self.right_stamp is None:
            missing.append("right_joint")
        if self.top_image is None or self.top_stamp is None:
            missing.append("top_image")
        if self.left_image is None or self.left_img_stamp is None:
            missing.append("left_wrist_image")
        if self.right_image is None or self.right_img_stamp is None:
            missing.append("right_wrist_image")
        return missing

    def _missing_streams(self) -> List[str]:
        with self.lock:
            return self._missing_streams_unlocked()

    def _snapshot(self) -> Optional[Snapshot]:
        with self.lock:
            if self._missing_streams_unlocked():
                return None
            return Snapshot(
                left_pos=self.left_pos.copy(),
                left_effort=self.left_effort.copy(),
                right_pos=self.right_pos.copy(),
                right_effort=self.right_effort.copy(),
                top_image=self.top_image.copy(),
                left_image=self.left_image.copy(),
                right_image=self.right_image.copy(),
                stamps=(
                    float(self.left_stamp),
                    float(self.right_stamp),
                    float(self.top_stamp),
                    float(self.left_img_stamp),
                    float(self.right_img_stamp),
                ),
            )

    def _debug_streams(self) -> None:
        with self.lock:
            rospy.loginfo_throttle(
                2.0,
                "debug streams raw=%s valid=%s drop=%s stamps(left,right,top,lw,rw)=(%s,%s,%s,%s,%s)",
                self.cb_raw_count,
                self.cb_count,
                self.cb_drop_count,
                "None" if self.left_stamp is None else f"{self.left_stamp:.3f}",
                "None" if self.right_stamp is None else f"{self.right_stamp:.3f}",
                "None" if self.top_stamp is None else f"{self.top_stamp:.3f}",
                "None" if self.left_img_stamp is None else f"{self.left_img_stamp:.3f}",
                "None" if self.right_img_stamp is None else f"{self.right_img_stamp:.3f}",
            )

    def _health_ok(self, snap: Snapshot, now_sec: float) -> Tuple[bool, str]:
        for ts in snap.stamps:
            if now_sec - ts > self.max_input_staleness_sec:
                return False, f"stale input (>{self.max_input_staleness_sec:.3f}s)"
        if max(snap.stamps) - min(snap.stamps) > self.max_sync_delta_sec:
            return False, f"sync delta too large (>{self.max_sync_delta_sec:.3f}s)"
        return True, ""

    def _infer_action(self, snap: Snapshot) -> np.ndarray:
        obs_state = np.concatenate(
            [snap.left_pos, snap.left_effort, snap.right_pos, snap.right_effort], axis=0
        ).astype(np.float32)
        state_1x = obs_state.reshape(1, 28)
        top_1x = snap.top_image.reshape(1, 3, self.image_height, self.image_width)
        left_1x = snap.left_image.reshape(1, 3, self.image_height, self.image_width)
        right_1x = snap.right_image.reshape(1, 3, self.image_height, self.image_width)

        obs = {
            self.features["state"]: state_1x,
            self.features["image_top"]: top_1x,
            self.features["image_left"]: left_1x,
            self.features["image_right"]: right_1x,
        }

        required_keys = set(self.policy.config.input_features.keys())
        alias_candidates = {
            "observation.state": state_1x,
            "observation.image": top_1x,
            "observation.image_top": top_1x,
            "observation.left_wrist_image": left_1x,
            "observation.image_left": left_1x,
            "observation.right_wrist_image": right_1x,
            "observation.image_right": right_1x,
        }
        inserted_aliases: List[str] = []
        for key in required_keys:
            if key not in obs and key in alias_candidates:
                obs[key] = alias_candidates[key]
                inserted_aliases.append(key)
        if inserted_aliases:
            rospy.logwarn_throttle(
                5.0,
                "Auto-filled obs keys for policy compatibility: %s",
                ",".join(sorted(inserted_aliases)),
            )

        processed = self.preprocessor(obs)
        model_inputs: Dict[str, Any] = {}
        for key in self.policy.config.input_features.keys():
            if key not in processed:
                raise RuntimeError(f"Preprocessor output missing required key: {key}")
            value = processed[key]
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(self.policy.config.device, non_blocking=True)
            else:
                model_inputs[key] = value

        with torch.inference_mode():
            action = self.policy.select_action(model_inputs)
        action = self.postprocessor(action)

        action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        action_np = action_np.astype(np.float32).reshape(-1)
        if action_np.size != self.action_dim:
            raise RuntimeError(f"Policy output must be {self.action_dim}-D, got {action_np.size}")
        if not np.all(np.isfinite(action_np)):
            raise RuntimeError("Policy output contains non-finite values")
        return action_np

    def _guard_single(self, raw: np.ndarray, prev: np.ndarray) -> np.ndarray:
        target = raw.copy()
        target[6] = np.clip(target[6], self.guard.gripper_min, self.guard.gripper_max)

        delta = target - prev
        delta[:6] = np.clip(delta[:6], -self.guard.max_joint_step, self.guard.max_joint_step)
        delta[6] = float(np.clip(delta[6], -self.guard.max_gripper_step, self.guard.max_gripper_step))
        stepped = prev + delta

        cmd = self.guard.ema_alpha * stepped + (1.0 - self.guard.ema_alpha) * prev
        cmd[6] = np.clip(cmd[6], self.guard.gripper_min, self.guard.gripper_max)
        return cmd.astype(np.float32)

    def _apply_guard(self, raw_left: np.ndarray, raw_right: np.ndarray, snap: Snapshot) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enable_gripper:
            raw_left = raw_left.copy()
            raw_right = raw_right.copy()
            raw_left[6] = snap.left_pos[6]
            raw_right[6] = snap.right_pos[6]

        if self.prev_left_cmd is None:
            self.prev_left_cmd = snap.left_pos.copy()
        if self.prev_right_cmd is None:
            self.prev_right_cmd = snap.right_pos.copy()

        left = self._guard_single(raw_left, self.prev_left_cmd)
        right = self._guard_single(raw_right, self.prev_right_cmd)
        self.prev_left_cmd = left
        self.prev_right_cmd = right
        return left, right

    def _publish_joint(self, pub: rospy.Publisher, position: np.ndarray, stamp: rospy.Time) -> None:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(self.joint_names)
        msg.position = [float(v) for v in position]
        msg.velocity = [self.velocity_default] * 6 + [self.gripper_velocity]
        msg.effort = [self.effort_default] * 6 + [self.gripper_effort]
        pub.publish(msg)

    def run(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("act_infer_2arm3cam_direct started (reference-like auto inference mode)")
        rospy.loginfo("publish-only mode: no mux/follow/teleop switching")
        rospy.loginfo(
            "Input topics: left_state=%s right_state=%s top_image=%s left_image=%s right_image=%s",
            self.left_state_topic,
            self.right_state_topic,
            self.top_image_topic,
            self.left_image_topic,
            self.right_image_topic,
        )
        rospy.loginfo("Output topics: left=%s right=%s", self.out_left_topic, self.out_right_topic)
        if self.startup_grace_sec > 0.0:
            rospy.loginfo("Startup grace: %.2f s", self.startup_grace_sec)
            rospy.sleep(self.startup_grace_sec)

        try:
            while not rospy.is_shutdown():
                snap = self._snapshot()
                if snap is None:
                    missing = ",".join(self._missing_streams())
                    rospy.loginfo_throttle(2.0, "Waiting for required streams: %s", missing)
                    if self.debug_streams:
                        self._debug_streams()
                    rate.sleep()
                    continue

                now_sec = rospy.Time.now().to_sec()
                healthy, reason = self._health_ok(snap, now_sec)
                if not healthy:
                    rospy.logwarn_throttle(2.0, "Input unhealthy, skip publish: %s", reason)
                    rate.sleep()
                    continue

                try:
                    action = self._infer_action(snap)
                except Exception as exc:
                    rospy.logerr_throttle(2.0, "Inference failed: %s", str(exc))
                    rate.sleep()
                    continue

                left_raw = action[:7]
                right_raw = action[7:14]
                left_cmd, right_cmd = self._apply_guard(left_raw, right_raw, snap)

                stamp = rospy.Time.now()
                stamp_sec = float(stamp.to_sec())
                self._publish_joint(self.pub_left, left_cmd, stamp)
                self._publish_joint(self.pub_right, right_cmd, stamp)
                self._write_output_csv(stamp_sec, snap, action, left_cmd, right_cmd)

                self.publish_count += 1
                if self.debug_streams:
                    rospy.loginfo_throttle(2.0, "Published command frames: %d", self.publish_count)
                rate.sleep()
        finally:
            self._close_output_csv()


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"bad config yaml: {path}")
    return cfg


def _set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.checkpoint_dir:
        _set_nested(cfg, "model.checkpoint_dir", args.checkpoint_dir)
    if args.device:
        _set_nested(cfg, "model.device", args.device)
    if args.rate_hz is not None:
        _set_nested(cfg, "runtime.rate_hz", args.rate_hz)
    if args.temporal_ensemble_coeff is not None:
        _set_nested(cfg, "model.temporal_ensemble_coeff", args.temporal_ensemble_coeff)
    if args.guard_profile:
        _set_nested(cfg, "runtime.guard_profile", args.guard_profile)
    if args.left_state_topic:
        _set_nested(cfg, "topics.left_state", args.left_state_topic)
    if args.right_state_topic:
        _set_nested(cfg, "topics.right_state", args.right_state_topic)
    if args.left_image_topic:
        _set_nested(cfg, "topics.left_image", args.left_image_topic)
    if args.right_image_topic:
        _set_nested(cfg, "topics.right_image", args.right_image_topic)
    if args.top_image_topic:
        _set_nested(cfg, "topics.top_image", args.top_image_topic)
    if args.left_image_compressed_topic:
        _set_nested(cfg, "topics.left_image_compressed", args.left_image_compressed_topic)
    if args.right_image_compressed_topic:
        _set_nested(cfg, "topics.right_image_compressed", args.right_image_compressed_topic)
    if args.top_image_compressed_topic:
        _set_nested(cfg, "topics.top_image_compressed", args.top_image_compressed_topic)
    if args.middle_image_topic:
        _set_nested(cfg, "topics.top_image", args.middle_image_topic)
    if args.left_cmd_topic:
        _set_nested(cfg, "topics.left_cmd", args.left_cmd_topic)
    if args.right_cmd_topic:
        _set_nested(cfg, "topics.right_cmd", args.right_cmd_topic)
    if args.debug_streams:
        _set_nested(cfg, "runtime.debug_streams", True)
    return cfg


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACT direct inference publisher for 2-arm 3-camera")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--rate-hz", type=float, default=None)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    parser.add_argument("--guard-profile", choices=sorted(GUARD_PRESETS.keys()), default="")
    parser.add_argument("--left-state-topic", default="")
    parser.add_argument("--right-state-topic", default="")
    parser.add_argument("--left-image-topic", default="")
    parser.add_argument("--right-image-topic", default="")
    parser.add_argument("--top-image-topic", default="")
    parser.add_argument("--left-image-compressed-topic", default="")
    parser.add_argument("--right-image-compressed-topic", default="")
    parser.add_argument("--top-image-compressed-topic", default="")
    parser.add_argument("--middle-image-topic", default="", help="legacy alias of --top-image-topic")
    parser.add_argument("--left-cmd-topic", default="")
    parser.add_argument("--right-cmd-topic", default="")
    parser.add_argument("--debug-streams", action="store_true")
    return parser


def main() -> None:
    args, _ = make_parser().parse_known_args()
    cfg = _load_yaml(args.config)
    cfg = _apply_overrides(cfg, args)

    rospy.init_node("act_infer_2arm3cam_direct")
    node = PolicyPublisherNode(cfg)
    node.run()


if __name__ == "__main__":
    main()
