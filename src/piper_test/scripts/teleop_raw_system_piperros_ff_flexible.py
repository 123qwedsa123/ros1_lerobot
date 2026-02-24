#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flexible recorder based on teleop_raw_system_piperros_ff.py

Goals:
1) Keep original teleop/camera/piper bringup behavior.
2) Make rosbag topic selection highly configurable.
3) New key bindings:
   - SPACE: start recording (if recording, stop and discard)
   - s: stop and save current episode
   - d: stop and discard current episode
   - q: quit (discard current episode if recording)
"""

import importlib.util
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import rospy


BASE_SCRIPT = Path(__file__).with_name("teleop_raw_system_piperros_ff.py")
if not BASE_SCRIPT.is_file():
    raise RuntimeError(f"Base script not found: {BASE_SCRIPT}")

_spec = importlib.util.spec_from_file_location("teleop_raw_system_piperros_ff_base", str(BASE_SCRIPT))
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load base script: {BASE_SCRIPT}")
_base_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_mod)

TeleopRawSystemBase = _base_mod.TeleopRawSystem
WINDOW_NAME = _base_mod.WINDOW_NAME


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


class TeleopRawSystemFlexible(TeleopRawSystemBase):
    def _topic_mode(self) -> str:
        rb_cfg = self.cfg.get("rosbag", {})
        return str(rb_cfg.get("topic_mode", "default")).strip().lower()

    def _compile_regex(self, patterns, tag):
        compiled = []
        for p in _as_list(patterns):
            p = str(p).strip()
            if not p:
                continue
            try:
                compiled.append(re.compile(p))
            except re.error as exc:
                rospy.logwarn("[record] invalid %s regex '%s': %s", tag, p, exc)
        return compiled

    def _list_all_topics(self):
        try:
            out = subprocess.check_output(["rostopic", "list"], stderr=subprocess.STDOUT, text=True)
            return [line.strip() for line in out.splitlines() if line.strip()]
        except Exception as exc:
            rospy.logwarn("[record] topic_mode=all failed to list rostopics: %s", exc)
            return []

    def _expand_topic_entry(self, template):
        topic = str(template).strip()
        if not topic:
            return []

        uses_arm = any(k in topic for k in ("{master}", "{slave}", "{pair}", "{master_can}", "{slave_can}"))
        uses_cam = any(k in topic for k in ("{camera}", "{name}", "{device_name}", "{serial_no}"))

        if not uses_arm and not uses_cam:
            return [topic]

        arm_ctxs = [{}]
        cam_ctxs = [{}]
        if uses_arm:
            arm_ctxs = []
            for pair in self.arm_pairs:
                arm_ctxs.append(
                    {
                        "master": pair.get("master", ""),
                        "slave": pair.get("slave", ""),
                        "pair": pair.get("name", ""),
                        "master_can": pair.get("master_can", ""),
                        "slave_can": pair.get("slave_can", ""),
                    }
                )
        if uses_cam:
            cam_ctxs = []
            for cam in self.cameras:
                cam_ctxs.append(
                    {
                        "camera": cam.get("name", ""),
                        "name": cam.get("name", ""),
                        "device_name": cam.get("device_name", ""),
                        "serial_no": str(cam.get("serial_no", "")),
                    }
                )

        out = []
        for ac in arm_ctxs:
            for cc in cam_ctxs:
                ctx = {}
                ctx.update(ac)
                ctx.update(cc)
                try:
                    out.append(topic.format(**ctx))
                except Exception as exc:
                    rospy.logwarn("[record] topic template format failed '%s': %s", topic, exc)
        return out

    def _append_topics(self, dst, src):
        for entry in _as_list(src):
            dst.extend(self._expand_topic_entry(entry))

    def _build_record_topics(self):
        rb_cfg = self.cfg.get("rosbag", {})
        mode = self._topic_mode()

        topics = []
        if mode in ("default", "default_plus_custom"):
            topics.extend(super()._build_record_topics())
        elif mode in ("custom", "all", "all_plus_custom"):
            pass
        else:
            rospy.logwarn("[record] unknown topic_mode=%s, fallback to default", mode)
            topics.extend(super()._build_record_topics())

        if mode in ("custom", "default_plus_custom", "all_plus_custom"):
            self._append_topics(topics, rb_cfg.get("record_topics", []))

        if mode in ("all", "all_plus_custom"):
            topics.extend(self._list_all_topics())

        self._append_topics(topics, rb_cfg.get("additional_topics", []))

        include_re = self._compile_regex(rb_cfg.get("include_topic_regex", []), "include")
        exclude_re = self._compile_regex(rb_cfg.get("exclude_topic_regex", []), "exclude")

        out = []
        seen = set()
        for t in topics:
            t = str(t).strip()
            if not t or t in seen:
                continue
            if include_re and not any(r.search(t) for r in include_re):
                continue
            if exclude_re and any(r.search(t) for r in exclude_re):
                continue
            seen.add(t)
            out.append(t)
        return out

    def _discard_episode_dir(self, ep_dir):
        if ep_dir is None:
            return
        ep_dir = Path(ep_dir)
        if not ep_dir.exists():
            return

        rb_cfg = self.cfg.get("rosbag", {})
        retries = max(1, int(rb_cfg.get("discard_delete_retries", 3)))
        wait_sec = max(0.05, float(rb_cfg.get("discard_delete_wait_sec", 0.2)))
        last_exc = None
        for i in range(retries):
            try:
                shutil.rmtree(ep_dir)
                rospy.loginfo("[record] DISCARD removed %s", ep_dir)
                return
            except Exception as exc:
                last_exc = exc
                if i < retries - 1:
                    time.sleep(wait_sec)
        rospy.logwarn("[record] DISCARD failed to remove %s: %s", ep_dir, last_exc)

    def _stop_episode(self, stop_reason="user_stop", save=True):
        """stop rosbag and either save metadata or discard episode folder."""
        if not self.is_recording:
            return

        ep_name = self.current_ep_name
        ep_dir = self.current_ep_dir
        started_utc = self.current_started_utc
        topics = list(self.current_topics)
        duration = time.time() - self.rec_start_time if self.rec_start_time else None

        rosbag_ok = False
        if self.rosbag_proc is not None:
            rosbag_ok = self._stop_process_group(self.rosbag_proc)
            self.rosbag_proc = None
        if hasattr(self, "_rosbag_log_fh") and self._rosbag_log_fh:
            self._rosbag_log_fh.flush()
            self._rosbag_log_fh.close()
            self._rosbag_log_fh = None

        ended_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.is_recording = False
        self.ep_count = self._next_ep_index()

        action = "SAVE" if save else "DISCARD"
        rospy.loginfo("[record] STOP %s action=%s reason=%s duration=%.2fs", ep_name, action, stop_reason, duration or 0.0)

        if save:
            job = {
                "ep_name": ep_name,
                "ep_dir": ep_dir,
                "started_utc": started_utc,
                "ended_utc": ended_utc,
                "duration_sec": duration,
                "stop_reason": stop_reason,
                "topics": topics,
                "rosbag_ok": rosbag_ok,
                "camera_transport": self.camera_transport,
                "lz4": self.rosbag_lz4,
                "rosbag_log_path": str(getattr(self, "_rosbag_log_path", "")),
            }
            try:
                self.save_q.put_nowait(job)
            except Exception:
                rospy.logwarn("[save] queue full, skip metadata write")
        else:
            self._discard_episode_dir(ep_dir)

        self.current_ep_name = None
        self.current_ep_dir = None
        self.current_started_utc = None
        self.current_topics = []
        self.rec_start_time = None
        self.frame_count = 0

    def _update_preview(self):
        pw, ph = 480, 360
        panels = []
        for cam in self.cameras:
            cn = cam["name"]
            slave_tag = self._camera_slave_tag(cn)
            cs = self.cam_state.get(cn)
            if cs and "color" in cs:
                panel = cv2.resize(cs["color"], (pw, ph))
            else:
                panel = np.zeros((ph, pw, 3), dtype=np.uint8)
                cv2.putText(panel, f"{cn}: waiting...", (pw // 6, ph // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            overlay = panel.copy()
            cv2.rectangle(overlay, (0, 0), (pw, 32), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, panel, 0.5, 0, panel)
            lbl_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
            cam_label = f"  {cn} ({cam['device_name']})"
            if slave_tag:
                cam_label += f" -> {slave_tag}"
            cv2.putText(panel, cam_label, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lbl_color, 2)
            panels.append(panel)

        mosaic = np.hstack(panels) if panels else np.zeros((ph, pw, 3), dtype=np.uint8)
        total_w = mosaic.shape[1]
        bar_h = 52
        bar = np.zeros((bar_h, total_w, 3), dtype=np.uint8)
        if self.is_recording:
            elapsed = time.time() - self.rec_start_time if self.rec_start_time else 0.0
            m, s = int(elapsed) // 60, int(elapsed) % 60
            txt = (
                f"  REC  ep={self.current_ep_name}  frames={self.frame_count}  t={m:02d}:{s:02d}"
                "  | [S]save [D]discard [SPACE]discard [Q]quit"
            )
            if int(elapsed * 2) % 2 == 0:
                cv2.circle(bar, (20, bar_h // 2), 8, (0, 0, 255), -1)
            cv2.putText(bar, txt, (36, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            txt = f"  IDLE next=episode_{self.ep_count:03d} saved={self.save_done}  | [SPACE]start [Q]quit"
            cv2.circle(bar, (20, bar_h // 2), 8, (0, 255, 0), -1)
            cv2.putText(bar, txt, (36, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
        cv2.imshow(WINDOW_NAME, np.vstack([mosaic, bar]))

    def _print_banner(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("Teleop Raw System PiperROS-FF  →  Rosbag (Flexible)")
        rospy.loginfo("  session_dir = %s", self.session_dir)
        rospy.loginfo("  rate        = %d Hz", self.rate_hz)
        rospy.loginfo("  arm_pairs   = %s", [p["name"] for p in self.arm_pairs])
        rospy.loginfo("  cameras     = %s", [c["name"] for c in self.cameras])
        rospy.loginfo("  transport   = %s  lz4=%s", self.camera_transport, self.rosbag_lz4)
        rospy.loginfo("  topic_mode  = %s", self._topic_mode())
        rospy.loginfo("  topics      = %s", self._build_record_topics())
        rospy.loginfo("[SPACE]开始录制  [S]保存停止  [D]丢弃停止  [Q]退出")
        rospy.loginfo("录制中再次按 [SPACE] 会直接丢弃当前 episode")
        rospy.loginfo("=" * 60)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("等待所有数据源就绪...")
        while not rospy.is_shutdown():
            self._update_preview()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                return
            if self._all_ready():
                rospy.loginfo("所有数据源就绪！")
                break
            rate.sleep()

        while not rospy.is_shutdown():
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if self.is_recording:
                    self._stop_episode("space_discard", save=False)
                else:
                    self._start_episode()
            elif key in (ord("s"), ord("S")):
                if self.is_recording:
                    self._stop_episode("user_save", save=True)
            elif key in (ord("d"), ord("D")):
                if self.is_recording:
                    self._stop_episode("user_discard", save=False)
            elif key in (ord("q"), ord("Q")):
                rospy.loginfo("退出...")
                if self.is_recording:
                    self._stop_episode("quit_discard", save=False)
                break

            if self.is_recording:
                self.frame_count += 1

            self._update_preview()
            rate.sleep()

        cv2.destroyAllWindows()
        self.save_q.join()


def main():
    try:
        TeleopRawSystemFlexible().run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
