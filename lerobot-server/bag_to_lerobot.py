#!/usr/bin/env python3
"""
ROS bag → LeRobot v2.1/v3.0 数据集转换器

两阶段设计（解决 ROS1 环境与 lerobot 环境不兼容问题）:
  Phase 1 (ROS1 环境): bag → parquet + mp4 + meta  (需要 rosbag, cv2, pyarrow)
  Phase 2 (lerobot 环境): v2.1 → v3.0 升级         (需要 lerobot)

用法:
  # 第一阶段（在 ROS1 环境下运行，无需激活 conda）
  source /opt/ros/noetic/setup.bash
  python bag_to_lerobot.py --config bag_convert_config.yaml --phase 1

  # 第二阶段（在 lerobot 环境下运行）
  conda activate lerobot-mujoco
  python bag_to_lerobot.py --config bag_convert_config.yaml --phase 2

  # 或用 convert.sh 一键执行两阶段
"""

from __future__ import annotations
import argparse, json, math, re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────── 配置 ───────────────────────────

@dataclass
class TopicConfig:
    robot_left:   str
    robot_right:  str
    teleop_left:  str
    teleop_right: str
    top_camera:   str
    left_wrist:   str
    right_wrist:  str

    @property
    def image_list(self): return [self.top_camera, self.left_wrist, self.right_wrist]
    @property
    def joint_list(self): return [self.robot_left, self.robot_right, self.teleop_left, self.teleop_right]


@dataclass
class ConvertConfig:
    session_dir:          Path
    output_root:          Path
    dataset_id:           str
    task:                 str
    fps:                  float
    min_frames:           int
    max_sync_lag_sec:     float
    pos_dim:              int
    effort_dim:           int
    action_dim:           int
    image_width:          int
    image_height:         int
    video_codec:          str
    image_use_bag_time:   bool
    topics:               TopicConfig
    clean_output:         bool
    delete_v21_backup_after_v30: bool
    convert_to_v30:       bool
    v30_push_to_hub:      bool
    v30_force_conversion: bool

    @property
    def dataset_dir(self): return self.output_root / self.dataset_id


def as_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in ("1", "true", "yes", "on"):
            return True
        if lv in ("0", "false", "no", "off"):
            return False
    return bool(default)


def load_config(yaml_path: str) -> ConvertConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    c = raw.get("convert", raw)
    t = c.get("topics", {})
    return ConvertConfig(
        session_dir          = Path(c["session_dir"]).expanduser(),
        output_root          = Path(c["output_root"]).expanduser(),
        dataset_id           = c["dataset_id"],
        task                 = c.get("task", "teleop"),
        fps                  = float(c.get("fps", 30)),
        min_frames           = int(c.get("min_frames", 32)),
        max_sync_lag_sec     = float(c.get("max_sync_lag_sec", 0.08)),
        pos_dim              = int(c.get("pos_dim_per_arm", 7)),
        effort_dim           = int(c.get("effort_dim_per_arm", 7)),
        action_dim           = int(c.get("action_dim_per_arm", 7)),
        image_width          = int(c.get("image_width", 256)),
        image_height         = int(c.get("image_height", 256)),
        video_codec          = c.get("video_codec", "mp4v"),
        image_use_bag_time   = as_bool(c.get("image_use_bag_time", False), False),
        topics               = TopicConfig(
            robot_left   = t.get("robot_left",   "/robot/arm_left/joint_states_single"),
            robot_right  = t.get("robot_right",  "/robot/arm_right/joint_states_single"),
            teleop_left  = t.get("teleop_left",  "/teleop/arm_left/joint_states_single"),
            teleop_right = t.get("teleop_right", "/teleop/arm_right/joint_states_single"),
            top_camera   = t.get("top_camera",   "/realsense_top/color/image_raw/compressed"),
            left_wrist   = t.get("left_wrist",   "/realsense_left/color/image_raw/compressed"),
            right_wrist  = t.get("right_wrist",  "/realsense_right/color/image_raw/compressed"),
        ),
        clean_output         = as_bool(c.get("clean_output", True), True),
        delete_v21_backup_after_v30 = as_bool(c.get("delete_v21_backup_after_v30", False), False),
        convert_to_v30       = as_bool(c.get("convert_to_v30", False), False),
        v30_push_to_hub      = as_bool(c.get("v30_push_to_hub", False), False),
        v30_force_conversion = as_bool(c.get("v30_force_conversion", False), False),
    )


# ─────────────────────────── 数据结构 ───────────────────────────

@dataclass
class EpisodeResult:
    episode_index: int
    length: int
    parquet_path: Path
    video_paths: Dict[str, Path]
    stats: Dict


class EpisodeSkipError(RuntimeError):
    pass


# ─────────────────────────── Phase 1: bag 读取 ───────────────────────────

class BagReader:
    def __init__(self, cfg: ConvertConfig, cv2, rosbag):
        self.cfg, self.cv2, self.rosbag = cfg, cv2, rosbag

    def _msg_time(self, msg, bag_t) -> float:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp:
            s = float(stamp.to_sec())
            if s > 0: return s
        return float(bag_t.to_sec())

    def read(self, bag_path: Path) -> Dict:
        cfg = self.cfg
        tp  = cfg.topics
        effort_topics = {tp.robot_left, tp.robot_right}

        joint_ts:   Dict[str, List] = {t: [] for t in tp.joint_list}
        joint_pos:  Dict[str, List] = {t: [] for t in tp.joint_list}
        joint_eff:  Dict[str, List] = {t: [] for t in [tp.robot_left, tp.robot_right]}
        img_ts:     Dict[str, List] = {t: [] for t in tp.image_list}
        img_frames: Dict[str, List] = {t: [] for t in tp.image_list}
        dropped:    Dict[str, int]  = {t: 0  for t in tp.joint_list}

        need_pos = {tp.robot_left: cfg.pos_dim, tp.robot_right: cfg.pos_dim,
                    tp.teleop_left: cfg.action_dim, tp.teleop_right: cfg.action_dim}

        bag = self.rosbag.Bag(str(bag_path), "r")
        try:
            for topic, msg, t in bag.read_messages(topics=tp.joint_list + tp.image_list):
                if topic in tp.joint_list:
                    dim = need_pos[topic]
                    pos = list(getattr(msg, "position", []))
                    if len(pos) < dim:
                        dropped[topic] += 1; continue
                    ts_val = self._msg_time(msg, t)
                    if topic in effort_topics:
                        eff = list(getattr(msg, "effort", []))
                        if len(eff) < cfg.effort_dim:
                            dropped[topic] += 1; continue
                        joint_eff[topic].append(np.array(eff[:cfg.effort_dim], np.float32))
                    joint_ts[topic].append(ts_val)
                    joint_pos[topic].append(np.array(pos[:dim], np.float32))
                elif topic in tp.image_list:
                    ts_val = float(t.to_sec()) if cfg.image_use_bag_time else self._msg_time(msg, t)
                    arr = np.frombuffer(msg.data, np.uint8)
                    if arr.size == 0: continue
                    img = self.cv2.imdecode(arr, self.cv2.IMREAD_COLOR)
                    if img is None: continue
                    img = self.cv2.resize(img, (cfg.image_width, cfg.image_height),
                                          interpolation=self.cv2.INTER_AREA)
                    img_ts[topic].append(ts_val)
                    img_frames[topic].append(img)
        finally:
            bag.close()

        for t, n in dropped.items():
            if n: print(f"  [warn] {bag_path.parent.name}: dropped {n} short msgs on {t}")

        return {
            "joint_ts":   {k: np.array(v, np.float64) for k, v in joint_ts.items()},
            "joint_pos":  {k: np.array(v, np.float32) for k, v in joint_pos.items()},
            "joint_eff":  {k: np.array(v, np.float32) for k, v in joint_eff.items()},
            "img_ts":     {k: np.array(v, np.float64) for k, v in img_ts.items()},
            "img_frames": img_frames,
        }


class TemporalSampler:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg

    @staticmethod
    def _nearest(src: np.ndarray, q: np.ndarray) -> np.ndarray:
        if src.size == 1: return np.zeros(q.size, np.int64)
        idx = np.clip(np.searchsorted(src, q), 1, src.size - 1)
        prev = idx - 1
        idx[np.abs(q - src[prev]) <= np.abs(src[idx] - q)] = prev[np.abs(q - src[prev]) <= np.abs(src[idx] - q)]
        return idx.astype(np.int64)

    def sample(self, data: Dict) -> Dict:
        cfg = self.cfg
        tp  = cfg.topics
        jts, img_ts = data["joint_ts"], data["img_ts"]

        for key in tp.joint_list + tp.image_list:
            src = jts if key in jts else img_ts
            if src[key].size == 0:
                raise EpisodeSkipError(f"空数据流: {key}")

        t_start = max(jts[tp.robot_left][0], jts[tp.robot_right][0],
                      jts[tp.teleop_left][0], jts[tp.teleop_right][0],
                      img_ts[tp.top_camera][0], img_ts[tp.left_wrist][0], img_ts[tp.right_wrist][0])
        t_end   = min(jts[tp.robot_left][-1], jts[tp.robot_right][-1],
                      jts[tp.teleop_left][-1], jts[tp.teleop_right][-1],
                      img_ts[tp.top_camera][-1], img_ts[tp.left_wrist][-1], img_ts[tp.right_wrist][-1])
        if t_end <= t_start:
            raise EpisodeSkipError("各流无重叠时间窗口")

        sample_ts = t_start + np.arange(
            int(math.floor((t_end - t_start) * cfg.fps)) + 1, dtype=np.float64) / cfg.fps
        idx = {k: self._nearest(jts[k], sample_ts) for k in tp.joint_list}
        idx.update({k: self._nearest(img_ts[k], sample_ts) for k in tp.image_list})

        lags = [np.abs(jts[k][idx[k]] - sample_ts) for k in tp.joint_list] + \
               [np.abs(img_ts[k][idx[k]] - sample_ts) for k in tp.image_list]
        keep = np.ones(len(sample_ts), bool)
        for lag in lags: keep &= lag <= cfg.max_sync_lag_sec

        n_drop = int((~keep).sum())
        if n_drop: print(f"  [info] 丢弃同步误差帧 {n_drop} 帧")
        if int(keep.sum()) < cfg.min_frames:
            raise EpisodeSkipError(f"有效帧不足: {int(keep.sum())} < {cfg.min_frames}")

        return {"sample_ts": sample_ts[keep], "idx": {k: v[keep] for k, v in idx.items()}}


class EpisodeWriter:
    def __init__(self, cfg: ConvertConfig, cv2):
        self.cfg, self.cv2 = cfg, cv2

    @staticmethod
    def _joint_name(i): return "gripper" if i == 6 else f"joint{i+1}"
    def _names(self, prefix, dim, suffix):
        return [f"{prefix}.{self._joint_name(i)}.{suffix}" for i in range(dim)]

    def _video_path(self, ep_idx, key):
        chunk = ep_idx // 1000
        skey  = key.replace("/", "_").replace(" ", "_")
        return self.cfg.dataset_dir / "videos" / f"chunk-{chunk:03d}" / skey / f"episode_{ep_idx:06d}.mp4"

    def _write_mp4(self, path: Path, frames, fps):
        h, w = frames[0].shape[:2]
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = self.cv2.VideoWriter_fourcc(*self.cfg.video_codec)
        wr = self.cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not wr.isOpened(): raise RuntimeError(f"无法打开视频写入器: {path}")
        try:
            for f in frames: wr.write(f)
        finally:
            wr.release()

    @staticmethod
    def _vstats(x): return {"min": x.min(0).tolist(), "max": x.max(0).tolist(),
                             "mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "count": [int(x.shape[0])]}
    @staticmethod
    def _sstats(x): return {"min": [float(x.min())], "max": [float(x.max())],
                             "mean": [float(x.mean())], "std": [float(x.std())], "count": [int(x.size)]}
    @staticmethod
    def _istats(frames):
        mn, mx, s, s2 = np.ones(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for f in frames:
            rgb = f[:,:,::-1].astype(np.float32)/255; flat = rgb.reshape(-1,3)
            mn = np.minimum(mn, flat.min(0)); mx = np.maximum(mx, flat.max(0))
            s += flat.sum(0); s2 += (flat*flat).sum(0)
        n = frames[0].shape[0]*frames[0].shape[1]*len(frames)
        mu = s/n; std = np.sqrt(np.maximum(s2/n - mu*mu, 0))
        stats = {
            k: [[[float(v[0])]], [[float(v[1])]], [[float(v[2])]]]
            for k, v in zip(("min", "max", "mean", "std"), (mn, mx, mu, std))
        }
        stats["count"] = [len(frames)]
        return stats

    def write(self, ep_idx, data, sampled) -> EpisodeResult:
        cfg = self.cfg; tp = cfg.topics; idx = sampled["idx"]; ts = sampled["sample_ts"]
        jpos, jeff = data["joint_pos"], data["joint_eff"]
        frames = data["img_frames"]

        obs = np.concatenate([jpos[tp.robot_left][idx[tp.robot_left]],
                               jeff[tp.robot_left][idx[tp.robot_left]],
                               jpos[tp.robot_right][idx[tp.robot_right]],
                               jeff[tp.robot_right][idx[tp.robot_right]]], axis=1).astype(np.float32)
        act = np.concatenate([jpos[tp.teleop_left][idx[tp.teleop_left]],
                               jpos[tp.teleop_right][idx[tp.teleop_right]]], axis=1).astype(np.float32)

        top_f = [frames[tp.top_camera][int(i)] for i in idx[tp.top_camera]]
        lw_f  = [frames[tp.left_wrist][int(i)]  for i in idx[tp.left_wrist]]
        rw_f  = [frames[tp.right_wrist][int(i)] for i in idx[tp.right_wrist]]

        chunk = ep_idx // 1000
        parquet_path = cfg.dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        rel_ts = (ts - ts[0]).astype(np.float32)
        frame_idx = np.arange(len(obs), dtype=np.int64)
        pd.DataFrame({
            "observation.state": [r.tolist() for r in obs],
            "action":            [r.tolist() for r in act],
            "timestamp":         rel_ts,
            "frame_index":       frame_idx,
            "episode_index":     np.full(len(obs), ep_idx, np.int64),
            "index":             np.zeros(len(obs), np.int64),
            "task_index":        np.zeros(len(obs), np.int64),
        }).to_parquet(parquet_path, index=False)

        video_paths = {
            "observation.image":             self._video_path(ep_idx, "observation.image"),
            "observation.left_wrist_image":  self._video_path(ep_idx, "observation.left_wrist_image"),
            "observation.right_wrist_image": self._video_path(ep_idx, "observation.right_wrist_image"),
        }
        self._write_mp4(video_paths["observation.image"],             top_f, cfg.fps)
        self._write_mp4(video_paths["observation.left_wrist_image"],  lw_f,  cfg.fps)
        self._write_mp4(video_paths["observation.right_wrist_image"], rw_f,  cfg.fps)

        stats = {
            "action": self._vstats(act), "observation.state": self._vstats(obs),
            "observation.image": self._istats(top_f),
            "observation.left_wrist_image": self._istats(lw_f),
            "observation.right_wrist_image": self._istats(rw_f),
            "timestamp": self._sstats(rel_ts.astype(np.float64)),
            "frame_index": self._sstats(frame_idx.astype(np.float64)),
            "episode_index": self._sstats(np.full(len(obs), ep_idx, np.float64)),
            "index": self._sstats(frame_idx.astype(np.float64)),
            "task_index": self._sstats(np.zeros(len(obs), np.float64)),
        }
        return EpisodeResult(ep_idx, len(obs), parquet_path, video_paths, stats)


class MetadataWriter:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg

    def _joint_name(self, i): return "gripper" if i == 6 else f"joint{i+1}"
    def _names(self, prefix, dim, suffix):
        return [f"{prefix}.{self._joint_name(i)}.{suffix}" for i in range(dim)]

    def write(self, results: List[EpisodeResult]):
        cfg = self.cfg
        meta = cfg.dataset_dir / "meta"
        meta.mkdir(parents=True, exist_ok=True)

        state_names  = (self._names("left", cfg.pos_dim, "pos") +
                        self._names("left", cfg.effort_dim, "effort_raw") +
                        self._names("right", cfg.pos_dim, "pos") +
                        self._names("right", cfg.effort_dim, "effort_raw"))
        action_names = (self._names("left", cfg.action_dim, "pos") +
                        self._names("right", cfg.action_dim, "pos"))

        vinfo = {"video.fps": cfg.fps, "video.height": cfg.image_height, "video.width": cfg.image_width,
                 "video.channels": 3, "video.codec": cfg.video_codec, "video.pix_fmt": "yuv420p",
                 "video.is_depth_map": False, "has_audio": False}
        vfeat = {"dtype": "video", "shape": [cfg.image_height, cfg.image_width, 3],
                 "names": ["height", "width", "channels"], "video_info": vinfo, "info": vinfo}

        info = {
            "codebase_version": "v2.1", "robot_type": "omy",
            "total_episodes": len(results), "total_frames": sum(r.length for r in results),
            "total_tasks": 1, "total_videos": len(results) * 3,
            "total_chunks": max(1, math.ceil(len(results) / 1000)), "chunks_size": 1000,
            "fps": cfg.fps, "splits": {"train": f"0:{len(results)}"},
            "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action":                        {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
                "observation.state":             {"dtype": "float32", "shape": [len(state_names)],  "names": state_names},
                "observation.image":             vfeat,
                "observation.left_wrist_image":  vfeat,
                "observation.right_wrist_image": vfeat,
                "timestamp":    {"dtype": "float32", "shape": [1], "names": None},
                "frame_index":  {"dtype": "int64",   "shape": [1], "names": None},
                "episode_index":{"dtype": "int64",   "shape": [1], "names": None},
                "index":        {"dtype": "int64",   "shape": [1], "names": None},
                "task_index":   {"dtype": "int64",   "shape": [1], "names": None},
            },
        }
        (meta / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": cfg.task}) + "\n", encoding="utf-8")

        with (meta / "episodes.jsonl").open("w") as f:
            for r in results:
                f.write(json.dumps({"episode_index": r.episode_index,
                                    "tasks": [cfg.task], "length": r.length}) + "\n")
        with (meta / "episodes_stats.jsonl").open("w") as f:
            for r in results:
                f.write(json.dumps({"episode_index": r.episode_index, "stats": r.stats}) + "\n")

        (cfg.dataset_dir / "README.md").write_text(
            f"# LeRobot v2.1 bimanual dataset\n\n- dataset_id: `{cfg.dataset_id}`\n"
            f"- total_episodes: `{len(results)}`\n- fps: `{cfg.fps}`\n", encoding="utf-8")


def rewrite_global_indices(results: List[EpisodeResult]):
    global_idx = 0
    for r in results:
        df = pd.read_parquet(r.parquet_path)
        n = len(df)
        idx = np.arange(global_idx, global_idx + n, dtype=np.int64)
        df["index"] = idx
        df.to_parquet(r.parquet_path, index=False)
        r.stats["index"] = {"min": [float(idx.min())], "max": [float(idx.max())],
                             "mean": [float(idx.mean())], "std": [float(idx.std())], "count": [n]}
        global_idx += n


# ─────────────────────────── Phase 2: v3.0 升级 ───────────────────────────

def run_phase2(cfg: ConvertConfig, delete_v21_backup: bool | None = None):
    """仅在 lerobot 环境下运行（conda activate lerobot-mujoco）"""
    try:
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    except ImportError as e:
        raise RuntimeError(
            "无法导入 lerobot。请确认已激活正确环境:\n"
            "  conda activate lerobot-mujoco\n"
            "  python bag_to_lerobot.py --config ... --phase 2"
        ) from e

    info_path = cfg.dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise RuntimeError(
            f"未找到 Phase 1 的输出: {info_path}\n"
            "请先在 ROS1 环境下运行 Phase 1。"
        )

    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected_eps = int(info.get("total_episodes", 0))
    expected_videos = expected_eps * 3
    actual_eps = len(list((cfg.dataset_dir / "data").rglob("episode_*.parquet")))
    actual_videos = len(list((cfg.dataset_dir / "videos").rglob("episode_*.mp4")))
    if expected_eps <= 0:
        raise RuntimeError(f"Phase 1 元数据异常: total_episodes={expected_eps}")
    if actual_eps != expected_eps or actual_videos != expected_videos:
        raise RuntimeError(
            "检测到 v2.1 输出不一致，通常是旧文件残留导致。\n"
            f"meta.total_episodes={expected_eps}, parquet={actual_eps}, mp4={actual_videos} (期望={expected_videos})\n"
            f"请删除目录后重跑 Phase 1: rm -rf {cfg.dataset_dir}"
        )

    print(f"[phase2] 升级 {cfg.dataset_id} 到 v3.0 ...")
    convert_dataset(
        repo_id=cfg.dataset_id,
        root=cfg.output_root,
        push_to_hub=cfg.v30_push_to_hub,
        force_conversion=cfg.v30_force_conversion,
    )

    delete_backup = cfg.delete_v21_backup_after_v30 if delete_v21_backup is None else delete_v21_backup
    if delete_backup:
        v30_info_path = cfg.dataset_dir / "meta" / "info.json"
        v30_info = json.loads(v30_info_path.read_text(encoding="utf-8"))
        codebase_version = str(v30_info.get("codebase_version", "unknown"))
        if codebase_version != "v3.0":
            raise RuntimeError(
                "Phase 2 结束后检测到输出不是 v3.0，已停止删除 v2.1 备份。\n"
                f"当前版本: {codebase_version}"
            )
        v21_backup_dir = cfg.dataset_dir.parent / f"{cfg.dataset_dir.name}_old"
        if v21_backup_dir.is_dir():
            shutil.rmtree(v21_backup_dir)
            print(f"[phase2] 已删除 v2.1 备份目录: {v21_backup_dir}")
        else:
            print(f"[phase2] 未找到 v2.1 备份目录，跳过删除: {v21_backup_dir}")

    print(f"[phase2] 完成: {cfg.dataset_dir}")


# ─────────────────────────── 主转换器 ───────────────────────────

class BagToLeRobotConverter:

    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        # Phase 1 才需要 rosbag / cv2，这里延迟导入
        try:
            import cv2, rosbag
        except ImportError as e:
            raise RuntimeError(
                "缺少 Phase 1 依赖（rosbag / cv2）。\n"
                "请在 ROS1 环境下运行:\n"
                "  source /opt/ros/noetic/setup.bash\n"
                "  python bag_to_lerobot.py --config ... --phase 1"
            ) from e
        try:
            import pyarrow  # noqa
        except ImportError:
            import fastparquet  # noqa
        self.reader  = BagReader(cfg, cv2, rosbag)
        self.sampler = TemporalSampler(cfg)
        self.writer  = EpisodeWriter(cfg, cv2)
        self.meta    = MetadataWriter(cfg)

    @staticmethod
    def _collect_bags(session_dir: Path) -> List[Path]:
        eps = sorted(
            [p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")],
            key=lambda p: int(m.group(1)) if (m := re.match(r"episode_(\d+)$", p.name)) else 10**9,
        )
        return [ep / "episode.bag" for ep in eps if (ep / "episode.bag").is_file()]

    def _prepare_output_dir(self):
        cfg = self.cfg
        if cfg.dataset_dir.exists() and cfg.clean_output:
            print(f"[phase1] 清理旧输出目录: {cfg.dataset_dir}")
            shutil.rmtree(cfg.dataset_dir)
        cfg.dataset_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        cfg = self.cfg
        self._prepare_output_dir()

        bags = self._collect_bags(cfg.session_dir)
        if not bags:
            raise RuntimeError(f"未找到 episode_*/episode.bag: {cfg.session_dir}")
        print(f"[phase1] 找到 {len(bags)} 个 bag episode")

        results, skips, ep_idx = [], {}, 0
        for bag in bags:
            ep_name = bag.parent.name
            print(f"[phase1] {ep_name} -> episode_{ep_idx:06d}")
            try:
                data    = self.reader.read(bag)
                sampled = self.sampler.sample(data)
                result  = self.writer.write(ep_idx, data, sampled)
                results.append(result)
                ep_idx += 1
            except EpisodeSkipError as e:
                skips[str(e)] = skips.get(str(e), 0) + 1
                print(f"  [skip] {ep_name}: {e}")
            except Exception as e:
                skips[str(e)] = skips.get(str(e), 0) + 1
                print(f"  [error] {ep_name}: {e}")

        if not results:
            raise RuntimeError(f"没有成功转换的 episode。跳过原因: {skips}")

        rewrite_global_indices(results)
        self.meta.write(results)

        print(f"\n[phase1] 完成: {len(results)} episodes → {cfg.dataset_dir} (v2.1)")
        if skips:
            for reason, cnt in sorted(skips.items(), key=lambda x: -x[1]):
                print(f"  跳过 {cnt}x: {reason}")


# ─────────────────────────── 入口 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ROS bag → LeRobot 数据集转换器")
    parser.add_argument("--config", default="bag_convert_config.yaml")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help=(
            "1 = bag→parquet+mp4+meta (ROS1 环境)\n"
            "2 = v2.1→v3.0 升级 (lerobot 环境)\n"
            "不填 = 按 config 里的 convert_to_v30 自动决定（仅单环境时用）"
        ),
    )
    parser.add_argument(
        "--delete-v21-backup",
        action="store_true",
        help="Phase 2 成功后删除 <dataset_id>_old（仅保留 v3.0）。",
    )
    parser.add_argument(
        "--keep-v21-backup",
        action="store_true",
        help="强制保留 <dataset_id>_old（覆盖配置里的 delete_v21_backup_after_v30）。",
    )
    args = parser.parse_args()

    if not Path(args.config).is_file():
        raise SystemExit(f"配置文件不存在: {args.config}")

    cfg = load_config(args.config)
    if args.delete_v21_backup and args.keep_v21_backup:
        raise SystemExit("参数冲突: --delete-v21-backup 与 --keep-v21-backup 不能同时使用。")
    delete_v21_backup = cfg.delete_v21_backup_after_v30
    if args.delete_v21_backup:
        delete_v21_backup = True
    if args.keep_v21_backup:
        delete_v21_backup = False

    if args.phase == 1:
        # 只跑 Phase 1（ROS1 环境）
        BagToLeRobotConverter(cfg).run()

    elif args.phase == 2:
        # 只跑 Phase 2（lerobot 环境）
        run_phase2(cfg, delete_v21_backup=delete_v21_backup)

    else:
        # 没有指定 phase：尝试 Phase 1，然后按 config 决定是否 Phase 2
        BagToLeRobotConverter(cfg).run()
        if cfg.convert_to_v30:
            run_phase2(cfg, delete_v21_backup=delete_v21_backup)


if __name__ == "__main__":
    main()
