#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fake deployment server for client-side debugging.

Purpose:
- Do NOT load model weights.
- Reuse dataset frames/actions to emulate the real server protocol:
  get_reset / infer / dataset_preview.
- Let you verify the ROS client pipeline and control path independently.

Usage:
  python fake_policy_server_2arm3cam_dataset.py --config config/pipeline_config.yaml
"""

import argparse
import json
import random
import time
import traceback
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import zmq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Reuse unified config parser from the real server so config semantics are identical.
from policy_server_2arm3cam_deploy import (
    build_state_spec,
    extract_server_config,
    load_yaml,
    resolve_paths,
)


def _to_np_float32(vec: Any, expected_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(f"{name} dim={arr.size}, expected {expected_dim}")
    return arr


def _image_to_jpg(img: Any, image_size: int, jpeg_quality: int) -> bytes:
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)

    # Support CHW or HWC.
    if img.ndim == 3 and img.shape[0] == 3:
        x = img.permute(1, 2, 0)
    elif img.ndim == 3 and img.shape[-1] == 3:
        x = img
    else:
        raise RuntimeError(f"bad image shape: {tuple(img.shape)}")

    x = x.detach().cpu().numpy()
    if x.dtype != np.uint8:
        if np.max(x) <= 1.5:
            x = np.clip(x, 0.0, 1.0) * 255.0
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)

    if image_size > 0:
        x = cv2.resize(x, (image_size, image_size), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("jpg encode failed")
    return buf.tobytes()


def _episode_bounds(ds: LeRobotDataset) -> Tuple[List[int], List[int]]:
    # Preferred API
    epi = getattr(ds, "episode_data_index", None)
    if epi is not None:
        try:
            ep_from = [int(x) for x in epi["from"]]
            ep_to = [int(x) for x in epi["to"]]
            if len(ep_from) == len(ep_to) and len(ep_from) > 0:
                return ep_from, ep_to
        except Exception:
            pass

    # Fallback: derive from episode_index column
    hf = getattr(ds, "hf_dataset", None)
    if hf is None:
        raise RuntimeError("dataset has neither episode_data_index nor hf_dataset")
    try:
        ep_col = [int(x) for x in hf["episode_index"]]
    except Exception:
        ep_col = [int(hf[i]["episode_index"]) for i in range(len(hf))]

    if not ep_col:
        return [], []

    ep_from: List[int] = []
    ep_to: List[int] = []
    start = 0
    prev = ep_col[0]
    for i in range(1, len(ep_col)):
        cur = ep_col[i]
        if cur != prev:
            ep_from.append(start)
            ep_to.append(i)
            start = i
            prev = cur
    ep_from.append(start)
    ep_to.append(len(ep_col))
    return ep_from, ep_to


class DatasetResponder:
    def __init__(
        self,
        ds: LeRobotDataset,
        features: Dict[str, str],
        state_dim: int,
        action_dim: int,
        preview_image_size: int,
        preview_jpeg_quality: int,
        infer_mode: str = "episode",
        episode_index: int = -1,
    ) -> None:
        self.ds = ds
        self.features = features
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.preview_image_size = int(preview_image_size)
        self.preview_jpeg_quality = int(preview_jpeg_quality)
        self.infer_mode = str(infer_mode).strip().lower()
        if self.infer_mode not in ("fixed", "cycle", "episode"):
            self.infer_mode = "episode"

        self.ep_from, self.ep_to = _episode_bounds(ds)
        self.num_episodes = len(self.ep_from)
        self.num_frames = int(len(ds))
        if self.num_frames <= 0:
            raise RuntimeError("dataset is empty")
        if self.num_episodes <= 0:
            raise RuntimeError("dataset has zero episodes")

        self._cursor = int(self.ep_from[0])  # for cycle mode
        # fixed 模式: 启动时随机抽一条 item，后续一直重播同一条
        self._fixed_idx = int(random.randrange(self.num_frames))
        _sample, _info = self._sample_by_frame(self._fixed_idx)
        self.fixed_info = {
            "frame_idx": int(_info["frame_idx"]),
            "episode": int(_info["episode"]),
            "t_in_episode": int(_info["t_in_episode"]),
        }

        # episode 模式: 选一个 episode，从头到尾顺序回放 action
        if int(episode_index) < 0:
            self._episode = int(random.randrange(self.num_episodes))
        else:
            self._episode = int(max(0, min(int(episode_index), self.num_episodes - 1)))
        self._episode_from = int(self.ep_from[self._episode])
        self._episode_to = int(self.ep_to[self._episode])  # exclusive
        self._episode_cursor = int(self._episode_from)
        self._episode_loop = 0
        self.episode_info = {
            "episode": int(self._episode),
            "from": int(self._episode_from),
            "to": int(self._episode_to),
            "len": int(self._episode_to - self._episode_from),
        }

    def _sample_by_frame(self, frame_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        idx = int(frame_idx)
        s = self.ds[idx]

        # Find episode/t_in_episode for metadata.
        ep = 0
        for i, (f, t) in enumerate(zip(self.ep_from, self.ep_to)):
            if f <= idx < t:
                ep = i
                break
        t_in_episode = int(idx - self.ep_from[ep])
        ep_len = int(self.ep_to[ep] - self.ep_from[ep])
        info = {
            "frame_idx": idx,
            "episode": ep,
            "t_in_episode": t_in_episode,
            "episode_len": ep_len,
            "is_last": bool(t_in_episode >= ep_len - 1),
            "num_frames": self.num_frames,
            "num_episodes": self.num_episodes,
        }
        return s, info

    def _pick_reset_frame(self, selection: str, fixed_episode: int) -> int:
        if self.infer_mode == "fixed":
            return int(self._fixed_idx)
        if self.infer_mode == "episode":
            return int(self._episode_from)

        sel = str(selection).strip()
        if sel == "fixed":
            ep = max(0, min(int(fixed_episode), self.num_episodes - 1))
            return int(self.ep_from[ep])
        if sel == "mean_initial":
            # Fake server: mean_initial fallback to random first frame.
            ep = random.randrange(self.num_episodes)
            return int(self.ep_from[ep])
        # random
        ep = random.randrange(self.num_episodes)
        return int(self.ep_from[ep])

    def _extract_state(self, sample: Dict[str, Any]) -> np.ndarray:
        key = self.features["state"]
        if key not in sample:
            raise KeyError(f"sample missing state key: {key}")
        return _to_np_float32(sample[key], self.state_dim, "state")

    def _extract_action(self, sample: Dict[str, Any]) -> np.ndarray:
        key = self.features["action"]
        if key not in sample:
            raise KeyError(f"sample missing action key: {key}")
        return _to_np_float32(sample[key], self.action_dim, "action")

    def _extract_jpg_triplet(self, sample: Dict[str, Any]) -> Tuple[bytes, bytes, bytes]:
        kl = self.features["image_left"]
        kr = self.features["image_right"]
        km = self.features["image_middle"]
        if kl not in sample or kr not in sample or km not in sample:
            raise KeyError(f"sample missing image keys: {kl}/{kr}/{km}")
        jl = _image_to_jpg(sample[kl], self.preview_image_size, self.preview_jpeg_quality)
        jr = _image_to_jpg(sample[kr], self.preview_image_size, self.preview_jpeg_quality)
        jm = _image_to_jpg(sample[km], self.preview_image_size, self.preview_jpeg_quality)
        return jl, jr, jm

    def make_get_reset(self, selection: str, fixed_episode: int) -> Tuple[Dict[str, Any], bytes]:
        frame_idx = self._pick_reset_frame(selection, fixed_episode)
        sample, info = self._sample_by_frame(frame_idx)
        action = self._extract_action(sample)
        meta = {
            "ok": True,
            "cmd": "get_reset",
            "ts": time.time(),
            "mode": "fake_dataset",
            "selection": str(selection),
            "episode": int(info["episode"]),
            "t_in_episode": int(info["t_in_episode"]),
            "control_target_format": "full",
        }
        return meta, action.astype(np.float32).tobytes()

    def make_infer(self) -> Tuple[Dict[str, Any], bytes]:
        if self.infer_mode == "cycle":
            sample, info = self._sample_by_frame(self._cursor)
            self._cursor += 1
            if self._cursor >= self.num_frames:
                self._cursor = 0
        elif self.infer_mode == "episode":
            wrapped = False
            sample, info = self._sample_by_frame(self._episode_cursor)
            self._episode_cursor += 1
            if self._episode_cursor >= self._episode_to:
                self._episode_cursor = int(self._episode_from)
                self._episode_loop += 1
                wrapped = True
        else:
            sample, info = self._sample_by_frame(self._fixed_idx)
            wrapped = False

        action = self._extract_action(sample)
        meta = {
            "ok": True,
            "cmd": "infer",
            "ts": time.time(),
            "mode": "fake_dataset",
            "episode": int(info["episode"]),
            "t_in_episode": int(info["t_in_episode"]),
        }
        if self.infer_mode == "episode":
            meta["episode_replay"] = True
            meta["episode_loop"] = int(self._episode_loop)
            meta["wrapped"] = bool(wrapped)
        return meta, action.astype(np.float32).tobytes()

    def make_dataset_preview(
        self,
        pick_mode: str,
        episode: int = 0,
        t_in_episode: int = 0,
        do_infer: bool = False,
    ) -> Tuple[Dict[str, Any], List[bytes]]:
        mode = str(pick_mode).strip()
        if mode == "episode_t":
            ep = max(0, min(int(episode), self.num_episodes - 1))
            t = max(0, int(t_in_episode))
            ep_len = int(self.ep_to[ep] - self.ep_from[ep])
            t = min(t, max(ep_len - 1, 0))
            frame_idx = int(self.ep_from[ep] + t)
        elif mode == "random_episode_first":
            ep = random.randrange(self.num_episodes)
            frame_idx = int(self.ep_from[ep])
        elif mode == "random_frame":
            frame_idx = random.randrange(self.num_frames)
        else:
            raise ValueError(f"invalid pick_mode={mode}")

        sample, info = self._sample_by_frame(frame_idx)
        state_vec = self._extract_state(sample)
        action = self._extract_action(sample)
        jpg_left, jpg_right, jpg_middle = self._extract_jpg_triplet(sample)
        pred = action if bool(do_infer) else np.zeros((self.action_dim,), dtype=np.float32)
        control = action

        meta = {
            "ok": True,
            "cmd": "dataset_preview",
            "ts": time.time(),
            "pick_mode": mode,
            "episode": int(info["episode"]),
            "t_in_episode": int(info["t_in_episode"]),
            "frame_idx": int(info["frame_idx"]),
            "episode_len": int(info["episode_len"]),
            "is_last": bool(info["is_last"]),
            "do_infer": bool(do_infer),
            "policy_reset": False,
            "infer_input_mode": "dataset_obs_dataset_state",
            "obs_source": "dataset",
            "state_source": "dataset",
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "num_frames": int(info["num_frames"]),
            "num_episodes": int(info["num_episodes"]),
            "control_target_format": "full",
        }
        payloads = [
            jpg_left,
            jpg_right,
            jpg_middle,
            state_vec.astype(np.float32).tobytes(),
            pred.astype(np.float32).tobytes(),
            control.astype(np.float32).tobytes(),
        ]
        return meta, payloads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/pipeline_config.yaml")
    ap.add_argument(
        "--infer-mode",
        type=str,
        default="episode",
        choices=["episode", "fixed", "cycle"],
        help="episode: replay one episode from t=0->end by action; fixed: replay one fixed item; cycle: iterate all frames",
    )
    ap.add_argument(
        "--episode-index",
        type=int,
        default=-1,
        help="used in infer-mode=episode, -1 means random episode",
    )
    args = ap.parse_args()

    cfg = extract_server_config(load_yaml(args.config))
    cfg = resolve_paths(cfg, args.config)

    server_cfg = cfg.get("server", {})
    features = cfg.get("features", {})
    state_cfg = cfg.get("state", {})
    model_cfg = cfg.get("model", {})
    model_ds_cfg = model_cfg.get("dataset", {})
    preview_cfg = cfg.get("dataset_preview", {})

    bind = str(server_cfg.get("bind", "0.0.0.0")).strip() or "0.0.0.0"
    port = int(server_cfg.get("port", 5577))
    endpoint = f"tcp://{bind}:{port}"

    state_spec = build_state_spec(state_cfg, default_num_arms=2)
    state_dim = int(model_cfg.get("state_dim", state_spec["state_dim"]))
    action_dim = int(model_cfg.get("action_dim", state_spec["control_dim"]))

    ds_repo = str(model_ds_cfg.get("repo_id", "")).strip()
    ds_root = str(model_ds_cfg.get("root", "")).strip()
    if not ds_repo or not ds_root:
        raise ValueError("model.dataset.repo_id / model.dataset.root 不能为空（从 pipeline 继承后仍为空）")

    # Required feature keys
    feat_keys = {
        "image_left": str(features.get("image_left", "observation.image_left")).strip(),
        "image_right": str(features.get("image_right", "observation.image_right")).strip(),
        "image_middle": str(features.get("image_middle", "observation.image_middle")).strip(),
        "state": str(features.get("state", "observation.state")).strip(),
        "action": str(features.get("action", "action")).strip(),
    }

    ds = LeRobotDataset(ds_repo, root=ds_root)
    responder = DatasetResponder(
        ds=ds,
        features=feat_keys,
        state_dim=state_dim,
        action_dim=action_dim,
        preview_image_size=int(preview_cfg.get("image_size", int(model_cfg.get("image_size", 256)))),
        preview_jpeg_quality=int(preview_cfg.get("jpeg_quality", 80)),
        infer_mode=args.infer_mode,
        episode_index=int(args.episode_index),
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(endpoint)

    print("=" * 72)
    print("[FAKE] dataset-backed fake policy server started")
    print(f"[FAKE] endpoint={endpoint}")
    print(f"[FAKE] dataset={ds_root} repo={ds_repo}")
    print(f"[FAKE] infer_mode={args.infer_mode}")
    if args.infer_mode == "fixed":
        fi = responder.fixed_info
        print(
            "[FAKE] fixed replay item: frame_idx={} episode={} t_in_episode={}".format(
                fi["frame_idx"], fi["episode"], fi["t_in_episode"]
            )
        )
    if args.infer_mode == "episode":
        ei = responder.episode_info
        print(
            "[FAKE] episode replay: episode={} len={} from={} to={}".format(
                ei["episode"], ei["len"], ei["from"], ei["to"]
            )
        )
    print(f"[FAKE] state_dim={state_dim} action_dim={action_dim}")
    print(f"[FAKE] features={feat_keys}")
    print("[FAKE] protocol: get_reset / infer / dataset_preview")
    print("=" * 72)

    step = 0
    while True:
        parts = sock.recv_multipart()
        if not parts:
            sock.send_multipart([json.dumps({"ok": False, "err": "empty request"}).encode("utf-8"), b""])
            continue

        try:
            req_meta = json.loads(parts[0].decode("utf-8"))
        except Exception as e:
            sock.send_multipart([json.dumps({"ok": False, "err": f"bad json: {e}"}).encode("utf-8"), b""])
            continue

        cmd = str(req_meta.get("cmd", "")).strip()
        try:
            if cmd == "get_reset":
                selection = str(req_meta.get("selection", "random")).strip()
                fixed_episode = int(req_meta.get("fixed_episode", 0))
                rsp, payload = responder.make_get_reset(selection=selection, fixed_episode=fixed_episode)
                sock.send_multipart([json.dumps(rsp).encode("utf-8"), payload])
                continue

            if cmd == "infer":
                rsp, payload = responder.make_infer()
                rsp["t"] = int(step)
                step += 1
                sock.send_multipart([json.dumps(rsp).encode("utf-8"), payload])
                continue

            if cmd == "dataset_preview":
                pick_mode = str(req_meta.get("pick_mode", "random_frame")).strip()
                episode = int(req_meta.get("episode", 0))
                t_in_episode = int(req_meta.get("t_in_episode", 0))
                do_infer = bool(req_meta.get("do_infer", False))
                rsp, payloads = responder.make_dataset_preview(
                    pick_mode=pick_mode,
                    episode=episode,
                    t_in_episode=t_in_episode,
                    do_infer=do_infer,
                )
                sock.send_multipart([json.dumps(rsp).encode("utf-8")] + payloads)
                continue

            sock.send_multipart(
                [json.dumps({"ok": False, "err": f"unknown cmd: {cmd}"}).encode("utf-8"), b""]
            )
        except Exception as e:
            traceback.print_exc()
            sock.send_multipart([json.dumps({"ok": False, "err": str(e)}).encode("utf-8"), b""])


if __name__ == "__main__":
    main()
