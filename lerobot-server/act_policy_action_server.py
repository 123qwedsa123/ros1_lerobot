#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT policy action server (ZMQ REP, dataset-driven observation).

This server keeps the same request/response protocol as hdf5_action_server.py:
  - hello : return server metadata
  - reset : reset policy + episode cursor, return reset target action
  - next  : run ACT policy inference for current sample, return predicted action

Response multipart:
  [json_meta, action_bytes(float32)]
"""

import argparse
import json
import random
import signal
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import zmq

try:
    # lerobot >= 0.4.x
    from lerobot import __version__ as LEROBOT_VERSION
except Exception:
    LEROBOT_VERSION = "unknown"

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType

    LEROBOT_IMPORT_STYLE = "lerobot.*"
except Exception:
    # compatibility fallback
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType

    LEROBOT_IMPORT_STYLE = "lerobot.common.*"


def _to_chw_float01(x: Any) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim != 3:
        raise RuntimeError(f"image tensor ndim must be 3, got {x.ndim}")

    # CHW
    if x.shape[0] == 3:
        chw = x
    # HWC
    elif x.shape[-1] == 3:
        chw = x.permute(2, 0, 1)
    else:
        raise RuntimeError(f"unsupported image shape={tuple(x.shape)}")

    chw = chw.float()
    if float(chw.max()) > 1.5:
        chw = chw / 255.0
    return chw.unsqueeze(0)


def _to_vec_batch(x: Any) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.float()
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2 and x.shape[0] == 1:
        return x
    return x.reshape(1, -1)


def _first_action(action_tensor: Any) -> np.ndarray:
    x = action_tensor
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 3:
        x = x[0, 0, :]
    elif x.ndim == 2:
        x = x[0, :]
    elif x.ndim == 1:
        pass
    else:
        raise RuntimeError(f"bad action shape={tuple(x.shape)}")
    return x.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _build_act_config(input_features, output_features, chunk_size: int, n_action_steps: int, temporal_ensemble_coeff: float):
    kwargs = {
        "input_features": input_features,
        "output_features": output_features,
        "chunk_size": int(chunk_size),
        "n_action_steps": int(n_action_steps),
    }
    if temporal_ensemble_coeff is not None:
        kwargs["temporal_ensemble_coeff"] = float(temporal_ensemble_coeff)
    try:
        return ACTConfig(**kwargs)
    except TypeError:
        kwargs.pop("temporal_ensemble_coeff", None)
        return ACTConfig(**kwargs)


def _get_episode_boundaries(ds) -> Dict[str, List[int]]:
    ep = getattr(ds, "episode_data_index", None)
    if ep is not None:
        try:
            ep_from = [int(x) for x in ep["from"]]
            ep_to = [int(x) for x in ep["to"]]
            if len(ep_from) == len(ep_to):
                return {"from": ep_from, "to": ep_to}
        except Exception:
            pass

    hf = getattr(ds, "hf_dataset", None)
    if hf is None:
        raise AttributeError("dataset has neither episode_data_index nor hf_dataset")

    try:
        episode_col = hf["episode_index"]
    except Exception:
        episode_col = [hf[i]["episode_index"] for i in range(len(hf))]

    episode_ids = [int(x) for x in episode_col]
    if len(episode_ids) == 0:
        return {"from": [], "to": []}

    ep_from = [0]
    ep_to = []
    prev = episode_ids[0]
    for i in range(1, len(episode_ids)):
        cur = episode_ids[i]
        if cur != prev:
            ep_to.append(i)
            ep_from.append(i)
            prev = cur
    ep_to.append(len(episode_ids))
    return {"from": ep_from, "to": ep_to}


class ActPolicyActionServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.seed is not None:
            random.seed(int(args.seed))
            np.random.seed(int(args.seed))
            torch.manual_seed(int(args.seed))

        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

        print("=" * 72, flush=True)
        print(f"[INIT] lerobot={LEROBOT_VERSION} imports={LEROBOT_IMPORT_STYLE}", flush=True)
        print(f"[INIT] device={self.device}", flush=True)
        print(f"[INIT] repo_id={args.repo_id}", flush=True)
        print(f"[INIT] dataset_root={args.dataset_root}", flush=True)
        print(f"[INIT] ckpt_dir={args.ckpt_dir}", flush=True)
        print("=" * 72, flush=True)

        self.dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
        feats = dataset_to_policy_features(self.dataset_metadata.features)
        self.out_feats = {k: ft for k, ft in feats.items() if ft.type is FeatureType.ACTION}
        self.in_feats = {k: ft for k, ft in feats.items() if k not in self.out_feats}

        if not self.out_feats:
            raise RuntimeError("no action feature found in dataset metadata")
        if not self.in_feats:
            raise RuntimeError("no input feature found in dataset metadata")

        self.act_cfg = _build_act_config(
            input_features=self.in_feats,
            output_features=self.out_feats,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        )
        delta_ts = resolve_delta_timestamps(self.act_cfg, self.dataset_metadata)
        self.ds = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=delta_ts)

        self.policy = ACTPolicy.from_pretrained(
            args.ckpt_dir,
            config=self.act_cfg,
            dataset_stats=self.dataset_metadata.stats,
        )
        self.policy.eval().to(self.device)
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        self.ep_bounds = _get_episode_boundaries(self.ds)
        self.num_episodes = len(self.ep_bounds["from"])
        if self.num_episodes <= 0:
            raise RuntimeError("dataset has zero episodes")

        # action dim from first sample action
        first_sample = self.ds[0]
        first_action = _first_action(first_sample["action"])
        self.action_dim = int(first_action.size)
        if self.action_dim <= 0:
            raise RuntimeError("action_dim <= 0")

        self.current_episode = -1
        self.current_indices: List[int] = []
        self.t = 0
        self.T = 0
        self.reset_target = np.zeros((self.action_dim,), dtype=np.float32)

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.RCVTIMEO, int(args.recv_timeout_ms))
        self.sock.setsockopt(zmq.SNDTIMEO, int(args.send_timeout_ms))
        self.sock.setsockopt(zmq.LINGER, int(args.linger_ms))
        self.sock.bind(f"tcp://{args.bind}:{args.port}")

        self.running = True
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, _frame) -> None:
        self.running = False
        print(f"[SIGNAL] {signum}, stopping ...", flush=True)

    def _rsp(self, meta: Dict[str, Any], payload: np.ndarray = None) -> List[bytes]:
        payload_bytes = b""
        if payload is not None:
            payload_bytes = np.asarray(payload, dtype=np.float32).reshape(-1).tobytes()
        return [json.dumps(meta).encode("utf-8"), payload_bytes]

    def _pick_episode(self) -> int:
        if self.args.episode >= 0:
            eid = int(self.args.episode)
            if eid >= self.num_episodes:
                raise RuntimeError(f"episode={eid} out of range [0,{self.num_episodes - 1}]")
            return eid
        return random.randrange(self.num_episodes)

    def _episode_indices(self, episode_idx: int) -> List[int]:
        s = int(self.ep_bounds["from"][episode_idx])
        e = int(self.ep_bounds["to"][episode_idx])
        return list(range(s, e))

    def _extract_reset_target(self, sample: Dict[str, Any]) -> np.ndarray:
        # Prefer first action-sized slice from observation.state.
        st = sample.get("observation.state", None)
        if st is not None:
            v = np.asarray(st, dtype=np.float32).reshape(-1)
            if v.size >= self.action_dim:
                return v[: self.action_dim].astype(np.float32)

        # Fallback to sample action itself.
        act = _first_action(sample["action"])
        if act.size != self.action_dim:
            raise RuntimeError(f"reset target dim mismatch: {act.size} != {self.action_dim}")
        return act.astype(np.float32)

    def _reset_episode(self) -> None:
        self.current_episode = self._pick_episode()
        self.current_indices = self._episode_indices(self.current_episode)
        self.T = len(self.current_indices)
        if self.T <= 0:
            raise RuntimeError(f"episode={self.current_episode} is empty")
        self.t = 0

        first_sample = self.ds[self.current_indices[0]]
        self.reset_target = self._extract_reset_target(first_sample)

        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _dt(self) -> float:
        hz = max(float(self.args.control_hz), 1e-6)
        return float(1.0 / hz)

    def _build_policy_batch(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, torch.Tensor] = {}
        for key in self.in_feats.keys():
            if key not in sample:
                raise KeyError(f"missing key in sample: {key}")
            v = sample[key]
            if "image" in key:
                x = _to_chw_float01(v)
            else:
                x = _to_vec_batch(v)
            batch[key] = x.to(self.device)
        return batch

    def _infer_action(self) -> np.ndarray:
        if self.t >= self.T:
            raise RuntimeError("episode cursor overflow")

        sample = self.ds[self.current_indices[self.t]]
        batch = self._build_policy_batch(sample)
        with torch.no_grad():
            pred = self.policy.select_action(batch)
        action = _first_action(pred)
        if action.size != self.action_dim:
            raise RuntimeError(f"bad predicted action dim: {action.size} != {self.action_dim}")
        return action.astype(np.float32)

    def _handle_hello(self) -> List[bytes]:
        meta = {
            "ok": True,
            "cmd": "hello",
            "mode": "act_policy",
            "repo_id": self.args.repo_id,
            "ckpt_dir": self.args.ckpt_dir,
            "num_episodes": int(self.num_episodes),
            "action_dim": int(self.action_dim),
            "chunk_size": int(self.args.chunk_size),
            "n_action_steps": int(self.args.n_action_steps),
            "temporal_ensemble_coeff": float(self.args.temporal_ensemble_coeff),
            "control_hz": float(self.args.control_hz),
            "loop": bool(self.args.loop),
            "ts": time.time(),
        }
        return self._rsp(meta)

    def _handle_reset(self) -> List[bytes]:
        self._reset_episode()
        meta = {
            "ok": True,
            "cmd": "reset",
            "episode": int(self.current_episode),
            "frame_idx": 0,
            "T": int(self.T),
            "action_dim": int(self.action_dim),
            "dt_sec": self._dt(),
            "ts": time.time(),
        }
        return self._rsp(meta, self.reset_target)

    def _handle_next(self) -> List[bytes]:
        if self.current_episode < 0:
            return self._rsp({"ok": False, "err": "must call reset first", "ts": time.time()})

        if self.t >= self.T:
            if self.args.loop:
                self._reset_episode()
            else:
                meta = {
                    "ok": True,
                    "cmd": "next",
                    "done": True,
                    "episode": int(self.current_episode),
                    "frame_idx": int(self.T),
                    "T": int(self.T),
                    "ts": time.time(),
                }
                return self._rsp(meta)

        frame_idx = int(self.t)
        action = self._infer_action()
        self.t += 1
        meta = {
            "ok": True,
            "cmd": "next",
            "done": False,
            "episode": int(self.current_episode),
            "frame_idx": frame_idx,
            "T": int(self.T),
            "action_dim": int(self.action_dim),
            "dt_sec": self._dt(),
            "ts": time.time(),
        }
        return self._rsp(meta, action)

    def _handle(self, req: Dict[str, Any]) -> List[bytes]:
        cmd = str(req.get("cmd", "")).strip().lower()
        if cmd == "hello":
            return self._handle_hello()
        if cmd in ("reset", "restart"):
            return self._handle_reset()
        if cmd == "next":
            return self._handle_next()
        return self._rsp({"ok": False, "err": f"unknown cmd: {cmd}", "ts": time.time()})

    def run(self) -> None:
        print("=" * 72, flush=True)
        print(f"[SERVER] bind=tcp://{self.args.bind}:{self.args.port}", flush=True)
        print(f"[SERVER] mode=ACT policy inference", flush=True)
        print(
            f"[SERVER] action_dim={self.action_dim}, episodes={self.num_episodes}, "
            f"chunk={self.args.chunk_size}, n_action_steps={self.args.n_action_steps}, "
            f"temporal_ensemble_coeff={self.args.temporal_ensemble_coeff}",
            flush=True,
        )
        print("=" * 72, flush=True)

        while self.running:
            try:
                parts = self.sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception as exc:
                print(f"[WARN] recv failed: {exc}", flush=True)
                continue

            try:
                if not parts:
                    rsp = self._rsp({"ok": False, "err": "empty request", "ts": time.time()})
                else:
                    req = json.loads(parts[0].decode("utf-8"))
                    rsp = self._handle(req)
            except Exception as exc:
                rsp = self._rsp({"ok": False, "err": str(exc), "ts": time.time()})

            try:
                self.sock.send_multipart(rsp)
            except Exception as exc:
                print(f"[WARN] send failed: {exc}", flush=True)

        try:
            self.sock.close(0)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ACT policy action server")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5560)

    ap.add_argument("--repo-id", default="baseline")
    ap.add_argument("--dataset-root", default="./lerobot_baseline_dataset")
    ap.add_argument("--ckpt-dir", default="./output_baseline")

    ap.add_argument("--episode", type=int, default=-1, help="-1 means random episode on reset")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--n-action-steps", type=int, default=1)
    ap.add_argument("--temporal-ensemble-coeff", type=float, default=0.01)
    ap.add_argument("--control-hz", type=float, default=30.0)

    ap.add_argument("--recv-timeout-ms", type=int, default=1000)
    ap.add_argument("--send-timeout-ms", type=int, default=1000)
    ap.add_argument("--linger-ms", type=int, default=0)

    args = ap.parse_args()
    if args.port < 1 or args.port > 65535:
        raise ValueError(f"bad port: {args.port}")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be > 0")
    if args.n_action_steps <= 0:
        raise ValueError("n-action-steps must be > 0")
    if args.control_hz <= 0:
        raise ValueError("control-hz must be > 0")
    return args


def main() -> int:
    try:
        args = parse_args()
        server = ActPolicyActionServer(args)
        server.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
