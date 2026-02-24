#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 action replay server (ZMQ REP).

Protocol (multipart):
  request : [json_meta]
  response: [json_meta, action_bytes]

Commands:
  - hello: return dataset metadata
  - reset: rewind to frame 0 and return first action
  - next : return current action + dt, then advance cursor

Action layout:
  concat(arm0[7], arm1[7], ...), float32
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np
import zmq


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _resolve_episode_path(data_dir: str, episode_id: int, hdf5_path: str) -> str:
    if hdf5_path:
        p = os.path.abspath(os.path.expanduser(hdf5_path))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"hdf5 not found: {p}")
        return p

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(data_dir):
        m = re.match(r"^episode_(\d+)\.hdf5$", name)
        if m:
            candidates.append((int(m.group(1)), os.path.join(data_dir, name)))

    if not candidates:
        raise FileNotFoundError(f"no episode_*.hdf5 in {data_dir}")

    candidates.sort(key=lambda x: x[0])
    if episode_id < 0:
        return candidates[-1][1]

    for eid, path in candidates:
        if eid == episode_id:
            return path
    raise FileNotFoundError(f"episode_{episode_id}.hdf5 not found in {data_dir}")


def _fix_len7(row: np.ndarray) -> np.ndarray:
    out = np.zeros((7,), dtype=np.float32)
    n = min(int(row.size), 7)
    out[:n] = row[:n].astype(np.float32)
    return out


def _parse_arm_keys(slaves_group: h5py.Group, arm_keys_raw: str) -> List[str]:
    if arm_keys_raw.strip():
        keys = [k.strip() for k in arm_keys_raw.split(",") if k.strip()]
        if not keys:
            raise ValueError("arm_keys is empty after parsing")
    else:
        keys = sorted(list(slaves_group.keys()))
    if not keys:
        raise ValueError("no arm keys found under /slaves")
    return keys


class Hdf5ActionServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.episode_path = _resolve_episode_path(args.data_dir, args.episode_id, args.hdf5_path)

        self.arm_keys: List[str] = []
        self.timestamps = np.zeros((0,), dtype=np.float64)
        self.actions = np.zeros((0, 0), dtype=np.float32)
        self.frame_count = 0
        self.action_dim = 0
        self.idx = 0
        self._load_hdf5()

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

    def _load_hdf5(self) -> None:
        with h5py.File(self.episode_path, "r") as f:
            if self.args.timestamps_key not in f:
                raise KeyError(f"missing /{self.args.timestamps_key}")
            if self.args.slaves_group not in f:
                raise KeyError(f"missing /{self.args.slaves_group}")

            ts = np.asarray(f[self.args.timestamps_key], dtype=np.float64).reshape(-1)
            slaves = f[self.args.slaves_group]
            self.arm_keys = _parse_arm_keys(slaves, self.args.arm_keys)

            arm_pos: Dict[str, np.ndarray] = {}
            lengths = [int(ts.shape[0])]
            for key in self.arm_keys:
                if key not in slaves:
                    raise KeyError(f"missing /{self.args.slaves_group}/{key}")
                grp = slaves[key]
                if "positions" not in grp:
                    raise KeyError(f"missing /{self.args.slaves_group}/{key}/positions")
                pos = np.asarray(grp["positions"], dtype=np.float32)
                if pos.ndim == 1:
                    pos = pos.reshape(1, -1)
                if pos.ndim != 2:
                    raise ValueError(f"bad positions shape for {key}: {pos.shape}")
                arm_pos[key] = pos
                lengths.append(int(pos.shape[0]))

        common_len = min(lengths)
        if common_len <= 0:
            raise ValueError("empty trajectory")

        self.timestamps = ts[:common_len]
        self.frame_count = common_len
        self.action_dim = len(self.arm_keys) * 7
        self.actions = np.zeros((self.frame_count, self.action_dim), dtype=np.float32)

        for i in range(self.frame_count):
            chunks = []
            for key in self.arm_keys:
                chunks.append(_fix_len7(arm_pos[key][i]))
            self.actions[i] = np.concatenate(chunks, axis=0).astype(np.float32)

    def _dt_for_index(self, idx: int) -> float:
        if idx >= self.frame_count - 1:
            return float(self.args.hold_last_sec)
        raw = float(self.timestamps[idx + 1] - self.timestamps[idx])
        if not np.isfinite(raw) or raw <= 0.0:
            raw = 1.0 / float(self.args.fallback_hz)
        scaled = raw / float(self.args.speed_scale)
        return float(_clamp(scaled, float(self.args.min_dt_sec), float(self.args.max_dt_sec)))

    def _rsp(self, meta: Dict, action: np.ndarray = None) -> List[bytes]:
        payload = b"" if action is None else np.asarray(action, np.float32).reshape(-1).tobytes()
        return [json.dumps(meta).encode("utf-8"), payload]

    def _handle_hello(self) -> List[bytes]:
        meta = {
            "ok": True,
            "cmd": "hello",
            "episode_path": self.episode_path,
            "frame_count": int(self.frame_count),
            "arm_keys": list(self.arm_keys),
            "arm_count": int(len(self.arm_keys)),
            "action_dim": int(self.action_dim),
            "speed_scale": float(self.args.speed_scale),
            "loop": bool(self.args.loop),
            "ts": time.time(),
        }
        return self._rsp(meta)

    def _handle_reset(self) -> List[bytes]:
        self.idx = 0
        action0 = self.actions[0]
        meta = {
            "ok": True,
            "cmd": "reset",
            "frame_idx": 0,
            "dt_sec": float(self._dt_for_index(0)),
            "action_dim": int(self.action_dim),
            "ts": time.time(),
        }
        return self._rsp(meta, action0)

    def _handle_next(self) -> List[bytes]:
        if self.idx >= self.frame_count:
            if self.args.loop:
                self.idx = 0
            else:
                meta = {
                    "ok": True,
                    "cmd": "next",
                    "done": True,
                    "frame_idx": int(self.frame_count),
                    "ts": time.time(),
                }
                return self._rsp(meta)

        i = self.idx
        action = self.actions[i]
        dt = self._dt_for_index(i)
        self.idx += 1
        meta = {
            "ok": True,
            "cmd": "next",
            "done": False,
            "frame_idx": int(i),
            "dt_sec": float(dt),
            "action_dim": int(self.action_dim),
            "ts": time.time(),
        }
        return self._rsp(meta, action)

    def _handle(self, req: Dict) -> List[bytes]:
        cmd = str(req.get("cmd", "")).strip().lower()
        if cmd == "hello":
            return self._handle_hello()
        if cmd == "reset":
            return self._handle_reset()
        if cmd == "next":
            return self._handle_next()
        if cmd == "restart":
            return self._handle_reset()
        meta = {"ok": False, "err": f"unknown cmd: {cmd}", "ts": time.time()}
        return self._rsp(meta)

    def run(self) -> None:
        print("=" * 72, flush=True)
        print(f"[SERVER] bind=tcp://{self.args.bind}:{self.args.port}", flush=True)
        print(f"[SERVER] episode={self.episode_path}", flush=True)
        print(
            f"[SERVER] frames={self.frame_count}, arms={self.arm_keys}, action_dim={self.action_dim}",
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
    ap = argparse.ArgumentParser(description="HDF5 action replay server")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--hdf5-path", default="")
    ap.add_argument("--data-dir", default="/workspace/piper_master_slave_ws/data")
    ap.add_argument("--episode-id", type=int, default=-1)
    ap.add_argument("--timestamps-key", default="timestamps")
    ap.add_argument("--slaves-group", default="slaves")
    ap.add_argument(
        "--arm-keys",
        default="",
        help="comma separated keys under /slaves, empty means sorted all keys",
    )
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--fallback-hz", type=float, default=30.0)
    ap.add_argument("--min-dt-sec", type=float, default=0.001)
    ap.add_argument("--max-dt-sec", type=float, default=0.2)
    ap.add_argument("--hold-last-sec", type=float, default=0.5)
    ap.add_argument("--recv-timeout-ms", type=int, default=1000)
    ap.add_argument("--send-timeout-ms", type=int, default=1000)
    ap.add_argument("--linger-ms", type=int, default=0)
    args = ap.parse_args()

    if args.port < 1 or args.port > 65535:
        raise ValueError(f"bad port: {args.port}")
    if args.speed_scale <= 0:
        raise ValueError("speed-scale must be > 0")
    if args.fallback_hz <= 0:
        raise ValueError("fallback-hz must be > 0")
    if args.min_dt_sec <= 0:
        raise ValueError("min-dt-sec must be > 0")
    if args.max_dt_sec < args.min_dt_sec:
        raise ValueError("max-dt-sec must be >= min-dt-sec")
    return args


def main() -> int:
    try:
        args = parse_args()
        server = Hdf5ActionServer(args)
        server.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
