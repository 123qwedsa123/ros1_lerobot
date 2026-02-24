#!/usr/bin/env python3
"""
Incremental bag -> LeRobot converter (producer-consumer style).

This script watches a teleop session directory for completed episodes and
converts each episode as soon as it is ready, without waiting for all bags.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bag_to_lerobot as b2l  # noqa: E402


STATE_VERSION = 1


def _resolve_config_relative(path: Path, config_dir: Path) -> Path:
    return path if path.is_absolute() else (config_dir / path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _episode_sort_key(path: Path) -> int:
    m = re.match(r"episode_(\d+)$", path.name)
    return int(m.group(1)) if m else 10**9


def _iter_episode_dirs(session_dir: Path) -> List[Path]:
    if not session_dir.is_dir():
        return []
    dirs = [p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    return sorted(dirs, key=_episode_sort_key)


def _is_bag_ready(bag_path: Path, settle_sec: float, require_metadata: bool) -> bool:
    if not bag_path.is_file():
        return False
    if require_metadata and not (bag_path.parent / "metadata.json").is_file():
        return False
    age = time.time() - bag_path.stat().st_mtime
    return age >= settle_sec


def _index_stats(start_idx: int, length: int) -> Dict[str, List[float]]:
    idx = np.arange(start_idx, start_idx + length, dtype=np.int64)
    return {
        "min": [float(idx.min())],
        "max": [float(idx.max())],
        "mean": [float(idx.mean())],
        "std": [float(idx.std())],
        "count": [int(length)],
    }


def _patch_global_index(parquet_path: Path, start_idx: int) -> Dict[str, List[float]]:
    df = pd.read_parquet(parquet_path)
    length = len(df)
    idx = np.arange(start_idx, start_idx + length, dtype=np.int64)
    df["index"] = idx
    df.to_parquet(parquet_path, index=False)
    return _index_stats(start_idx, length)


def _state_template(cfg: b2l.ConvertConfig) -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "session_dir": str(cfg.session_dir.resolve()),
        "dataset_dir": str(cfg.dataset_dir.resolve()),
        "dataset_id": cfg.dataset_id,
        "converted": [],
        "failed": {},
    }


def _load_state(state_path: Path, cfg: b2l.ConvertConfig) -> Dict[str, Any]:
    if not state_path.is_file():
        dataset_data_dir = cfg.dataset_dir / "data"
        if dataset_data_dir.exists() and any(dataset_data_dir.rglob("episode_*.parquet")):
            raise RuntimeError(
                f"Dataset already exists but state file is missing: {state_path}\n"
                "Use a new dataset_id/output_root, or remove existing dataset output first."
            )
        return _state_template(cfg)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("version", -1)) != STATE_VERSION:
        raise RuntimeError(f"Unsupported state version in {state_path}")

    expected_session = str(cfg.session_dir.resolve())
    expected_dataset = str(cfg.dataset_dir.resolve())
    if state.get("session_dir") != expected_session:
        raise RuntimeError(
            f"State session_dir mismatch.\nstate: {state.get('session_dir')}\nconfig: {expected_session}"
        )
    if state.get("dataset_dir") != expected_dataset:
        raise RuntimeError(
            f"State dataset_dir mismatch.\nstate: {state.get('dataset_dir')}\nconfig: {expected_dataset}"
        )

    state.setdefault("converted", [])
    state.setdefault("failed", {})
    return state


def _record_to_result(cfg: b2l.ConvertConfig, writer: b2l.EpisodeWriter, rec: Dict[str, Any]) -> b2l.EpisodeResult:
    ep_idx = int(rec["episode_index"])
    chunk = ep_idx // 1000
    parquet_path = cfg.dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    video_paths = {
        "observation.image": writer._video_path(ep_idx, "observation.image"),
        "observation.left_wrist_image": writer._video_path(ep_idx, "observation.left_wrist_image"),
        "observation.right_wrist_image": writer._video_path(ep_idx, "observation.right_wrist_image"),
    }
    return b2l.EpisodeResult(
        episode_index=ep_idx,
        length=int(rec["length"]),
        parquet_path=parquet_path,
        video_paths=video_paths,
        stats=rec["stats"],
    )


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    # Keep deterministic order in state file.
    state["converted"] = sorted(state["converted"], key=lambda x: int(x["episode_index"]))
    _atomic_write_json(state_path, state)


def _next_episode_index(converted: List[Dict[str, Any]]) -> int:
    if not converted:
        return 0
    return max(int(item["episode_index"]) for item in converted) + 1


def _converted_total_frames(converted: List[Dict[str, Any]]) -> int:
    return sum(int(item["length"]) for item in converted)


def _format_sec(sec: float) -> str:
    return f"{sec:.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch teleop bags and convert incrementally to LeRobot v2.1"
    )
    parser.add_argument("--config", default="config/bag_convert_config.yaml")
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--settle-sec", type=float, default=3.0)
    parser.add_argument("--require-metadata", action="store_true", default=True)
    parser.add_argument("--no-require-metadata", dest="require_metadata", action="store_false")
    parser.add_argument("--once", action="store_true", help="Convert ready episodes once, then exit")
    parser.add_argument(
        "--idle-exit-sec",
        type=float,
        default=0.0,
        help="Exit after this many seconds with no new conversion (0 = never)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry episodes that previously failed in state file",
    )
    parser.add_argument(
        "--phase2-on-exit",
        action="store_true",
        help="Run v2.1 -> v3.0 conversion once when watcher exits (if config convert_to_v30=true)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    cfg = b2l.load_config(str(config_path))
    cfg.session_dir = _resolve_config_relative(cfg.session_dir, config_path.parent).resolve()
    cfg.output_root = _resolve_config_relative(cfg.output_root, config_path.parent).resolve()

    if cfg.clean_output:
        print("[watch] clean_output=true in config, forcing clean_output=false for incremental mode")
        cfg.clean_output = False

    if not cfg.session_dir.is_dir():
        raise SystemExit(f"Session dir not found: {cfg.session_dir}")

    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file).expanduser().resolve() if args.state_file else (
        cfg.dataset_dir / ".stream_convert_state.json"
    )

    state = _load_state(state_path, cfg)
    converted: List[Dict[str, Any]] = state["converted"]
    failed: Dict[str, Dict[str, Any]] = state["failed"]

    converter = b2l.BagToLeRobotConverter(cfg)
    reader = converter.reader
    sampler = converter.sampler
    writer = converter.writer
    meta = converter.meta

    converted_bags = {str(Path(item["source_bag"]).resolve()) for item in converted}
    if args.retry_failed:
        failed = {}
        state["failed"] = failed

    results = [_record_to_result(cfg, writer, rec) for rec in sorted(converted, key=lambda x: int(x["episode_index"]))]
    next_ep_idx = _next_episode_index(converted)
    total_frames = _converted_total_frames(converted)
    last_activity_t = time.time()

    print(f"[watch] session_dir: {cfg.session_dir}")
    print(f"[watch] dataset_dir: {cfg.dataset_dir}")
    print(f"[watch] state_file : {state_path}")
    print(f"[watch] already converted: {len(converted)} episodes")
    if failed:
        print(f"[watch] known failed episodes: {len(failed)}")

    while True:
        ready_bags: List[Path] = []
        for ep_dir in _iter_episode_dirs(cfg.session_dir):
            bag_path = ep_dir / "episode.bag"
            bag_key = str(bag_path.resolve())
            if bag_key in converted_bags:
                continue
            if bag_key in failed and not args.retry_failed:
                continue
            if _is_bag_ready(bag_path, args.settle_sec, args.require_metadata):
                ready_bags.append(bag_path)

        if not ready_bags:
            if args.once:
                print("[watch] no ready episodes in --once mode, exiting")
                break
            if args.idle_exit_sec > 0.0:
                idle = time.time() - last_activity_t
                if idle >= args.idle_exit_sec:
                    print(f"[watch] idle timeout reached ({_format_sec(idle)}), exiting")
                    break
            time.sleep(max(0.1, args.poll_sec))
            continue

        for bag_path in ready_bags:
            bag_key = str(bag_path.resolve())
            if bag_key in converted_bags:
                continue

            src_episode = bag_path.parent.name
            print(f"[watch] converting {src_episode} -> episode_{next_ep_idx:06d}")
            try:
                data = reader.read(bag_path)
                sampled = sampler.sample(data)
                result = writer.write(next_ep_idx, data, sampled)
                result.stats["index"] = _patch_global_index(result.parquet_path, total_frames)

                record = {
                    "source_bag": bag_key,
                    "source_episode": src_episode,
                    "source_size": int(bag_path.stat().st_size),
                    "source_mtime": float(bag_path.stat().st_mtime),
                    "episode_index": int(next_ep_idx),
                    "length": int(result.length),
                    "stats": result.stats,
                }
                state["converted"].append(record)
                converted_bags.add(bag_key)
                if bag_key in failed:
                    failed.pop(bag_key, None)

                results.append(result)
                next_ep_idx += 1
                total_frames += result.length
                meta.write(results)
                _save_state(state_path, state)

                last_activity_t = time.time()
                print(
                    f"[watch] done {src_episode}, frames={result.length}, "
                    f"total_episodes={len(results)}, total_frames={total_frames}"
                )

            except b2l.EpisodeSkipError as exc:
                failed[bag_key] = {
                    "source_episode": src_episode,
                    "reason": str(exc),
                    "time": time.time(),
                }
                _save_state(state_path, state)
                print(f"[watch] skip {src_episode}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failed[bag_key] = {
                    "source_episode": src_episode,
                    "reason": str(exc),
                    "time": time.time(),
                }
                _save_state(state_path, state)
                print(f"[watch] error {src_episode}: {exc}")

        if args.once:
            break

    if args.phase2_on_exit and cfg.convert_to_v30:
        print("[watch] phase2_on_exit enabled, starting v2.1 -> v3.0 conversion")
        b2l.run_phase2(cfg)
    elif args.phase2_on_exit:
        print("[watch] phase2_on_exit enabled but convert_to_v30=false in config, skipped")

    print("[watch] finished")


if __name__ == "__main__":
    main()
