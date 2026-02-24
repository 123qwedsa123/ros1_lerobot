#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Server - 三相机版本（state 配置化）
特征名: image_left (cam1), image_right (cam2), image_middle (cam3)

四种模式:
1) dataset       : obs 和 state 都来自 dataset，用于验证 policy
2) client        : obs 和 state 都来自 client，实际部署
3) client_obs    : obs 来自 client，state 来自 dataset
4) client_state  : obs 来自 dataset，state 来自 client

Reset 流程:
1) 随机抽取一条 episode
2) 读取第一帧 state（按 config 的 state_dim）
3) 从 state 中提取 positions，发给 client 复位（通常 14 维）
4) policy.reset()
5) client 复位成功后开始 infer

Client 发送:
- mode=client       : [meta, jpg1, jpg2, jpg3, state_bytes]
- mode=client_obs   : [meta, jpg1, jpg2, jpg3]
- mode=client_state : [meta, state_bytes]
- get_reset         : client 可以发空 state，server 不读取
"""

import argparse
import json
import time
import random
import os
from typing import Any, Dict, List

import numpy as np
import zmq
import cv2
import torch
import yaml
from PIL import Image as PILImage

try:
    from lerobot import __version__ as LEROBOT_VERSION
except Exception:
    LEROBOT_VERSION = "unknown"

try:
    # lerobot >= 0.4.x (例如 0.4.3)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType
    LEROBOT_IMPORT_STYLE = "lerobot.*"
except Exception:
    # 兼容旧路径
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import FeatureType
    LEROBOT_IMPORT_STYLE = "lerobot.common.*"

DEFAULT_REPO_ID = "baseline"
DEFAULT_DATASET_ROOT = "./lerobot_baseline_dataset"
DEFAULT_CKPT_DIR = "./output_baseline"
DEFAULT_CONFIG_PATH = "config/pipeline_config.yaml"
IMAGE_SIZE = 256
DEFAULT_SAVE_DIR = "./client_data_saved_3cams"
VALID_STATE_FIELD_KEYS = {"positions", "velocities", "efforts"}

MODE_DATASET = "dataset"
MODE_CLIENT = "client"
MODE_CLIENT_OBS = "client_obs"
MODE_CLIENT_STATE = "client_state"
VALID_MODES = [MODE_DATASET, MODE_CLIENT, MODE_CLIENT_OBS, MODE_CLIENT_STATE]
CLIENT_ACTION_SOURCE_POLICY = "policy"
CLIENT_ACTION_SOURCE_DATASET = "dataset"
VALID_CLIENT_ACTION_SOURCES = [CLIENT_ACTION_SOURCE_POLICY, CLIENT_ACTION_SOURCE_DATASET]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def build_act_config(input_features, output_features, chunk_size: int, n_action_steps: int, temporal_ensemble_coeff: float):
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


def get_episode_boundaries(ds) -> Dict[str, List[int]]:
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
        raise AttributeError(
            "LeRobotDataset has neither episode_data_index nor hf_dataset to infer episode boundaries."
        )

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


def _resolve_indices(field_def: Dict[str, Any], total_dim: int) -> List[int]:
    indices = field_def.get("joint_indices", None)
    exclude = field_def.get("exclude_indices", None)
    if indices is not None and exclude is not None:
        raise ValueError("joint_indices and exclude_indices are mutually exclusive")

    if indices is not None:
        out = [int(i) for i in indices]
    elif exclude is not None:
        ex = {int(i) for i in exclude}
        out = [i for i in range(total_dim) if i not in ex]
    else:
        out = list(range(total_dim))

    for idx in out:
        if idx < 0 or idx >= total_dim:
            raise ValueError(f"joint index out of range: {idx}, total_dim={total_dim}")
    return out


def build_state_spec(state_cfg: Dict[str, Any], default_num_arms: int = 2) -> Dict[str, Any]:
    num_arms = int(state_cfg.get("num_arms", default_num_arms))
    per_arm_dim = int(state_cfg.get("per_arm_dim", 7))
    history_steps = int(state_cfg.get("history_steps", 0))
    if num_arms <= 0:
        raise ValueError("state.num_arms must be > 0")
    if per_arm_dim <= 0:
        raise ValueError("state.per_arm_dim must be > 0")
    if history_steps < 0:
        raise ValueError("state.history_steps must be >= 0")

    raw_fields = state_cfg.get("fields", None)
    if not isinstance(raw_fields, list) or len(raw_fields) == 0:
        raise ValueError("state.fields must be a non-empty list")

    fields: List[Dict[str, Any]] = []
    for idx, raw_field in enumerate(raw_fields):
        if isinstance(raw_field, str):
            raw_field = {"key": raw_field}
        if not isinstance(raw_field, dict):
            raise ValueError(f"state.fields[{idx}] must be dict|str")
        if not bool(raw_field.get("enabled", True)):
            continue
        key = str(raw_field.get("key", "")).strip()
        if key not in VALID_STATE_FIELD_KEYS:
            raise ValueError(
                f"state.fields[{idx}].key must be one of {sorted(VALID_STATE_FIELD_KEYS)}, got {key}"
            )
        joint_indices = _resolve_indices(raw_field, per_arm_dim)
        fields.append({"key": key, "joint_indices": joint_indices})

    if len(fields) == 0:
        raise ValueError("state.fields has no enabled entries")

    base_dim_per_arm = 0
    positions_offset_within_arm = None
    positions_joint_indices: List[int] = []
    for field in fields:
        if field["key"] == "positions" and positions_offset_within_arm is None:
            positions_offset_within_arm = base_dim_per_arm
            positions_joint_indices = list(field["joint_indices"])
        base_dim_per_arm += len(field["joint_indices"])

    if base_dim_per_arm <= 0:
        raise ValueError("state base dimension is zero")

    base_dim = base_dim_per_arm * num_arms
    state_dim = base_dim * (history_steps + 1)

    current_chunk_offset = history_steps * base_dim
    state_position_indices: List[int] = []
    control_joint_indices: List[int] = []
    if positions_offset_within_arm is not None:
        for arm_idx in range(num_arms):
            state_arm_base = current_chunk_offset + arm_idx * base_dim_per_arm + positions_offset_within_arm
            for j_local_idx, joint_idx in enumerate(positions_joint_indices):
                state_position_indices.append(int(state_arm_base + j_local_idx))
                control_joint_indices.append(int(arm_idx * per_arm_dim + joint_idx))

    control_dim = num_arms * per_arm_dim
    is_full_control = (
        len(control_joint_indices) == control_dim
        and list(control_joint_indices) == list(range(control_dim))
    )

    return {
        "state_dim": state_dim,
        "control_dim": control_dim,
        "fields": fields,
        "state_position_indices": state_position_indices,
        "control_joint_indices": control_joint_indices,
        "is_full_control": is_full_control,
    }


def load_runtime_state_config(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"bad yaml config: {config_path}")

    if "deploy_direct" in cfg or "deploy_server" in cfg:
        deploy_direct = _as_dict(cfg.get("deploy_direct", {}))
        deploy_server = _as_dict(cfg.get("deploy_server", {}))
        model_cfg = _deep_merge_dict(_as_dict(deploy_direct.get("model", {})), _as_dict(deploy_server.get("model", {})))
        state_cfg = _deep_merge_dict(_as_dict(deploy_direct.get("state", {})), _as_dict(deploy_server.get("state", {})))
    else:
        model_cfg = _as_dict(cfg.get("model", {}))
        state_cfg = _as_dict(cfg.get("state", {}))

    if not state_cfg:
        raise ValueError("missing state config (expect deploy_direct.state or state)")
    state_spec = build_state_spec(state_cfg, default_num_arms=2)

    model_state_dim = int(model_cfg.get("state_dim", state_spec["state_dim"]))
    if model_state_dim != int(state_spec["state_dim"]):
        raise ValueError(
            f"model.state_dim={model_state_dim} mismatches derived state dim={state_spec['state_dim']}"
        )

    model_action_dim = int(model_cfg.get("action_dim", state_spec["control_dim"]))
    return {
        "state_dim": model_state_dim,
        "action_dim": model_action_dim,
        "control_dim": int(state_spec["control_dim"]),
        "state_position_indices": list(state_spec["state_position_indices"]),
        "control_joint_indices": list(state_spec["control_joint_indices"]),
        "is_full_control": bool(state_spec["is_full_control"]),
        "fields": state_spec["fields"],
    }


def to_state_np(x: Any, expected_dim: int, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy().astype(np.float32).reshape(-1)
    else:
        arr = np.asarray(x, np.float32).reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(f"{name} size={arr.size}, expected {expected_dim}")
    return arr


def extract_reset_positions(state_vec: np.ndarray, runtime_state_cfg: Dict[str, Any]) -> np.ndarray:
    state_pos_idx = runtime_state_cfg["state_position_indices"]
    if len(state_pos_idx) == 0:
        raise ValueError("state config has no enabled positions field; cannot build reset payload")
    if not runtime_state_cfg["is_full_control"]:
        raise ValueError(
            "current protocol only supports full reset target (14 joints). "
            "please enable all positions joints in state.fields"
        )
    reset_pos = np.asarray(state_vec, np.float32).reshape(-1)[state_pos_idx]
    control_dim = int(runtime_state_cfg["control_dim"])
    if reset_pos.size != control_dim:
        raise ValueError(f"reset position size={reset_pos.size}, expected control_dim={control_dim}")
    return reset_pos.astype(np.float32)

class ClientDataSaver:
    """保存客户端发送的数据（三相机 state 配置化）"""
    def __init__(self, save_dir: str, enabled: bool = True):
        self.save_dir = save_dir
        self.enabled = enabled
        self.current_episode_dir = None
        self.current_episode_idx = 0
        self.frame_idx = 0

        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)
            existing = [d for d in os.listdir(self.save_dir) if d.startswith("episode_")]
            if existing:
                indices = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
                self.current_episode_idx = max(indices) + 1 if indices else 0
            print(f"📁 ClientDataSaver: {self.save_dir}, next_ep={self.current_episode_idx}")

    def start_new_episode(self, episode_info: dict = None):
        if not self.enabled:
            return

        self.current_episode_dir = os.path.join(self.save_dir, f"episode_{self.current_episode_idx:04d}")
        os.makedirs(self.current_episode_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_episode_dir, "images_left"), exist_ok=True)
        os.makedirs(os.path.join(self.current_episode_dir, "images_right"), exist_ok=True)
        os.makedirs(os.path.join(self.current_episode_dir, "images_middle"), exist_ok=True)

        self.frame_idx = 0
        meta = {
            "episode_idx": self.current_episode_idx,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_timestamp": time.time(),
            **(episode_info or {})
        }
        with open(os.path.join(self.current_episode_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"📁 new episode: {self.current_episode_dir}")
        self.current_episode_idx += 1

    def save_frame(self, jpg1_bytes=None, jpg2_bytes=None, jpg3_bytes=None, state_vec=None, pred_action=None, extra_info=None):
        if not self.enabled or self.current_episode_dir is None:
            return

        # 保存左侧相机
        if jpg1_bytes:
            img1_bgr = cv2.imdecode(np.frombuffer(jpg1_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img1_bgr is not None:
                img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
                PILImage.fromarray(img1_rgb).save(
                    os.path.join(self.current_episode_dir, "images_left", f"frame_{self.frame_idx:05d}.png")
                )

        # 保存右侧相机
        if jpg2_bytes:
            img2_bgr = cv2.imdecode(np.frombuffer(jpg2_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img2_bgr is not None:
                img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
                PILImage.fromarray(img2_rgb).save(
                    os.path.join(self.current_episode_dir, "images_right", f"frame_{self.frame_idx:05d}.png")
                )

        # 保存中间相机
        if jpg3_bytes:
            img3_bgr = cv2.imdecode(np.frombuffer(jpg3_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img3_bgr is not None:
                img3_rgb = cv2.cvtColor(img3_bgr, cv2.COLOR_BGR2RGB)
                PILImage.fromarray(img3_rgb).save(
                    os.path.join(self.current_episode_dir, "images_middle", f"frame_{self.frame_idx:05d}.png")
                )

        row = {"frame_idx": self.frame_idx, "timestamp": time.time()}
        if state_vec is not None:
            row["state"] = np.asarray(state_vec, np.float32).flatten().tolist()
        if pred_action is not None:
            row["pred_action"] = np.asarray(pred_action, np.float32).flatten().tolist()
        if extra_info:
            row.update(extra_info)

        with open(os.path.join(self.current_episode_dir, "states.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.frame_idx += 1

    def finalize_episode(self):
        if not self.enabled or self.current_episode_dir is None:
            return

        meta_path = os.path.join(self.current_episode_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta["end_timestamp"] = time.time()
            meta["total_frames"] = self.frame_idx
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"📁 episode done, frames={self.frame_idx}")
        self.current_episode_dir = None

def first_action14(x) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 3:
        x = x[0, 0, :]
    elif x.ndim == 2:
        x = x[0, :]
    elif x.ndim != 1:
        raise RuntimeError(f"bad action ndim={x.ndim}, shape={tuple(x.shape)}")
    return x.detach().cpu().numpy().astype(np.float32)

def preprocess_image_from_client(jpg_bytes: bytes, size: int = IMAGE_SIZE) -> torch.Tensor:
    img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode JPEG image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = PILImage.fromarray(img_rgb).resize((size, size))
    img_np = np.asarray(img_pil, dtype=np.uint8)
    img_chw = img_np.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(img_chw).unsqueeze(0)

def preprocess_image_from_dataset(img) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    if img.ndim == 3 and img.shape[0] == 3:
        x = img
    elif img.ndim == 3 and img.shape[-1] == 3:
        x = img.permute(2, 0, 1)
    else:
        raise RuntimeError(f"bad image shape={tuple(img.shape)}")
    x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    return x.unsqueeze(0)

def preprocess_state(state) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state)
    state = state.float()
    if state.ndim == 1:
        state = state.unsqueeze(0)
    return state

def get_mode_description(mode: str) -> str:
    return {
        MODE_DATASET: "dataset: obs+state 都来自 dataset",
        MODE_CLIENT: "client: obs+state 都来自 client",
        MODE_CLIENT_OBS: "client_obs: obs 来自 client, state 来自 dataset",
        MODE_CLIENT_STATE: "client_state: obs 来自 dataset, state 来自 client",
    }.get(mode, mode)

def get_episode_indices(ds, episode_idx: int):
    ep = get_episode_boundaries(ds)
    s = int(ep["from"][episode_idx])
    e = int(ep["to"][episode_idx])
    return list(range(s, e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="pipeline config yaml (for state spec)")
    ap.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5577)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--mode", type=str, default=MODE_CLIENT, choices=VALID_MODES)
    ap.add_argument(
        "--client_action_source",
        type=str,
        default=CLIENT_ACTION_SOURCE_POLICY,
        choices=VALID_CLIENT_ACTION_SOURCES,
        help="when mode=client, choose action source: policy or dataset",
    )

    ap.add_argument("--chunk_size", type=int, default=100)
    ap.add_argument("--n_action_steps", type=int, default=1)
    ap.add_argument("--temporal_ensemble_coeff", type=float, default=0.01)
    ap.add_argument("--print_every", type=int, default=1)

    ap.add_argument("--save_client_data", action="store_true")
    ap.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)

    args = ap.parse_args()

    runtime_state_cfg = load_runtime_state_config(args.config)
    state_dim = int(runtime_state_cfg["state_dim"])
    reset_dim = int(runtime_state_cfg["control_dim"])
    action_dim_cfg = int(runtime_state_cfg["action_dim"])

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"⚙️  state config: {args.config}")
    print(f"state_dim={state_dim}, reset_dim={reset_dim}, action_dim(cfg)={action_dim_cfg}")
    print(f"lerobot={LEROBOT_VERSION}, imports={LEROBOT_IMPORT_STYLE}")

    print("=" * 70)
    print("📂 load metadata")
    print(f"repo_id={args.repo_id}")
    print(f"root={args.dataset_root}")
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    feats = dataset_to_policy_features(dataset_metadata.features)
    out_feats = {k: ft for k, ft in feats.items() if ft.type is FeatureType.ACTION}
    in_feats = {k: ft for k, ft in feats.items() if k not in out_feats}
    print(f"in_feats={list(in_feats.keys())}")
    print(f"out_feats={list(out_feats.keys())}")

    cfg = build_act_config(
        input_features=in_feats,
        output_features=out_feats,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    delta_ts = resolve_delta_timestamps(cfg, dataset_metadata)

    need_policy = not (
        args.mode == MODE_CLIENT and args.client_action_source == CLIENT_ACTION_SOURCE_DATASET
    )
    policy = None
    if need_policy:
        print("=" * 70)
        print(f"🔧 load policy: {args.ckpt_dir}")
        policy = ACTPolicy.from_pretrained(args.ckpt_dir, config=cfg, dataset_stats=dataset_metadata.stats)
        policy.eval().to(device)
        if hasattr(policy, "reset"):
            policy.reset()
        print(f"✅ policy ready, device={device}")
    else:
        print("=" * 70)
        print("⏭️ skip policy load (mode=client + client_action_source=dataset)")

    print("=" * 70)
    print("📂 load dataset")
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=delta_ts)
    num_episodes, num_frames = ds.num_episodes, ds.num_frames
    print(f"✅ dataset: {num_episodes} eps, {num_frames} frames")

    should_save = args.save_client_data and args.mode in [MODE_CLIENT, MODE_CLIENT_OBS, MODE_CLIENT_STATE]
    data_saver = ClientDataSaver(args.save_dir, enabled=should_save)

    current_ep = None
    current_indices = None
    T = 0

    t = 0
    mae_avg = 0.0
    n_mae = 0
    reset_done = False

    def select_new_episode():
        nonlocal current_ep, current_indices, T
        current_ep = args.episode if args.episode is not None else random.randrange(num_episodes)
        current_indices = get_episode_indices(ds, current_ep)
        T = len(current_indices)
        print(f"🎯 episode={current_ep}, T={T}")

    def do_reset():
        nonlocal t, mae_avg, n_mae, reset_done
        select_new_episode()

        first_sample = ds[current_indices[0]]
        state_vec = to_state_np(first_sample["observation.state"], state_dim, "dataset first state")
        reset_pos = extract_reset_positions(state_vec, runtime_state_cfg)

        if policy is not None and hasattr(policy, "reset"):
            policy.reset()

        t = 0
        mae_avg = 0.0
        n_mae = 0
        reset_done = True

        return reset_pos

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    endpoint = f"tcp://{args.bind}:{args.port}"
    sock.bind(endpoint)

    print("=" * 70)
    print(f"🚀 server: {endpoint}")
    print(f"mode: {get_mode_description(args.mode)}")
    if args.mode == MODE_CLIENT:
        print(f"client_action_source: {args.client_action_source}")
    print("⚠️  必须先 cmd=get_reset")
    print("=" * 70)

    with torch.no_grad():
        while True:
            parts = sock.recv_multipart()
            if not parts:
                sock.send_multipart([json.dumps({"ok": False, "err": "empty request"}).encode(), b""])
                continue

            try:
                meta_req = json.loads(parts[0].decode())
            except Exception as e:
                sock.send_multipart([json.dumps({"ok": False, "err": f"bad json: {e}"}).encode(), b""])
                continue

            cmd = meta_req.get("cmd", "")

            if cmd == "get_reset":
                data_saver.finalize_episode()
                reset_pos = do_reset()

                episode_info = {
                    "mode": args.mode,
                    "source_episode": current_ep,
                    "source_T": T,
                    "reset_pos": reset_pos.tolist(),
                    "reset_dim": int(reset_pos.size),
                }
                if args.mode == MODE_CLIENT:
                    episode_info["client_action_source"] = args.client_action_source
                data_saver.start_new_episode(episode_info)

                rsp = {"ok": True, "cmd": "get_reset", "episode": current_ep, "T": T, "mode": args.mode, "ts": time.time()}
                sock.send_multipart([json.dumps(rsp).encode(), reset_pos.tobytes()])
                print(f"[RESET] episode={current_ep} T={T} reset_pos[:4]={reset_pos[:4]} dim={reset_pos.size}")
                continue

            if cmd != "infer":
                sock.send_multipart([json.dumps({"ok": False, "err": f"unknown cmd: {cmd}"}).encode(), b""])
                continue

            if not reset_done:
                sock.send_multipart([json.dumps({"ok": False, "err": "must call get_reset first"}).encode(), b""])
                continue

            if args.mode in [MODE_DATASET, MODE_CLIENT_OBS, MODE_CLIENT_STATE] and t >= T:
                if args.loop:
                    data_saver.finalize_episode()
                    reset_pos = do_reset()
                    data_saver.start_new_episode({
                        "mode": args.mode,
                        "source_episode": current_ep,
                        "source_T": T,
                        "reset_pos": reset_pos.tolist(),
                        "reset_dim": int(reset_pos.size),
                    })
                    rsp = {"ok": True, "need_reset": True, "episode": current_ep, "T": T, "ts": time.time()}
                    sock.send_multipart([json.dumps(rsp).encode(), reset_pos.tobytes()])
                    print(f"[LOOP] need_reset -> episode={current_ep}")
                    continue
                else:
                    sock.send_multipart([json.dumps({"ok": False, "done": True, "t": t, "T": T}).encode(), b""])
                    continue

            try:
                if args.mode == MODE_DATASET:
                    sample = ds[current_indices[t]]
                    batch = {}
                    for k in in_feats.keys():
                        v = sample[k]
                        if "image" in k:
                            batch[k] = preprocess_image_from_dataset(v).to(device)
                        else:
                            batch[k] = preprocess_state(v).to(device)

                    pred14 = first_action14(policy.select_action(batch))
                    gt14 = first_action14(sample["action"])

                    mae = float(np.mean(np.abs(pred14 - gt14)))
                    mae_avg = (mae_avg * n_mae + mae) / (n_mae + 1)
                    n_mae += 1

                    if (t % max(args.print_every, 1)) == 0:
                        print(f"[t={t:05d}/{T-1}] MAE={mae:.6f} AVG={mae_avg:.6f} pred[:2]={pred14[:2]} gt[:2]={gt14[:2]}")

                    rsp = {"ok": True, "cmd": "infer", "t": t, "T": T, "mae": mae, "mae_avg": mae_avg, "ts": time.time()}
                    sock.send_multipart([json.dumps(rsp).encode(), pred14.tobytes()])
                    t += 1

                elif args.mode == MODE_CLIENT:
                    if len(parts) < 5:
                        rsp = {"ok": False, "err": f"expected 5 parts [meta,jpg1,jpg2,jpg3,state], got {len(parts)}"}
                        sock.send_multipart([json.dumps(rsp).encode(), b""])
                        continue

                    jpg1_bytes, jpg2_bytes, jpg3_bytes = parts[1], parts[2], parts[3]
                    state_vec = np.frombuffer(parts[4], dtype=np.float32).copy()
                    if state_vec.size != state_dim:
                        raise ValueError(f"state size={state_vec.size}, expected {state_dim}")

                    if args.client_action_source == CLIENT_ACTION_SOURCE_POLICY:
                        if policy is None:
                            raise RuntimeError("policy is not loaded but client_action_source=policy")
                        img1 = preprocess_image_from_client(jpg1_bytes, IMAGE_SIZE).to(device)
                        img2 = preprocess_image_from_client(jpg2_bytes, IMAGE_SIZE).to(device)
                        img3 = preprocess_image_from_client(jpg3_bytes, IMAGE_SIZE).to(device)

                        # ✅ FIXED: 使用 image_left/right/middle 匹配数据集特征名
                        batch = {
                            "observation.image_left": img1,
                            "observation.image_right": img2,
                            "observation.image_middle": img3,
                            "observation.state": preprocess_state(state_vec).to(device),
                        }
                        pred14 = first_action14(policy.select_action(batch))
                        extra_info = {"t": t, "action_source": CLIENT_ACTION_SOURCE_POLICY}
                        if (t % max(args.print_every, 1)) == 0:
                            print(f"[t={t:05d}] state[:4]={state_vec[:4]} pred[:4]={pred14[:4]} (policy)")
                    else:
                        if T <= 0:
                            raise RuntimeError("empty episode indices for dataset action replay")
                        replay_t = int(t % T)
                        sample = ds[current_indices[replay_t]]
                        pred14 = first_action14(sample["action"])
                        extra_info = {
                            "t": t,
                            "action_source": CLIENT_ACTION_SOURCE_DATASET,
                            "dataset_episode": int(current_ep),
                            "dataset_t": replay_t,
                            "dataset_T": int(T),
                        }
                        if (t % max(args.print_every, 1)) == 0:
                            print(
                                f"[t={t:05d}] state[:4]={state_vec[:4]} action[:4]={pred14[:4]} "
                                f"(dataset replay ep={current_ep} t={replay_t}/{T-1})"
                            )

                    pred14 = np.asarray(pred14, np.float32).reshape(-1)
                    if pred14.size != action_dim_cfg:
                        raise ValueError(f"action size={pred14.size}, expected {action_dim_cfg}")

                    data_saver.save_frame(
                        jpg1_bytes=jpg1_bytes,
                        jpg2_bytes=jpg2_bytes,
                        jpg3_bytes=jpg3_bytes,
                        state_vec=state_vec,
                        pred_action=pred14,
                        extra_info=extra_info,
                    )

                    rsp = {
                        "ok": True,
                        "cmd": "infer",
                        "t": t,
                        "ts": time.time(),
                        "action_source": args.client_action_source,
                    }
                    if args.client_action_source == CLIENT_ACTION_SOURCE_DATASET:
                        rsp["dataset_episode"] = int(current_ep)
                        rsp["dataset_t"] = int(t % T)
                        rsp["dataset_T"] = int(T)
                    sock.send_multipart([json.dumps(rsp).encode(), pred14.tobytes()])
                    t += 1

                elif args.mode == MODE_CLIENT_OBS:
                    if len(parts) < 4:
                        rsp = {"ok": False, "err": f"expected >=4 parts [meta,jpg1,jpg2,jpg3], got {len(parts)}"}
                        sock.send_multipart([json.dumps(rsp).encode(), b""])
                        continue

                    jpg1_bytes, jpg2_bytes, jpg3_bytes = parts[1], parts[2], parts[3]
                    sample = ds[current_indices[t]]
                    state_vec = to_state_np(sample["observation.state"], state_dim, "dataset state")

                    img1 = preprocess_image_from_client(jpg1_bytes, IMAGE_SIZE).to(device)
                    img2 = preprocess_image_from_client(jpg2_bytes, IMAGE_SIZE).to(device)
                    img3 = preprocess_image_from_client(jpg3_bytes, IMAGE_SIZE).to(device)

                    # ✅ FIXED: 使用 image_left/right/middle 匹配数据集特征名
                    batch = {
                        "observation.image_left": img1,
                        "observation.image_right": img2,
                        "observation.image_middle": img3,
                        "observation.state": preprocess_state(state_vec).to(device),
                    }

                    pred14 = first_action14(policy.select_action(batch))
                    gt14 = first_action14(sample["action"])
                    mae = float(np.mean(np.abs(pred14 - gt14)))
                    mae_avg = (mae_avg * n_mae + mae) / (n_mae + 1)
                    n_mae += 1

                    data_saver.save_frame(jpg1_bytes=jpg1_bytes, jpg2_bytes=jpg2_bytes, jpg3_bytes=jpg3_bytes,
                                        state_vec=state_vec, pred_action=pred14, extra_info={"t": t, "mae": mae})

                    if (t % max(args.print_every, 1)) == 0:
                        print(f"[t={t:05d}/{T-1}] MAE={mae:.6f} AVG={mae_avg:.6f} pred[:2]={pred14[:2]} gt[:2]={gt14[:2]} (client_obs)")

                    rsp = {"ok": True, "cmd": "infer", "t": t, "T": T, "mae": mae, "mae_avg": mae_avg, "ts": time.time()}
                    sock.send_multipart([json.dumps(rsp).encode(), pred14.tobytes()])
                    t += 1

                elif args.mode == MODE_CLIENT_STATE:
                    if len(parts) < 2:
                        rsp = {"ok": False, "err": f"expected >=2 parts [meta,state], got {len(parts)}"}
                        sock.send_multipart([json.dumps(rsp).encode(), b""])
                        continue

                    state_vec = np.frombuffer(parts[-1], dtype=np.float32).copy()
                    if state_vec.size != state_dim:
                        raise ValueError(f"state size={state_vec.size}, expected {state_dim}")

                    sample = ds[current_indices[t]]
                    # ✅ FIXED: 从 dataset 读取时使用 image_left/right/middle
                    img1 = preprocess_image_from_dataset(sample["observation.image_left"]).to(device)
                    img2 = preprocess_image_from_dataset(sample["observation.image_right"]).to(device)
                    img3 = preprocess_image_from_dataset(sample["observation.image_middle"]).to(device)

                    batch = {
                        "observation.image_left": img1,
                        "observation.image_right": img2,
                        "observation.image_middle": img3,
                        "observation.state": preprocess_state(state_vec).to(device),
                    }

                    pred14 = first_action14(policy.select_action(batch))
                    gt14 = first_action14(sample["action"])
                    mae = float(np.mean(np.abs(pred14 - gt14)))
                    mae_avg = (mae_avg * n_mae + mae) / (n_mae + 1)
                    n_mae += 1

                    data_saver.save_frame(state_vec=state_vec, pred_action=pred14, extra_info={"t": t, "mae": mae})

                    if (t % max(args.print_every, 1)) == 0:
                        print(f"[t={t:05d}/{T-1}] MAE={mae:.6f} AVG={mae_avg:.6f} pred[:2]={pred14[:2]} gt[:2]={gt14[:2]} (client_state)")

                    rsp = {"ok": True, "cmd": "infer", "t": t, "T": T, "mae": mae, "mae_avg": mae_avg, "ts": time.time()}
                    sock.send_multipart([json.dumps(rsp).encode(), pred14.tobytes()])
                    t += 1

            except Exception as e:
                import traceback
                traceback.print_exc()
                sock.send_multipart([json.dumps({"ok": False, "err": str(e)}).encode(), b""])

if __name__ == "__main__":
    main()
