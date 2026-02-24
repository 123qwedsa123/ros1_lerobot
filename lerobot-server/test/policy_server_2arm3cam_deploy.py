#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT deployment server for 2-arm 3-camera deployment.

Protocol:
- get_reset: return control target for reset (dataset/home_pose) or empty payload (none).
- infer: request multipart [meta_json, jpg_left, jpg_right, jpg_middle, state_bytes]
         response payload is float32[action_dim] action.
- dataset_preview: sample dataset frame with pick_mode:
  random_frame | random_episode_first | episode_t.
  Returns 3 jpeg frames + state + pred_action(optional) + control_target(optional).
"""

import argparse
import bisect
import copy
import json
import os
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
import zmq
from PIL import Image as PILImage

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

VALID_RESET_MODES = {"dataset", "home_pose", "none"}
VALID_RESET_DATASET_SELECTIONS = {"random", "fixed", "mean_initial"}
INFER_MODE_CURRENT = "current_then_live"
INFER_MODE_RESET = "reset_then_live"
VALID_INFER_MODES = {INFER_MODE_CURRENT, INFER_MODE_RESET}
VALID_STATE_FIELD_KEYS = {"positions", "velocities", "efforts"}
DP_INPUT_MODE_DATASET_OBS_DATASET_STATE = "dataset_obs_dataset_state"
DP_INPUT_MODE_DATASET_OBS_LIVE_STATE = "dataset_obs_live_state"
DP_INPUT_MODE_LIVE_OBS_DATASET_STATE = "live_obs_dataset_state"
VALID_DP_INFER_INPUT_MODES = {
    DP_INPUT_MODE_DATASET_OBS_DATASET_STATE,
    DP_INPUT_MODE_DATASET_OBS_LIVE_STATE,
    DP_INPUT_MODE_LIVE_OBS_DATASET_STATE,
}
DP_INFER_INPUT_MODE_TO_SOURCES = {
    DP_INPUT_MODE_DATASET_OBS_DATASET_STATE: ("dataset", "dataset"),
    DP_INPUT_MODE_DATASET_OBS_LIVE_STATE: ("dataset", "live"),
    DP_INPUT_MODE_LIVE_OBS_DATASET_STATE: ("live", "dataset"),
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _pipeline_legacy_relpath(raw_path: str) -> str:
    p = str(raw_path or "").strip()
    if not p or os.path.isabs(p):
        return p
    if p.startswith("../"):
        return p
    if p.startswith("./"):
        p = p[2:]
    return os.path.normpath(os.path.join("..", p))


def _looks_like_server_cfg(cfg: Dict[str, Any]) -> bool:
    required = ("server", "infer", "model", "state", "dataset_preview", "features", "reset", "save_client_data")
    for k in required:
        if k not in cfg or not isinstance(cfg[k], dict):
            return False
    return True


def _inherit_dict_if_missing(dst: Dict[str, Any], src: Dict[str, Any], key: str) -> None:
    if key not in dst and isinstance(src.get(key), dict):
        dst[key] = copy.deepcopy(src[key])


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_dict_section(base_cfg: Dict[str, Any], override_cfg: Dict[str, Any], key: str) -> None:
    """
    Merge dict section as: override wins, missing keys inherited from base.
    This is important for unified pipeline where deploy_server.model often only
    overrides ckpt_dir but should still inherit deploy_direct.model.policy/dataset.
    """
    base_val = base_cfg.get(key)
    override_val = override_cfg.get(key)
    if isinstance(base_val, dict) and isinstance(override_val, dict):
        override_cfg[key] = _deep_merge_dict(base_val, override_val)
    elif key not in override_cfg and isinstance(base_val, dict):
        override_cfg[key] = copy.deepcopy(base_val)


def extract_server_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support:
    1) Original dedicated server yaml (root has server/infer/model/...).
    2) Unified pipeline yaml (root has deploy_server section).
    """
    if not isinstance(raw_cfg, dict):
        raise ValueError("config must be a dict")

    if _looks_like_server_cfg(raw_cfg):
        cfg = copy.deepcopy(raw_cfg)
        cfg["_config_source"] = "server_root"
        return cfg

    deploy_server = raw_cfg.get("deploy_server", None)
    if not isinstance(deploy_server, dict):
        raise ValueError(
            "config is neither server config nor pipeline config with `deploy_server` section"
        )
    cfg = copy.deepcopy(deploy_server)
    cfg["_config_source"] = "pipeline.deploy_server"

    shared = _as_dict(raw_cfg.get("shared", {}))
    convert = _as_dict(raw_cfg.get("convert", {}))
    train_act = _as_dict(raw_cfg.get("train_act", {}))
    deploy_direct = _as_dict(raw_cfg.get("deploy_direct", {}))

    # Reuse common deploy fields from deploy_direct to avoid duplicated config.
    # Use deep-merge so partial deploy_server.model overrides (e.g., ckpt_dir only)
    # won't accidentally drop deploy_direct.model.policy/dataset.
    for k in ("model", "state", "features", "reset"):
        _merge_dict_section(deploy_direct, cfg, k)

    scfg = cfg.setdefault("server", {})
    icfg = cfg.setdefault("infer", {})
    mcfg = cfg.setdefault("model", {})
    mpcfg = mcfg.setdefault("policy", {})
    mdcfg = mcfg.setdefault("dataset", {})
    dpcfg = cfg.setdefault("dataset_preview", {})
    fcfg = cfg.setdefault("features", {})
    rcfg = cfg.setdefault("reset", {})
    rdcfg = rcfg.setdefault("dataset", {})
    dcfg = cfg.setdefault("save_client_data", {})

    train_policy = _as_dict(train_act.get("policy", {}))
    train_output = _as_dict(train_act.get("output", {}))
    train_deploy_export = _as_dict(train_act.get("deploy_export", {}))

    if _is_empty_value(mcfg.get("device")) and not _is_empty_value(shared.get("device")):
        mcfg["device"] = shared.get("device")
    if _is_empty_value(mcfg.get("device")):
        mcfg["device"] = "auto"

    if _is_empty_value(mdcfg.get("seed")) and not _is_empty_value(shared.get("seed")):
        mdcfg["seed"] = shared.get("seed")
    if _is_empty_value(mdcfg.get("seed")):
        mdcfg["seed"] = 42

    if _is_empty_value(mdcfg.get("root")) and not _is_empty_value(convert.get("output_root_dir")):
        mdcfg["root"] = _pipeline_legacy_relpath(str(convert.get("output_root_dir")))
    if _is_empty_value(mdcfg.get("repo_id")) and not _is_empty_value(convert.get("repo_id")):
        mdcfg["repo_id"] = str(convert.get("repo_id"))

    if "chunk_size" not in mpcfg and "chunk_size" in train_policy:
        mpcfg["chunk_size"] = train_policy["chunk_size"]
    if "n_action_steps" not in mpcfg and "n_action_steps" in train_policy:
        mpcfg["n_action_steps"] = train_policy["n_action_steps"]
    if "temporal_ensemble_coeff" not in mpcfg and "temporal_ensemble_coeff" in train_policy:
        mpcfg["temporal_ensemble_coeff"] = train_policy["temporal_ensemble_coeff"]

    auto_ckpt_dir = ""
    if not _is_empty_value(train_deploy_export.get("dir")):
        auto_ckpt_dir = _pipeline_legacy_relpath(str(train_deploy_export.get("dir")))
    elif not _is_empty_value(train_output.get("final_dir")):
        auto_ckpt_dir = _pipeline_legacy_relpath(str(train_output.get("final_dir")))
    if _is_empty_value(mcfg.get("ckpt_dir")) and auto_ckpt_dir:
        mcfg["ckpt_dir"] = auto_ckpt_dir

    if _is_empty_value(mcfg.get("state_dim")):
        mcfg["state_dim"] = 28
    if _is_empty_value(mcfg.get("action_dim")):
        mcfg["action_dim"] = 14
    if _is_empty_value(mcfg.get("image_size")):
        mcfg["image_size"] = 256
    if "strict_load" not in mcfg:
        mcfg["strict_load"] = True

    if _is_empty_value(scfg.get("bind")):
        scfg["bind"] = "0.0.0.0"
    if _is_empty_value(scfg.get("port")):
        scfg["port"] = 5577
    if _is_empty_value(icfg.get("mode")):
        icfg["mode"] = INFER_MODE_RESET

    if _is_empty_value(dpcfg.get("image_size")):
        dpcfg["image_size"] = int(mcfg.get("image_size", 256))
    if _is_empty_value(dpcfg.get("jpeg_quality")):
        dpcfg["jpeg_quality"] = 80

    fcfg.setdefault("image_left", "observation.image_left")
    fcfg.setdefault("image_right", "observation.image_right")
    fcfg.setdefault("image_middle", "observation.image_middle")
    fcfg.setdefault("state", "observation.state")
    fcfg.setdefault("action", "action")

    if _is_empty_value(rcfg.get("mode")):
        rcfg["mode"] = "dataset"
    if _is_empty_value(rdcfg.get("repo_id")) and not _is_empty_value(mdcfg.get("repo_id")):
        rdcfg["repo_id"] = mdcfg.get("repo_id")
    if _is_empty_value(rdcfg.get("root")) and not _is_empty_value(mdcfg.get("root")):
        rdcfg["root"] = mdcfg.get("root")
    if _is_empty_value(rdcfg.get("selection")):
        rdcfg["selection"] = "random"
    if _is_empty_value(rdcfg.get("seed")) and not _is_empty_value(mdcfg.get("seed")):
        rdcfg["seed"] = mdcfg.get("seed")

    if "enabled" not in dcfg:
        dcfg["enabled"] = False
    if _is_empty_value(dcfg.get("save_dir")):
        dcfg["save_dir"] = "./client_data_saved_2arm3cam_deploy"

    return cfg


def _resolve_path(base_dir: str, raw_path: str) -> str:
    p = str(raw_path).strip()
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def resolve_paths(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(config_path))
    cfg["model"]["ckpt_dir"] = _resolve_path(base_dir, cfg["model"]["ckpt_dir"])
    model_dataset = cfg.get("model", {}).get("dataset", {})
    if isinstance(model_dataset, dict) and str(model_dataset.get("root", "")).strip():
        cfg["model"]["dataset"]["root"] = _resolve_path(base_dir, cfg["model"]["dataset"]["root"])

    mode = str(cfg["reset"].get("mode", "")).strip()
    if mode == "dataset":
        cfg["reset"]["dataset"]["root"] = _resolve_path(base_dir, cfg["reset"]["dataset"]["root"])

    save_dir = str(cfg["save_client_data"].get("save_dir", "")).strip()
    if save_dir:
        cfg["save_client_data"]["save_dir"] = _resolve_path(base_dir, save_dir)
    return cfg


def _must_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    val = cfg.get(key)
    if not isinstance(val, dict):
        raise ValueError(f"Missing or invalid section: {key}")
    return val


def _to_float32_vec(data: Any, expected_len: int, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if arr.size != expected_len:
        raise ValueError(f"{name} size={arr.size}, expected {expected_len}")
    return arr


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

    if len(out) == 0:
        return []
    for idx in out:
        if idx < 0 or idx >= total_dim:
            raise ValueError(f"joint index out of range: {idx}, total_dim={total_dim}")
    return out


def build_state_spec(state_cfg: Dict[str, Any], default_num_arms: int = 2) -> Dict[str, Any]:
    num_arms = int(state_cfg.get("num_arms", default_num_arms))
    per_arm_dim = int(state_cfg.get("per_arm_dim", 7))
    history_steps = int(state_cfg.get("history_steps", 0))
    history_padding = str(state_cfg.get("history_padding", "repeat")).strip().lower()
    if num_arms <= 0:
        raise ValueError("state.num_arms must be > 0")
    if per_arm_dim <= 0:
        raise ValueError("state.per_arm_dim must be > 0")
    if history_steps < 0:
        raise ValueError("state.history_steps must be >= 0")
    if history_padding not in ("repeat", "zero"):
        raise ValueError("state.history_padding must be repeat|zero")

    raw_fields = state_cfg.get("fields", None)
    if not isinstance(raw_fields, list) or len(raw_fields) == 0:
        raise ValueError("state.fields must be a non-empty list")

    fields: List[Dict[str, Any]] = []
    for idx, raw_field in enumerate(raw_fields):
        if not isinstance(raw_field, dict):
            raise ValueError(f"state.fields[{idx}] must be a dict")
        if not bool(raw_field.get("enabled", True)):
            continue
        key = str(raw_field.get("key", "")).strip()
        if key not in VALID_STATE_FIELD_KEYS:
            raise ValueError(f"state.fields[{idx}].key must be one of {sorted(VALID_STATE_FIELD_KEYS)}, got {key}")
        joint_indices = _resolve_indices(raw_field, per_arm_dim)
        fields.append({"key": key, "joint_indices": joint_indices})

    if len(fields) == 0:
        raise ValueError("state.fields has no enabled entries")

    base_dim_per_arm = 0
    positions_offset_within_arm: Optional[int] = None
    positions_joint_indices: List[int] = []
    for field in fields:
        if field["key"] == "positions" and positions_offset_within_arm is None:
            positions_offset_within_arm = base_dim_per_arm
            positions_joint_indices = list(field["joint_indices"])
        base_dim_per_arm += len(field["joint_indices"])
    if base_dim_per_arm <= 0:
        raise ValueError("state base dimension is zero; check fields/joint indices")

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
    return {
        "num_arms": num_arms,
        "per_arm_dim": per_arm_dim,
        "history_steps": history_steps,
        "history_padding": history_padding,
        "fields": fields,
        "base_dim_per_arm": base_dim_per_arm,
        "base_dim": base_dim,
        "state_dim": state_dim,
        "control_dim": control_dim,
        "state_position_indices": state_position_indices,
        "control_joint_indices": control_joint_indices,
    }


def state_vec_to_control_target(state_vec: np.ndarray, state_spec: Dict[str, Any]) -> Dict[str, Any]:
    state = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if state.size != int(state_spec["state_dim"]):
        raise ValueError(f"state size={state.size}, expected {state_spec['state_dim']}")

    state_pos_idx = state_spec["state_position_indices"]
    control_joint_idx = state_spec["control_joint_indices"]
    if len(state_pos_idx) == 0:
        return {"format": "none", "joint_indices": [], "values": np.zeros((0,), dtype=np.float32)}

    values = state[state_pos_idx].astype(np.float32)
    control_dim = int(state_spec["control_dim"])
    is_full = (
        len(control_joint_idx) == control_dim
        and list(control_joint_idx) == list(range(control_dim))
    )
    fmt = "full" if is_full else "sparse"
    return {
        "format": fmt,
        "joint_indices": list(control_joint_idx),
        "values": values,
    }


def build_control_target_meta(meta: Dict[str, Any], target: Dict[str, Any], prefix: str) -> None:
    fmt_key = f"{prefix}_target_format"
    joints_key = f"{prefix}_joint_indices"
    meta[fmt_key] = str(target.get("format", "none"))
    if meta[fmt_key] == "sparse":
        meta[joints_key] = [int(i) for i in target.get("joint_indices", [])]


def _pick_device(device_str: str) -> torch.device:
    d = str(device_str).strip().lower()
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but unavailable, fallback to cpu")
        d = "cpu"
    if d not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported device: {device_str}")
    return torch.device(d)


def _parse_temporal_ensemble_coeff(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    val = float(raw)
    if val < 0.0:
        raise ValueError("temporal_ensemble_coeff must be >= 0 or null")
    return val


def _first_action_vec(x: Any, action_dim: int) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 3:
        x = x[0, 0, :]
    elif x.ndim == 2:
        x = x[0, :]
    elif x.ndim != 1:
        raise RuntimeError(f"Bad action ndim={x.ndim}, shape={tuple(x.shape)}")
    arr = x.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if arr.size != action_dim:
        raise RuntimeError(f"Action dim mismatch: got {arr.size}, expected {action_dim}")
    return arr


def _preprocess_jpeg(jpg_bytes: bytes, image_size: int) -> torch.Tensor:
    img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode JPEG image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    chw = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(chw).unsqueeze(0)


def _preprocess_state(state_vec: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0)


def _preprocess_image_from_dataset(img: Any) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    if img.ndim == 3 and img.shape[0] == 3:
        x = img
    elif img.ndim == 3 and img.shape[-1] == 3:
        x = img.permute(2, 0, 1)
    else:
        raise RuntimeError(f"Bad dataset image shape={tuple(img.shape)}")
    x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    return x.unsqueeze(0)


def _encode_dataset_image_to_jpg(img: Any, out_size: int, jpeg_quality: int) -> bytes:
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    if img.ndim == 3 and img.shape[0] == 3:
        x = img.permute(1, 2, 0)
    elif img.ndim == 3 and img.shape[-1] == 3:
        x = img
    else:
        raise RuntimeError(f"Bad dataset image shape={tuple(img.shape)}")

    x = x.detach().cpu().numpy()
    if x.dtype != np.uint8:
        if np.max(x) <= 1.5:
            x = np.clip(x, 0.0, 1.0) * 255.0
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    if out_size > 0:
        x = cv2.resize(x, (out_size, out_size), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode dataset image to jpg")
    return buf.tobytes()


def _extract_state_from_dataset_sample(sample: Dict[str, Any], state_key: str, state_dim: int) -> np.ndarray:
    if state_key not in sample:
        raise KeyError(f"Dataset sample missing key: {state_key}")
    state = sample[state_key]
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().numpy().astype(np.float32).reshape(-1)
    else:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.size != state_dim:
        raise ValueError(f"Dataset state dim={arr.size}, expected {state_dim}")
    return arr


def _parse_dp_infer_input_mode(req_meta: Dict[str, Any]) -> Tuple[str, str, str]:
    mode = str(req_meta.get("infer_input_mode", DP_INPUT_MODE_DATASET_OBS_DATASET_STATE)).strip()
    if not mode:
        mode = DP_INPUT_MODE_DATASET_OBS_DATASET_STATE
    if mode not in VALID_DP_INFER_INPUT_MODES:
        raise ValueError(
            f"invalid infer_input_mode={mode}, valid={sorted(VALID_DP_INFER_INPUT_MODES)}"
        )
    obs_source, state_source = DP_INFER_INPUT_MODE_TO_SOURCES[mode]
    return mode, obs_source, state_source


def _extract_live_state_from_request(parts: List[bytes], state_dim: int) -> np.ndarray:
    if len(parts) < 5:
        raise ValueError("dataset_preview requires 5 request parts when using live state")
    raw = parts[4]
    if not raw:
        raise ValueError("dataset_preview live state payload is empty")
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    if arr.size != state_dim:
        raise ValueError(f"live state dim={arr.size}, expected {state_dim}")
    return arr


def _extract_live_jpeg_from_request(parts: List[bytes], idx: int, camera_name: str) -> bytes:
    if len(parts) <= idx:
        raise ValueError(f"dataset_preview request missing live {camera_name} image payload")
    raw = parts[idx]
    if not raw:
        raise ValueError(f"dataset_preview live {camera_name} image payload is empty")
    return raw


def _get_episode_boundaries(ds: LeRobotDataset) -> Dict[str, List[int]]:
    """
    Compatibility helper for LeRobot >=0.4.x dataset API changes.

    Priority:
    1) Use ds.episode_data_index when available.
    2) Fallback to deriving contiguous episode ranges from hf_dataset['episode_index'].
    """
    cache = getattr(ds, "_episode_boundaries_cache", None)
    if isinstance(cache, dict) and "from" in cache and "to" in cache:
        return cache

    ep = getattr(ds, "episode_data_index", None)
    if ep is not None:
        try:
            ep_from = [int(x) for x in ep["from"]]
            ep_to = [int(x) for x in ep["to"]]
            if len(ep_from) == len(ep_to) and len(ep_from) > 0:
                out = {"from": ep_from, "to": ep_to}
                setattr(ds, "_episode_boundaries_cache", out)
                return out
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
        out = {"from": [], "to": []}
        setattr(ds, "_episode_boundaries_cache", out)
        return out

    ep_from: List[int] = []
    ep_to: List[int] = []
    start = 0
    prev = episode_ids[0]
    for i in range(1, len(episode_ids)):
        cur = episode_ids[i]
        if cur != prev:
            ep_from.append(start)
            ep_to.append(i)
            start = i
            prev = cur
    ep_from.append(start)
    ep_to.append(len(episode_ids))

    out = {"from": ep_from, "to": ep_to}
    setattr(ds, "_episode_boundaries_cache", out)
    return out


def _dataset_num_episodes(ds: LeRobotDataset) -> int:
    try:
        n = int(ds.num_episodes)
        if n > 0:
            return n
    except Exception:
        pass
    bounds = _get_episode_boundaries(ds)
    return int(len(bounds["from"]))


def _dataset_num_frames(ds: LeRobotDataset) -> int:
    try:
        n = int(ds.num_frames)
        if n > 0:
            return n
    except Exception:
        pass
    try:
        return int(len(ds))
    except Exception:
        hf = getattr(ds, "hf_dataset", None)
        if hf is None:
            raise
        return int(len(hf))


def _episode_indices(ds: LeRobotDataset, episode_idx: int) -> List[int]:
    bounds = _get_episode_boundaries(ds)
    n = len(bounds["from"])
    if not (0 <= int(episode_idx) < n):
        raise IndexError(f"episode_idx out of range: {episode_idx}, num_episodes={n}")
    start = int(bounds["from"][episode_idx])
    end = int(bounds["to"][episode_idx])
    return list(range(start, end))


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    scfg = _must_dict(cfg, "server")
    icfg = _must_dict(cfg, "infer")
    mcfg = _must_dict(cfg, "model")
    mpcfg = _must_dict(mcfg, "policy")
    mdcfg = _must_dict(mcfg, "dataset")
    stcfg = _must_dict(cfg, "state")
    dpcfg = _must_dict(cfg, "dataset_preview")
    fcfg = _must_dict(cfg, "features")
    rcfg = _must_dict(cfg, "reset")
    dcfg = _must_dict(cfg, "save_client_data")
    state_spec = build_state_spec(stcfg, default_num_arms=2)

    bind = str(scfg.get("bind", "")).strip()
    if not bind:
        raise ValueError("server.bind is required")
    port = int(scfg.get("port", 0))
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid server.port: {port}")

    infer_mode = str(icfg.get("mode", "")).strip()
    if infer_mode not in VALID_INFER_MODES:
        raise ValueError(f"infer.mode must be one of {sorted(VALID_INFER_MODES)}, got {infer_mode}")

    ckpt_dir = str(mcfg.get("ckpt_dir", "")).strip()
    if not ckpt_dir:
        raise ValueError("model.ckpt_dir is required")
    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"model.ckpt_dir not found: {ckpt_dir}")

    state_dim = int(mcfg.get("state_dim", -1))
    action_dim = int(mcfg.get("action_dim", -1))
    if state_dim <= 0:
        raise ValueError("model.state_dim must be > 0")
    if action_dim <= 0:
        raise ValueError("model.action_dim must be > 0")
    if state_dim != int(state_spec["state_dim"]):
        raise ValueError(
            f"model.state_dim={state_dim} mismatches derived state spec dim={state_spec['state_dim']}"
        )
    control_dim = int(state_spec["control_dim"])
    if action_dim != control_dim:
        raise ValueError(
            f"model.action_dim={action_dim} mismatches robot control dim={control_dim}"
        )

    image_size = int(mcfg.get("image_size", 0))
    if image_size <= 0:
        raise ValueError("model.image_size must be > 0")
    strict_load = mcfg.get("strict_load", True)
    if not isinstance(strict_load, bool):
        raise ValueError("model.strict_load must be a boolean")
    cfg["model"]["strict_load"] = strict_load

    chunk_size = int(mpcfg.get("chunk_size", 0))
    n_action_steps = int(mpcfg.get("n_action_steps", 0))
    if chunk_size <= 0:
        raise ValueError("model.policy.chunk_size must be > 0")
    if n_action_steps <= 0:
        raise ValueError("model.policy.n_action_steps must be > 0")
    _ = _parse_temporal_ensemble_coeff(mpcfg.get("temporal_ensemble_coeff", None))

    model_repo_id = str(mdcfg.get("repo_id", "")).strip()
    model_root = str(mdcfg.get("root", "")).strip()
    if not model_repo_id or not model_root:
        raise ValueError("model.dataset.repo_id and model.dataset.root are required")
    _ = int(mdcfg.get("seed", 42))

    dp_image_size = int(dpcfg.get("image_size", 0))
    dp_jpeg_quality = int(dpcfg.get("jpeg_quality", 0))
    if dp_image_size <= 0:
        raise ValueError("dataset_preview.image_size must be > 0")
    if not (1 <= dp_jpeg_quality <= 100):
        raise ValueError("dataset_preview.jpeg_quality must be in [1,100]")

    mode = str(rcfg.get("mode", "")).strip()
    if mode not in VALID_RESET_MODES:
        raise ValueError(f"reset.mode must be one of {sorted(VALID_RESET_MODES)}, got {mode}")
    if infer_mode == INFER_MODE_RESET and mode != "dataset":
        raise ValueError("infer.mode=reset_then_live requires reset.mode=dataset")

    for key in ("image_left", "image_right", "image_middle", "state", "action"):
        val = str(fcfg.get(key, "")).strip()
        if not val:
            raise ValueError(f"features.{key} is required")

    if mode == "home_pose":
        _ = _to_float32_vec(rcfg.get("home_pose14", []), control_dim, "reset.home_pose14")
    if mode == "dataset":
        ds_cfg = _must_dict(rcfg, "dataset")
        repo_id = str(ds_cfg.get("repo_id", "")).strip()
        root = str(ds_cfg.get("root", "")).strip()
        if not repo_id or not root:
            raise ValueError("reset.dataset.repo_id and reset.dataset.root are required for dataset mode")
        sel = str(ds_cfg.get("selection", "random")).strip()
        if sel not in VALID_RESET_DATASET_SELECTIONS:
            raise ValueError(
                "reset.dataset.selection must be one of "
                f"{sorted(VALID_RESET_DATASET_SELECTIONS)}, got {sel}"
            )
        if sel == "fixed":
            _ = int(ds_cfg.get("fixed_episode", 0))

    _ = bool(dcfg.get("enabled", False))
    _ = str(dcfg.get("save_dir", "")).strip()
    cfg["_state_spec"] = state_spec
    return cfg


def validate_model_config(cfg: Dict[str, Any]) -> None:
    ckpt_dir = cfg["model"]["ckpt_dir"]
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise ValueError(f"Missing model config: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    input_features = model_cfg.get("input_features", {})
    output_features = model_cfg.get("output_features", {})
    features = cfg["features"]
    state_dim = int(cfg["model"]["state_dim"])
    action_dim = int(cfg["model"]["action_dim"])

    for k in (features["image_left"], features["image_right"], features["image_middle"], features["state"]):
        if k not in input_features:
            raise ValueError(f"Model config missing input feature: {k}")
    if features["action"] not in output_features:
        raise ValueError(f"Model config missing output feature: {features['action']}")

    state_shape = input_features[features["state"]].get("shape", [])
    action_shape = output_features[features["action"]].get("shape", [])
    if not state_shape or int(state_shape[0]) != state_dim:
        raise ValueError(f"Model state shape mismatch: got {state_shape}, expected [{state_dim}]")
    if not action_shape or int(action_shape[0]) != action_dim:
        raise ValueError(f"Model action shape mismatch: got {action_shape}, expected [{action_dim}]")


def build_policy_context(cfg: Dict[str, Any]) -> Tuple[LeRobotDatasetMetadata, ACTConfig, LeRobotDataset]:
    model_cfg = cfg["model"]
    policy_cfg = model_cfg["policy"]
    ds_cfg = model_cfg["dataset"]

    repo_id = str(ds_cfg["repo_id"])
    root = str(ds_cfg["root"])
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=root)
    feats = dataset_to_policy_features(dataset_metadata.features)
    out_feats = {k: ft for k, ft in feats.items() if ft.type is FeatureType.ACTION}
    in_feats = {k: ft for k, ft in feats.items() if k not in out_feats}

    temporal_coeff = _parse_temporal_ensemble_coeff(policy_cfg.get("temporal_ensemble_coeff", None))
    act_kwargs = {
        "input_features": in_feats,
        "output_features": out_feats,
        "chunk_size": int(policy_cfg["chunk_size"]),
        "n_action_steps": int(policy_cfg["n_action_steps"]),
    }
    if temporal_coeff is not None:
        act_kwargs["temporal_ensemble_coeff"] = temporal_coeff
    try:
        act_cfg = ACTConfig(**act_kwargs)
    except TypeError:
        act_kwargs.pop("temporal_ensemble_coeff", None)
        act_cfg = ACTConfig(**act_kwargs)
    delta_ts = resolve_delta_timestamps(act_cfg, dataset_metadata)
    model_dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_ts)
    return dataset_metadata, act_cfg, model_dataset


class ClientDataSaver:
    def __init__(self, save_dir: str, enabled: bool) -> None:
        self.save_dir = save_dir
        self.enabled = bool(enabled)
        self.current_episode_dir = None
        self.current_episode_idx = 0
        self.frame_idx = 0

        if not self.enabled:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        existing = [d for d in os.listdir(self.save_dir) if d.startswith("episode_")]
        if existing:
            idxs = [int(x.split("_")[1]) for x in existing if x.split("_")[1].isdigit()]
            self.current_episode_idx = max(idxs) + 1 if idxs else 0
        print(f"[SAVE] enabled, dir={self.save_dir}, next_episode={self.current_episode_idx}")

    def start_new_episode(self, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self.current_episode_dir = os.path.join(self.save_dir, f"episode_{self.current_episode_idx:04d}")
        os.makedirs(self.current_episode_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_episode_dir, "images_left"), exist_ok=True)
        os.makedirs(os.path.join(self.current_episode_dir, "images_right"), exist_ok=True)
        os.makedirs(os.path.join(self.current_episode_dir, "images_middle"), exist_ok=True)
        self.frame_idx = 0

        metadata = {
            "episode_idx": self.current_episode_idx,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_timestamp": time.time(),
        }
        if meta:
            metadata.update(meta)
        with open(os.path.join(self.current_episode_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.current_episode_idx += 1

    def _save_image(self, jpg_bytes: bytes, rel_dir: str) -> None:
        if not jpg_bytes:
            return
        img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        PILImage.fromarray(img_rgb).save(
            os.path.join(self.current_episode_dir, rel_dir, f"frame_{self.frame_idx:05d}.png")
        )

    def save_frame(
        self,
        jpg_left: bytes,
        jpg_right: bytes,
        jpg_middle: bytes,
        state_vec: np.ndarray,
        pred_action: np.ndarray,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or self.current_episode_dir is None:
            return

        self._save_image(jpg_left, "images_left")
        self._save_image(jpg_right, "images_right")
        self._save_image(jpg_middle, "images_middle")

        row = {
            "frame_idx": self.frame_idx,
            "timestamp": time.time(),
            "state": np.asarray(state_vec, dtype=np.float32).tolist(),
            "pred_action": np.asarray(pred_action, dtype=np.float32).tolist(),
        }
        if extra_info:
            row.update(extra_info)

        with open(os.path.join(self.current_episode_dir, "states.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.frame_idx += 1

    def finalize_episode(self) -> None:
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
        self.current_episode_dir = None


class ResetProvider:
    def __init__(self, cfg: Dict[str, Any], state_key: str, state_dim: int, state_spec: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.mode = str(cfg["reset"]["mode"])
        self.state_key = state_key
        self.state_dim = state_dim
        self.state_spec = state_spec
        self.control_dim = int(state_spec["control_dim"])
        self.home_pose = None
        self.ds = None
        self.num_episodes = 0
        self.dataset_selection = "random"
        self.fixed_episode = 0
        self.mean_initial_state = None
        self.mean_initial_used_episodes = 0

        if self.mode == "home_pose":
            self.home_pose = _to_float32_vec(cfg["reset"]["home_pose14"], self.control_dim, "reset.home_pose14")
        elif self.mode == "dataset":
            ds_cfg = cfg["reset"]["dataset"]
            seed = int(ds_cfg.get("seed", 42))
            random.seed(seed)
            policy_cfg = cfg["model"]["policy"]
            temporal_coeff = _parse_temporal_ensemble_coeff(policy_cfg.get("temporal_ensemble_coeff", None))

            repo_id = str(ds_cfg["repo_id"])
            root = str(ds_cfg["root"])
            metadata = LeRobotDatasetMetadata(repo_id, root=root)
            feats = dataset_to_policy_features(metadata.features)
            out_feats = {k: ft for k, ft in feats.items() if ft.type is FeatureType.ACTION}
            in_feats = {k: ft for k, ft in feats.items() if k not in out_feats}
            act_kwargs = {
                "input_features": in_feats,
                "output_features": out_feats,
                "chunk_size": int(policy_cfg["chunk_size"]),
                "n_action_steps": int(policy_cfg["n_action_steps"]),
            }
            if temporal_coeff is not None:
                act_kwargs["temporal_ensemble_coeff"] = temporal_coeff
            try:
                act_cfg = ACTConfig(**act_kwargs)
            except TypeError:
                act_kwargs.pop("temporal_ensemble_coeff", None)
                act_cfg = ACTConfig(**act_kwargs)
            delta_ts = resolve_delta_timestamps(act_cfg, metadata)
            self.ds = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_ts)
            self.num_episodes = _dataset_num_episodes(self.ds)
            if self.num_episodes <= 0:
                raise ValueError("Dataset has zero episodes")
            self.dataset_selection = str(ds_cfg.get("selection", "random")).strip()
            if self.dataset_selection not in VALID_RESET_DATASET_SELECTIONS:
                raise ValueError(
                    "reset.dataset.selection must be one of "
                    f"{sorted(VALID_RESET_DATASET_SELECTIONS)}, got {self.dataset_selection}"
                )
            self.fixed_episode = int(ds_cfg.get("fixed_episode", 0))
            if self.dataset_selection == "mean_initial":
                self._ensure_mean_initial_state()

    def _compute_mean_initial_state(self) -> Tuple[np.ndarray, int]:
        if self.ds is None:
            raise RuntimeError("dataset is not initialized")

        accum = np.zeros((self.state_dim,), dtype=np.float64)
        used = 0
        for episode in range(self.num_episodes):
            indices = _episode_indices(self.ds, episode)
            if not indices:
                continue
            sample = self.ds[indices[0]]
            state_vec = _extract_state_from_dataset_sample(sample, self.state_key, self.state_dim)
            accum += state_vec.astype(np.float64)
            used += 1

        if used <= 0:
            raise ValueError("No non-empty episodes found for selection=mean_initial")
        return (accum / float(used)).astype(np.float32), used

    def _ensure_mean_initial_state(self) -> None:
        if self.mean_initial_state is not None:
            return
        self.mean_initial_state, self.mean_initial_used_episodes = self._compute_mean_initial_state()
        print(
            "[RESET] prepared mean_initial target: "
            f"episodes_used={self.mean_initial_used_episodes}/{self.num_episodes}"
        )

    def _normalize_selection(self, selection_override: Optional[str]) -> str:
        if selection_override is None:
            return self.dataset_selection
        sel = str(selection_override).strip()
        if not sel:
            return self.dataset_selection
        if sel not in VALID_RESET_DATASET_SELECTIONS:
            raise ValueError(
                "reset selection override must be one of "
                f"{sorted(VALID_RESET_DATASET_SELECTIONS)}, got {sel}"
            )
        return sel

    def get_reset(
        self,
        selection_override: Optional[str] = None,
        fixed_episode_override: Optional[int] = None,
    ) -> Tuple[bytes, Dict[str, Any]]:
        if self.mode == "none":
            return b"", {"mode": "none", "skip_reset": True}
        if self.mode == "home_pose":
            target = {"format": "full", "joint_indices": list(range(self.control_dim)), "values": self.home_pose.copy()}
            extra = {"mode": "home_pose"}
            build_control_target_meta(extra, target, prefix="control")
            return target["values"].astype(np.float32).tobytes(), extra

        selection = self._normalize_selection(selection_override)
        if selection == "mean_initial":
            self._ensure_mean_initial_state()
            state_vec = self.mean_initial_state.copy()
            extra = {
                "mode": "dataset",
                "selection": "mean_initial",
                "episodes_used": int(self.mean_initial_used_episodes),
            }
        else:
            if selection == "fixed":
                episode = int(self.fixed_episode if fixed_episode_override is None else fixed_episode_override)
                if not (0 <= episode < self.num_episodes):
                    raise ValueError(f"fixed_episode out of range: {episode}, num_episodes={self.num_episodes}")
            else:
                episode = random.randrange(self.num_episodes)

            indices = _episode_indices(self.ds, episode)
            if not indices:
                raise ValueError(f"Episode {episode} has no frames")
            sample = self.ds[indices[0]]
            state_vec = _extract_state_from_dataset_sample(sample, self.state_key, self.state_dim)
            extra = {"mode": "dataset", "selection": selection, "episode": episode, "T": len(indices)}
            if selection == "fixed":
                extra["fixed_episode"] = int(episode)

        target = state_vec_to_control_target(state_vec, self.state_spec)
        build_control_target_meta(extra, target, prefix="control")
        if target["format"] == "none":
            extra["skip_reset"] = True
            return b"", extra
        return target["values"].astype(np.float32).tobytes(), extra


class ModelDatasetProvider:
    VALID_PICK_MODES = {"random_frame", "random_episode_first", "episode_t"}

    def __init__(self, ds: LeRobotDataset, cfg: Dict[str, Any], features: Dict[str, str], state_dim: int) -> None:
        self.ds = ds
        self.features = features
        self.state_dim = state_dim
        ds_cfg = cfg["model"]["dataset"]
        self.rng = random.Random(int(ds_cfg.get("seed", 42)))
        self.num_frames = _dataset_num_frames(ds)
        ep_idx = _get_episode_boundaries(ds)
        self.ep_from = [int(x) for x in ep_idx["from"]]
        self.ep_to = [int(x) for x in ep_idx["to"]]
        self.num_episodes = int(len(self.ep_from))
        self.ep_len = [self.ep_to[i] - self.ep_from[i] for i in range(self.num_episodes)]
        if self.num_frames <= 0:
            raise ValueError("model.dataset has zero frames")
        if self.num_episodes <= 0:
            raise ValueError("model.dataset has zero episodes")

    def _frame_to_episode(self, frame_idx: int) -> Tuple[int, int]:
        ep = bisect.bisect_right(self.ep_to, frame_idx)
        if ep >= self.num_episodes:
            ep = self.num_episodes - 1
        t_in_episode = frame_idx - self.ep_from[ep]
        return ep, t_in_episode

    def _sample_by_frame(self, frame_idx: int, episode: int, t_in_episode: int, pick_mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not (0 <= episode < self.num_episodes):
            raise ValueError(f"episode out of range: {episode}")
        episode_len = int(self.ep_len[episode])
        if episode_len <= 0:
            raise ValueError(f"episode {episode} has zero length")
        if not (0 <= t_in_episode < episode_len):
            raise ValueError(
                f"t_in_episode out of range: {t_in_episode}, episode_len={episode_len}, episode={episode}"
            )
        if not (0 <= frame_idx < self.num_frames):
            raise ValueError(f"frame_idx out of range: {frame_idx}, num_frames={self.num_frames}")

        sample = self.ds[int(frame_idx)]
        info = {
            "pick_mode": str(pick_mode),
            "episode": int(episode),
            "frame_idx": int(frame_idx),
            "t_in_episode": int(t_in_episode),
            "episode_len": int(episode_len),
            "is_last": bool(t_in_episode >= (episode_len - 1)),
            "num_frames": self.num_frames,
            "num_episodes": self.num_episodes,
        }
        return sample, info

    def sample_random_frame(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        frame_idx = self.rng.randrange(self.num_frames)
        episode, t_in_episode = self._frame_to_episode(frame_idx)
        return self._sample_by_frame(frame_idx, episode, t_in_episode, "random_frame")

    def sample_random_episode_first(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        episode = self.rng.randrange(self.num_episodes)
        return self.sample_episode_t(episode, 0)

    def sample_episode_t(self, episode: int, t_in_episode: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ep = int(episode)
        t = int(t_in_episode)
        if not (0 <= ep < self.num_episodes):
            raise ValueError(f"episode out of range: {ep}, num_episodes={self.num_episodes}")
        episode_len = int(self.ep_len[ep])
        if not (0 <= t < episode_len):
            raise ValueError(f"t_in_episode out of range: {t}, episode_len={episode_len}, episode={ep}")
        frame_idx = int(self.ep_from[ep] + t)
        return self._sample_by_frame(frame_idx, ep, t, "episode_t")

    def sample(self, pick_mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        mode = str(pick_mode).strip()
        if mode not in self.VALID_PICK_MODES:
            raise ValueError(f"invalid pick_mode={mode}, valid={sorted(self.VALID_PICK_MODES)}")
        if mode == "random_episode_first":
            sample, info = self.sample_random_episode_first()
            info["pick_mode"] = mode
            return sample, info
        if mode == "episode_t":
            raise ValueError("episode_t mode requires episode and t_in_episode")
        return self.sample_random_frame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to server yaml")
    args = ap.parse_args()

    cfg = extract_server_config(load_yaml(args.config))
    cfg = resolve_paths(cfg, args.config)
    cfg = validate_config(cfg)
    validate_model_config(cfg)
    dataset_metadata, act_cfg, model_dataset = build_policy_context(cfg)
    state_spec = cfg.get("_state_spec") or build_state_spec(cfg["state"], default_num_arms=2)

    device = _pick_device(cfg["model"]["device"])
    ckpt_dir = cfg["model"]["ckpt_dir"]
    image_size = int(cfg["model"]["image_size"])
    state_dim = int(cfg["model"]["state_dim"])
    action_dim = int(cfg["model"]["action_dim"])
    print_every = max(int(cfg["model"].get("print_every", 30)), 1)
    strict_load = bool(cfg["model"].get("strict_load", True))
    features = cfg["features"]
    policy_cfg = cfg["model"]["policy"]
    infer_mode = str(cfg["infer"]["mode"]).strip()
    dataset_preview_cfg = cfg["dataset_preview"]
    dataset_preview_size = int(dataset_preview_cfg["image_size"])
    dataset_preview_jpeg_quality = int(dataset_preview_cfg["jpeg_quality"])

    print("=" * 72)
    print("[INIT] loading policy")
    print(f"[INIT] config_source={cfg.get('_config_source', 'unknown')}")
    print(f"[INIT] ckpt_dir={ckpt_dir}")
    print(f"[INIT] device={device}")
    print(
        "[INIT] state_spec: "
        f"num_arms={state_spec['num_arms']} per_arm_dim={state_spec['per_arm_dim']} "
        f"history_steps={state_spec['history_steps']} state_dim={state_spec['state_dim']}"
    )
    print(
        "[INIT] policy hyperparams: "
        f"chunk_size={policy_cfg['chunk_size']} "
        f"n_action_steps={policy_cfg['n_action_steps']} "
        f"temporal_ensemble_coeff={policy_cfg.get('temporal_ensemble_coeff', None)}"
    )
    print(f"[INIT] strict_load={strict_load}")
    try:
        policy = ACTPolicy.from_pretrained(
            ckpt_dir,
            config=act_cfg,
            dataset_stats=dataset_metadata.stats,
            strict=strict_load,
        )
    except TypeError:
        policy = ACTPolicy.from_pretrained(
            ckpt_dir,
            config=act_cfg,
            dataset_stats=dataset_metadata.stats,
        )
    policy.eval().to(device)
    if hasattr(policy, "reset"):
        policy.reset()
    print("[INIT] policy ready")

    reset_provider = None
    if infer_mode == INFER_MODE_RESET:
        reset_provider = ResetProvider(cfg, state_key=features["state"], state_dim=state_dim, state_spec=state_spec)
    model_dataset_provider = ModelDatasetProvider(model_dataset, cfg, features, state_dim=state_dim)
    reset_mode = reset_provider.mode if reset_provider is not None else "skip"
    require_get_reset = infer_mode == INFER_MODE_RESET

    save_cfg = cfg["save_client_data"]
    saver = ClientDataSaver(
        save_dir=str(save_cfg.get("save_dir", "./client_data_saved_2arm3cam_deploy")),
        enabled=bool(save_cfg.get("enabled", False)),
    )

    bind = cfg["server"]["bind"]
    port = int(cfg["server"]["port"])
    endpoint = f"tcp://{bind}:{port}"

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(endpoint)
    print("=" * 72)
    print(f"[RUN] server bind={endpoint}")
    print(f"[RUN] infer_mode={infer_mode}")
    print(f"[RUN] reset_mode={reset_mode}")
    print("[RUN] protocol: get_reset / infer / dataset_preview")
    print("=" * 72)

    step = 0
    reset_done = not require_get_reset
    episode_opened = False

    with torch.no_grad():
        while True:
            parts = sock.recv_multipart()
            if not parts:
                sock.send_multipart([json.dumps({"ok": False, "err": "empty request"}).encode(), b""])
                continue

            try:
                req_meta = json.loads(parts[0].decode("utf-8"))
            except Exception as e:
                sock.send_multipart([json.dumps({"ok": False, "err": f"bad json: {e}"}).encode(), b""])
                continue

            cmd = req_meta.get("cmd", "")

            if cmd == "get_reset":
                try:
                    if hasattr(policy, "reset"):
                        policy.reset()
                    reset_done = True
                    step = 0

                    saver.finalize_episode()
                    ep_meta = {"reset_mode": reset_mode, "infer_mode": infer_mode}

                    if infer_mode == INFER_MODE_CURRENT:
                        # current_then_live: server 仅做 policy reset，不给物理复位目标
                        reset_target = {"format": "none", "joint_indices": [], "values": np.zeros((0,), dtype=np.float32)}
                        reset_extra = {"mode": "current_then_live", "skip_reset": True}
                        build_control_target_meta(reset_extra, reset_target, prefix="control")
                        payload = b""
                    else:
                        # reset_then_live: 从 dataset 抽取 reset 控制目标
                        if reset_provider is None:
                            raise RuntimeError("reset_provider is not initialized in reset_then_live mode")
                        selection_override = req_meta.get("selection", None)
                        fixed_episode_override = req_meta.get("fixed_episode", None)
                        payload, reset_extra = reset_provider.get_reset(
                            selection_override=selection_override,
                            fixed_episode_override=fixed_episode_override,
                        )

                    ep_meta.update(reset_extra)
                    saver.start_new_episode(ep_meta)
                    episode_opened = True

                    rsp = {"ok": True, "cmd": "get_reset", "ts": time.time(), "infer_mode": infer_mode}
                    rsp.update(reset_extra)
                    sock.send_multipart([json.dumps(rsp).encode("utf-8"), payload])
                    continue
                except Exception as e:
                    traceback.print_exc()
                    sock.send_multipart([json.dumps({"ok": False, "err": str(e)}).encode("utf-8"), b""])
                    continue

            if cmd == "dataset_preview":
                try:
                    pick_mode = str(req_meta.get("pick_mode", "random_frame"))
                    do_infer = bool(req_meta.get("do_infer", False))
                    policy_reset = bool(req_meta.get("policy_reset", True))
                    infer_input_mode, obs_source, state_source = _parse_dp_infer_input_mode(req_meta)
                    if pick_mode == "episode_t":
                        if "episode" not in req_meta or "t_in_episode" not in req_meta:
                            raise ValueError("episode_t requires episode and t_in_episode")
                        sample, sample_info = model_dataset_provider.sample_episode_t(
                            int(req_meta["episode"]), int(req_meta["t_in_episode"])
                        )
                    elif pick_mode == "random_episode_first":
                        sample, sample_info = model_dataset_provider.sample_random_episode_first()
                        sample_info["pick_mode"] = pick_mode
                    elif pick_mode == "random_frame":
                        sample, sample_info = model_dataset_provider.sample_random_frame()
                    else:
                        raise ValueError(
                            f"invalid pick_mode={pick_mode}, valid={sorted(ModelDatasetProvider.VALID_PICK_MODES)}"
                        )
                    state_vec = _extract_state_from_dataset_sample(sample, features["state"], state_dim)
                    control_target = state_vec_to_control_target(state_vec, state_spec)
                    infer_state_vec = state_vec
                    if do_infer and state_source == "live":
                        infer_state_vec = _extract_live_state_from_request(parts, state_dim)

                    if policy_reset and hasattr(policy, "reset"):
                        policy.reset()

                    pred_action = None
                    if do_infer:
                        if obs_source == "live":
                            live_jpg_left = _extract_live_jpeg_from_request(parts, 1, "left")
                            live_jpg_right = _extract_live_jpeg_from_request(parts, 2, "right")
                            live_jpg_middle = _extract_live_jpeg_from_request(parts, 3, "middle")
                            image_left = _preprocess_jpeg(live_jpg_left, image_size).to(device)
                            image_right = _preprocess_jpeg(live_jpg_right, image_size).to(device)
                            image_middle = _preprocess_jpeg(live_jpg_middle, image_size).to(device)
                        else:
                            image_left = _preprocess_image_from_dataset(sample[features["image_left"]]).to(device)
                            image_right = _preprocess_image_from_dataset(sample[features["image_right"]]).to(device)
                            image_middle = _preprocess_image_from_dataset(sample[features["image_middle"]]).to(device)
                        batch = {
                            features["image_left"]: image_left,
                            features["image_right"]: image_right,
                            features["image_middle"]: image_middle,
                            features["state"]: _preprocess_state(infer_state_vec).to(device),
                        }
                        pred_action = _first_action_vec(policy.select_action(batch), action_dim=action_dim)

                    jpg_left = _encode_dataset_image_to_jpg(
                        sample[features["image_left"]], dataset_preview_size, dataset_preview_jpeg_quality
                    )
                    jpg_right = _encode_dataset_image_to_jpg(
                        sample[features["image_right"]], dataset_preview_size, dataset_preview_jpeg_quality
                    )
                    jpg_middle = _encode_dataset_image_to_jpg(
                        sample[features["image_middle"]], dataset_preview_size, dataset_preview_jpeg_quality
                    )

                    rsp = {
                        "ok": True,
                        "cmd": "dataset_preview",
                        "ts": time.time(),
                        "pick_mode": str(sample_info["pick_mode"]),
                        "episode": int(sample_info["episode"]),
                        "t_in_episode": int(sample_info["t_in_episode"]),
                        "frame_idx": int(sample_info["frame_idx"]),
                        "episode_len": int(sample_info["episode_len"]),
                        "is_last": bool(sample_info["is_last"]),
                        "do_infer": do_infer,
                        "policy_reset": policy_reset,
                        "infer_input_mode": infer_input_mode,
                        "obs_source": obs_source,
                        "state_source": state_source,
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                    }
                    rsp["num_frames"] = int(sample_info["num_frames"])
                    rsp["num_episodes"] = int(sample_info["num_episodes"])
                    build_control_target_meta(rsp, control_target, prefix="control")
                    pred_bytes = b"" if pred_action is None else pred_action.astype(np.float32).tobytes()
                    control_bytes = (
                        b""
                        if control_target["format"] == "none"
                        else control_target["values"].astype(np.float32).tobytes()
                    )
                    sock.send_multipart(
                        [
                            json.dumps(rsp).encode("utf-8"),
                            jpg_left,
                            jpg_right,
                            jpg_middle,
                            state_vec.astype(np.float32).tobytes(),
                            pred_bytes,
                            control_bytes,
                        ]
                    )
                    continue
                except Exception as e:
                    traceback.print_exc()
                    sock.send_multipart([json.dumps({"ok": False, "err": str(e)}).encode("utf-8"), b""])
                    continue

            if cmd != "infer":
                sock.send_multipart([json.dumps({"ok": False, "err": f"unknown cmd: {cmd}"}).encode("utf-8"), b""])
                continue

            if require_get_reset and not reset_done:
                sock.send_multipart(
                    [json.dumps({"ok": False, "err": "must call get_reset first"}).encode("utf-8"), b""]
                )
                continue

            try:
                if len(parts) < 5:
                    raise ValueError(
                        f"expected 5 parts [meta,jpg_left,jpg_right,jpg_middle,state], got {len(parts)}"
                    )

                jpg_left, jpg_right, jpg_middle = parts[1], parts[2], parts[3]
                state_vec = np.frombuffer(parts[4], dtype=np.float32).copy()
                if state_vec.size != state_dim:
                    raise ValueError(f"state size={state_vec.size}, expected {state_dim}")

                img_left = _preprocess_jpeg(jpg_left, image_size).to(device)
                img_right = _preprocess_jpeg(jpg_right, image_size).to(device)
                img_middle = _preprocess_jpeg(jpg_middle, image_size).to(device)
                batch = {
                    features["image_left"]: img_left,
                    features["image_right"]: img_right,
                    features["image_middle"]: img_middle,
                    features["state"]: _preprocess_state(state_vec).to(device),
                }

                pred_action = _first_action_vec(policy.select_action(batch), action_dim=action_dim)

                if saver.enabled and not episode_opened:
                    saver.start_new_episode({"reset_mode": reset_mode, "implicit_start": "infer"})
                    episode_opened = True
                saver.save_frame(
                    jpg_left=jpg_left,
                    jpg_right=jpg_right,
                    jpg_middle=jpg_middle,
                    state_vec=state_vec,
                    pred_action=pred_action,
                    extra_info={"t": step},
                )

                if (step % print_every) == 0:
                    print(f"[INFER] t={step} state[:4]={state_vec[:4]} pred[:4]={pred_action[:4]}")

                rsp = {"ok": True, "cmd": "infer", "t": step, "ts": time.time()}
                sock.send_multipart([json.dumps(rsp).encode("utf-8"), pred_action.tobytes()])
                step += 1
            except Exception as e:
                traceback.print_exc()
                sock.send_multipart([json.dumps({"ok": False, "err": str(e)}).encode("utf-8"), b""])


if __name__ == "__main__":
    main()
