#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 → LeRobot 格式转换脚本 (lerobot >= 0.4.0 / Dataset v3.0)
用法:
  python convert_to_lerobot.py --config config/pipeline_config.yaml
"""

import os, sys, glob, argparse, yaml, warnings
import numpy as np
import h5py
from PIL import Image as PILImage
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================
# 工具函数
# ============================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_merge_dict(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _ensure_dict(value, name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"配置项 {name} 必须是字典")
    return value


def _normalize_fields(fields, default_key="positions"):
    if fields is None:
        fields = [default_key]
    if isinstance(fields, (str, dict)):
        fields = [fields]
    if not isinstance(fields, list):
        raise TypeError("state.fields / action.fields 必须是列表、字符串或字典")
    out = []
    for item in fields:
        if isinstance(item, str):
            f = {"key": item, "enabled": True, "joint_indices": None}
        elif isinstance(item, dict):
            if "key" not in item:
                raise ValueError(f"field 缺少 key: {item}")
            f = dict(item)
            f.setdefault("enabled", True)
            f.setdefault("joint_indices", None)
        else:
            raise TypeError(f"未知 field 类型: {type(item)}")
        out.append(f)
    if not out:
        out = [{"key": default_key, "enabled": True, "joint_indices": None}]
    return out


def _normalize_arm_pairs(arm_pairs, action_prefix):
    if arm_pairs is None:
        arm_pairs = ["pair1", "pair2"]
    if not isinstance(arm_pairs, list) or not arm_pairs:
        raise ValueError("convert.arm_pairs 必须是非空列表")
    out = []
    for idx, item in enumerate(arm_pairs, start=1):
        if isinstance(item, str):
            pair_name = item
            state_label = f"s{idx}"
            action_label = f"{action_prefix}{idx}"
        elif isinstance(item, dict):
            pair_name = item.get("pair_name", item.get("name"))
            if not pair_name:
                raise ValueError(f"arm_pairs[{idx - 1}] 缺少 pair_name/name")
            state_label = item.get("state_label", f"s{idx}")
            action_label = item.get("action_label", f"{action_prefix}{idx}")
        else:
            raise TypeError(f"未知 arm_pairs[{idx - 1}] 类型: {type(item)}")
        out.append({
            "pair_name": str(pair_name),
            "state_label": str(state_label),
            "action_label": str(action_label),
        })
    return out


def _normalize_cameras(cameras):
    if cameras is None:
        cameras = ["left", "right", "top"]
    if isinstance(cameras, (str, dict)):
        cameras = [cameras]
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("convert.cameras 必须是非空列表")
    out = []
    for idx, item in enumerate(cameras):
        if isinstance(item, str):
            name = item
            camera_cfg = {"name": name, "obs_key": f"observation.image_{name}"}
        elif isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError(f"cameras[{idx}] 缺少 name")
            camera_cfg = {
                "name": str(name),
                "obs_key": str(item.get("obs_key", f"observation.image_{name}")),
            }
            if "hdf5_name" in item:
                camera_cfg["hdf5_name"] = str(item["hdf5_name"])
        else:
            raise TypeError(f"未知 cameras[{idx}] 类型: {type(item)}")
        out.append(camera_cfg)
    return out


def build_default_convert_config():
    return {
        "input_data_dir": "./test_data",
        "output_root_dir": "./lerobot_test_data",
        "repo_id": "test",
        "hdf5_layout": "masters_slaves",
        "fps": 30,
        "robot_type": "piper",
        "task_name": "teleop_task",
        "image_size": [256, 256],
        "image_resample": "BILINEAR",
        "per_arm_dim": 7,
        "state":  {"history_steps": 0, "history_padding": "repeat", "fields": ["positions"]},
        "action": {"fields": ["positions"]},
        "validation": {
            "check_nan": True,
            "check_length_consistency": True,
            "check_timestamp_jumps": True,
            "max_timestamp_gap_sec": 0.1,
            "skip_bad_episodes": True,
        },
        "parallel": {"num_workers": 8, "chunksize": 1},
        "resume_if_possible": False,  # v3.0 不支持文件级续跑
    }


def normalize_convert_config(cfg):
    cfg = deep_merge_dict(build_default_convert_config(), cfg)

    layout = str(cfg.get("hdf5_layout", "masters_slaves")).strip().lower()
    layout_defaults = {
        "masters_slaves": {
            "timestamps": "timestamps",
            "state_group": "slaves",
            "action_group": "masters",
            "cameras_group": "cameras",
        },
        "state_action": {
            "timestamps": "timestamps",
            "state_group": "slaves",
            "action_group": "actions",
            "cameras_group": "cameras",
        },
    }
    base_keys = layout_defaults.get(layout, layout_defaults["masters_slaves"])
    cfg["hdf5_keys"] = deep_merge_dict(base_keys, _ensure_dict(cfg.get("hdf5_keys"), "hdf5_keys"))

    action_group = cfg["hdf5_keys"]["action_group"]
    action_prefix = "a" if str(action_group) == "actions" else "m"

    cfg["arm_pairs"] = _normalize_arm_pairs(cfg.get("arm_pairs"), action_prefix=action_prefix)
    cfg["cameras"]   = _normalize_cameras(cfg.get("cameras"))

    cfg["state"]  = deep_merge_dict({"history_steps": 0, "history_padding": "repeat", "fields": ["positions"]}, _ensure_dict(cfg.get("state"), "state"))
    cfg["action"] = deep_merge_dict({"fields": ["positions"]}, _ensure_dict(cfg.get("action"), "action"))
    cfg["state"]["fields"]  = _normalize_fields(cfg["state"].get("fields"),  default_key="positions")
    cfg["action"]["fields"] = _normalize_fields(cfg["action"].get("fields"), default_key="positions")

    cfg["validation"] = deep_merge_dict(build_default_convert_config()["validation"], _ensure_dict(cfg.get("validation"), "validation"))
    cfg["parallel"]   = deep_merge_dict(build_default_convert_config()["parallel"],   _ensure_dict(cfg.get("parallel"),   "parallel"))

    cfg["input_data_dir"]  = str(cfg.get("input_data_dir", "")).strip()
    cfg["output_root_dir"] = str(cfg.get("output_root_dir", "")).strip()
    if not cfg["input_data_dir"]:
        raise ValueError("convert.input_data_dir 不能为空")
    if not cfg["output_root_dir"]:
        raise ValueError("convert.output_root_dir 不能为空")

    cfg["repo_id"] = str(cfg.get("repo_id", "")).strip()
    if not cfg["repo_id"]:
        cfg["repo_id"] = os.path.basename(os.path.normpath(cfg["output_root_dir"])) or "lerobot_dataset"

    return cfg


def resolve_convert_config(raw_cfg):
    if not isinstance(raw_cfg, dict):
        raise TypeError("配置文件内容必须是字典")
    convert_section = raw_cfg.get("convert")
    if convert_section is None:
        raise ValueError("配置文件必须包含 convert section")
    if not isinstance(convert_section, dict):
        raise TypeError("配置项 convert 必须是字典")
    shared = raw_cfg.get("shared", {}) or {}
    if not isinstance(shared, dict):
        raise TypeError("配置项 shared 必须是字典")
    merged = deep_merge_dict(shared, convert_section)
    return normalize_convert_config(merged)


def get_episode_num(filepath):
    fname = os.path.basename(filepath)
    return int(fname.split("_")[1].split(".")[0])


def resolve_indices(field_def, total_dim):
    indices = field_def.get("joint_indices")
    exclude = field_def.get("exclude_indices")
    if indices is not None:
        return list(indices)
    elif exclude is not None:
        return [i for i in range(total_dim) if i not in exclude]
    else:
        return list(range(total_dim))


def resolve_role_field_dataset(h5_file, group_key, pair_name, field_key):
    pair_path = f"{group_key}/{pair_name}"
    if pair_path not in h5_file:
        raise KeyError(f"缺少 HDF5 组/数据集: {pair_path}")
    node = h5_file[pair_path]
    if isinstance(node, h5py.Dataset):
        if field_key != "positions":
            raise KeyError(f"缺少数据集: {pair_path}/{field_key} (当前布局为直接存 positions)")
        return node, pair_path
    if field_key not in node:
        raise KeyError(f"缺少数据集: {pair_path}/{field_key}")
    return node[field_key], f"{pair_path}/{field_key}"


def resolve_role_group_key(keys, role):
    if role == "state":
        key = keys.get("state_group")
    elif role == "action":
        key = keys.get("action_group")
    else:
        raise ValueError(f"未知 role: {role}")
    if not key:
        raise KeyError(f"hdf5_keys 缺少 role={role} 对应组名配置")
    return key


def build_base_feature_names(cfg, role):
    role_cfg  = cfg[role]
    arm_pairs = cfg["arm_pairs"]
    dim       = cfg["per_arm_dim"]
    names = []
    for pair in arm_pairs:
        label = pair["state_label"] if role == "state" else pair["action_label"]
        for field_def in role_cfg["fields"]:
            if not field_def.get("enabled", True):
                continue
            key = field_def["key"]
            for j in resolve_indices(field_def, dim):
                names.append(f"{label}_{key}_j{j}")
    return names


def build_feature_names(cfg, role):
    base_names    = build_base_feature_names(cfg, role)
    history_steps = cfg[role].get("history_steps", 0) if role == "state" else 0
    if history_steps <= 0:
        return base_names
    all_names = []
    for h in range(history_steps, -1, -1):
        suffix = "t0" if h == 0 else f"t-{h}"
        for n in base_names:
            all_names.append(f"{n}_{suffix}")
    return all_names


def get_resample_method(name):
    mapping = {
        "NEAREST":  PILImage.NEAREST,
        "BILINEAR": PILImage.BILINEAR,
        "BICUBIC":  PILImage.BICUBIC,
        "LANCZOS":  PILImage.LANCZOS,
    }
    return mapping.get(name.upper(), PILImage.BILINEAR)


# ============================================================
# 数据校验
# ============================================================

def validate_episode(h5_path, cfg):
    val_cfg = cfg["validation"]
    issues  = []
    try:
        with h5py.File(h5_path, "r") as f:
            keys = cfg["hdf5_keys"]
            ts   = f[keys["timestamps"]][()]
            T    = len(ts)
            if T == 0:
                return False, "空 episode (0帧)"
            lengths = {"timestamps": T}

            if val_cfg.get("check_timestamp_jumps") and T > 1:
                diffs   = np.diff(ts)
                max_gap = val_cfg.get("max_timestamp_gap_sec", 0.1)
                jumps   = np.where(diffs > max_gap)[0]
                if len(jumps) > 0:
                    issues.append(f"时间戳跳变 {len(jumps)} 处, 最大间隔 {diffs.max():.4f}s")

            for pair in cfg["arm_pairs"]:
                pn = pair["pair_name"]
                for role_key in ("state", "action"):
                    group = resolve_role_group_key(keys, role_key)
                    for field_def in cfg[role_key]["fields"]:
                        if not field_def.get("enabled", True):
                            continue
                        try:
                            ds, ds_path = resolve_role_field_dataset(f, group, pn, field_def["key"])
                        except KeyError as e:
                            issues.append(str(e))
                            continue
                        lengths[ds_path] = ds.shape[0]
                        if val_cfg.get("check_nan"):
                            nan_count = np.isnan(ds[()]).sum()
                            if nan_count > 0:
                                issues.append(f"{ds_path} 含 {nan_count} 个 NaN")

            cam_group = keys["cameras_group"]
            for cam in cfg["cameras"]:
                hdf5_cn    = cam.get("hdf5_name", cam["name"])
                color_path = f"{cam_group}/{hdf5_cn}/color"
                if color_path not in f:
                    issues.append(f"缺少相机数据: {color_path}")
                    continue
                lengths[color_path] = f[color_path].shape[0]

            if val_cfg.get("check_length_consistency"):
                if len(set(lengths.values())) > 1:
                    issues.append(f"长度不一致: {dict(lengths)}")

    except Exception as e:
        return False, f"读取异常: {e}"

    if issues:
        msg = "; ".join(issues)
        has_critical = any("缺少" in i or "NaN" in i or "长度不一致" in i for i in issues)
        return (not has_critical), f"[WARN] {msg}"
    return True, "OK"


# ============================================================
# 单 episode 处理（可并行）
# ============================================================

def extract_role_vector(h5_file, cfg, pair_name, role):
    keys      = cfg["hdf5_keys"]
    group_key = resolve_role_group_key(keys, role)
    dim       = cfg["per_arm_dim"]
    parts     = []
    for field_def in cfg[role]["fields"]:
        if not field_def.get("enabled", True):
            continue
        ds, _ = resolve_role_field_dataset(h5_file, group_key, pair_name, field_def["key"])
        data  = ds[()]
        data  = data[:, resolve_indices(field_def, dim)]
        parts.append(data.astype(np.float32))
    if not parts:
        grp = h5_file[f"{group_key}/{pair_name}"]
        T   = grp.shape[0] if isinstance(grp, h5py.Dataset) else list(grp.values())[0].shape[0]
        return np.zeros((T, 0), dtype=np.float32)
    return np.concatenate(parts, axis=1)


def apply_history_stacking(states, history_steps, padding="repeat"):
    if history_steps <= 0:
        return states
    T, D    = states.shape
    stacked = np.zeros((T, (history_steps + 1) * D), dtype=np.float32)
    for i in range(T):
        chunks = []
        for h in range(history_steps, -1, -1):
            past_idx = i - h
            if past_idx < 0:
                chunks.append(np.zeros(D, dtype=np.float32) if padding == "zero" else states[0].copy())
            else:
                chunks.append(states[past_idx])
        stacked[i] = np.concatenate(chunks)
    return stacked


def process_episode(h5_path, cfg):
    try:
        keys     = cfg["hdf5_keys"]
        img_size = tuple(cfg["image_size"])
        resample = get_resample_method(cfg.get("image_resample", "BILINEAR"))

        with h5py.File(h5_path, "r") as f:
            ts = f[keys["timestamps"]][()]
            T  = len(ts)
            if T == 0:
                return None, f"空 episode: {h5_path}"

            state_parts  = []
            action_parts = []
            for pair in cfg["arm_pairs"]:
                pn = pair["pair_name"]
                state_parts.append(extract_role_vector(f, cfg, pn, "state"))
                action_parts.append(extract_role_vector(f, cfg, pn, "action"))
            states  = np.concatenate(state_parts,  axis=1)
            actions = np.concatenate(action_parts, axis=1)

            cam_images = {}
            cam_group  = keys["cameras_group"]
            for cam in cfg["cameras"]:
                hdf5_cn = cam.get("hdf5_name", cam["name"])
                cam_images[cam["obs_key"]] = f[f"{cam_group}/{hdf5_cn}/color"][()]

        history_steps   = cfg["state"].get("history_steps", 0)
        history_padding = cfg["state"].get("history_padding", "repeat")
        states = apply_history_stacking(states, history_steps, history_padding)

        cam_resized = {}
        for obs_key, raw_imgs in cam_images.items():
            resized = []
            for img in raw_imgs:
                pil_img = PILImage.fromarray(img[:, :, ::-1]).resize(img_size, resample)  # BGR→RGB
                resized.append(np.asarray(pil_img, dtype=np.uint8))
            cam_resized[obs_key] = resized

        frames = []
        for i in range(T):
            frame = {"observation.state": states[i], "action": actions[i]}
            for obs_key, imgs in cam_resized.items():
                frame[obs_key] = imgs[i]
            frames.append(frame)
        return frames, None

    except KeyError as e:
        return None, f"HDF5 键缺失 {h5_path}: {e}"
    except Exception as e:
        return None, f"处理失败 {h5_path}: {e}"


# ============================================================
# 主转换流程
# ============================================================

def convert(cfg):
    input_dir  = cfg["input_data_dir"]
    output_dir = cfg["output_root_dir"]
    repo_id    = cfg["repo_id"]
    fps        = cfg["fps"]
    img_size   = tuple(cfg["image_size"])

    state_names  = build_feature_names(cfg, "state")
    action_names = build_feature_names(cfg, "action")
    state_dim    = len(state_names)
    action_dim   = len(action_names)

    # ---- FIX 1: dtype="video"（v3.0 相机必须用 video）----
    # ---- FIX 2: names 用 {"motors": [...]} 字典格式 ----
    features = {}
    for cam in cfg["cameras"]:
        features[cam["obs_key"]] = {
            "dtype": "video",                           # FIX 1: image → video
            "shape": (img_size[1], img_size[0], 3),
            "names": ["height", "width", "channel"],
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": {"motors": state_names},              # FIX 2: list → dict
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": {"motors": action_names},             # FIX 2: list → dict
    }

    history_steps = cfg["state"].get("history_steps", 0)
    print("=" * 60)
    print(f"State:  {state_dim} 维  (history_steps={history_steps})")
    print(f"  enabled fields: {[f['key'] for f in cfg['state']['fields'] if f.get('enabled', True)]}")
    print(f"Action: {action_dim} 维")
    print(f"  enabled fields: {[f['key'] for f in cfg['action']['fields'] if f.get('enabled', True)]}")
    print(f"Cameras: {[c['obs_key'] for c in cfg['cameras']]}")
    print(f"Image size: {img_size[0]}x{img_size[1]}")
    print("=" * 60)

    h5_files = sorted(
        glob.glob(os.path.join(input_dir, "episode_*.hdf5")),
        key=get_episode_num,
    )
    if not h5_files:
        print(f"错误: {input_dir} 中没有 episode_*.hdf5")
        sys.exit(1)
    print(f"找到 {len(h5_files)} 个 episode")

    print("\n数据校验中...")
    valid_files = []
    for h5p in tqdm(h5_files, desc="校验", ncols=80):
        ok, msg = validate_episode(h5p, cfg)
        if not ok:
            print(f"  跳过 {os.path.basename(h5p)}: {msg}")
            if not cfg["validation"].get("skip_bad_episodes", True):
                print("skip_bad_episodes=false，终止")
                sys.exit(1)
        else:
            if msg != "OK":
                print(f"  {os.path.basename(h5p)}: {msg}")
            valid_files.append(h5p)
    print(f"校验通过: {len(valid_files)}/{len(h5_files)}")

    # ---- FIX 3: 移除断点续跑逻辑，直接全量创建 ----
    # v3.0 多 episode 合并到同一个 parquet/mp4，无法安全截断续跑
    print("\n创建 LeRobot 数据集...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type=cfg.get("robot_type", "piper"),
        features=features,
    )

    task_name   = cfg.get("task_name", "teleop_task")
    num_workers = int(cfg["parallel"].get("num_workers", 0))
    chunksize   = cfg["parallel"].get("chunksize", 1)
    if num_workers < 0:
        num_workers = max(cpu_count() - 1, 1)

    process_fn = partial(process_episode, cfg=cfg)
    failed     = []

    if num_workers > 0:
        print(f"\n并行处理 (workers={num_workers})...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, valid_files, chunksize=chunksize),
                total=len(valid_files), desc="处理", ncols=80,
            ))
    else:
        print("\n单进程处理...")
        results = [process_fn(h5p) for h5p in tqdm(valid_files, desc="处理", ncols=80)]

    print("\n写入 LeRobot 数据集...")
    total_frames = 0
    try:
        for h5p, (frames, err) in tqdm(
            zip(valid_files, results), total=len(valid_files), desc="写入", ncols=80
        ):
            if err:
                failed.append((h5p, err))
                continue
            if not frames:
                continue
            for frame in frames:
                frame["task"] = task_name
                dataset.add_frame(frame)
            dataset.save_episode()
            total_frames += len(frames)
    finally:
        dataset.finalize()

    print("\n" + "=" * 60)
    print(f"转换完成!")
    print(f"  输出目录:   {output_dir}")
    print(f"  Episodes:   {dataset.num_episodes}")
    print(f"  总帧数:     {dataset.num_frames}")
    print(f"  State维度:  {state_dim}  (history_steps={history_steps})")
    print(f"  Action维度: {action_dim}")
    if failed:
        print(f"\n失败 ({len(failed)}):")
        for p, e in failed:
            print(f"  {os.path.basename(p)}: {e}")
    print("=" * 60)


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HDF5 → LeRobot 转换 (v3.0)")
    parser.add_argument("--config", type=str,
                        default="config/pipeline_config.yaml",
                        help="YAML 配置文件路径")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)

    raw_cfg = load_config(args.config)
    cfg     = resolve_convert_config(raw_cfg)
    convert(cfg)


if __name__ == "__main__":
    main()