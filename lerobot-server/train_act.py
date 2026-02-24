# train_act.py
# 运行方式:
#   python train_act.py --config config/pipeline_config.yaml
#
# 兼容 lerobot >= 0.4.0 (Dataset v3.0)
# 相比旧版改动:
#   FIX 1: 移除 resolve_delta_timestamps 导入，改为手动构建 delta_timestamps
#           (0.4.x 中该函数已从 lerobot.datasets.factory 移走)
#   FIX 2: validate_local_lerobot_dataset 增加 v3.0 file-*.parquet 匹配模式
#   FIX 3: stats 获取方式兼容 v3.0 (dataset_metadata.stats -> dataset.meta.stats)

import os
import glob
import argparse
import traceback
import re
import json
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import yaml
import torch
from torch.utils.data import DataLoader, Sampler

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

import matplotlib.pyplot as plt
from torchvision import transforms


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge_dict(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _ensure_dict(value: Any, name: str) -> Dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"配置项 {name} 必须是字典")
    return value


def build_default_train_config(convert_cfg: Dict) -> Dict:
    convert_cfg   = _ensure_dict(convert_cfg, "convert")
    dataset_root  = str(convert_cfg.get("output_root_dir", "./lerobot_test_data")).strip()
    dataset_repo  = str(convert_cfg.get("repo_id", "test")).strip()
    if not dataset_repo:
        dataset_repo = os.path.basename(os.path.normpath(dataset_root)) or "test"
    return {
        "seed": 42,
        "device": "cuda",
        "dataset": {"root": dataset_root, "repo": dataset_repo, "video_backend": "auto"},
        "policy": {"chunk_size": 30, "n_action_steps": 30},
        "train":  {"steps": 5000, "log_freq": 100, "lr": 1e-4, "grad_clip_norm": 0},
        "dataloader": {
            "batch_size": 8, "num_workers": 4,
            "shuffle": True, "drop_last": True, "pin_memory": True,
        },
        "transform": {
            "gaussian_noise": {"enable": False, "mean": 0.0, "std": 0.02},
            "clamp":          {"enable": False, "min": 0.0, "max": 1.0},
        },
        "checkpoint": {
            "dir": "./checkpoints",
            "save_every_steps": 5000,
            "save_on_train_end": True,
            "max_to_keep": 5,
            "auto_resume": True,
            "auto_resume_scan_output": True,
            "resume_from": "",
        },
        "loss_plot": {
            "enable": False,
            "dir": "./output/loss_trend",
            "filename_prefix": "loss_trend",
            "save_on_checkpoint": True,
            "save_on_train_end": True,
            "checkpoint_subdir": False,
            "checkpoint_subdir_prefix": "step",
            "smooth_window": 200,
            "log_scale": False,
        },
        "output": {"final_dir": "./output"},
        "deploy_export": {
            "dir": "./output",
            "export_on_checkpoint": False,
            "export_on_train_end": True,
            "backfill_historical_checkpoints": False,
            "checkpoint_subdir": True,
            "checkpoint_subdir_prefix": "step",
            "export_train_end_snapshot": True,
            "train_end_subdir_prefix": "final_step",
        },
        "eval": {
            "enable": False, "episode_index": 0,
            "batch_size": 1, "num_workers": 4, "use_train_transform": False,
        },
        "plot": {"enable": False, "action_dim": "auto", "show": False, "save_path": ""},
    }


def normalize_train_config(cfg: Dict, convert_cfg: Dict) -> Dict:
    merged = deep_merge_dict(build_default_train_config(convert_cfg), cfg)
    merged["dataset"] = deep_merge_dict(
        build_default_train_config(convert_cfg)["dataset"],
        _ensure_dict(merged.get("dataset"), "dataset"),
    )
    merged["output"] = deep_merge_dict(
        {"final_dir": "./output"},
        _ensure_dict(merged.get("output"), "output"),
    )
    merged["deploy_export"] = deep_merge_dict(
        {
            "dir": merged["output"]["final_dir"],
            "export_on_checkpoint": False,
            "export_on_train_end": True,
            "backfill_historical_checkpoints": False,
            "checkpoint_subdir": True,
            "checkpoint_subdir_prefix": "step",
            "export_train_end_snapshot": True,
            "train_end_subdir_prefix": "final_step",
        },
        _ensure_dict(merged.get("deploy_export"), "deploy_export"),
    )
    if not str(merged["deploy_export"].get("dir", "")).strip():
        merged["deploy_export"]["dir"] = merged["output"]["final_dir"]
    if not str(merged["dataset"].get("root", "")).strip():
        raise ValueError("train_act.dataset.root 不能为空")
    if not str(merged["dataset"].get("repo", "")).strip():
        raise ValueError("train_act.dataset.repo 不能为空")
    return merged


def resolve_train_config(raw_cfg: Dict) -> Dict:
    if not isinstance(raw_cfg, dict):
        raise TypeError("配置文件内容必须是字典")
    shared = raw_cfg.get("shared", {}) or {}
    convert_cfg = _ensure_dict(raw_cfg.get("convert"), "convert")
    train_section = raw_cfg.get("train_act")
    if train_section is None:
        raise ValueError("配置文件必须包含 train_act section")
    merged = deep_merge_dict(shared, _ensure_dict(train_section, "train_act"))
    return normalize_train_config(merged, convert_cfg)


def parse_video_backend(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "auto", "none", "null"}:
        return None
    allowed = {"torchcodec", "pyav", "video_reader"}
    if text not in allowed:
        raise ValueError(f"不支持的视频后端: {value}. 可选: auto|torchcodec|pyav|video_reader")
    return text


def resolve_video_backend(value: Any) -> str:
    backend = parse_video_backend(value)
    if backend is None:
        try:
            from torchcodec.decoders import VideoDecoder  # noqa: F401
            return "torchcodec"
        except Exception as e:
            print(f"[WARN] torchcodec 不可用，回退到 pyav ({type(e).__name__})")
            return "pyav"
    if backend == "torchcodec":
        try:
            from torchcodec.decoders import VideoDecoder  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "配置了 video_backend=torchcodec，但当前环境不可用。"
                "请安装匹配版本的 torchcodec/FFmpeg，或改为 pyav/auto。"
            ) from e
    return backend


def suppress_torchvision_video_deprecation_warning() -> None:
    message = "The video decoding and encoding capabilities of torchvision are deprecated.*"
    warnings.filterwarnings(
        "ignore",
        message=message,
        category=UserWarning,
    )
    rule = "ignore:The video decoding and encoding capabilities of torchvision are deprecated"
    existing = os.environ.get("PYTHONWARNINGS", "").strip()
    if rule not in existing.split(","):
        os.environ["PYTHONWARNINGS"] = f"{existing},{rule}".strip(",")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_local_lerobot_dataset(root: str) -> None:
    root = str(root).strip()
    # FIX 2: v3.0 元数据文件变了 (stats.json 替代 stats.safetensors)
    required_files = ["meta/info.json", "meta/tasks.parquet", "meta/stats.json"]
    missing = [r for r in required_files if not os.path.isfile(os.path.join(root, r))]

    # v3.0: file-*.parquet；v2.1: episode_*.parquet
    has_parquet = (
        bool(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet")))
        or bool(glob.glob(os.path.join(root, "data", "chunk-*", "episode_*.parquet")))
        or bool(glob.glob(os.path.join(root, "data", "file-*.parquet")))
        or bool(glob.glob(os.path.join(root, "data", "episode_*.parquet")))
    )
    if not has_parquet:
        missing.append("data/**/file-*.parquet 或 episode_*.parquet")

    if missing:
        raise FileNotFoundError(
            f"本地 LeRobot 数据集不完整: root={os.path.abspath(root)}; 缺少 {', '.join(missing)}. "
            "请先重新运行 convert_to_lerobot.py。"
        )


# FIX 1: 不再依赖 resolve_delta_timestamps，直接从 ACTConfig 的 delta_indices 手动构建
def build_delta_timestamps(act_cfg: ACTConfig, fps: float) -> Optional[Dict[str, List[float]]]:
    delta: Dict[str, List[float]] = {}

    obs_indices = act_cfg.observation_delta_indices
    if obs_indices is not None:
        delta["observation.state"] = [i / fps for i in obs_indices]

    action_indices = act_cfg.action_delta_indices
    if action_indices is not None:
        delta["action"] = [i / fps for i in action_indices]

    return delta or None


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = float(mean)
        self.std  = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean


def build_image_transform(cfg: Dict) -> Optional[transforms.Compose]:
    tcfg  = cfg.get("transform", {})
    parts = []
    gn = tcfg.get("gaussian_noise", {})
    if gn.get("enable", False):
        parts.append(AddGaussianNoise(mean=gn.get("mean", 0.0), std=gn.get("std", 0.01)))
    clamp = tcfg.get("clamp", {})
    if clamp.get("enable", False):
        mn, mx = float(clamp.get("min", 0.0)), float(clamp.get("max", 1.0))
        parts.append(transforms.Lambda(lambda x: x.clamp(mn, mx)))
    return transforms.Compose(parts) if parts else None


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_checkpoint(ckpt_dir, step, policy, optimizer, cfg,
                    max_to_keep=5, loss_steps=None, loss_values=None) -> str:
    ensure_dir(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{step:07d}.pt")
    payload = {
        "step": step,
        "policy_state_dict":    policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    if loss_steps is not None and loss_values is not None and len(loss_steps) == len(loss_values):
        payload["loss_steps"]  = [int(s)   for s in loss_steps]
        payload["loss_values"] = [float(v) for v in loss_values]
    try:
        torch.save(payload, ckpt_path)
    except Exception as e:
        print(f"[WARN] checkpoint 保存失败: {ckpt_path}\n{type(e).__name__}: {e}")
        return ""
    try:
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_step_*.pt")))
        if max_to_keep > 0 and len(ckpts) > max_to_keep:
            for p in ckpts[:-max_to_keep]:
                try: os.remove(p)
                except OSError: pass
    except Exception:
        pass
    return ckpt_path


def load_checkpoint(resume_path, policy, optimizer, device) -> Tuple[int, List, List]:
    ckpt = torch.load(resume_path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    step        = int(ckpt.get("step", 0))
    loss_steps  = ckpt.get("loss_steps",  [])
    loss_values = ckpt.get("loss_values", [])
    try:
        if isinstance(loss_steps, list) and isinstance(loss_values, list) and len(loss_steps) == len(loss_values):
            loss_steps  = [int(s)   for s in loss_steps]
            loss_values = [float(v) for v in loss_values]
        else:
            loss_steps, loss_values = [], []
    except Exception:
        loss_steps, loss_values = [], []
    return step, loss_steps, loss_values


def parse_ckpt_step(path: str) -> int:
    m = re.search(r"ckpt_step_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_checkpoints(search_dirs: List[str]) -> List[str]:
    candidates = []
    for d in search_dirs:
        d = str(d).strip()
        if d:
            candidates.extend(glob.glob(os.path.join(d, "ckpt_step_*.pt")))
    candidates = list(set(candidates))
    candidates.sort(key=lambda p: (parse_ckpt_step(p), os.path.getmtime(p)))
    return candidates


def find_latest_checkpoint(search_dirs: List[str]) -> Optional[str]:
    c = list_checkpoints(search_dirs)
    return c[-1] if c else None


def is_valid_pretrained_export_dir(out_dir: str) -> bool:
    out_dir = str(out_dir).strip()
    if not out_dir or not os.path.isdir(out_dir):
        return False
    if not os.path.isfile(os.path.join(out_dir, "config.json")):
        return False

    has_model = False
    for pat in ["model.safetensors", "*.safetensors", "pytorch_model.bin", "*.bin"]:
        if glob.glob(os.path.join(out_dir, pat)):
            has_model = True
            break
    if not has_model:
        return False

    processor_configs = [
        f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    ]
    for name in processor_configs:
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
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
                if state_file and not os.path.isfile(os.path.join(out_dir, state_file)):
                    return False
        except Exception:
            return False

    return True


def step_subdir(base_dir: str, prefix: str, step: int) -> str:
    return os.path.join(base_dir, f"{(prefix or 'step').strip()}_{int(step):07d}")


def export_pretrained(policy, out_dir: str, dataset_stats=None, tag: str = "") -> bool:
    ensure_dir(out_dir)
    suffix = f" ({tag})" if tag else ""
    try:
        policy.save_pretrained(out_dir)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_stats,
        )
        preprocessor.save_pretrained(
            out_dir,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        )
        postprocessor.save_pretrained(
            out_dir,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        )
        print(f"[INFO] pretrained export{suffix}: {out_dir}")
        return True
    except Exception as e:
        print(f"[WARN] pretrained export failed{suffix}: {out_dir}\n{type(e).__name__}: {e}")
        return False


def collect_missing_checkpoint_exports(ckpt_paths, deploy_dir,
                                       checkpoint_subdir, checkpoint_subdir_prefix):
    if not ckpt_paths:
        return []
    if not checkpoint_subdir:
        step = parse_ckpt_step(ckpt_paths[-1])
        return [(step, ckpt_paths[-1], deploy_dir)] if step >= 0 and not is_valid_pretrained_export_dir(deploy_dir) else []
    missing = []
    for p in ckpt_paths:
        step = parse_ckpt_step(p)
        if step < 0:
            continue
        out = step_subdir(deploy_dir, checkpoint_subdir_prefix, step)
        if not is_valid_pretrained_export_dir(out):
            missing.append((step, p, out))
    return missing


def backfill_historical_checkpoint_exports(ckpt_paths, deploy_dir, checkpoint_subdir,
                                           checkpoint_subdir_prefix, policy, optimizer, device,
                                           dataset_stats=None):
    missing = collect_missing_checkpoint_exports(ckpt_paths, deploy_dir,
                                                 checkpoint_subdir, checkpoint_subdir_prefix)
    if not missing:
        return 0, 0, 0
    print(f"[INFO] 历史 checkpoint 缺失导出: {len(missing)} 个，开始补齐...")
    exported, failed = 0, 0
    for idx, (step, ckpt_path, out_dir) in enumerate(missing, 1):
        print(f"[INFO] backfill {idx}/{len(missing)} | step={step} -> {out_dir}")
        try:
            load_checkpoint(ckpt_path, policy, optimizer, device)
            if export_pretrained(policy, out_dir, dataset_stats=dataset_stats, tag=f"backfill step={step}"):
                exported += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] backfill step={step} 失败: {type(e).__name__}: {e}")
    return len(missing), exported, failed


def moving_average(values: List[float], window: int) -> List[float]:
    out, q, running = [], deque(), 0.0
    for v in values:
        q.append(float(v)); running += float(v)
        if len(q) > window: running -= q.popleft()
        out.append(running / len(q))
    return out


def save_loss_trend_plot(steps, losses, save_path, smooth_window=1,
                         log_scale=False, title="Training Loss Trend") -> bool:
    if not steps or not losses or len(steps) != len(losses):
        return False
    ensure_dir(os.path.dirname(save_path) or ".")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, label="loss", linewidth=1.0, alpha=0.4)
    if smooth_window > 1:
        ax.plot(steps, moving_average(losses, smooth_window),
                label=f"loss_ma{smooth_window}", linewidth=1.5)
    ax.set_title(title); ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    if log_scale: ax.set_yscale("log")
    ax.legend()
    try:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] loss trend plot saved: {save_path}")
        return True
    except Exception as e:
        print(f"[WARN] 保存 plot 失败: {e}")
        return False
    finally:
        plt.close(fig)


class EpisodeSampler(Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx   = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)
    def __iter__(self): return iter(self.frame_ids)
    def __len__(self): return len(self.frame_ids)


def to_device_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml")
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    cfg     = resolve_train_config(raw_cfg)
    suppress_torchvision_video_deprecation_warning()

    set_seed(int(cfg.get("seed", 42)))
    device = pick_device(str(cfg.get("device", "cuda")))
    print(f"[INFO] device = {device}")

    dcfg = cfg["dataset"]
    repo = dcfg["repo"]
    root = dcfg["root"]
    video_backend = resolve_video_backend(dcfg.get("video_backend", "auto"))
    print(f"[INFO] video_backend = {video_backend}")

    validate_local_lerobot_dataset(root)

    # ===== Dataset metadata & feature mapping =====
    try:
        dataset_metadata = LeRobotDatasetMetadata(repo, root=root)
        features         = dataset_to_policy_features(dataset_metadata.features)
        output_features  = {k: ft for k, ft in features.items() if ft.type == FeatureType.ACTION}
        input_features   = {k: ft for k, ft in features.items() if k not in output_features}
        if not output_features:
            raise RuntimeError("没有找到 ACTION 输出特征，请检查 dataset features 定义。")
    except Exception as e:
        print("[ERROR] 构建 dataset metadata/features 失败")
        traceback.print_exc(); raise

    # ===== Policy config =====
    pcfg    = cfg["policy"]
    act_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=int(pcfg.get("chunk_size", 30)),
        n_action_steps=int(pcfg.get("n_action_steps", 30)),
    )

    # FIX 1: 手动构建 delta_timestamps，不依赖已移走的 resolve_delta_timestamps
    delta_timestamps = build_delta_timestamps(act_cfg, fps=float(dataset_metadata.fps))

    # ===== Dataset & DataLoader =====
    train_transform = build_image_transform(cfg)
    eval_cfg        = cfg.get("eval", {})
    eval_transform  = train_transform if bool(eval_cfg.get("use_train_transform", False)) else None

    try:
        train_dataset = LeRobotDataset(
            repo,
            delta_timestamps=delta_timestamps,
            root=root,
            image_transforms=train_transform,
            video_backend=video_backend,
        )
    except Exception as e:
        print("[ERROR] 构建训练集失败")
        traceback.print_exc(); raise

    # FIX 3: v3.0 stats 通过 dataset.meta.stats 获取
    try:
        stats = dataset_metadata.stats
    except AttributeError:
        stats = train_dataset.meta.stats

    policy = ACTPolicy(act_cfg, dataset_stats=stats)
    policy.train()
    policy.to(device)

    tcfg      = cfg["train"]
    ocfg      = cfg["dataloader"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(tcfg.get("lr", 1e-4)))
    pin_mem   = bool(ocfg.get("pin_memory", True)) and (device.type != "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(ocfg.get("batch_size", 8)),
        shuffle=bool(ocfg.get("shuffle", True)),
        num_workers=int(ocfg.get("num_workers", 4)),
        pin_memory=pin_mem,
        drop_last=bool(ocfg.get("drop_last", True)),
    )

    # ===== Config aliases =====
    training_steps = int(tcfg.get("steps", 5000))
    log_freq       = max(int(tcfg.get("log_freq", 100)), 1)
    grad_clip      = float(tcfg.get("grad_clip_norm", 0.0))

    ckcfg           = cfg["checkpoint"]
    ckpt_dir        = str(ckcfg.get("dir", "./checkpoints"))
    auto_resume     = bool(ckcfg.get("auto_resume", True))
    scan_output     = bool(ckcfg.get("auto_resume_scan_output", True))
    save_every      = int(ckcfg.get("save_every_steps", 0))
    save_on_end     = bool(ckcfg.get("save_on_train_end", True))
    max_keep        = int(ckcfg.get("max_to_keep", 5))

    deploy_cfg      = cfg.get("deploy_export", {})
    final_dir       = str(cfg.get("output", {}).get("final_dir", "./output"))
    deploy_dir      = str(deploy_cfg.get("dir", final_dir))
    exp_on_ckpt     = bool(deploy_cfg.get("export_on_checkpoint", True))
    exp_on_end      = bool(deploy_cfg.get("export_on_train_end", True))
    backfill        = bool(deploy_cfg.get("backfill_historical_checkpoints", False))
    ckpt_subdir     = bool(deploy_cfg.get("checkpoint_subdir", True))
    ckpt_prefix     = str(deploy_cfg.get("checkpoint_subdir_prefix", "step"))
    exp_snap        = bool(deploy_cfg.get("export_train_end_snapshot", True))
    snap_prefix     = str(deploy_cfg.get("train_end_subdir_prefix", "final_step"))

    lpcfg           = cfg.get("loss_plot", {})
    lp_enable       = bool(lpcfg.get("enable", False))
    lp_dir          = str(lpcfg.get("dir", "./output/loss_trend"))
    lp_prefix       = str(lpcfg.get("filename_prefix", "loss_trend")).strip() or "loss_trend"
    lp_on_ckpt      = bool(lpcfg.get("save_on_checkpoint", True))
    lp_on_end       = bool(lpcfg.get("save_on_train_end", True))
    lp_ckpt_subdir  = bool(lpcfg.get("checkpoint_subdir", False))
    lp_ckpt_prefix  = str(lpcfg.get("checkpoint_subdir_prefix", "step"))
    lp_smooth       = max(int(lpcfg.get("smooth_window", 1)), 1)
    lp_log_scale    = bool(lpcfg.get("log_scale", False))

    # ===== Resume =====
    step, resume_loaded        = 0, False
    loss_steps: List[int]      = []
    loss_values: List[float]   = []
    resume_path = str(ckcfg.get("resume_from", "")).strip()

    if not resume_path and auto_resume:
        scan_dirs  = [ckpt_dir]
        if scan_output:
            scan_dirs.extend([final_dir, deploy_dir])
        auto_ckpt = find_latest_checkpoint(scan_dirs)
        if auto_ckpt:
            resume_path = auto_ckpt
            print(f"[INFO] 自动检测到 checkpoint: {resume_path}")

    if resume_path:
        try:
            step, loaded_ls, loaded_lv = load_checkpoint(resume_path, policy, optimizer, device)
            resume_loaded = True
            if loaded_ls and loaded_lv:
                loss_steps, loss_values = loaded_ls, loaded_lv
                print(f"[INFO] 已恢复 loss 历史: {len(loss_steps)} points")
            print(f"[INFO] 续训 checkpoint: {resume_path} (resume_step={step})")
        except Exception as e:
            print(f"[WARN] 恢复 checkpoint 失败，改为重新训练\n{type(e).__name__}: {e}")
            step, resume_loaded = 0, False
            loss_steps, loss_values = [], []
    else:
        print("[INFO] 未找到可恢复 checkpoint，重新训练 (step=0)")

    # ===== Backfill historical exports =====
    if backfill:
        if not resume_loaded:
            print("[INFO] 跳过历史补齐：未从 checkpoint 恢复。")
        else:
            mt, ex, fa = backfill_historical_checkpoint_exports(
                list_checkpoints([ckpt_dir]), deploy_dir, ckpt_subdir, ckpt_prefix,
                policy, optimizer, device, stats,
            )
            print(f"[INFO] 历史补齐完成: 缺失={mt} 成功={ex} 失败={fa}")
            if mt > 0:
                try:
                    step, loaded_ls, loaded_lv = load_checkpoint(resume_path, policy, optimizer, device)
                    if loaded_ls: loss_steps, loss_values = loaded_ls, loaded_lv
                except Exception as e:
                    print("[ERROR] 补齐后恢复续训状态失败，终止。"); raise

    last_ckpt_step: Optional[int] = None

    if step >= training_steps:
        print(f"[INFO] 已达到训练目标 step={step}/{training_steps}，跳过训练循环。")

    # ===== Train loop =====
    done = False
    while not done and step < training_steps:
        for batch in train_loader:
            try:
                inp = to_device_batch(batch, device)
                loss, _ = policy.forward(inp)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                print(f"[ERROR] 训练步异常 (step={step})\n{type(e).__name__}: {e}")
                try:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"[DEBUG] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                except Exception:
                    pass
                traceback.print_exc(); raise

            step += 1
            loss_steps.append(step)
            loss_values.append(float(loss.item()))

            if step == 1 or step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")

            if save_every > 0 and step % save_every == 0:
                saved = save_checkpoint(ckpt_dir, step, policy, optimizer, cfg,
                                        max_to_keep=max_keep,
                                        loss_steps=loss_steps, loss_values=loss_values)
                if saved:
                    last_ckpt_step = step
                    print(f"[INFO] checkpoint saved: {saved}")
                    if exp_on_ckpt:
                        out = step_subdir(deploy_dir, ckpt_prefix, step) if ckpt_subdir else deploy_dir
                        export_pretrained(policy, out, dataset_stats=stats, tag=f"step={step}")
                    if lp_enable and lp_on_ckpt:
                        lp_path = os.path.join(
                            step_subdir(lp_dir, lp_ckpt_prefix, step) if lp_ckpt_subdir else lp_dir,
                            f"{lp_prefix}.png"
                        )
                        save_loss_trend_plot(loss_steps, loss_values, lp_path,
                                             lp_smooth, lp_log_scale, f"Loss (step={step})")

            if step >= training_steps:
                done = True; break

    # ===== Final checkpoint =====
    if save_on_end and step > 0 and step != last_ckpt_step:
        saved = save_checkpoint(ckpt_dir, step, policy, optimizer, cfg,
                                max_to_keep=max_keep,
                                loss_steps=loss_steps, loss_values=loss_values)
        if saved:
            last_ckpt_step = step
            print(f"[INFO] final checkpoint saved: {saved}")

    if lp_enable and lp_on_end and step > 0:
        save_loss_trend_plot(loss_steps, loss_values,
                             os.path.join(lp_dir, f"{lp_prefix}_final_step_{step:07d}.png"),
                             lp_smooth, lp_log_scale, f"Loss (final step={step})")

    # ===== Export pretrained =====
    if exp_on_end:
        export_pretrained(policy, final_dir, dataset_stats=stats, tag=f"train_end step={step}")
        if exp_snap:
            snap_dir = step_subdir(deploy_dir, snap_prefix, step)
            if os.path.abspath(snap_dir) != os.path.abspath(final_dir):
                export_pretrained(policy, snap_dir, dataset_stats=stats, tag=f"train_end_snapshot step={step}")
    else:
        print(f"[INFO] skip export (export_on_train_end=false)")

    # ===== Eval (optional) =====
    if not bool(eval_cfg.get("enable", False)):
        return

    try:
        eval_dataset = LeRobotDataset(repo, delta_timestamps=delta_timestamps,
                                      root=root, image_transforms=eval_transform,
                                      video_backend=video_backend)
    except Exception as e:
        print("[ERROR] 构建评估集失败"); traceback.print_exc(); raise

    policy.eval()
    ep_idx    = int(eval_cfg.get("episode_index", 0))
    ep_sampler = EpisodeSampler(eval_dataset, ep_idx)
    eval_loader = DataLoader(
        eval_dataset, batch_size=int(eval_cfg.get("batch_size", 1)),
        shuffle=False, num_workers=int(eval_cfg.get("num_workers", 4)),
        pin_memory=(device.type != "cpu"), sampler=ep_sampler,
    )

    actions, gt_actions = [], []
    try: policy.reset()
    except Exception: pass

    with torch.no_grad():
        for batch in eval_loader:
            inp = to_device_batch(batch, device)
            actions.append(policy.select_action(inp))
            gt_actions.append(inp["action"][:, 0, :])

    actions    = torch.cat(actions,    dim=0)
    gt_actions = torch.cat(gt_actions, dim=0)
    n = min(actions.shape[0], gt_actions.shape[0])
    mae = torch.mean(torch.abs(actions[:n] - gt_actions[:n])).item()
    print(f"Mean action error (MAE): {mae:.3f}")

    # ===== Plot (optional) =====
    pcfg_plot = cfg.get("plot", {})
    if not bool(pcfg_plot.get("enable", False)):
        return

    ad = pcfg_plot.get("action_dim", "auto")
    action_dim = int(actions.shape[1]) if ad == "auto" else min(int(ad), int(actions.shape[1]))
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, max(4, action_dim * 1.2)))
    if action_dim == 1: axs = [axs]
    for i in range(action_dim):
        axs[i].plot(actions[:n, i].cpu().numpy(), label="pred")
        axs[i].plot(gt_actions[:n, i].cpu().numpy(), label="gt")
        axs[i].legend()

    sp = str(pcfg_plot.get("save_path", "")).strip()
    if sp:
        try: fig.savefig(sp, bbox_inches="tight", dpi=150); print(f"[INFO] plot saved: {sp}")
        except Exception as e: print(f"[WARN] 保存 plot 失败: {e}")
    if bool(pcfg_plot.get("show", False)): plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
