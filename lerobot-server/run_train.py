#!/usr/bin/env python
# run_train.py  用法: python run_train.py --config config/train_config.yaml

import argparse
import os
import shlex
import shutil
import subprocess
from typing import Any

import yaml


class _StrictFormatDict(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"模板变量 '{key}' 未定义，请在 variables 中配置。")


def _resolve_templates(obj: Any, context: dict[str, Any]) -> Any:
    if isinstance(obj, str):
        return obj.format_map(_StrictFormatDict(context))
    if isinstance(obj, list):
        return [_resolve_templates(i, context) for i in obj]
    if isinstance(obj, dict):
        return {k: _resolve_templates(v, context) for k, v in obj.items()}
    return obj


def _to_cli_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


def _flatten_train_args(args: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in args.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_train_args(v, key))
        else:
            out[key] = v
    return out


def _normalize_command(command: Any) -> list[str]:
    if isinstance(command, str):
        parts = shlex.split(command)
        if not parts:
            raise ValueError("runtime.command 不能为空字符串")
        return parts
    if isinstance(command, list):
        if not command:
            raise ValueError("runtime.command 不能为空列表")
        return [str(i) for i in command]
    raise TypeError("runtime.command 必须是字符串或字符串列表")


def _validate_command_exists(cmd0: str, env: dict[str, str]) -> None:
    if "/" in cmd0:
        expanded = os.path.expanduser(cmd0)
        if not (os.path.isfile(expanded) and os.access(expanded, os.X_OK)):
            raise RuntimeError(
                f"训练命令不可执行: {cmd0}\n"
                "请在 config/train_config.yaml 的 runtime.command 中填写正确可执行路径。"
            )
        return

    found = shutil.which(cmd0, path=env.get("PATH"))
    if found is None:
        raise RuntimeError(
            f"在 PATH 中找不到训练命令: {cmd0}\n"
            "请激活包含 lerobot 的环境，或将 runtime.command 设为绝对路径，例如:\n"
            "  /home/jinhe/miniconda3/envs/lerobot-mujoco/bin/lerobot-train"
        )


def _legacy_train_args(cfg: dict[str, Any]) -> dict[str, Any]:
    # 兼容旧版 train_config.yaml
    args: dict[str, Any] = {
        "dataset.repo_id": cfg.get("dataset_id"),
        "dataset.video_backend": cfg.get("video_backend"),
        "policy.type": cfg.get("policy_type"),
        "policy.device": cfg.get("device"),
        "policy.push_to_hub": cfg.get("push_to_hub"),
        "policy.use_amp": cfg.get("use_amp"),
        "policy.n_obs_steps": cfg.get("n_obs_steps"),
        "policy.chunk_size": cfg.get("chunk_size"),
        "policy.n_action_steps": cfg.get("n_action_steps"),
        "steps": cfg.get("steps"),
        "batch_size": cfg.get("batch_size"),
        "num_workers": cfg.get("num_workers"),
        "save_freq": cfg.get("save_freq"),
        "log_freq": cfg.get("log_freq"),
        "wandb.mode": cfg.get("wandb_mode"),
        "resume": cfg.get("resume", False),
    }

    if cfg.get("dataset_root") and cfg.get("dataset_id"):
        args["dataset.root"] = f"{cfg['dataset_root']}/{cfg['dataset_id']}"

    if cfg.get("run_name"):
        args["policy.repo_id"] = f"local/{cfg['run_name']}"

    if cfg.get("output_root") and cfg.get("run_name"):
        args["output_dir"] = f"{cfg['output_root']}/{cfg['run_name']}"

    if cfg.get("resume_config_path"):
        args["config_path"] = cfg["resume_config_path"]

    return {k: v for k, v in args.items() if v is not None}


def _build_command(cfg: dict[str, Any]) -> tuple[list[str], dict[str, str], bool, str | None]:
    runtime = cfg.get("runtime", {})
    dry_run = bool(runtime.get("dry_run", False))
    check_command = bool(runtime.get("check_command", True))
    cwd = runtime.get("cwd")

    variables = cfg.get("variables", {})
    if not isinstance(variables, dict):
        raise TypeError("variables 必须是字典")

    scalar_legacy_context = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    context = {**scalar_legacy_context, **variables}

    command = _resolve_templates(runtime.get("command", "lerobot-train"), context)
    command_parts = _normalize_command(command)

    env = os.environ.copy()
    env_overrides = _resolve_templates(runtime.get("env", {}), context)
    if not isinstance(env_overrides, dict):
        raise TypeError("runtime.env 必须是字典，例如: {CUDA_VISIBLE_DEVICES: '0'}")

    for k, v in env_overrides.items():
        env[str(k)] = str(v)

    # 旧配置兼容: 未设置 runtime.env 时沿用 cuda_device
    if "CUDA_VISIBLE_DEVICES" not in env_overrides and "cuda_device" in cfg:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_device"])

    if cwd is not None:
        cwd = os.path.abspath(os.path.expanduser(str(_resolve_templates(cwd, context))))

    train_args_raw = cfg.get("train_args")
    if train_args_raw is None:
        train_args_raw = _legacy_train_args(cfg)

    if not isinstance(train_args_raw, dict):
        raise TypeError("train_args 必须是字典，键是 CLI 参数名（不带 --）。")

    train_args = _resolve_templates(train_args_raw, context)
    flags = _resolve_templates(cfg.get("flags", []), context)
    raw_args = _resolve_templates(cfg.get("raw_args", []), context)

    if not isinstance(flags, list) or not all(isinstance(x, str) for x in flags):
        raise TypeError("flags 必须是字符串列表，例如: ['help'] -> --help")
    if not isinstance(raw_args, list) or not all(isinstance(x, str) for x in raw_args):
        raise TypeError("raw_args 必须是字符串列表，将按原样附加到命令末尾。")

    cmd: list[str] = list(command_parts)

    train_args_flat = _flatten_train_args(train_args)

    for key, value in train_args_flat.items():
        if value is None:
            continue

        opt = f"--{key}"
        if isinstance(value, list):
            if len(value) == 0:
                continue
            cmd.append(opt)
            cmd.extend(_to_cli_value(i) for i in value)
        else:
            cmd.extend([opt, _to_cli_value(value)])

    for flag in flags:
        cmd.append(f"--{flag}")

    cmd.extend(raw_args)

    if check_command:
        _validate_command_exists(cmd[0], env)

    return cmd, env, dry_run, cwd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cmd, env, cfg_dry_run, cwd = _build_command(cfg)

    print("[run_train] command:")
    print("  " + shlex.join(cmd))

    runtime_env = cfg.get("runtime", {}).get("env", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    if runtime_env:
        print("[run_train] env overrides:")
        for k, v in runtime_env.items():
            print(f"  {k}={v}")
    elif "cuda_device" in cfg:
        print(f"[run_train] env overrides:\n  CUDA_VISIBLE_DEVICES={cfg['cuda_device']}")
    if cwd is not None:
        print(f"[run_train] cwd:\n  {cwd}")

    if args.dry_run or cfg_dry_run:
        print("[run_train] dry-run mode, skipped execution.")
        return

    subprocess.run(cmd, check=True, env=env, cwd=cwd)


if __name__ == "__main__":
    main()
