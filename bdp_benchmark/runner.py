"""Reusable train/evaluate orchestration for the driving benchmark."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from critic_based_rl.model import get_algorithm_class
from critic_based_rl.runner_core import train_sb3_model

from .config import resolved_config
from .env_factory import make_vector_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _safe_name(value: object) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text).strip("_")


def _run_label(args) -> str:
    if args.run_name:
        return _safe_name(args.run_name)
    sampler = "categorical" if args.policy_mode == "builtin" else args.candidate_sampler
    return _safe_name(f"{args.simulator}_{args.trajectory_execution_mode}_{args.policy_mode}_{sampler}")


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def prepare_run_directory(args, *, timestamp: str | None = None) -> tuple[Path, Path]:
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    log_root = Path(args.log_path).expanduser() if args.log_path else PROJECT_ROOT / "logs"
    if not log_root.is_absolute():
        log_root = PROJECT_ROOT / log_root
    log_dir = log_root / f"{timestamp}_{_run_label(args)}"
    model_dir = log_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=False)
    source_config = Path(args.config).expanduser() if args.config else None
    if source_config is not None:
        shutil.copyfile(source_config, log_dir / "config.yaml")
    else:
        (log_dir / "config.yaml").write_text(
            yaml.safe_dump(resolved_config(args), sort_keys=False),
            encoding="utf-8",
        )
    (log_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config(args), sort_keys=False),
        encoding="utf-8",
    )
    versions = {}
    for distribution in ("bdp-benchmark", "critic-based-rl", "highway-env", "metadrive-simulator", "stable-baselines3", "torch", "gymnasium"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed-as-distribution"
    reproduction = {
        "benchmark_git_head": _git_head(PROJECT_ROOT),
        "arguments": vars(args),
        "versions": versions,
    }
    (log_dir / "reproduction.yaml").write_text(yaml.safe_dump(reproduction, sort_keys=False), encoding="utf-8")
    return log_dir, model_dir


def run_training(args) -> Path:
    log_dir, model_dir = prepare_run_directory(args)
    env = make_vector_env(args, log_dir=log_dir, is_train=True)
    eval_env = None
    if int(args.eval_freq) > 0:
        eval_env = make_vector_env(args, log_dir=log_dir, is_train=False)
    train_sb3_model(args, env, log_dir, model_dir, eval_env=eval_env, close_env=True)
    return log_dir


def run_evaluation(args) -> list[float]:
    if not args.model_path:
        raise ValueError("--model_path is required for evaluation")
    log_dir = Path(args.log_path).expanduser() if args.log_path else PROJECT_ROOT / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = make_vector_env(args, log_dir=log_dir, is_train=False)
    model = get_algorithm_class(args.sb3_algorithm).load(args.model_path, env=env, device=args.device)
    returns: list[float] = []
    running_return = np.zeros(env.num_envs, dtype=np.float64)
    obs = env.reset()
    try:
        while len(returns) < int(args.evaluate_episodes):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = env.step(action)
            running_return += rewards
            for env_idx, done in enumerate(dones):
                if done:
                    returns.append(float(running_return[env_idx]))
                    running_return[env_idx] = 0.0
                    if len(returns) >= int(args.evaluate_episodes):
                        break
    finally:
        env.close()
    print(f"episodes={len(returns)} mean_return={np.mean(returns):.6f} std_return={np.std(returns):.6f}")
    return returns
