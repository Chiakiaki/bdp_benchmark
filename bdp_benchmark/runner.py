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
from critic_based_rl.inference import predict_discrete_with_scores

from .config import resolved_config
from .env_factory import make_vector_env
from .visualization import make_overlay


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


def collect_evaluation_returns(
    model,
    env,
    *,
    args,
    variant_ids: list[str],
    episodes_per_variant: int,
) -> dict[str, list[float]]:
    if len(variant_ids) != env.num_envs:
        raise ValueError(
            f"Evaluation variant count {len(variant_ids)} does not match env.num_envs={env.num_envs}"
        )
    returns_by_variant = {variant: [] for variant in dict.fromkeys(variant_ids)}
    running_return = np.zeros(env.num_envs, dtype=np.float64)
    obs = env.reset()
    while any(len(values) < episodes_per_variant for values in returns_by_variant.values()):
        action, _ = model.predict(obs, deterministic=args.inference_mode == "deterministic")
        obs, rewards, dones, infos = env.step(action)
        running_return += rewards
        for env_idx, done in enumerate(dones):
            if not done:
                continue
            variant = str(infos[env_idx].get("environment_variant", variant_ids[env_idx]))
            if variant not in returns_by_variant:
                raise ValueError(f"Evaluation worker reported unexpected environment variant: {variant}")
            if len(returns_by_variant[variant]) < episodes_per_variant:
                returns_by_variant[variant].append(float(running_return[env_idx]))
            running_return[env_idx] = 0.0
    return returns_by_variant


def run_evaluation(args) -> list[float]:
    if not args.model_path:
        raise ValueError("--model_path is required for evaluation")
    if bool(getattr(args, "visual_check", False)):
        return run_visual_check(args)
    log_dir = Path(args.log_path).expanduser() if args.log_path else PROJECT_ROOT / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = make_vector_env(args, log_dir=log_dir, is_train=False)
    model = get_algorithm_class(args.sb3_algorithm).load(args.model_path, env=env, device=args.device)
    configured_variants = list(getattr(args, "environment_variants", ()) or ())
    variant_ids = configured_variants if configured_variants else [str(args.env_id)]
    try:
        returns_by_variant = collect_evaluation_returns(
            model,
            env,
            args=args,
            variant_ids=variant_ids,
            episodes_per_variant=int(args.evaluate_episodes),
        )
    finally:
        env.close()
    returns = [value for variant_returns in returns_by_variant.values() for value in variant_returns]
    for variant, variant_returns in returns_by_variant.items():
        print(
            f"variant={variant} episodes={len(variant_returns)} "
            f"mean_return={np.mean(variant_returns):.6f} std_return={np.std(variant_returns):.6f}"
        )
    print(f"episodes={len(returns)} mean_return={np.mean(returns):.6f} std_return={np.std(returns):.6f}")
    return returns


def _unwrap_visual_env(vector_env):
    """Find the benchmark sidecar below DummyVecEnv/Monitor/BDP wrappers."""
    if not hasattr(vector_env, "envs") or len(vector_env.envs) != 1:
        raise ValueError("visual_check requires exactly one DummyVecEnv environment")
    current = vector_env.envs[0]
    while current is not None and not hasattr(current, "get_visual_candidate_set"):
        current = getattr(current, "env", None)
    if current is None:
        raise RuntimeError("Could not find a benchmark visual environment below the vector wrapper")
    return current


def _candidate_features_for_observation(args, candidate_set, observation) -> np.ndarray:
    """Validate that the visible candidate geometry matches the policy input contract."""
    if getattr(args, "policy_mode", "builtin") != "bdp":
        return np.asarray(candidate_set.features, dtype=np.float32)
    if not isinstance(observation, dict) or "candidates" not in observation:
        raise RuntimeError("BDP visual check expected a Dict observation with candidates")
    observed = np.asarray(observation["candidates"])[0]
    candidate_count = int(candidate_set.labels.shape[0])
    observed = observed[:candidate_count]
    if str(args.candidate_sampler) in ("frenet", "native_action_frenet"):
        expected = np.asarray(candidate_set.features, dtype=np.float32)
    else:
        expected = np.eye(candidate_count, dtype=np.float32)
    if observed.shape != expected.shape or not np.allclose(observed, expected, atol=1.0e-5):
        raise RuntimeError(
            "Visual candidate payload does not match the candidate features supplied to the policy: "
            f"observed={observed.shape}, expected={expected.shape}."
        )
    return expected


def run_visual_check(args) -> list[float]:
    """Run one-environment native rendering with candidate score overlays."""
    log_dir = Path(args.log_path).expanduser() if args.log_path else PROJECT_ROOT / "visual_check"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = make_vector_env(args, log_dir=log_dir, is_train=False)
    model = get_algorithm_class(args.sb3_algorithm).load(args.model_path, env=env, device=args.device)
    benchmark_env = _unwrap_visual_env(env)
    returns: list[float] = []
    running_return = np.zeros(env.num_envs, dtype=np.float64)
    obs = env.reset()
    deterministic = args.inference_mode == "deterministic"
    try:
        while len(returns) < int(args.evaluate_episodes):
            candidate_set = benchmark_env.get_visual_candidate_set()
            features = _candidate_features_for_observation(args, candidate_set, obs)
            actions, scores = predict_discrete_with_scores(model, obs, deterministic=deterministic)
            selected = int(np.asarray(actions).reshape(-1)[0])
            pid_target = benchmark_env.preview_pid_target(selected)
            overlay = make_overlay(
                labels=np.asarray(candidate_set.labels),
                features=features,
                trajectories=np.asarray(candidate_set.trajectories),
                scores=np.asarray(scores)[0],
                selected_index=selected,
                pid_target_xy=pid_target,
            )
            benchmark_env.set_visual_overlay(overlay)
            benchmark_env.render()
            obs, rewards, dones, _infos = env.step(actions)
            running_return += rewards
            for env_idx, done in enumerate(dones):
                if done:
                    returns.append(float(running_return[env_idx]))
                    running_return[env_idx] = 0.0
                    if len(returns) >= int(args.evaluate_episodes):
                        break
    finally:
        env.close()
    print(f"visual episodes={len(returns)} mean_return={np.mean(returns):.6f} std_return={np.std(returns):.6f}")
    return returns
