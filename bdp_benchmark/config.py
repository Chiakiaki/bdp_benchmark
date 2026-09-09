"""Shared CLI and YAML configuration for BDP driving benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from critic_based_rl.args import add_sb3_bdp_args


def _flatten_sections(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict) and key != "environment_config":
            flattened.update(value)
        else:
            flattened[key] = value
    if "name" in flattened and "sb3_algorithm" not in flattened:
        flattened["sb3_algorithm"] = flattened.pop("name")
    return flattened


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BDP candidate policies on public driving simulators.")
    add_sb3_bdp_args(
        parser,
        include_environment=False,
        include_carla=False,
        include_runtime=False,
    )
    parser.add_argument("--config", type=str, default=None, help="Benchmark job YAML.")
    parser.add_argument("--run_mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--visual_check", action="store_true", default=False)
    parser.add_argument(
        "--inference_mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
        help="Action selection used by evaluation and visual check.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_model_path", type=str, default=None)
    parser.add_argument("--simulator", choices=("highway", "metadrive"), default="highway")
    parser.add_argument("--env_id", type=str, default="highway-fast-v0")
    parser.add_argument(
        "--trajectory_execution_mode",
        choices=("native_controller", "frenet_pid", "frenet_pid_v2"),
        default="native_controller",
    )
    parser.add_argument(
        "--environment_config",
        type=json.loads,
        default={},
        help="Simulator configuration overrides as JSON; YAML may use a mapping.",
    )
    parser.add_argument("--trajectory_horizon_s", type=float, default=2.0)
    parser.add_argument("--trajectory_sample_count", type=int, default=11)
    parser.add_argument("--position_feature_scale_m", type=float, default=50.0)
    parser.add_argument("--speed_feature_scale_mps", type=float, default=40.0)
    parser.add_argument("--native_lateral_span_m", type=float, default=3.5)
    parser.add_argument("--native_speed_span_mps", type=float, default=5.0)
    parser.add_argument("--minimum_target_speed_mps", type=float, default=0.0)
    parser.add_argument("--maximum_target_speed_mps", type=float, default=40.0)
    parser.add_argument("--frenet_lane_change_width_scale", type=float, default=1.0)
    parser.add_argument("--frenet_speed_delta_mps", type=float, default=5.0)
    parser.add_argument("--pid_lookahead_points", type=int, default=2)
    parser.add_argument("--pid_heading_kp", type=float, default=1.2)
    parser.add_argument("--pid_heading_ki", type=float, default=0.0)
    parser.add_argument("--pid_heading_kd", type=float, default=0.08)
    parser.add_argument("--pid_cross_track_kp", type=float, default=0.35)
    parser.add_argument("--pid_speed_kp", type=float, default=1.0)
    parser.add_argument("--pid_speed_ki", type=float, default=0.05)
    parser.add_argument("--pid_speed_kd", type=float, default=0.0)
    parser.add_argument("--pid_max_steering_rad", type=float, default=0.7)
    parser.add_argument("--pid_max_accel_mps2", type=float, default=5.0)
    parser.add_argument("--pid_max_decel_mps2", type=float, default=8.0)
    parser.add_argument("--evaluate_episodes", type=int, default=10)
    parser.add_argument("--model_path", type=str, default=None)
    parser.set_defaults(
        sb3_algorithm="PPO",
        policy_mode="builtin",
        candidate_sampler="one_hot",
        builtin_policy="MlpPolicy",
        monitor_mode="env",
        vec_env="dummy",
    )
    return parser


def _load_yaml_defaults(parser: argparse.ArgumentParser, config_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Benchmark config must be a YAML mapping: {config_path}")
    flattened = _flatten_sections(config)
    destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(flattened) - destinations)
    if unknown:
        raise ValueError(f"Unknown benchmark config keys in {config_path}: {', '.join(unknown)}")
    parser.set_defaults(**flattened)


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.trajectory_horizon_s <= 0.0:
        raise ValueError("trajectory_horizon_s must be positive")
    if args.trajectory_sample_count < 2:
        raise ValueError("trajectory_sample_count must be at least 2")
    if args.position_feature_scale_m <= 0.0 or args.speed_feature_scale_mps <= 0.0:
        raise ValueError("candidate feature scales must be positive")
    if args.minimum_target_speed_mps < 0.0:
        raise ValueError("minimum_target_speed_mps must be non-negative")
    if args.maximum_target_speed_mps <= args.minimum_target_speed_mps:
        raise ValueError("maximum_target_speed_mps must exceed minimum_target_speed_mps")
    if args.n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    if args.simulator == "metadrive" and args.n_envs > 1 and args.vec_env != "subproc":
        raise ValueError("MetaDrive with n_envs > 1 requires vec_env=subproc")
    if args.policy_mode == "bdp":
        allowed = {
            "native_controller": {"one_hot", "native_action_frenet"},
            "frenet_pid": {"one_hot", "frenet"},
            "frenet_pid_v2": {"one_hot", "frenet"},
        }[args.trajectory_execution_mode]
        if args.candidate_sampler not in allowed:
            required = (
                "candidate_sampler=frenet"
                if args.trajectory_execution_mode in ("frenet_pid", "frenet_pid_v2")
                else "one_hot or native_action_frenet"
            )
            raise ValueError(
                f"{args.trajectory_execution_mode} BDP mode requires {required}; got {args.candidate_sampler}"
            )
    return args


def parse_args(argv: Sequence[str] | None = None, *, validate: bool = True) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = build_parser()
    if pre_args.config:
        config_path = Path(pre_args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Benchmark config does not exist: {config_path}")
        _load_yaml_defaults(parser, config_path)
        parser.set_defaults(config=str(config_path))
    argv_list = list(argv) if argv is not None else list(sys.argv[1:])
    if "--visual_check" in argv_list:
        defaults: dict[str, Any] = {}
        if not _option_in_args(argv_list, "--run_mode"):
            defaults["run_mode"] = "evaluate"
        if not _option_in_args(argv_list, "--n_envs"):
            defaults["n_envs"] = 1
        if not _option_in_args(argv_list, "--vec_env"):
            defaults["vec_env"] = "dummy"
        if not _option_in_args(argv_list, "--render_mode"):
            defaults["render_mode"] = "human"
        if not _option_in_args(argv_list, "--inference_mode"):
            defaults["inference_mode"] = "deterministic"
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.num_timesteps = int(args.num_timesteps)
    args.environment_config = dict(args.environment_config or {})
    args.dwb_stop_candidate_index = -1
    if args.simulator == "metadrive" and args.env_id == "highway-fast-v0":
        args.env_id = "MetaDrive-v0"
    if args.visual_check and args.simulator == "metadrive" and not _option_in_args(argv_list, "--environment_config"):
        args.environment_config["use_render"] = True
    if args.visual_check:
        if not _option_in_args(argv_list, "--run_mode") and args.run_mode == "train":
            args.run_mode = "evaluate"
        if not _option_in_args(argv_list, "--n_envs"):
            args.n_envs = 1
        if not _option_in_args(argv_list, "--vec_env"):
            args.vec_env = "dummy"
        if not _option_in_args(argv_list, "--render_mode") and args.render_mode is None:
            args.render_mode = "human"
        if not _option_in_args(argv_list, "--inference_mode"):
            args.inference_mode = "deterministic"
    return validate_args(args) if validate else args


def _option_in_args(args: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in args)


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime": {
            "run_mode": args.run_mode,
            "run_name": args.run_name,
            "visual_check": args.visual_check,
            "inference_mode": args.inference_mode,
        },
        "environment": {
            "simulator": args.simulator,
            "env_id": args.env_id,
            "trajectory_execution_mode": args.trajectory_execution_mode,
            "environment_config": dict(args.environment_config),
        },
        "frenet": {
            "trajectory_horizon_s": args.trajectory_horizon_s,
            "trajectory_sample_count": args.trajectory_sample_count,
            "position_feature_scale_m": args.position_feature_scale_m,
            "speed_feature_scale_mps": args.speed_feature_scale_mps,
            "native_lateral_span_m": args.native_lateral_span_m,
            "native_speed_span_mps": args.native_speed_span_mps,
            "minimum_target_speed_mps": args.minimum_target_speed_mps,
            "maximum_target_speed_mps": args.maximum_target_speed_mps,
            "frenet_lane_change_width_scale": args.frenet_lane_change_width_scale,
            "frenet_speed_delta_mps": args.frenet_speed_delta_mps,
        },
        "pid": {
            key: getattr(args, key)
            for key in (
                "pid_lookahead_points",
                "pid_heading_kp",
                "pid_heading_ki",
                "pid_heading_kd",
                "pid_cross_track_kp",
                "pid_speed_kp",
                "pid_speed_ki",
                "pid_speed_kd",
                "pid_max_steering_rad",
                "pid_max_accel_mps2",
                "pid_max_decel_mps2",
            )
        },
        "algorithm": {
            key: getattr(args, key)
            for key in (
                "sb3_algorithm",
                "num_timesteps",
                "learning_rate",
                "schedule",
                "trpo_timesteps_per_batch",
                "batch_size",
                "gamma",
                "gae_lambda",
                "ent_coef",
                "target_kl",
                "cg_max_steps",
                "cg_damping",
                "n_critic_updates",
                "sub_sampling_factor",
                "ppo_epochs",
                "ppo_clip_range",
                "value_loss_coef",
                "max_grad_norm",
                "seed",
                "device",
                "n_envs",
                "vec_env",
            )
        },
        "model_architecture": {
            key: getattr(args, key)
            for key in (
                "policy_mode",
                "builtin_policy",
                "candidate_sampler",
                "candidate_scorer",
                "observation_format",
                "policy_layers",
                "value_layers",
                "activation",
                "share_features_extractor",
                "features_extractor",
                "features_dim",
                "features_extractor_kwargs",
                "allow_bdp_candidate_count_change",
            )
        },
        "logging": {
            key: getattr(args, key)
            for key in ("log_path", "log_interval", "save_freq", "monitor_mode", "eval_freq", "n_eval_episodes")
        },
        "evaluation": {
            "evaluate_episodes": args.evaluate_episodes,
            "model_path": args.model_path,
            "resume_model_path": args.resume_model_path,
        },
        "render": {
            "render_mode": args.render_mode,
            "render_train": args.render_train,
            "render_freq": args.render_freq,
            "play_mode": args.play_mode,
        },
    }
