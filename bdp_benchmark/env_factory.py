"""Construct benchmark environments without modifying simulator packages."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from critic_based_rl.envs import GenericDiscreteCandidateEnv
from critic_based_rl.monitoring import wrap_prebuilt_vec_env_with_monitor
from critic_based_rl.samplers import DiscreteOneHotExternalSampler

from .common.candidates import CandidateGenerationConfig, FrenetCandidateSampler
from .common.tracking import TrackerConfig


def generation_config_from_args(args) -> CandidateGenerationConfig:
    return CandidateGenerationConfig(
        horizon_s=float(args.trajectory_horizon_s),
        sample_count=int(args.trajectory_sample_count),
        position_scale_m=float(args.position_feature_scale_m),
        speed_scale_mps=float(args.speed_feature_scale_mps),
        native_lateral_span_m=float(args.native_lateral_span_m),
        native_speed_span_mps=float(args.native_speed_span_mps),
        minimum_target_speed_mps=float(args.minimum_target_speed_mps),
        maximum_target_speed_mps=float(args.maximum_target_speed_mps),
        lane_change_width_scale=float(args.frenet_lane_change_width_scale),
        speed_delta_mps=float(args.frenet_speed_delta_mps),
    )


def tracker_config_from_args(args) -> TrackerConfig:
    return TrackerConfig(
        lookahead_points=int(args.pid_lookahead_points),
        heading_kp=float(args.pid_heading_kp),
        heading_ki=float(args.pid_heading_ki),
        heading_kd=float(args.pid_heading_kd),
        cross_track_kp=float(args.pid_cross_track_kp),
        speed_kp=float(args.pid_speed_kp),
        speed_ki=float(args.pid_speed_ki),
        speed_kd=float(args.pid_speed_kd),
        max_steering_rad=float(args.pid_max_steering_rad),
        max_accel_mps2=float(args.pid_max_accel_mps2),
        max_decel_mps2=float(args.pid_max_decel_mps2),
    )


def make_raw_benchmark_env(args, *, render_mode: str | None = None):
    generation_config = generation_config_from_args(args)
    tracker_config = tracker_config_from_args(args)
    if args.simulator == "highway":
        from .highway.env import HighwayBenchmarkEnv

        return HighwayBenchmarkEnv(
            env_id=args.env_id,
            execution_mode=args.trajectory_execution_mode,
            generation_config=generation_config,
            tracker_config=tracker_config,
            env_config=args.environment_config,
            render_mode=render_mode,
        )
    if args.simulator == "metadrive":
        from .metadrive.env import MetaDriveBenchmarkEnv

        return MetaDriveBenchmarkEnv(
            env_id=args.env_id,
            execution_mode=args.trajectory_execution_mode,
            generation_config=generation_config,
            tracker_config=tracker_config,
            env_config=args.environment_config,
            render_mode=render_mode,
        )
    raise ValueError(f"Unsupported simulator: {args.simulator}")


def make_single_env(args, *, monitor_path: Path | None = None, rank: int = 0):
    should_render = bool(getattr(args, "render_train", False)) or args.run_mode == "evaluate" or bool(args.play_mode)
    render_mode = args.render_mode if should_render else None
    raw_env = make_raw_benchmark_env(args, render_mode=render_mode)
    if bool(getattr(args, "visual_check", False)):
        raw_env.enable_visualization()
    if args.policy_mode == "builtin":
        env = raw_env
    else:
        if args.candidate_sampler == "one_hot":
            sampler = DiscreteOneHotExternalSampler(raw_env.action_space)
        else:
            sampler = FrenetCandidateSampler(
                raw_env.action_space,
                feature_dim=generation_config_from_args(args).feature_dim,
            )
        env = GenericDiscreteCandidateEnv(raw_env, sampler=sampler, max_candidates=int(raw_env.action_space.n))
    if monitor_path is not None:
        monitor_path.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(monitor_path / f"env_{rank}"))
    return env


def make_vector_env(args, *, log_dir: Path, is_train: bool):
    n_envs = int(args.n_envs) if is_train else 1
    inner_monitor = args.monitor_mode in ("auto", "env")
    if args.monitor_mode == "vec":
        inner_monitor = False

    def factory(rank: int) -> Callable:
        def _make():
            local_args = copy(args)
            if not is_train:
                local_args.run_mode = "evaluate"
            monitor_path = (log_dir / ("monitor" if is_train else "evaluation") / f"rank_{rank}") if inner_monitor else None
            return make_single_env(local_args, monitor_path=monitor_path, rank=rank)

        return _make

    factories = [factory(rank) for rank in range(n_envs)]
    if n_envs == 1 or args.vec_env == "dummy":
        env = DummyVecEnv(factories)
    else:
        start_method = "spawn" if args.simulator == "metadrive" else None
        env = SubprocVecEnv(factories, start_method=start_method)
    if args.monitor_mode == "vec":
        env = wrap_prebuilt_vec_env_with_monitor(env, log_dir / ("monitor" if is_train else "evaluation"))
    return env
