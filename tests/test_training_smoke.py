from __future__ import annotations

from pathlib import Path

import pytest

from bdp_benchmark.config import parse_args
from bdp_benchmark.runner import run_training
from bdp_benchmark.env_factory import make_single_env
from critic_based_rl.model import get_algorithm_class


@pytest.mark.parametrize("bundle", ["benchmark", "benchmark_metadrive_12mps"])
@pytest.mark.parametrize("policy_mode", ["builtin", "bdp"])
@pytest.mark.parametrize("version", ["v3", "v3_legacy"])
def test_v3_two_worker_train_save_reload(bundle, policy_mode, version, tmp_path):
    import yaml
    job_root = Path(__file__).resolve().parents[1] / "scripts/job_scripts" / bundle
    suffix = "bdp_frenet" if policy_mode == "bdp" else "builtin"
    args = parse_args([
        "--config", str(job_root / f"metadrive_frenet_pid_{version}_{suffix}_benchmark.yaml"),
        "--n_envs", "2", "--num_timesteps", "64", "--trpo_timesteps_per_batch", "16",
        "--batch_size", "16", "--ppo_epochs", "1", "--save_freq", "32", "--eval_freq", "0",
        "--device", "cpu", "--log_path", str(tmp_path),
    ])
    log_dir = run_training(args)
    path = log_dir / "models" / f"PPO_{'BDP' if policy_mode == 'bdp' else 'builtin'}_final_model.zip"
    assert path.is_file()
    assert len(list((log_dir / "models").rglob("*.zip"))) > 1  # During-training checkpoints too.
    effective = yaml.safe_load((log_dir / "resolved_config.yaml").read_text())
    assert effective["environment"]["trajectory_execution_mode"] == f"frenet_pid_{version}"
    assert effective["frenet"]["frenet_speed_command_time_s"] == 1
    env = make_single_env(args)
    try:
        count = 25
        model = get_algorithm_class("PPO").load(path, env=env, device="cpu")
        obs, _ = env.reset(seed=41)
        assert env.action_space.n == count
        if policy_mode == "bdp":
            assert obs["candidates"].shape == (count, 55)
        action, _ = model.predict(obs, deterministic=True)
        env.step(action)
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["frenet_pid", "frenet_pid_v2"])
@pytest.mark.parametrize("policy_mode", ["builtin", "bdp"])
def test_curvature_metadrive_two_worker_train_save_and_reload(mode, policy_mode, tmp_path):
    job_root = Path(__file__).resolve().parents[1] / "scripts/job_scripts/benchmark_metadrive_12mps"
    suffix = "bdp_frenet" if policy_mode == "bdp" else "builtin"
    args = parse_args([
        "--config", str(job_root / f"metadrive_{mode}_{suffix}_benchmark.yaml"),
        "--n_envs", "2", "--num_timesteps", "32", "--trpo_timesteps_per_batch", "8",
        "--batch_size", "8", "--ppo_epochs", "1", "--save_freq", "16", "--eval_freq", "0",
        "--device", "cpu", "--log_path", str(tmp_path),
    ])
    # Keep the job's map, traffic, observation and reward for this short integration check.
    log_dir = run_training(args)
    path = log_dir / "models" / f"PPO_{'BDP' if policy_mode == 'bdp' else 'builtin'}_final_model.zip"
    assert path.is_file()
    import yaml

    effective = yaml.safe_load((log_dir / "resolved_config.yaml").read_text())
    assert effective["frenet"]["frenet_include_curvature_candidates"] is True
    env = make_single_env(args)
    try:
        model = get_algorithm_class("PPO").load(path, env=env, device="cpu")
        obs, _ = env.reset(seed=41)
        assert env.action_space.n == 25
        if policy_mode == "bdp":
            assert obs["candidates"].shape == (25, 55)
        action, _ = model.predict(obs, deterministic=True)
        assert 0 <= int(action) < 25
        env.step(action)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("policy_mode", "candidate_sampler"),
    [
        ("builtin", "one_hot"),
        ("bdp", "one_hot"),
        ("bdp", "native_action_frenet"),
    ],
)
def test_tiny_highway_ppo_training(policy_mode: str, candidate_sampler: str, tmp_path: Path) -> None:
    args = parse_args(
        [
            "--simulator",
            "highway",
            "--env_id",
            "highway-fast-v0",
            "--trajectory_execution_mode",
            "native_controller",
            "--policy_mode",
            policy_mode,
            "--candidate_sampler",
            candidate_sampler,
            "--candidate_scorer",
            "dot",
            "--num_timesteps",
            "16",
            "--trpo_timesteps_per_batch",
            "8",
            "--batch_size",
            "8",
            "--ppo_epochs",
            "1",
            "--save_freq",
            "0",
            "--eval_freq",
            "0",
            "--log_interval",
            "1",
            "--log_path",
            str(tmp_path),
            "--environment_config",
            '{"vehicles_count": 0, "duration": 2, "initial_lane_id": 1}',
        ]
    )

    log_dir = run_training(args)

    model_tag = "BDP" if policy_mode == "bdp" else "builtin"
    assert (log_dir / "models" / f"PPO_{model_tag}_final_model.zip").exists()
