from __future__ import annotations

from pathlib import Path

import pytest

from bdp_benchmark.config import parse_args
from bdp_benchmark.runner import run_training


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
