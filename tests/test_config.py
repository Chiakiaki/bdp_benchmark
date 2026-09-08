from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bdp_benchmark.config import parse_args, resolved_config, validate_args


def test_cli_overrides_yaml_and_is_reflected_in_resolved_config(tmp_path: Path) -> None:
    config_path = tmp_path / "job.yaml"
    config_path.write_text(
        """
environment:
  simulator: metadrive
  env_id: MetaDrive-v0
  trajectory_execution_mode: native_controller
frenet:
  trajectory_sample_count: 9
algorithm:
  n_envs: 3
  vec_env: subproc
model_architecture:
  policy_mode: bdp
  candidate_sampler: native_action_frenet
""",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--trajectory_sample_count",
            "7",
            "--n_envs",
            "2",
        ]
    )

    assert args.simulator == "metadrive"
    assert args.trajectory_sample_count == 7
    assert args.n_envs == 2
    resolved = resolved_config(args)
    assert resolved["frenet"]["trajectory_sample_count"] == 7
    assert resolved["algorithm"]["n_envs"] == 2


def test_config_rejects_sampler_that_does_not_match_execution_mode() -> None:
    args = parse_args(
        [
            "--simulator",
            "highway",
            "--trajectory_execution_mode",
            "frenet_pid",
            "--policy_mode",
            "bdp",
            "--candidate_sampler",
            "native_action_frenet",
        ],
        validate=False,
    )

    with pytest.raises(ValueError, match="frenet_pid.*candidate_sampler=frenet"):
        validate_args(args)


def test_config_rejects_invalid_horizon_sampling() -> None:
    args = parse_args(["--trajectory_horizon_s", "0.0"], validate=False)
    with pytest.raises(ValueError, match="trajectory_horizon_s"):
        validate_args(args)

    args = parse_args(["--trajectory_sample_count", "1"], validate=False)
    with pytest.raises(ValueError, match="trajectory_sample_count"):
        validate_args(args)


def test_resolved_config_is_yaml_serializable() -> None:
    config = resolved_config(parse_args([]))
    assert yaml.safe_load(yaml.safe_dump(config)) == config


def test_all_reproducible_job_configs_parse() -> None:
    jobs = sorted((Path(__file__).resolve().parents[1] / "scripts" / "job_scripts").glob("*.yaml"))
    assert len(jobs) == 12
    for job in jobs:
        args = parse_args(["--config", str(job)])
        assert args.num_timesteps > 0, job.name
