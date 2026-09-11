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
    job_root = Path(__file__).resolve().parents[1] / "scripts" / "job_scripts"
    jobs = sorted(job_root.rglob("*.yaml"))
    assert len(jobs) >= 12
    for job in jobs:
        args = parse_args(["--config", str(job)])
        assert args.num_timesteps > 0, job.name
    assert any(job.name == "visual_check_highway_bdp.yaml" for job in jobs)
    assert any(job.name == "visual_check_metadrive_bdp.yaml" for job in jobs)


def test_visual_check_sets_single_rendered_deterministic_evaluation_defaults() -> None:
    args = parse_args(["--simulator", "highway", "--visual_check"], validate=True)

    assert args.visual_check is True
    assert args.run_mode == "evaluate"
    assert args.n_envs == 1
    assert args.vec_env == "dummy"
    assert args.render_mode == "human"
    assert args.inference_mode == "deterministic"
    assert args.visual_check_variant_index == 0
    assert resolved_config(args)["runtime"]["visual_check_variant_index"] == 0


def test_visual_check_preserves_explicit_stochastic_mode() -> None:
    args = parse_args(
        ["--simulator", "highway", "--visual_check", "--inference_mode", "stochastic"],
        validate=True,
    )

    assert args.inference_mode == "stochastic"


def test_visual_check_job_config_also_enables_human_evaluation_defaults() -> None:
    job_root = Path(__file__).resolve().parents[1] / "scripts" / "job_scripts"
    job = next(job_root.rglob("visual_check_highway_bdp.yaml"))
    args = parse_args(["--config", str(job)])

    assert args.visual_check is True
    assert args.run_mode == "evaluate"
    assert args.render_mode == "human"
    assert args.n_envs == 1


def test_frenet_pid_v2_is_an_explicit_supported_execution_mode() -> None:
    args = parse_args(
        [
            "--simulator",
            "highway",
            "--trajectory_execution_mode",
            "frenet_pid_v2",
            "--policy_mode",
            "bdp",
            "--candidate_sampler",
            "frenet",
        ]
    )

    assert args.trajectory_execution_mode == "frenet_pid_v2"


def test_highway_environment_variants_use_shared_ego_relative_observation(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed.yaml"
    config_path.write_text(
        """
environment:
  simulator: highway
  environment_variants:
    - highway-fast-v0
    - merge-v1
    - roundabout-v1
algorithm:
  n_envs: 6
  vec_env: subproc
""",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    assert args.environment_variants == ["highway-fast-v0", "merge-v1", "roundabout-v1"]
    assert args.environment_config["observation"] == {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "normalize": True,
        "absolute": False,
        "order": "sorted",
    }
    assert resolved_config(args)["environment"]["environment_variants"] == args.environment_variants


def test_cli_environment_variants_override_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed.yaml"
    config_path.write_text(
        """
environment:
  simulator: highway
  environment_variants: [highway-fast-v0, merge-v1, roundabout-v1]
algorithm:
  n_envs: 6
""",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--environment_variants",
            "merge-v1",
            "roundabout-v1",
        ]
    )

    assert args.environment_variants == ["merge-v1", "roundabout-v1"]


def test_omitted_environment_variants_preserve_single_environment_behavior() -> None:
    args = parse_args([])

    assert args.env_id == "highway-fast-v0"
    assert args.environment_variants == []
    assert args.environment_config == {}


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--simulator", "metadrive", "--environment_variants", "MetaDrive-v0"],
            "HighwayEnv-only",
        ),
        (
            ["--environment_variants", "merge-v1", "merge-v1", "--n_envs", "2"],
            "must not contain duplicates",
        ),
        (
            [
                "--environment_variants",
                "highway-fast-v0",
                "merge-v1",
                "roundabout-v1",
                "--n_envs",
                "2",
            ],
            "at least one worker per variant",
        ),
        (
            [
                "--environment_variants",
                "highway-fast-v0",
                "merge-v1",
                "roundabout-v1",
                "--n_envs",
                "3",
                "--environment_config",
                '{"observation":{"type":"Kinematics","absolute":true}}',
            ],
            "shared ego-relative observation contract",
        ),
    ],
)
def test_environment_variants_reject_invalid_configuration(argv: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_args(argv)


def test_environment_variants_allow_single_worker_evaluation() -> None:
    args = parse_args(
        [
            "--run_mode",
            "evaluate",
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--n_envs",
            "1",
        ]
    )

    assert args.n_envs == 1


def test_environment_variants_warn_about_uneven_training_assignment() -> None:
    with pytest.warns(UserWarning, match="not divisible"):
        args = parse_args(
            [
                "--environment_variants",
                "highway-fast-v0",
                "merge-v1",
                "roundabout-v1",
                "--n_envs",
                "4",
            ]
        )

    assert args.n_envs == 4


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            [
                "--visual_check",
                "--environment_variants",
                "highway-fast-v0",
                "merge-v1",
                "roundabout-v1",
                "--visual_check_variant_index",
                "-1",
            ],
            "non-negative",
        ),
        (
            ["--visual_check", "--visual_check_variant_index", "4"],
            "requires environment_variants",
        ),
        (
            [
                "--environment_variants",
                "highway-fast-v0",
                "merge-v1",
                "roundabout-v1",
                "--n_envs",
                "3",
                "--visual_check_variant_index",
                "4",
            ],
            "only applies to --visual_check",
        ),
    ],
)
def test_visual_check_variant_index_rejects_invalid_use(argv: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_args(argv)
