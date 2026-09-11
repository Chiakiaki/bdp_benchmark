from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml


JOB_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "job_scripts"


def test_highway_mixed_comparison_launcher_is_valid_and_ordered() -> None:
    script = JOB_ROOT / "run_highway_mixed_native_comparison.sh"

    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    source = script.read_text(encoding="utf-8")
    bdp_position = source.index("highway_mixed_native_bdp_frenet_10hz_benchmark.yaml")
    builtin_position = source.index("highway_mixed_native_builtin_10hz_benchmark.yaml")
    assert bdp_position < builtin_position
    assert source.count('"$@"') == 2


def test_mixed_highway_jobs_preserve_official_timing_and_add_10hz_pair() -> None:
    original_names = (
        "highway_mixed_native_bdp_frenet_benchmark.yaml",
        "highway_mixed_native_builtin_benchmark.yaml",
    )
    ten_hz_names = (
        "highway_mixed_native_bdp_frenet_10hz_benchmark.yaml",
        "highway_mixed_native_bdp_one_hot_10hz_benchmark.yaml",
        "highway_mixed_native_builtin_10hz_benchmark.yaml",
    )

    for name in original_names:
        config = yaml.safe_load((JOB_ROOT / name).read_text(encoding="utf-8"))
        assert "simulation_frequency" not in config["environment"]["environment_config"]
        assert "policy_frequency" not in config["environment"]["environment_config"]
        assert config["algorithm"]["gamma"] == 0.80
        assert config["algorithm"]["num_timesteps"] == 2_000_000

    for name in ten_hz_names:
        config = yaml.safe_load((JOB_ROOT / name).read_text(encoding="utf-8"))
        env_config = config["environment"]["environment_config"]
        assert env_config["simulation_frequency"] == 20
        assert env_config["policy_frequency"] == 10
        assert "duration" not in env_config
        assert config["algorithm"]["gamma"] == 0.98
        assert config["algorithm"]["num_timesteps"] == 10_000_000


def test_mixed_highway_one_hot_is_a_controlled_frenet_ablation() -> None:
    frenet = yaml.safe_load(
        (JOB_ROOT / "highway_mixed_native_bdp_frenet_10hz_benchmark.yaml").read_text(encoding="utf-8")
    )
    one_hot = yaml.safe_load(
        (JOB_ROOT / "highway_mixed_native_bdp_one_hot_10hz_benchmark.yaml").read_text(encoding="utf-8")
    )

    assert one_hot["model_architecture"]["candidate_sampler"] == "one_hot"
    assert frenet["model_architecture"]["candidate_sampler"] == "native_action_frenet"
    del one_hot["runtime"]["run_name"]
    del frenet["runtime"]["run_name"]
    del one_hot["model_architecture"]["candidate_sampler"]
    del frenet["model_architecture"]["candidate_sampler"]
    assert one_hot == frenet


@pytest.mark.parametrize("mode", ["frenet_pid", "frenet_pid_v2"])
def test_metadrive_frenet_pid_benchmark_pair_has_matched_training_settings(mode: str) -> None:
    builtin = yaml.safe_load(
        (JOB_ROOT / f"metadrive_{mode}_builtin_benchmark.yaml").read_text(encoding="utf-8")
    )
    bdp = yaml.safe_load(
        (JOB_ROOT / f"metadrive_{mode}_bdp_frenet_benchmark.yaml").read_text(encoding="utf-8")
    )

    assert builtin["environment"] == bdp["environment"]
    assert builtin["frenet"] == bdp["frenet"]
    assert builtin["algorithm"] == bdp["algorithm"]
    assert builtin["logging"] == bdp["logging"]
    assert builtin["environment"]["trajectory_execution_mode"] == mode
    assert builtin["model_architecture"]["policy_mode"] == "builtin"
    assert bdp["model_architecture"]["policy_mode"] == "bdp"
    assert bdp["model_architecture"]["candidate_sampler"] == "frenet"
    assert bdp["model_architecture"]["candidate_scorer"] == "dot"


@pytest.mark.parametrize(
    ("script_name", "first_job", "second_job"),
    [
        (
            "run_metadrive_bdp_frenet_comparison.sh",
            "metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml",
            "metadrive_native_bdp_frenet_benchmark.yaml",
        ),
        (
            "run_metadrive_builtin_comparison.sh",
            "metadrive_frenet_pid_v2_builtin_benchmark.yaml",
            "metadrive_native_builtin_benchmark.yaml",
        ),
    ],
)
def test_metadrive_comparison_launchers_are_valid_and_ordered(
    script_name: str,
    first_job: str,
    second_job: str,
) -> None:
    script = JOB_ROOT / script_name

    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    source = script.read_text(encoding="utf-8")
    assert source.index(first_job) < source.index(second_job)
    assert source.count('"$@"') == 2


def test_active_metadrive_benchmarks_use_requested_parallel_batch_configuration() -> None:
    names = (
        "metadrive_frenet_pid_bdp_frenet_benchmark.yaml",
        "metadrive_frenet_pid_builtin_benchmark.yaml",
        "metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml",
        "metadrive_frenet_pid_v2_builtin_benchmark.yaml",
        "metadrive_native_bdp_frenet_benchmark.yaml",
        "metadrive_native_builtin_benchmark.yaml",
    )

    for name in names:
        config = yaml.safe_load((JOB_ROOT / name).read_text(encoding="utf-8"))
        algorithm = config["algorithm"]
        assert algorithm["n_envs"] == 8, name
        assert algorithm["batch_size"] == 1000, name
        assert algorithm["trpo_timesteps_per_batch"] * algorithm["n_envs"] % algorithm["batch_size"] == 0, name
