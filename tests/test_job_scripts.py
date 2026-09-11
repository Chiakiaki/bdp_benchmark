from __future__ import annotations

import subprocess
from pathlib import Path

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
