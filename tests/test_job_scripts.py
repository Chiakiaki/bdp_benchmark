from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml


JOB_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "job_scripts"
METADRIVE_BENCHMARK_JOB_ROOT = JOB_ROOT / "benchmark"
METADRIVE_12MPS_JOB_ROOT = JOB_ROOT / "benchmark_metadrive_12mps"


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
        (METADRIVE_BENCHMARK_JOB_ROOT / f"metadrive_{mode}_builtin_benchmark.yaml").read_text(encoding="utf-8")
    )
    bdp = yaml.safe_load(
        (METADRIVE_BENCHMARK_JOB_ROOT / f"metadrive_{mode}_bdp_frenet_benchmark.yaml").read_text(encoding="utf-8")
    )

    assert builtin["environment"] == bdp["environment"]
    assert builtin["frenet"] == bdp["frenet"]
    assert builtin["algorithm"] == bdp["algorithm"]
    assert builtin["logging"] == bdp["logging"]
    assert builtin["environment"]["trajectory_execution_mode"] == mode
    assert builtin["model_architecture"]["policy_mode"] == "builtin"
    assert bdp["model_architecture"]["policy_mode"] == "bdp"
    assert bdp["model_architecture"]["candidate_sampler"] == "frenet_route_continuous"
    assert bdp["model_architecture"]["candidate_scorer"] == "dot"


@pytest.mark.parametrize(
    ("bundle_name", "script_name", "first_job", "second_job"),
    [
        (
            "benchmark",
            "run_metadrive_bdp_frenet_comparison.sh",
            "benchmark/metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml",
            "benchmark/metadrive_native_bdp_frenet_benchmark.yaml",
        ),
        (
            "benchmark",
            "run_metadrive_builtin_comparison.sh",
            "benchmark/metadrive_frenet_pid_v2_builtin_benchmark.yaml",
            "benchmark/metadrive_native_builtin_benchmark.yaml",
        ),
        (
            "benchmark",
            "run_metadrive_frenet_pid_comparison.sh",
            "benchmark/metadrive_frenet_pid_bdp_frenet_benchmark.yaml",
            "benchmark/metadrive_frenet_pid_builtin_benchmark.yaml",
        ),
        (
            "benchmark_metadrive_12mps",
            "run_metadrive_bdp_frenet_comparison.sh",
            "benchmark_metadrive_12mps/metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml",
            "benchmark_metadrive_12mps/metadrive_native_bdp_frenet_benchmark.yaml",
        ),
        (
            "benchmark_metadrive_12mps",
            "run_metadrive_builtin_comparison.sh",
            "benchmark_metadrive_12mps/metadrive_frenet_pid_v2_builtin_benchmark.yaml",
            "benchmark_metadrive_12mps/metadrive_native_builtin_benchmark.yaml",
        ),
    ],
)
def test_metadrive_comparison_launchers_are_valid_and_ordered(
    bundle_name: str,
    script_name: str,
    first_job: str,
    second_job: str,
) -> None:
    script = JOB_ROOT / bundle_name / script_name

    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    source = script.read_text(encoding="utf-8")
    assert (JOB_ROOT / first_job).is_file()
    assert (JOB_ROOT / second_job).is_file()
    assert source.index(first_job) < source.index(second_job)
    assert source.count('"$@"') == 2


def test_active_metadrive_benchmarks_use_paper_aligned_ppo_configuration() -> None:
    names = (
        "metadrive_frenet_pid_bdp_frenet_benchmark.yaml",
        "metadrive_frenet_pid_builtin_benchmark.yaml",
        "metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml",
        "metadrive_frenet_pid_v2_builtin_benchmark.yaml",
        "metadrive_native_bdp_frenet_benchmark.yaml",
        "metadrive_native_builtin_benchmark.yaml",
        "metadrive_native_continuous_builtin_benchmark.yaml",
    )

    for name in names:
        config = yaml.safe_load((METADRIVE_BENCHMARK_JOB_ROOT / name).read_text(encoding="utf-8"))
        algorithm = config["algorithm"]
        assert algorithm["n_envs"] == 8, name
        assert algorithm["trpo_timesteps_per_batch"] == 1000, name
        assert algorithm["batch_size"] == 100, name
        assert algorithm["ppo_epochs"] == 20, name
        assert algorithm["learning_rate"] == 5.0e-5, name
        assert algorithm["gamma"] == 0.99, name
        assert algorithm["gae_lambda"] == 0.95, name
        assert algorithm["ppo_clip_range"] == 0.2, name
        rollout_size = algorithm["trpo_timesteps_per_batch"] * algorithm["n_envs"]
        assert rollout_size == 8000, name
        assert rollout_size // algorithm["batch_size"] * algorithm["ppo_epochs"] == 1600, name
        assert algorithm["num_timesteps"] // rollout_size * 1600 == 400_000, name
        assert config["environment"]["environment_config"]["store_map"] is False, name
        assert config["logging"]["log_interval"] == 1, name


def test_all_metadrive_job_scripts_use_demo_termination_overrides() -> None:
    metadrive_jobs = []
    for path in JOB_ROOT.rglob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config["environment"]["simulator"] == "metadrive":
            metadrive_jobs.append(path)
            env_config = config["environment"]["environment_config"]
            assert env_config["out_of_route_done"] is True, path
            assert env_config["on_continuous_line_done"] is False, path

    assert metadrive_jobs


def test_active_metadrive_job_scripts_use_expert_nearby_vehicle_observation() -> None:
    metadrive_jobs = []
    for bundle_root in (METADRIVE_BENCHMARK_JOB_ROOT, METADRIVE_12MPS_JOB_ROOT):
        for path in bundle_root.glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if config["environment"]["simulator"] == "metadrive":
                metadrive_jobs.append(path)
                lidar_config = config["environment"]["environment_config"]["vehicle_config"]["lidar"]
                assert lidar_config["num_others"] == 4, path

    assert metadrive_jobs


def test_active_metadrive_frenet_pid_jobs_share_adaptive_pursuit_configuration() -> None:
    reference = yaml.safe_load(
        (METADRIVE_BENCHMARK_JOB_ROOT / "metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml").read_text(encoding="utf-8")
    )["pid"]
    assert reference["pid_controller_mode"] == "adaptive_pursuit"

    jobs = []
    for bundle_root in (METADRIVE_BENCHMARK_JOB_ROOT, METADRIVE_12MPS_JOB_ROOT):
        for path in bundle_root.glob("metadrive_*_benchmark.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if config["environment"]["trajectory_execution_mode"] in {"frenet_pid", "frenet_pid_v2"}:
                jobs.append(path)
                assert config["pid"] == reference, path

    assert jobs


@pytest.mark.parametrize(
    ("bundle_root", "physical_speed_km_h", "target_speed_mps", "frenet_speed_delta_mps"),
    [
        (METADRIVE_BENCHMARK_JOB_ROOT, 80.0, 22.222222, 5.0),
        (METADRIVE_12MPS_JOB_ROOT, 43.2, 12.0, 2.5),
    ],
)
def test_metadrive_bundles_have_consistent_speed_contracts(
    bundle_root: Path,
    physical_speed_km_h: float,
    target_speed_mps: float,
    frenet_speed_delta_mps: float,
) -> None:
    jobs = []
    for path in bundle_root.glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        jobs.append(path)
        env_config = config["environment"]["environment_config"]
        assert env_config["agent_configs"]["default_agent"]["max_speed_km_h"] == physical_speed_km_h, path

        mode = config["environment"]["trajectory_execution_mode"]
        policy_mode = config["model_architecture"]["policy_mode"]
        if mode in {"frenet_pid", "frenet_pid_v2"}:
            assert config["frenet"]["maximum_target_speed_mps"] == target_speed_mps, path
            assert config["frenet"]["frenet_speed_delta_mps"] == frenet_speed_delta_mps, path
        elif policy_mode == "bdp":
            assert config["frenet"]["maximum_target_speed_mps"] == target_speed_mps, path

    assert jobs


@pytest.mark.parametrize("bundle_root", [METADRIVE_BENCHMARK_JOB_ROOT, METADRIVE_12MPS_JOB_ROOT])
def test_continuous_native_job_is_a_controlled_action_space_reference(bundle_root: Path) -> None:
    discrete = yaml.safe_load(
        (bundle_root / "metadrive_native_builtin_benchmark.yaml").read_text(encoding="utf-8")
    )
    continuous = yaml.safe_load(
        (bundle_root / "metadrive_native_continuous_builtin_benchmark.yaml").read_text(encoding="utf-8")
    )

    assert continuous["environment"]["environment_config"]["discrete_action"] is False
    del discrete["runtime"]["run_name"]
    del continuous["runtime"]["run_name"]
    del continuous["environment"]["environment_config"]["discrete_action"]
    assert continuous == discrete


@pytest.mark.parametrize(
    ("actual_name", "nominal_name"),
    [
        ("metadrive_frenet_pid_bdp_frenet_benchmark.yaml", "metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml"),
        ("metadrive_frenet_pid_builtin_benchmark.yaml", "metadrive_frenet_pid_v2_builtin_benchmark.yaml"),
    ],
)
def test_frenet_pid_v1_and_v2_differ_only_by_trajectory_origin(
    actual_name: str,
    nominal_name: str,
) -> None:
    actual = yaml.safe_load((METADRIVE_BENCHMARK_JOB_ROOT / actual_name).read_text(encoding="utf-8"))
    nominal = yaml.safe_load((METADRIVE_BENCHMARK_JOB_ROOT / nominal_name).read_text(encoding="utf-8"))

    assert actual["environment"]["trajectory_execution_mode"] == "frenet_pid"
    assert nominal["environment"]["trajectory_execution_mode"] == "frenet_pid_v2"
    del actual["runtime"]["run_name"]
    del nominal["runtime"]["run_name"]
    del actual["environment"]["trajectory_execution_mode"]
    del nominal["environment"]["trajectory_execution_mode"]
    assert actual == nominal


@pytest.mark.parametrize("bundle_root", [METADRIVE_BENCHMARK_JOB_ROOT, METADRIVE_12MPS_JOB_ROOT])
def test_active_metadrive_frenet_jobs_use_route_continuous_sampler(bundle_root: Path) -> None:
    selected = []
    for path in bundle_root.glob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        mode = config["environment"]["trajectory_execution_mode"]
        policy_mode = config["model_architecture"]["policy_mode"]
        uses_frenet_geometry = mode in {"frenet_pid", "frenet_pid_v2"} or (
            mode == "native_controller" and policy_mode == "bdp"
        )
        sampler = config["model_architecture"].get("candidate_sampler")
        if uses_frenet_geometry:
            selected.append(path)
            assert sampler == "frenet_route_continuous", path
        else:
            assert sampler != "frenet_route_continuous", path

    assert selected
