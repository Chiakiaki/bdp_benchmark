"""V3 job contracts and parser/effective-configuration integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


JOB_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "job_scripts"
BUNDLE_SOURCES = {
    "benchmark_metadrive_12mps": "benchmark_metadrive_12mps",
    "benchmark": "benchmark",
}
SPEED_CONTRACTS = {
    "benchmark_metadrive_12mps": (43.2, 12.0, True),
    "benchmark": (80.0, 22.222222, True),
}
POLICIES = ("bdp_frenet", "builtin")
V3_SPEED_SETTINGS = {
    "frenet_speed_command_time_s": 1.0,
    "frenet_speed_delta_rate_mps2": 8.0,
    "frenet_speed_action_scales": [-1, -0.5, 0, 0.5, 1],
}


def _job_path(bundle: str, policy: str, version: str = "v3") -> Path:
    return JOB_ROOT / bundle / f"metadrive_frenet_pid_{version}_{policy}_benchmark.yaml"


def _load_job(bundle: str, policy: str, version: str = "v3") -> dict:
    path = _job_path(bundle, policy, version)
    assert path.is_file(), f"Missing benchmark job: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("bundle", BUNDLE_SOURCES)
@pytest.mark.parametrize("policy", POLICIES)
def test_v3_job_preserves_its_v2_source(bundle: str, policy: str) -> None:
    expected = _load_job(BUNDLE_SOURCES[bundle], policy, "v2")
    assert expected["environment"]["trajectory_execution_mode"] == "frenet_pid_v2"
    expected["runtime"]["run_name"] = expected["runtime"]["run_name"].replace("_v2_", "_v3_")
    expected["environment"]["trajectory_execution_mode"] = "frenet_pid_v3"
    del expected["frenet"]["frenet_speed_delta_mps"]
    expected["frenet"].update(V3_SPEED_SETTINGS)
    expected["frenet"].update(frenet_include_curvature_candidates=True, frenet_curvature_steering_fraction=0.9)

    # Whole-job equality also protects inherited reward/observation overrides,
    # PPO, PID, model architecture, diagnostics, and logging settings.
    assert _load_job(bundle, policy) == expected


@pytest.mark.parametrize("bundle", BUNDLE_SOURCES)
def test_v3_pair_has_matched_settings(bundle: str) -> None:
    builtin = _load_job(bundle, "builtin")
    bdp = _load_job(bundle, "bdp_frenet")
    builtin_name = builtin["runtime"].pop("run_name")
    bdp_name = bdp["runtime"].pop("run_name")
    assert builtin_name.replace("_builtin_", "_bdp_frenet_") == bdp_name
    assert "_v3_" in builtin_name
    assert builtin["model_architecture"].pop("policy_mode") == "builtin"
    assert bdp["model_architecture"].pop("policy_mode") == "bdp"
    assert bdp["model_architecture"].pop("candidate_scorer") == "dot"
    assert builtin == bdp


@pytest.mark.parametrize("bundle", BUNDLE_SOURCES)
@pytest.mark.parametrize("policy", POLICIES)
def test_v3_job_has_approved_settings(bundle: str, policy: str) -> None:
    job = _load_job(bundle, policy)
    physical_speed, target_speed, curves = SPEED_CONTRACTS[bundle]
    environment = job["environment"]["environment_config"]
    assert environment["agent_configs"]["default_agent"]["max_speed_km_h"] == physical_speed
    assert environment["vehicle_config"]["lidar"]["num_others"] == 4
    assert environment["out_of_route_done"] is True
    assert environment["on_continuous_line_done"] is False
    assert environment["store_map"] is False
    assert job["environment"]["trajectory_execution_mode"] == "frenet_pid_v3"
    frenet = job["frenet"]
    assert {key: frenet[key] for key in V3_SPEED_SETTINGS} == V3_SPEED_SETTINGS
    assert "frenet_speed_delta_mps" not in frenet
    assert frenet["maximum_target_speed_mps"] == target_speed
    assert frenet.get("frenet_include_curvature_candidates", False) is curves
    if curves:
        assert frenet["frenet_curvature_steering_fraction"] == 0.9
    assert frenet["trajectory_horizon_s"] == 2.0
    assert frenet["trajectory_sample_count"] == 11
    assert frenet["position_feature_scale_m"] == 50.0
    assert frenet["speed_feature_scale_mps"] == 40.0
    assert job["model_architecture"]["candidate_sampler"] == "frenet_route_continuous"
    assert job["pid"]["pid_controller_mode"] == "adaptive_pursuit"
    assert job["pid"]["pid_lookahead_time_s"] == 0.7
    assert job["pid"]["pid_speed_preview_s"] == 0.4
    assert job["algorithm"] == {
        "name": "PPO",
        "num_timesteps": 2000000,
        "learning_rate": 5.0e-5,
        "schedule": "fixed",
        "trpo_timesteps_per_batch": 1000,
        "batch_size": 100,
        "ppo_epochs": 20,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip_range": 0.2,
        "target_kl": None,
        "n_envs": 8,
        "vec_env": "subproc",
        "seed": 41,
        "device": "auto",
    }


@pytest.mark.parametrize("bundle", BUNDLE_SOURCES)
@pytest.mark.parametrize("policy", POLICIES)
def test_v3_job_documents_active_speed_preview(bundle: str, policy: str) -> None:
    text = _job_path(bundle, policy).read_text(encoding="utf-8")
    preview_line = next(line for line in text.splitlines() if line.strip().startswith("pid_speed_preview_s:"))
    assert "ignored" not in preview_line.lower()
    assert "preview" in preview_line.lower()


@pytest.mark.parametrize("bundle", BUNDLE_SOURCES)
@pytest.mark.parametrize("policy", POLICIES)
def test_v3_job_parses_and_resolves(bundle: str, policy: str) -> None:
    from bdp_benchmark.config import parse_args, resolved_config

    source_args = parse_args(["--config", str(_job_path(BUNDLE_SOURCES[bundle], policy, "v2"))])
    expected = resolved_config(source_args)
    # Keep real validation enabled: missing v3 wiring must remain a visible
    # integration failure, not a skip, xfail, or mocked parser success.
    args = parse_args(["--config", str(_job_path(bundle, policy))])
    actual = resolved_config(args)
    expected["runtime"]["run_name"] = expected["runtime"]["run_name"].replace("_v2_", "_v3_")
    expected["environment"]["trajectory_execution_mode"] = "frenet_pid_v3"
    expected["frenet"].update(V3_SPEED_SETTINGS)
    expected["frenet"].update(frenet_include_curvature_candidates=True, frenet_curvature_steering_fraction=0.9)
    for key, value in V3_SPEED_SETTINGS.items():
        assert getattr(args, key) == value
        assert actual["frenet"][key] == value
    assert args.frenet_include_curvature_candidates is SPEED_CONTRACTS[bundle][2]
    # Resolved configs may retain the shared, inactive legacy default. Only the
    # new job YAMLs must omit it; it has no meaning in the v3 contract.
    actual["frenet"].pop("frenet_speed_delta_mps", None)
    expected["frenet"].pop("frenet_speed_delta_mps", None)
    assert actual == expected
