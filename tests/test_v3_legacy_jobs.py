"""V3 sidecar copies, real parser integration, and saved-config compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


JOB_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "job_scripts"
BUNDLES = ("benchmark", "benchmark_metadrive_12mps")
POLICIES = ("bdp_frenet", "builtin")
VERSIONS = ("v3", "v3_legacy")


def _job_path(bundle: str, policy: str, version: str = "v3") -> Path:
    return JOB_ROOT / bundle / f"metadrive_frenet_pid_{version}_{policy}_benchmark.yaml"


def _load_job(bundle: str, policy: str, version: str = "v3") -> dict:
    path = _job_path(bundle, policy, version)
    assert path.is_file(), f"Missing benchmark job: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("policy", POLICIES)
def test_v3_legacy_job_preserves_v3_fields(bundle: str, policy: str) -> None:
    expected = _load_job(bundle, policy)
    assert expected["environment"]["trajectory_execution_mode"] == "frenet_pid_v3"
    assert "_v3_" in expected["runtime"]["run_name"]
    expected["environment"]["trajectory_execution_mode"] = "frenet_pid_v3_legacy"
    expected["runtime"]["run_name"] = expected["runtime"]["run_name"].replace("_v3_", "_v3_legacy_")

    # YAML parsing ignores comments; every setting beyond mode/name must match.
    assert _load_job(bundle, policy, "v3_legacy") == expected
    assert expected["frenet"]["frenet_include_curvature_candidates"] is True
    assert expected["frenet"]["frenet_curvature_steering_fraction"] == 0.9
    assert expected["frenet"]["frenet_speed_action_scales"] == [-1, -0.5, 0, 0.5, 1]
    assert expected["frenet"]["frenet_speed_command_time_s"] == 1.0
    assert expected["pid"]["pid_lookahead_time_s"] == 0.7
    assert expected["pid"]["pid_speed_preview_s"] == 0.4


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("version", VERSIONS)
def test_v3_job_parser_backup_roundtrip(
    bundle: str, policy: str, version: str, tmp_path: Path
) -> None:
    from bdp_benchmark.config import parse_args, resolved_config

    source = resolved_config(parse_args(["--config", str(_job_path(bundle, policy))]))
    mode = f"frenet_pid_{version}"
    source["environment"]["trajectory_execution_mode"] = mode
    source["runtime"]["run_name"] = source["runtime"]["run_name"].replace("_v3_", f"_{version}_")

    # Missing legacy parser/validation wiring is a failure, not a skip or xfail.
    args = parse_args(["--config", str(_job_path(bundle, policy, version))])
    assert args.trajectory_execution_mode == mode
    assert args.run_name == source["runtime"]["run_name"]
    assert resolved_config(args) == source

    backup = tmp_path / "resolved_config.yaml"
    backup.write_text(yaml.safe_dump(source), encoding="utf-8")
    restored = parse_args(["--config", str(backup)])
    assert restored.trajectory_execution_mode == mode
    assert restored.run_name == args.run_name
    assert resolved_config(restored) == source


@pytest.mark.parametrize("bundle", BUNDLES)
@pytest.mark.parametrize("policy", POLICIES)
def test_old_v3_backup_requires_explicit_legacy_cli_override(
    bundle: str, policy: str, tmp_path: Path
) -> None:
    from bdp_benchmark.config import parse_args, resolved_config

    saved = resolved_config(parse_args(["--config", str(_job_path(bundle, policy))]))
    backup = tmp_path / "archived_v3_config.yaml"
    backup.write_text(yaml.safe_dump(saved), encoding="utf-8")
    original_bytes = backup.read_bytes()

    unchanged = parse_args(["--config", str(backup)])
    assert unchanged.trajectory_execution_mode == "frenet_pid_v3"
    assert resolved_config(unchanged) == saved

    legacy = parse_args([
        "--config", str(backup),
        "--trajectory_execution_mode", "frenet_pid_v3_legacy",
    ])
    saved["environment"]["trajectory_execution_mode"] = "frenet_pid_v3_legacy"
    assert legacy.trajectory_execution_mode == "frenet_pid_v3_legacy"
    assert legacy.run_name == unchanged.run_name
    assert resolved_config(legacy) == saved
    assert backup.read_bytes() == original_bytes
