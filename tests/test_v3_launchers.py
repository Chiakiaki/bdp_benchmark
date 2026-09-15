import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("bundle", ["benchmark", "benchmark_metadrive_12mps"])
def test_v3_pair_launcher_runs_bdp_then_builtin_from_arbitrary_directory(bundle, tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/job_scripts" / bundle / "run_metadrive_frenet_pid_v3_comparison.sh"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    result = subprocess.run([str(script), "--n_envs", "2"], cwd=tmp_path,
                            env={**os.environ, "BDP_BENCHMARK_PYTHON": "/bin/echo"},
                            capture_output=True, text=True, check=True)
    commands = [line for line in result.stdout.splitlines() if line.startswith("scripts/train.py")]
    assert len(commands) == 2
    for command, policy in zip(commands, ("bdp_frenet", "builtin")):
        config = f"scripts/job_scripts/{bundle}/metadrive_frenet_pid_v3_{policy}_benchmark.yaml"
        assert (root / config).is_file()
        assert command == f"scripts/train.py --config {config} --n_envs 2"


def test_old_80kmh_v3_bundle_was_merged_without_replacing_main_contract():
    root = Path(__file__).resolve().parents[1] / "scripts/job_scripts"
    assert not (root / "benchmark_metadrive").exists()
    contract = (root / "benchmark/config_contract.md").read_text()
    assert "PPO Training Contract" in contract
    assert "frenet_pid_v3" in contract
