import os
from pathlib import Path
import subprocess

import yaml


def test_12mps_native_launcher_runs_only_requested_jobs_in_order(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bundle = Path("scripts/job_scripts/benchmark_metadrive_12mps")
    script = root / bundle / "run_metadrive_native_comparison.sh"
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    result = subprocess.run([str(script), "--n_envs", "2"], cwd=tmp_path,
                            env={**os.environ, "BDP_BENCHMARK_PYTHON": "/bin/echo"},
                            capture_output=True, text=True, check=True)
    commands = [line for line in result.stdout.splitlines() if line.startswith("scripts/train.py")]
    names = ["metadrive_native_bdp_frenet_benchmark.yaml",
             "metadrive_native_builtin_benchmark.yaml",
             "metadrive_native_continuous_builtin_benchmark.yaml"]
    assert len(commands) == len(names)
    for command, name in zip(commands, names):
        path = bundle / name
        assert command == f"scripts/train.py --config {path} --n_envs 2"
        config = yaml.safe_load((root / path).read_text())
        assert config["environment"]["trajectory_execution_mode"] == "native_controller"
        assert config["environment"]["environment_config"]["agent_configs"]["default_agent"]["max_speed_km_h"] == 43.2
