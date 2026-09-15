import json
from pathlib import Path
import subprocess
import sys


def test_tracking_diagnostic_accepts_v3_job(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/diagnose_tracking.py"),
               "--config", str(root / "scripts/job_scripts/benchmark/metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml"),
               "--tracking_output", str(tmp_path), "--tracking_center_action", "--tracking_steps", "2",
               "--tracking_seeds", "5000", "--device", "cpu"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["steps"] == 2
    assert summary["config"]["environment"]["trajectory_execution_mode"] == "frenet_pid_v3"
