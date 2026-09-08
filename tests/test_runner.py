from __future__ import annotations

from pathlib import Path

import yaml

from bdp_benchmark.config import parse_args
from bdp_benchmark.runner import prepare_run_directory


def test_prepare_run_directory_backs_up_source_and_effective_config(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(
        """
environment:
  simulator: highway
  env_id: highway-fast-v0
  trajectory_execution_mode: native_controller
algorithm:
  num_timesteps: 100
model_architecture:
  policy_mode: builtin
""",
        encoding="utf-8",
    )
    args = parse_args(["--config", str(source), "--num_timesteps", "120", "--log_path", str(tmp_path / "logs")])

    log_dir, model_dir = prepare_run_directory(args, timestamp="20260908_120000")

    assert model_dir == log_dir / "models"
    assert (log_dir / "config.yaml").read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    effective = yaml.safe_load((log_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert effective["algorithm"]["num_timesteps"] == 120
