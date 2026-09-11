from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from bdp_benchmark.config import parse_args
from bdp_benchmark.runner import collect_evaluation_returns, prepare_run_directory


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


class _ConstantEvaluationModel:
    def predict(self, obs, deterministic: bool):
        del deterministic
        return np.zeros(obs.shape[0], dtype=np.int64), None


class _MixedEvaluationEnv:
    num_envs = 3

    def reset(self):
        return np.zeros((3, 1), dtype=np.float32)

    def step(self, actions):
        del actions
        return (
            np.zeros((3, 1), dtype=np.float32),
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([True, True, True]),
            [
                {"environment_variant": "highway-fast-v0"},
                {"environment_variant": "merge-v1"},
                {"environment_variant": "roundabout-v1"},
            ],
        )


def test_collect_evaluation_returns_tracks_each_variant_independently() -> None:
    args = SimpleNamespace(inference_mode="deterministic")

    returns = collect_evaluation_returns(
        _ConstantEvaluationModel(),
        _MixedEvaluationEnv(),
        args=args,
        variant_ids=["highway-fast-v0", "merge-v1", "roundabout-v1"],
        episodes_per_variant=2,
    )

    assert returns == {
        "highway-fast-v0": [1.0, 1.0],
        "merge-v1": [2.0, 2.0],
        "roundabout-v1": [3.0, 3.0],
    }
