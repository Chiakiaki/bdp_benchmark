from __future__ import annotations

import numpy as np
import pytest

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.tracking import TrackerConfig
from bdp_benchmark.metadrive.env import MetaDriveBenchmarkEnv


METADRIVE_CONFIG = {
    "use_render": False,
    "traffic_density": 0.0,
    "num_scenarios": 1,
    "map": "S",
    "horizon": 20,
    "log_level": 50,
}


@pytest.mark.parametrize(
    ("execution_mode", "expected_actions"),
    [("native_controller", 25), ("frenet_pid", 5)],
)
def test_metadrive_modes_build_candidates_and_step_headlessly(execution_mode: str, expected_actions: int) -> None:
    env = MetaDriveBenchmarkEnv(
        execution_mode=execution_mode,
        generation_config=CandidateGenerationConfig(horizon_s=0.5, sample_count=5),
        tracker_config=TrackerConfig(),
        env_config=METADRIVE_CONFIG,
    )
    try:
        # SB3 seeds are experiment seeds, while MetaDrive treats reset seeds as
        # scenario indexes. The wrapper maps arbitrary experiment seeds into
        # the configured scenario bank deterministically.
        obs, _ = env.reset(seed=41)
        candidates = env.build_candidate_set()
        next_obs, reward, terminated, truncated, info = env.step(expected_actions // 2)

        assert env.action_space.n == expected_actions
        assert candidates.features.shape == (expected_actions, 25)
        assert candidates.trajectories.shape == (expected_actions, 5, 4)
        if execution_mode == "native_controller":
            point_features = candidates.features.reshape(expected_actions, 5, 5)
            assert point_features[10, -1, 1] > point_features[14, -1, 1]
        assert next_obs.shape == obs.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()
