from __future__ import annotations

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.tracking import TrackerConfig
from bdp_benchmark.highway.env import HighwayBenchmarkEnv


def test_native_highway_wrapper_matches_upstream_seeded_transitions() -> None:
    config = {"vehicles_count": 8, "duration": 5, "initial_lane_id": 1}
    upstream = gym.make("highway-fast-v0", config=config)
    wrapped = HighwayBenchmarkEnv(
        env_id="highway-fast-v0",
        execution_mode="native_controller",
        generation_config=CandidateGenerationConfig(horizon_s=1.0, sample_count=7),
        tracker_config=TrackerConfig(),
        env_config=config,
    )
    try:
        upstream_obs, _ = upstream.reset(seed=123)
        wrapped_obs, _ = wrapped.reset(seed=123)
        np.testing.assert_allclose(wrapped_obs, upstream_obs)

        for action in [1, 3, 0, 1, 4]:
            wrapped.build_candidate_set()
            expected = upstream.step(action)
            actual = wrapped.step(action)
            np.testing.assert_allclose(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1])
            assert actual[2:4] == expected[2:4]
    finally:
        upstream.close()
        wrapped.close()
