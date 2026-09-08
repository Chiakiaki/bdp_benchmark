from __future__ import annotations

import numpy as np

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.tracking import TrackerConfig
from bdp_benchmark.highway.env import HighwayBenchmarkEnv


def test_visual_candidate_set_is_the_same_set_used_by_structured_candidate_generation() -> None:
    env = HighwayBenchmarkEnv(
        env_id="highway-fast-v0",
        execution_mode="native_controller",
        generation_config=CandidateGenerationConfig(horizon_s=0.5, sample_count=5),
        tracker_config=TrackerConfig(),
        env_config={"vehicles_count": 0, "duration": 2, "initial_lane_id": 1},
    )
    try:
        env.reset(seed=2)
        candidate_set = env.build_candidate_set()
        visual_set = env.get_visual_candidate_set()

        assert visual_set is candidate_set
        assert candidate_set.features.shape == (5, 25)
        assert candidate_set.trajectories.shape == (5, 5, 4)
        np.testing.assert_allclose(visual_set.features, candidate_set.features)
    finally:
        env.close()


def test_visual_capture_does_not_change_native_candidate_step() -> None:
    env = HighwayBenchmarkEnv(
        env_id="highway-fast-v0",
        execution_mode="native_controller",
        generation_config=CandidateGenerationConfig(horizon_s=0.5, sample_count=5),
        tracker_config=TrackerConfig(),
        env_config={"vehicles_count": 0, "duration": 2, "initial_lane_id": 1},
    )
    try:
        env.reset(seed=3)
        env.build_candidate_set()
        _obs, reward, terminated, truncated, _info = env.step(1)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()
