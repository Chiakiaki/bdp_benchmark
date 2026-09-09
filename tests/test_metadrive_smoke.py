from __future__ import annotations

import numpy as np
import pytest

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import EgoState
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


def test_metadrive_v2_commits_selected_reference_and_uses_nominal_projection() -> None:
    env = MetaDriveBenchmarkEnv(
        execution_mode="frenet_pid_v2",
        generation_config=CandidateGenerationConfig(horizon_s=0.5, sample_count=5),
        tracker_config=TrackerConfig(),
        env_config=METADRIVE_CONFIG,
    )
    try:
        env.reset(seed=0)
        first = env.build_candidate_set()
        assert env._nominal_state is not None
        assert env._nominal_state.history_valid is False
        env.step(1)
        assert env._nominal_state.history_valid is True
        second = env.build_candidate_set()
        projection = env._nominal_state.last_projection
        assert projection is not None
        assert projection.history_valid is True
        assert second.features.shape == first.features.shape
    finally:
        env.close()


def test_metadrive_v2_candidate_generation_uses_nominal_heading_not_actual_heading() -> None:
    env = MetaDriveBenchmarkEnv(
        execution_mode="frenet_pid_v2",
        generation_config=CandidateGenerationConfig(horizon_s=0.5, sample_count=5),
        tracker_config=TrackerConfig(),
        env_config=METADRIVE_CONFIG,
    )
    try:
        env.reset(seed=0)
        actual = env.adapter.ego_state()
        nominal = EgoState(actual.x + 1.0, actual.y, 0.75, actual.speed)
        candidates = env.adapter.build_candidate_set("frenet_pid_v2", nominal)

        np.testing.assert_allclose(candidates.trajectories[0, 0, :2], [nominal.x, nominal.y], atol=1e-5)
        assert candidates.trajectories[0, 0, 2] == pytest.approx(nominal.heading)
        assert env.adapter.ego_state() == actual
    finally:
        env.close()
