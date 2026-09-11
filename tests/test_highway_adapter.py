from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.tracking import TrackerConfig
from bdp_benchmark.highway.adapter import HighwayAdapter
from bdp_benchmark.highway.env import HighwayBenchmarkEnv


def _make_env(*, execution_mode: str = "native_controller", initial_lane_id: int = 1) -> HighwayBenchmarkEnv:
    return HighwayBenchmarkEnv(
        env_id="highway-fast-v0",
        execution_mode=execution_mode,
        generation_config=CandidateGenerationConfig(horizon_s=1.0, sample_count=7),
        tracker_config=TrackerConfig(),
        env_config={"vehicles_count": 0, "initial_lane_id": initial_lane_id, "duration": 5},
    )


def test_highway_native_targets_preserve_five_semantic_action_order() -> None:
    env = _make_env(initial_lane_id=1)
    try:
        env.reset(seed=7)
        target_d, target_speed = env.adapter.action_targets("native_controller")

        assert env.action_space.n == 5
        assert target_d[0] < target_d[1] < target_d[2]
        np.testing.assert_allclose(target_d[1], target_d[3:])
        assert target_speed[4] < target_speed[1] < target_speed[3]
        np.testing.assert_allclose(target_speed[:3], target_speed[1])
    finally:
        env.close()


def test_highway_unavailable_lane_change_is_idle_equivalent() -> None:
    env = _make_env(initial_lane_id=0)
    try:
        env.reset(seed=8)
        target_d, target_speed = env.adapter.action_targets("native_controller")
        np.testing.assert_allclose(target_d[0], target_d[1])
        np.testing.assert_allclose(target_speed[0], target_speed[1])
    finally:
        env.close()


def test_highway_candidate_features_are_state_dependent_and_do_not_mutate_vehicle() -> None:
    env = _make_env()
    try:
        env.reset(seed=9)
        vehicle = env.unwrapped.vehicle
        target_lane_before = vehicle.target_lane_index
        target_speed_before = vehicle.target_speed

        candidate_set = env.build_candidate_set()

        assert candidate_set.labels.tolist() == [0, 1, 2, 3, 4]
        assert candidate_set.features.shape == (5, 35)
        assert candidate_set.trajectories.shape == (5, 7, 4)
        assert np.all(np.isfinite(candidate_set.features))
        assert vehicle.target_lane_index == target_lane_before
        assert vehicle.target_speed == target_speed_before
    finally:
        env.close()


def test_native_descriptor_starts_in_actual_chassis_heading_not_velocity_heading() -> None:
    lane = SimpleNamespace(
        local_coordinates=lambda _position: (0.0, 0.0),
        heading_at=lambda _longitudinal: 0.0,
    )
    raw = SimpleNamespace(
        vehicle=SimpleNamespace(
            lane_index=("a", "b", 0),
            position=np.asarray([0.0, 0.0]),
            heading=0.8,
            speed=8.0,
        ),
        road=SimpleNamespace(network=SimpleNamespace(get_lane=lambda _index: lane)),
    )
    adapter = HighwayAdapter(SimpleNamespace(unwrapped=raw), CandidateGenerationConfig())

    _lane_index, _lane, _longitudinal, state, _position = adapter._base_lane_state()

    assert state.s_dot == pytest.approx(8.0 * np.cos(0.8))
    assert state.d_dot == pytest.approx(8.0 * np.sin(0.8))


@pytest.mark.parametrize("execution_mode", ["native_controller", "frenet_pid", "frenet_pid_v2"])
def test_highway_steps_report_environment_variant(execution_mode: str) -> None:
    env = _make_env(execution_mode=execution_mode)
    try:
        if execution_mode == "frenet_pid":
            assert env._nominal_state is None
        obs, _ = env.reset(seed=10)
        next_obs, reward, terminated, truncated, info = env.step(1)

        assert next_obs.shape == obs.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["environment_variant"] == "highway-fast-v0"
    finally:
        env.close()


def test_highway_v2_generation_uses_supplied_nominal_pose_and_heading() -> None:
    env = _make_env(execution_mode="frenet_pid_v2")
    try:
        env.reset(seed=11)
        actual = env.adapter.ego_state()
        nominal = EgoState(
            x=actual.x + 2.0,
            y=actual.y,
            heading=0.65,
            speed=actual.speed,
        )
        candidates = env.adapter.build_candidate_set("frenet_pid_v2", nominal)

        np.testing.assert_allclose(candidates.trajectories[0, 0, :2], [nominal.x, nominal.y], atol=1e-5)
        assert candidates.trajectories[0, 0, 2] == pytest.approx(nominal.heading)
        assert not np.allclose(candidates.features, env.adapter.build_candidate_set("frenet_pid", None).features)
        assert env.adapter.ego_state() == actual
    finally:
        env.close()


def test_highway_v2_commits_selected_path_then_projects_actual_position_on_next_plan() -> None:
    env = _make_env(execution_mode="frenet_pid_v2")
    try:
        env.reset(seed=12)
        first = env.build_candidate_set()
        assert env._nominal_state is not None
        assert env._nominal_state.history_valid is False
        env.step(1)
        assert env._nominal_state.history_valid is True

        second = env.build_candidate_set()
        projection = env._nominal_state.last_projection
        assert projection is not None
        assert projection.history_valid is True
        assert second.trajectories.shape == first.trajectories.shape
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["merge-v1", "roundabout-v1"])
def test_highway_adapter_is_shared_by_other_five_action_environments(env_id: str) -> None:
    env = HighwayBenchmarkEnv(
        env_id=env_id,
        execution_mode="native_controller",
        generation_config=CandidateGenerationConfig(horizon_s=0.5, sample_count=5),
        tracker_config=TrackerConfig(),
    )
    try:
        obs, _ = env.reset(seed=4)
        candidates = env.build_candidate_set()
        next_obs, reward, terminated, truncated, _ = env.step(1)

        assert candidates.features.shape == (5, 25)
        assert next_obs.shape == obs.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_roundabout_reference_clamps_route_lane_id_at_single_lane_exit() -> None:
    env = HighwayBenchmarkEnv(
        env_id="roundabout-v1",
        execution_mode="native_controller",
        generation_config=CandidateGenerationConfig(horizon_s=2.0, sample_count=11),
        tracker_config=TrackerConfig(),
    )
    try:
        env.reset(seed=42)
        for _ in range(7):
            env.build_candidate_set()
            _obs, _reward, terminated, truncated, _info = env.step(1)
            assert not terminated
            assert not truncated

        candidates = env.build_candidate_set()
        assert candidates.features.shape == (5, 55)
        assert np.all(np.isfinite(candidates.features))
    finally:
        env.close()
