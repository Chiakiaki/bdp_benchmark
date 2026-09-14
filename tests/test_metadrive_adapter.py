from __future__ import annotations

from types import SimpleNamespace

from gymnasium import spaces
import numpy as np
import pytest
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane

from bdp_benchmark.metadrive.adapter import (
    MetaDriveAdapter,
    decode_discrete_action_grid,
    frenet_pid_target_grid,
    native_action_targets,
)
from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.tracking import TrackerConfig
from bdp_benchmark.metadrive.env import MetaDriveBenchmarkEnv


def _transition_adapter(reference_mode: str) -> MetaDriveAdapter:
    straight = StraightLane([0.0, 0.0], [10.0, 0.0])
    straight.index = ("start", "junction", 0)
    curve = CircularLane(
        center=(10.0, 5.0),
        radius=5.0,
        start_phase=-np.pi / 2.0,
        angle=np.pi / 2.0,
        clockwise=False,
    )
    curve.index = ("junction", "finish", 0)
    graph = {
        "start": {"junction": [straight]},
        "junction": {"finish": [curve]},
    }
    navigation = SimpleNamespace(
        current_lane=straight,
        checkpoints=["start", "junction", "finish"],
        _target_checkpoints_index=[0, 1],
        map=SimpleNamespace(road_network=SimpleNamespace(graph=graph)),
    )
    vehicle = SimpleNamespace(
        position=np.asarray([8.0, 0.0]),
        heading_theta=0.0,
        speed=8.0,
        navigation=navigation,
    )
    env = SimpleNamespace(
        agent=vehicle,
        action_space=spaces.Discrete(15),
        config={"discrete_steering_dim": 5, "discrete_throttle_dim": 5},
    )
    return MetaDriveAdapter(
        env,
        CandidateGenerationConfig(horizon_s=2.0, sample_count=11, speed_delta_mps=5.0),
        reference_mode=reference_mode,
    )


def test_metadrive_action_grid_matches_env_input_policy_order() -> None:
    steering, throttle = decode_discrete_action_grid(steering_dim=5, throttle_dim=5)

    assert steering.shape == (25,)
    assert throttle.shape == (25,)
    np.testing.assert_allclose(steering[:5], [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(throttle[[0, 5, 10, 15, 20]], [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_route_continuous_adapter_follows_successor_while_legacy_extrapolates() -> None:
    legacy = _transition_adapter("lane_segment").build_candidate_set("frenet_pid")
    route_continuous = _transition_adapter("route_continuous").build_candidate_set("frenet_pid")

    center_candidate = 7
    assert legacy.trajectories[center_candidate, -1, 1] == pytest.approx(0.0, abs=1.0e-5)
    assert route_continuous.trajectories[center_candidate, -1, 1] > 1.0
    assert route_continuous.trajectories.shape == legacy.trajectories.shape == (15, 11, 4)
    assert route_continuous.features.shape == legacy.features.shape


def test_metadrive_adapter_rejects_unknown_reference_mode() -> None:
    with pytest.raises(ValueError, match="reference_mode"):
        _transition_adapter("unknown")


def test_all_five_metadrive_steering_bins_have_distinct_continuous_lateral_targets() -> None:
    target_d, target_speed = native_action_targets(
        current_d=0.2,
        current_speed=8.0,
        steering_dim=5,
        throttle_dim=5,
        lateral_span_m=3.5,
        speed_span_mps=4.0,
        minimum_speed_mps=0.0,
        maximum_speed_mps=20.0,
    )

    center_throttle_targets = target_d[10:15]
    assert len(np.unique(center_throttle_targets)) == 5
    # MetaDrive steering is positive left, while its lane-local lateral axis is
    # positive right. The descriptor must account for that sign difference.
    np.testing.assert_allclose(center_throttle_targets, [3.7, 1.95, 0.2, -1.55, -3.3])
    np.testing.assert_allclose(target_speed[10:15], 8.0)


def test_frenet_pid_grid_combines_three_lane_centers_with_five_speed_offsets() -> None:
    target_d, target_speed = frenet_pid_target_grid(
        lane_centers=np.asarray([-3.5, 0.0, 3.5]),
        current_speed=8.0,
        speed_delta_mps=4.0,
        minimum_speed_mps=0.0,
        maximum_speed_mps=20.0,
    )

    assert target_d.shape == (15,)
    assert target_speed.shape == (15,)
    np.testing.assert_allclose(target_d.reshape(5, 3), np.tile([-3.5, 0.0, 3.5], (5, 1)))
    np.testing.assert_allclose(target_speed.reshape(5, 3)[:, 0], [4.0, 6.0, 8.0, 10.0, 12.0])
    np.testing.assert_allclose(
        target_speed.reshape(5, 3),
        np.repeat(np.asarray([[4.0], [6.0], [8.0], [10.0], [12.0]]), 3, axis=1),
    )


def test_frenet_pid_lane_centers_use_navigation_lanes_and_virtual_boundary_lane() -> None:
    env = MetaDriveBenchmarkEnv(
        execution_mode="frenet_pid",
        generation_config=CandidateGenerationConfig(horizon_s=0.5, sample_count=5),
        tracker_config=TrackerConfig(),
        env_config={
            "use_render": False,
            "traffic_density": 0.0,
            "num_scenarios": 1,
            "map": 3,
            "horizon": 20,
            "log_level": 50,
        },
    )
    try:
        env.reset(seed=0)
        navigation = env.env.agent.navigation
        assert navigation.current_ref_lanes.index(navigation.current_lane) == 0
        lane_width = navigation.current_lane.width_at(0.0)

        target_d, _target_speed = env.adapter.action_targets("frenet_pid")
        lateral_grid = target_d.reshape(5, 3)

        np.testing.assert_allclose(lateral_grid[:, 0], -lane_width, atol=1e-5)
        np.testing.assert_allclose(lateral_grid[:, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(lateral_grid[:, 2], lane_width, atol=1e-5)
    finally:
        env.close()


def test_frenet_pid_lane_centers_use_current_road_when_route_references_advance_first() -> None:
    env = MetaDriveBenchmarkEnv(
        execution_mode="frenet_pid_v2",
        generation_config=CandidateGenerationConfig(horizon_s=0.5, sample_count=5),
        tracker_config=TrackerConfig(),
        env_config={
            "use_render": False,
            "traffic_density": 0.0,
            "random_traffic": False,
            "num_scenarios": 1,
            "start_seed": 5000,
            "map": 3,
            "horizon": 20,
            "log_level": 50,
        },
    )
    try:
        env.reset(seed=5041)
        navigation = env.env.agent.navigation
        assert navigation.current_lane not in navigation.next_ref_lanes
        navigation.current_ref_lanes = navigation.next_ref_lanes

        target_d, _target_speed = env.adapter.action_targets("frenet_pid_v2")

        lane_width = navigation.current_lane.width_at(0.0)
        np.testing.assert_allclose(target_d.reshape(5, 3)[0], [-lane_width, 0.0, lane_width], atol=1e-5)
    finally:
        env.close()


def test_native_descriptor_starts_in_actual_chassis_heading_not_velocity_heading() -> None:
    env = MetaDriveBenchmarkEnv(
        execution_mode="native_controller",
        generation_config=CandidateGenerationConfig(horizon_s=2.0, sample_count=11),
        tracker_config=TrackerConfig(),
        env_config={
            "use_render": False,
            "traffic_density": 0.0,
            "num_scenarios": 1,
            "map": "S",
            "horizon": 20,
            "log_level": 50,
        },
    )
    try:
        env.reset(seed=0)
        vehicle = env.env.agent
        actual_heading = 0.8
        velocity_heading = 0.2
        vehicle.set_heading_theta(actual_heading)
        vehicle.set_velocity([8.0 * np.cos(velocity_heading), 8.0 * np.sin(velocity_heading)])

        candidates = env.build_candidate_set()
        # Action 22 is the zero-steering, positive-throttle grid row.
        delta = candidates.trajectories[22, 1, :2] - candidates.trajectories[22, 0, :2]
        path_heading = float(np.arctan2(delta[1], delta[0]))

        assert np.arctan2(np.sin(path_heading - actual_heading), np.cos(path_heading - actual_heading)) == pytest.approx(
            0.0, abs=0.08
        )
    finally:
        env.close()
