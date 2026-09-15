import numpy as np
import pytest
from dataclasses import replace
from gymnasium import spaces

from bdp_benchmark.common import retiming
from bdp_benchmark.common.contracts import EgoState
from test_metadrive_adapter import _transition_adapter


def profile(tau=1.0, speed=8.0):
    return retiming.speed_command_profile(speed, np.array([-1, 0, 1]), tau=tau, rate=8,
                                          minimum_speed=0, maximum_speed=12, horizon_s=2, sample_count=11)


@pytest.mark.parametrize("tau", [1.0, 0.7])
def test_outer_paths_turn_until_tau_then_follow_exact_tangent(tau):
    p = profile(tau)
    origin = EgoState(3, 4, .3, 8)
    curvatures = np.array([.2, -.2])
    world = retiming.turn_then_straight_trajectories(origin, p, curvatures, turn_duration_s=tau)
    times = np.linspace(0, 2, 11)
    distance_at_turn_end = .5 * (8 + p.targets) * tau
    end_heading = origin.heading + distance_at_turn_end[:, None] * curvatures
    assert world.shape == (6, 11, 4)
    np.testing.assert_allclose(world[..., 3], np.repeat(p.speed, 2, axis=0))
    tail = times >= tau
    np.testing.assert_allclose(world[:, tail, 2], np.repeat(end_heading.reshape(-1, 1), tail.sum(), axis=1))
    full_circle = retiming.retimed_circular_trajectories(origin, p, curvatures)
    np.testing.assert_allclose(world[:, times <= tau], full_circle[:, times <= tau])
    expected_delta = np.repeat((p.distance[:, -1] - p.distance[:, -2])[:, None], 2, axis=0)
    unit_tangent = np.column_stack((np.cos(end_heading.ravel()), np.sin(end_heading.ravel())))
    np.testing.assert_allclose(world[:, -1, :2] - world[:, -2, :2], expected_delta * unit_tangent, atol=1e-12)
    # Exact boundary geometry even when tau falls between trajectory sample times.
    distance = distance_at_turn_end[:, None]
    angle = distance * curvatures
    chord = distance * np.sinc(angle / (2 * np.pi))
    end_xy = origin.xy + np.stack((chord * np.cos(origin.heading + angle/2),
                                   chord * np.sin(origin.heading + angle/2)), axis=-1)
    remaining = np.repeat(p.distance[:, -1] - distance_at_turn_end, 2)
    np.testing.assert_allclose(world[:, -1, :2], end_xy.reshape(-1, 2) + remaining[:, None] * unit_tangent)


def test_zero_curvature_stopped_and_full_horizon_turn_remain_valid():
    p = profile(tau=2, speed=0)
    origin = EgoState(0, 0, .5, 0)
    for curvature in ([0], [.2, -.2]):
        got = retiming.turn_then_straight_trajectories(origin, p, np.array(curvature), turn_duration_s=2)
        old = retiming.retimed_circular_trajectories(origin, p, np.array(curvature))
        np.testing.assert_allclose(got, old)
        assert np.isfinite(got).all()


def test_only_active_v3_wires_the_straight_tail():
    adapter = _transition_adapter("route_continuous")
    adapter.config = replace(adapter.config, include_curvature_candidates=True)
    adapter.vehicle.FRONT_WHEELBASE = 1.0
    adapter.vehicle.REAR_WHEELBASE = 1.5
    adapter.vehicle.max_steering = 40.0
    adapter.env.action_space = spaces.Discrete(25)
    current = adapter.build_candidate_set("frenet_pid_v3")
    legacy = adapter.build_candidate_set("frenet_pid_v3_legacy")
    assert current.features.shape == (25, 55)
    np.testing.assert_allclose(current.trajectories[15:, 5:, 2],
                               np.repeat(current.trajectories[15:, 5:6, 2], 6, axis=1))
    assert not np.allclose(current.trajectories[23, 5:, 2], legacy.trajectories[23, 5:, 2])
    np.testing.assert_allclose(current.trajectories[15:, :, 3], legacy.trajectories[15:, :, 3])
