import importlib

import numpy as np
import pytest

from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.frenet import FrenetState, ReferencePath, generate_frenet_trajectories
from bdp_benchmark.common.retiming import speed_command_profile
from bdp_benchmark.common.tracking import TrackerConfig, TrajectoryPIDTracker


def motion(v0=10, initial=None, scales=(-1, -.5, 0, .5, 1)):
    profile = speed_command_profile(v0, np.array(scales), tau=1, rate=8, minimum_speed=0,
                                    maximum_speed=12, horizon_s=2, sample_count=11)
    initial = initial or FrenetState(0, v0, 0, 0, 0, 0)
    api = importlib.import_module("bdp_benchmark.common.independent_frenet")
    frenet = api.independent_frenet_motion(initial, np.array([-3.5, 0, 3.5]), profile, tau=1, horizon_s=2)
    return api, profile, frenet


def test_longitudinal_ramp_and_original_timed_lateral_quintic():
    _, p, result = motion()
    old = generate_frenet_trajectories(FrenetState(0, 10, 0, 0, 0, 0),
                                     np.tile([-3.5, 0, 3.5], 5), np.repeat(p.targets, 3),
                                     horizon_s=2, sample_count=11)
    assert result.shape == (15, 11, 6)
    np.testing.assert_allclose(result[..., 0], np.repeat(p.distance, 3, axis=0))
    np.testing.assert_allclose(result[..., 2], np.repeat(p.speed, 3, axis=0))
    np.testing.assert_allclose(result[..., 4], np.repeat(p.acceleration, 3, axis=0))
    np.testing.assert_array_equal(result[..., [1, 3, 5]], old[..., [1, 3, 5]])
    # Each speed reaches the lane center at H, but at different longitudinal distances.
    np.testing.assert_allclose(result[:, -1, 1], np.tile([-3.5, 0, 3.5], 5))


def test_non_aligned_initial_longitudinal_projection_is_retained():
    initial = FrenetState(0, 8, 0, 0, 6, 0)
    _, p, result = motion(initial=initial)
    np.testing.assert_allclose(result[:, 0, 2], 8)
    np.testing.assert_allclose(result[:, 0, 3], 6)
    np.testing.assert_allclose(result[:, -1, 2], np.repeat(p.targets, 3))
    expected_acceleration = np.repeat((p.targets - 8)[:, None], 3, axis=0)
    np.testing.assert_allclose(result[..., 4][:, :5], np.broadcast_to(expected_acceleration, (15, 5)))


def test_geometry_can_move_laterally_while_desired_speed_and_control_stay_zero():
    api, p, frenet = motion(v0=0, scales=(0,))
    world = api.independent_frenet_to_world(
        ReferencePath.from_xy(np.array([[0, 0], [40, 0]]), lateral_normal_sign=-1),
        EgoState(0, 0, 0, 0), frenet, p, initial_reference_heading=0,
    )
    assert world.shape == (3, 11, 4)
    assert world[0, -1, 1] == pytest.approx(3.5)
    assert np.max(np.abs(frenet[..., 3])) > 3
    np.testing.assert_allclose(world[..., 3], 0)  # Desired speed, never hypot(s_dot,d_dot).
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode="adaptive_pursuit", speed_kp=2, speed_ki=0))
    tracker.set_reference(world[0])
    result = tracker.step(EgoState(0, 0, 0, 0), dt=.1)
    assert result.acceleration_mps2 == 0


def test_linear_speed_preview_requests_6p8_not_terminal_2():
    api, p, frenet = motion()
    world = api.independent_frenet_to_world(
        ReferencePath.from_xy(np.array([[0, 0], [40, 0]]), lateral_normal_sign=-1),
        EgoState(0, 0, 0, 10), frenet, p, initial_reference_heading=0,
    )
    tracker = TrajectoryPIDTracker(TrackerConfig(controller_mode="adaptive_pursuit", speed_kp=2, speed_ki=0))
    tracker.set_reference(world[1])
    control = tracker.step(EgoState(0, 0, 0, 10), dt=.1)
    assert tracker.last_diagnostics["target_speed_mps"] == pytest.approx(6.8)
    assert control.acceleration_mps2 == pytest.approx(-6.4)


def test_curved_reference_and_backward_nominal_heading_are_finite():
    api, p, frenet = motion(initial=FrenetState(0, -4, 0, 0, 3, 0), v0=5)
    heading = np.arctan2(3, -4)
    reference = ReferencePath.from_xy(np.array([[0, 0], [10, 0], [20, 5], [25, 15]]))
    world = api.independent_frenet_to_world(reference, EgoState(0, 0, heading, 5), frenet, p,
                                             initial_reference_heading=0)
    assert np.isfinite(world).all()
    assert world[1, 1, 0] < 0
    np.testing.assert_allclose(world[:, 0, 2], heading)
