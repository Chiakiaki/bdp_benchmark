import importlib

import numpy as np
import pytest

from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.frenet import ReferencePath


def api():
    return importlib.import_module("bdp_benchmark.common.retiming")


def profile(v0=10.0, scales=(-1, -0.5, 0, 0.5, 1), **kwargs):
    options = dict(tau=1.0, rate=8.0, minimum_speed=0.0, maximum_speed=12.0,
                   horizon_s=2.0, sample_count=11)
    options.update(kwargs)
    return api().speed_command_profile(v0, np.array(scales), **options)


def test_accelerate_then_cruise_clipping_and_exact_integral():
    p = profile()
    np.testing.assert_allclose(p.targets, [2, 6, 10, 12, 12])
    np.testing.assert_allclose(p.speed[0], [10, 8.4, 6.8, 5.2, 3.6, 2, 2, 2, 2, 2, 2])
    np.testing.assert_allclose(p.distance[0, [0, 2, 5, 10]], [0, 3.36, 6, 8])
    np.testing.assert_allclose(p.acceleration[0, :5], -8)
    np.testing.assert_allclose(p.acceleration[:, 5:], 0)
    np.testing.assert_allclose(p.speed[-1, :6], [10, 10.4, 10.8, 11.2, 11.6, 12])
    assert p.distance[-1, -1] == pytest.approx(23)


def test_tau_need_not_align_with_sample_times_and_no_division_by_action_scale():
    p = profile(tau=0.7)
    assert p.targets[0] == pytest.approx(4.4)
    np.testing.assert_allclose(p.speed[0, :5], [10, 8.4, 6.8, 5.2, 4.4])
    assert p.distance[0, -1] == pytest.approx(10.76)
    assert p.targets[2] == 10
    np.testing.assert_allclose(p.acceleration[2], 0)
    np.testing.assert_allclose(profile(v0=0, scales=(0,)).distance, 0)


def test_stopping_freezes_longitudinal_distance_and_preserves_initial_overspeed():
    p = profile(v0=4)
    assert p.targets[0] == 0
    np.testing.assert_allclose(p.speed[0, 5:], 0)
    np.testing.assert_allclose(p.distance[0, 5:], 2)
    over = profile(v0=13, scales=(0,))
    assert over.speed[0, 0] == 13
    assert over.speed[0, -1] == 12
    assert np.isfinite(over.distance).all()


@pytest.mark.parametrize("options", [{"tau": 0}, {"tau": 3}, {"rate": float("nan")}, {"rate": -1}])
def test_profile_rejects_bad_parameters(options):
    with pytest.raises(ValueError):
        profile(**options)


def lane_paths(p, origin=EgoState(0, 0, 0, 4), curved=False):
    if curved:
        angles = np.linspace(0, 1.4, 300)
        xy = np.column_stack((20 * np.sin(angles), 20 * (1 - np.cos(angles))))
    else:
        xy = np.array([[0, 0], [100, 0]], dtype=float)
    reference = ReferencePath.from_xy(xy, lateral_normal_sign=-1)
    return api().retimed_lane_trajectories(
        reference, origin, initial_d=0, lane_targets=np.array([-3.5, 0, 3.5]),
        initial_reference_heading=0, shape_length=8, profile=p, geometry_sample_count=256,
    )


def test_retimed_lane_change_stops_laterally_and_derivatives_are_zero_after_stop():
    p = profile(v0=4)
    world, frenet = lane_paths(p)
    assert world.shape == (15, 11, 4)
    assert frenet.shape == (15, 11, 6)
    np.testing.assert_allclose(world[:3, 5:, :2], np.repeat(world[:3, 5:6, :2], 6, axis=1))
    np.testing.assert_allclose(frenet[:3, 5:, 2:], 0)
    np.testing.assert_allclose(world[:, :, 3], np.repeat(p.speed, 3, axis=0))
    assert abs(world[0, -1, 1]) < 3.5  # A stopped vehicle does not finish a lane change sideways.


def test_zero_speed_freezes_all_candidate_positions_and_tangents():
    p = profile(v0=0, scales=(0,))
    world, frenet = lane_paths(p, origin=EgoState(0, 0, 0.2, 0))
    np.testing.assert_allclose(world[..., :2], 0)
    np.testing.assert_allclose(world[..., 2], 0.2)
    np.testing.assert_allclose(frenet[..., 2:], 0)


def test_curve_uses_actual_path_distance_not_reference_s_as_speed():
    p = profile(v0=6, scales=(0,), sample_count=201)
    world, frenet = lane_paths(p, origin=EgoState(0, 0, 0, 6), curved=True)
    traveled = np.linalg.norm(np.diff(world[..., :2], axis=1), axis=-1).sum(axis=1)
    np.testing.assert_allclose(traveled, [12] * 3, atol=0.03)
    np.testing.assert_allclose(world[..., 3], 6)
    assert world[1, -1, 1] > 1
    assert not np.allclose(frenet[0, :, 2], 6)  # Reference speed differs during lane changing/curvature.


def test_circular_candidates_share_profile_and_freeze_on_stop():
    p = profile(v0=4)
    world = api().retimed_circular_trajectories(EgoState(0, 0, 0, 4), p, np.array([0.2, -0.2]))
    assert world.shape == (10, 11, 4)
    np.testing.assert_allclose(world[..., 3], np.repeat(p.speed, 2, axis=0))
    np.testing.assert_allclose(world[0, :, 1:3], -world[1, :, 1:3])
    np.testing.assert_allclose(world[:2, 5:, :2], np.repeat(world[:2, 5:6, :2], 6, axis=1))


@pytest.mark.parametrize("heading", [2.0, -2.0, np.pi])
def test_lane_shape_preserves_backward_facing_nominal_tangent(heading):
    p = profile(v0=6, scales=(0,), sample_count=201)
    world, _ = lane_paths(p, origin=EgoState(0, 0, heading, 6))
    first_motion = world[:, 1, :2] - world[:, 0, :2]
    motion_heading = np.arctan2(first_motion[:, 1], first_motion[:, 0])
    error = np.arctan2(np.sin(motion_heading - heading), np.cos(motion_heading - heading))
    np.testing.assert_allclose(error, 0, atol=0.005)
