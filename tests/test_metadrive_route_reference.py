from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.straight_lane import StraightLane

from bdp_benchmark.metadrive.route_reference import build_route_reference, resolve_route_lanes


def _straight_to_curve_route():
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
    disconnected = StraightLane([10.0, 3.0], [15.0, 3.0])
    disconnected.index = ("junction", "finish", 1)
    graph = {
        "start": {"junction": [straight]},
        "junction": {"finish": [disconnected, curve]},
    }
    navigation = SimpleNamespace(
        checkpoints=["start", "junction", "finish"],
        _target_checkpoints_index=[0, 1],
        map=SimpleNamespace(road_network=SimpleNamespace(graph=graph)),
    )
    return navigation, straight, curve


def test_resolve_route_lanes_selects_connected_successor_on_assigned_route() -> None:
    navigation, straight, curve = _straight_to_curve_route()

    lanes = resolve_route_lanes(navigation, straight)

    assert lanes == (straight, curve)


def test_resolve_route_lanes_reanchors_off_route_lane_to_assigned_route() -> None:
    navigation, straight, curve = _straight_to_curve_route()
    off_route = StraightLane([0.0, 2.0], [10.0, 2.0])
    off_route.index = ("other", "edge", 0)

    lanes = resolve_route_lanes(
        navigation,
        off_route,
        planning_xy=np.asarray([8.0, 0.2]),
    )

    assert lanes == (straight, curve)


def test_route_reference_follows_curve_after_straight_segment_ends() -> None:
    navigation, straight, curve = _straight_to_curve_route()
    lanes = resolve_route_lanes(navigation, straight)

    result = build_route_reference(
        lanes,
        planning_xy=np.asarray([8.0, 0.0]),
        required_length=8.0,
        point_count=64,
    )

    x, y, heading = result.reference.sample_center(np.asarray([4.0]))
    assert result.start_lane is straight
    assert x.item() < 14.0
    assert y.item() > 0.1
    assert heading.item() > 0.05
    assert np.linalg.norm(result.reference.xy[-1] - np.asarray(curve.position(6.0, 0.0))) < 0.1


def test_route_reference_follows_straight_after_curve_segment_ends() -> None:
    curve = CircularLane(
        center=(0.0, 0.0),
        radius=5.0,
        start_phase=-np.pi / 2.0,
        angle=np.pi / 2.0,
        clockwise=False,
    )
    curve.index = ("start", "junction", 0)
    straight = StraightLane([5.0, 0.0], [5.0, 10.0])
    straight.index = ("junction", "finish", 0)
    planning_xy = np.asarray(curve.position(curve.length - 2.0, 0.0), dtype=np.float64)

    result = build_route_reference(
        [curve, straight],
        planning_xy=planning_xy,
        required_length=6.0,
        point_count=64,
    )
    x, y, heading = result.reference.sample_center(np.asarray([4.0]))

    assert x.item() == pytest.approx(5.0, abs=0.05)
    assert y.item() > 1.5
    assert heading.item() == pytest.approx(np.pi / 2.0, abs=0.05)


def test_nominal_position_on_successor_starts_reference_on_successor() -> None:
    navigation, straight, curve = _straight_to_curve_route()
    lanes = resolve_route_lanes(navigation, straight)
    nominal_xy = np.asarray(curve.position(2.0, 0.0), dtype=np.float64)

    result = build_route_reference(
        lanes,
        planning_xy=nominal_xy,
        required_length=3.0,
        point_count=32,
    )

    assert result.start_lane is curve
    assert result.start_longitudinal == pytest.approx(2.0, abs=1.0e-5)
    np.testing.assert_allclose(result.reference.xy[0], nominal_xy, atol=1.0e-5)


def test_route_reference_clamps_at_final_route_endpoint() -> None:
    navigation, straight, curve = _straight_to_curve_route()
    lanes = resolve_route_lanes(navigation, straight)

    result = build_route_reference(
        lanes,
        planning_xy=np.asarray([8.0, 0.0]),
        required_length=100.0,
        point_count=64,
    )
    x, y, _heading = result.reference.sample_center(np.asarray([100.0]))

    np.testing.assert_allclose([x.item(), y.item()], np.asarray(curve.end), atol=1.0e-5)
    assert result.reference.length == pytest.approx(2.0 + curve.length, rel=2.0e-3)
