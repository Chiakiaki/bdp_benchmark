from __future__ import annotations

import numpy as np

from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.tracking import TrackerConfig, TrajectoryPIDTracker


def _trajectory(y: float = 0.0, speed: float = 8.0) -> np.ndarray:
    x = np.linspace(0.0, 12.0, 7)
    return np.stack(
        [x, np.full_like(x, y), np.zeros_like(x), np.full_like(x, speed)],
        axis=-1,
    )


def test_tracker_accelerates_toward_speed_and_keeps_straight_reference() -> None:
    tracker = TrajectoryPIDTracker(TrackerConfig(lookahead_points=2))
    tracker.set_reference(_trajectory(speed=8.0))

    control = tracker.step(EgoState(x=0.0, y=0.0, heading=0.0, speed=4.0), dt=0.1)

    assert abs(control.steering_rad) < 1e-8
    assert control.acceleration_mps2 > 0.0


def test_tracker_steers_toward_left_and_right_references() -> None:
    tracker = TrajectoryPIDTracker(TrackerConfig(lookahead_points=2))
    ego = EgoState(x=0.0, y=0.0, heading=0.0, speed=5.0)

    tracker.set_reference(_trajectory(y=2.0))
    left = tracker.step(ego, dt=0.1)
    tracker.set_reference(_trajectory(y=-2.0))
    right = tracker.step(ego, dt=0.1)

    assert left.steering_rad > 0.0
    assert right.steering_rad < 0.0


def test_tracker_retains_reference_until_replaced_and_resets_pid_state() -> None:
    tracker = TrajectoryPIDTracker(TrackerConfig(heading_ki=0.5))
    reference = _trajectory(y=1.0)
    tracker.set_reference(reference)
    first = tracker.step(EgoState(x=0.0, y=0.0, heading=0.0, speed=5.0), dt=0.1)
    second = tracker.step(EgoState(x=0.0, y=0.0, heading=0.0, speed=5.0), dt=0.1)
    assert tracker.reference is not None
    assert second.steering_rad > first.steering_rad

    tracker.reset()
    assert tracker.reference is None
