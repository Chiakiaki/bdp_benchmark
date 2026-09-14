from __future__ import annotations

import numpy as np
import pytest

from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.tracking import TrackerConfig, TrajectoryPIDTracker


def _trajectory(y: float = 0.0, speed: float = 8.0, heading: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 12.0, 7)
    return np.stack(
        [x, np.full_like(x, y), np.full_like(x, heading), np.full_like(x, speed)],
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


def test_tracker_preview_target_matches_step_target_without_mutating_pid_state() -> None:
    tracker = TrajectoryPIDTracker(TrackerConfig(lookahead_points=2))
    reference = _trajectory(y=1.0)
    ego = EgoState(x=0.0, y=0.0, heading=0.0, speed=5.0)
    tracker.set_reference(reference)

    preview = tracker.preview_target(ego)
    assert preview is not None
    assert tracker._steering_pid.previous_error is None
    tracker.step(ego, dt=0.1)
    np.testing.assert_allclose(tracker.last_target_point, preview)


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [
        ("combined", 0.2 + np.arctan2(0.35 * 2.0, 5.0)),
        ("lateral_only", np.arctan2(0.35 * 2.0, 5.0)),
        ("lateral_only_no_speed_normalization", 0.35 * 2.0),
        ("heading_only", 0.2),
    ],
)
def test_tracker_selects_configured_steering_error(mode: str, expected_error: float) -> None:
    tracker = TrajectoryPIDTracker(
        TrackerConfig(
            lookahead_points=2,
            steering_error_mode=mode,
            heading_kp=1.0,
            heading_ki=0.0,
            heading_kd=0.0,
            max_steering_rad=2.0,
        )
    )
    tracker.set_reference(_trajectory(y=2.0, heading=0.2))

    control = tracker.step(EgoState(x=0.0, y=0.0, heading=0.0, speed=5.0), dt=0.1)

    assert control.steering_rad == pytest.approx(expected_error)


def test_tracker_rejects_unknown_steering_error_mode() -> None:
    with pytest.raises(ValueError, match="steering_error_mode"):
        TrackerConfig(steering_error_mode="unknown")


def test_reference_change_can_preserve_integrals_but_always_resets_derivatives() -> None:
    tracker = TrajectoryPIDTracker(
        TrackerConfig(
            integral_reset_on_reference_change=False,
            heading_ki=0.5,
            heading_kd=0.2,
            speed_ki=0.5,
            speed_kd=0.2,
        )
    )
    ego = EgoState(x=0.0, y=0.0, heading=0.0, speed=5.0)
    tracker.set_reference(_trajectory(y=1.0, speed=8.0))
    tracker.step(ego, dt=0.1)
    steering_integral = tracker._steering_pid.integral
    speed_integral = tracker._speed_pid.integral
    assert tracker._steering_pid.previous_error is not None
    assert tracker._speed_pid.previous_error is not None

    tracker.set_reference(_trajectory(y=-1.0, speed=6.0))

    assert tracker._steering_pid.integral == steering_integral
    assert tracker._speed_pid.integral == speed_integral
    assert tracker._steering_pid.previous_error is None
    assert tracker._speed_pid.previous_error is None


def test_reference_change_resets_integrals_by_default() -> None:
    tracker = TrajectoryPIDTracker(TrackerConfig(heading_ki=0.5, speed_ki=0.5))
    ego = EgoState(x=0.0, y=0.0, heading=0.0, speed=5.0)
    tracker.set_reference(_trajectory(y=1.0, speed=8.0))
    tracker.step(ego, dt=0.1)
    assert tracker._steering_pid.integral != 0.0
    assert tracker._speed_pid.integral != 0.0

    tracker.set_reference(_trajectory(y=-1.0, speed=6.0))

    assert tracker._steering_pid.integral == 0.0
    assert tracker._speed_pid.integral == 0.0


def test_episode_reset_clears_preserved_integrals() -> None:
    tracker = TrajectoryPIDTracker(
        TrackerConfig(integral_reset_on_reference_change=False, heading_ki=0.5, speed_ki=0.5)
    )
    tracker.set_reference(_trajectory(y=1.0, speed=8.0))
    tracker.step(EgoState(x=0.0, y=0.0, heading=0.0, speed=5.0), dt=0.1)

    tracker.reset()

    assert tracker._steering_pid.integral == 0.0
    assert tracker._speed_pid.integral == 0.0
    assert tracker._steering_pid.previous_error is None
    assert tracker._speed_pid.previous_error is None
