from __future__ import annotations

import numpy as np

from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.nominal import NominalTrajectoryState


def _trajectory() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.1, 2.0],
            [1.0, 0.0, 0.2, 3.0],
            [2.0, 0.0, 0.3, 4.0],
        ],
        dtype=np.float64,
    )


def test_nominal_state_initializes_from_actual_pose_and_heading() -> None:
    actual = EgoState(x=4.0, y=5.0, heading=1.2, speed=0.7)
    state = NominalTrajectoryState()

    nominal = state.planning_state(actual)

    assert nominal == actual
    assert state.history_valid is False


def test_nominal_state_uses_nearest_previous_trajectory_pose_not_actual_heading() -> None:
    state = NominalTrajectoryState()
    state.planning_state(EgoState(x=0.0, y=0.0, heading=0.0, speed=1.0))
    state.commit(_trajectory())
    actual = EgoState(x=1.08, y=0.04, heading=-2.0, speed=99.0)

    nominal = state.planning_state(actual)

    assert nominal.x == 1.0
    assert nominal.y == 0.0
    assert nominal.heading == 0.2
    assert nominal.speed == 3.0
    assert state.history_valid is True
    assert actual.heading == -2.0
    assert actual.speed == 99.0


def test_nominal_state_reset_discards_previous_trajectory() -> None:
    state = NominalTrajectoryState()
    state.commit(_trajectory())
    actual = EgoState(x=7.0, y=8.0, heading=0.9, speed=2.0)

    state.reset()
    nominal = state.planning_state(actual)

    assert nominal == actual
    assert state.history_valid is False
