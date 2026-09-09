"""Per-environment nominal planning state for trajectory stitching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import EgoState


@dataclass(frozen=True)
class NominalProjection:
    """Nominal pose obtained by projecting an actual position onto a prior path."""

    state: EgoState
    nearest_index: int
    projection_error_m: float
    history_valid: bool


class NominalTrajectoryState:
    """Store one selected trajectory and derive the next planning pose from it."""

    def __init__(self) -> None:
        self._selected_trajectory: np.ndarray | None = None
        self.last_projection: NominalProjection | None = None

    @property
    def history_valid(self) -> bool:
        return self._selected_trajectory is not None

    def reset(self) -> None:
        self._selected_trajectory = None
        self.last_projection = None

    def commit(self, trajectory: np.ndarray) -> None:
        value = np.asarray(trajectory, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] < 4 or value.shape[0] < 2:
            raise ValueError(f"selected trajectory must have shape [H,4+] with H >= 2, got {value.shape}")
        if not np.isfinite(value[:, :4]).all():
            raise ValueError("selected trajectory must contain finite x/y/heading/speed values")
        self._selected_trajectory = value[:, :4].copy()

    def planning_state(self, actual: EgoState) -> EgoState:
        """Return x_nominal; initial x_nominal is exactly x_actual."""
        if self._selected_trajectory is None:
            self.last_projection = NominalProjection(actual, 0, 0.0, False)
            return actual

        delta = self._selected_trajectory[:, :2] - actual.xy
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        nearest_index = int(np.argmin(distance_sq))
        point = self._selected_trajectory[nearest_index]
        projection = NominalProjection(
            state=EgoState(
                x=float(point[0]),
                y=float(point[1]),
                heading=float(point[2]),
                speed=float(point[3]),
            ),
            nearest_index=nearest_index,
            projection_error_m=float(np.sqrt(distance_sq[nearest_index])),
            history_valid=True,
        )
        self.last_projection = projection
        return projection.state
