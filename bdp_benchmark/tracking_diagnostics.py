"""Tracking measurements, independent of policy observations and rewards."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common.features import wrap_angle
from .common.tracking_geometry import project_trajectory


def tracking_errors(trajectory, actual) -> dict[str, float]:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    _, projected, _ = project_trajectory(trajectory, actual.xy)
    nominal = trajectory[np.argmin(np.sum((trajectory[:, :2] - actual.xy) ** 2, axis=1))]
    return {
        "nominal_error_m": float(np.linalg.norm(nominal[:2] - actual.xy)),
        "path_error_m": float(np.linalg.norm(projected[:2] - actual.xy)),
        "heading_error_rad": abs(float(wrap_angle(projected[2] - actual.heading))),
        "speed_error_mps": abs(float(projected[3] - actual.speed)),
        "actual_speed_mps": float(actual.speed),
        "reference_speed_mps": float(projected[3]),
    }


class TrackingRecorder:
    """Write bounded-memory per-step CSV diagnostics, including terminal samples."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", newline="", encoding="utf-8")
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.stream.close()
