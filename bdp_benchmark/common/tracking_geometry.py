"""Polyline projection and interpolation for feedback tracking and diagnostics."""

import numpy as np

from .features import wrap_angle


def project_trajectory(trajectory: np.ndarray, xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Nearest bounded polyline projection, interpolated pose, and cumulative distance."""
    delta = np.diff(trajectory[:, :2], axis=0)
    length_sq = np.einsum("ij,ij->i", delta, delta)
    fraction = np.clip(
        np.einsum("ij,ij->i", xy - trajectory[:-1, :2], delta) / np.maximum(length_sq, 1e-12), 0, 1
    )
    points = trajectory[:-1, :2] + fraction[:, None] * delta
    index = int(np.argmin(np.sum((points - xy) ** 2, axis=1)))
    arc = np.r_[0.0, np.cumsum(np.sqrt(length_sq))]
    pose = (1 - fraction[index]) * trajectory[index] + fraction[index] * trajectory[index + 1]
    pose[2] = trajectory[index, 2] + fraction[index] * wrap_angle(trajectory[index + 1, 2] - trajectory[index, 2])
    return float(arc[index] + fraction[index] * np.sqrt(length_sq[index])), pose, arc


def sample_trajectory(trajectory, arc, distance):
    query = float(np.clip(distance, 0.0, arc[-1]))
    result = np.array([np.interp(query, arc, trajectory[:, i]) for i in range(4)])
    result[2] = np.interp(query, arc, np.unwrap(trajectory[:, 2]))
    return result
