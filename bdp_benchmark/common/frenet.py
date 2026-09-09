"""Vectorized Frenet polynomial generation and reference-path conversion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrenetState:
    s: Any
    s_dot: Any
    s_ddot: Any
    d: Any
    d_dot: Any
    d_ddot: Any


@dataclass(frozen=True)
class ReferencePath:
    xy: np.ndarray
    cumulative_s: np.ndarray
    segment_heading: np.ndarray
    lateral_normal_sign: float = 1.0

    @classmethod
    def from_xy(cls, xy: np.ndarray, *, lateral_normal_sign: float = 1.0) -> "ReferencePath":
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise ValueError(f"Reference path must have shape [P, 2] with P >= 2, got {points.shape}")
        segment = np.diff(points, axis=0)
        length = np.linalg.norm(segment, axis=-1)
        keep = np.concatenate(([True], length > 1e-9))
        points = points[keep]
        if points.shape[0] < 2:
            raise ValueError("Reference path must contain at least two distinct points")
        segment = np.diff(points, axis=0)
        length = np.linalg.norm(segment, axis=-1)
        cumulative_s = np.concatenate(([0.0], np.cumsum(length)))
        heading = np.unwrap(np.arctan2(segment[:, 1], segment[:, 0]))
        if lateral_normal_sign not in (-1.0, 1.0):
            raise ValueError("lateral_normal_sign must be -1 or 1")
        return cls(
            xy=points,
            cumulative_s=cumulative_s,
            segment_heading=heading,
            lateral_normal_sign=float(lateral_normal_sign),
        )

    @property
    def length(self) -> float:
        return float(self.cumulative_s[-1])

    def sample_center(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        x = np.interp(query.reshape(-1), self.cumulative_s, self.xy[:, 0]).reshape(query.shape)
        y = np.interp(query.reshape(-1), self.cumulative_s, self.xy[:, 1]).reshape(query.shape)
        segment_idx = np.searchsorted(self.cumulative_s[1:], query, side="right")
        segment_idx = np.clip(segment_idx, 0, self.segment_heading.size - 1)
        heading = self.segment_heading[segment_idx]
        return x, y, heading


def _broadcast_initial(value: Any, target_shape: tuple[int, ...]) -> np.ndarray:
    base_shape = target_shape[:-1]
    value_array = np.asarray(value, dtype=np.float64)
    try:
        value_array = np.broadcast_to(value_array, base_shape)
    except ValueError as exc:
        raise ValueError(f"Initial Frenet state shape {value_array.shape} cannot broadcast to {base_shape}") from exc
    return np.broadcast_to(value_array[..., None], target_shape)


def _quintic_coefficients(
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    end_position: np.ndarray,
    duration: float,
) -> np.ndarray:
    t = float(duration)
    a0 = start_position
    a1 = start_velocity
    a2 = start_acceleration / 2.0
    rhs = np.stack(
        [
            end_position - (a0 + a1 * t + a2 * t**2),
            -(a1 + 2.0 * a2 * t),
            -(2.0 * a2),
        ],
        axis=-1,
    )
    tail = np.einsum("ij,...j->...i", _quintic_inverse(t), rhs)
    return np.concatenate([np.stack([a0, a1, a2], axis=-1), tail], axis=-1)


def _quartic_coefficients(
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    end_velocity: np.ndarray,
    duration: float,
) -> np.ndarray:
    t = float(duration)
    a0 = start_position
    a1 = start_velocity
    a2 = start_acceleration / 2.0
    rhs = np.stack(
        [end_velocity - (a1 + 2.0 * a2 * t), -(2.0 * a2)],
        axis=-1,
    )
    tail = np.einsum("ij,...j->...i", _quartic_inverse(t), rhs)
    return np.concatenate([np.stack([a0, a1, a2], axis=-1), tail], axis=-1)


@lru_cache(maxsize=32)
def _quintic_inverse(duration: float) -> np.ndarray:
    t = float(duration)
    matrix = np.asarray(
        [[t**3, t**4, t**5], [3.0 * t**2, 4.0 * t**3, 5.0 * t**4], [6.0 * t, 12.0 * t**2, 20.0 * t**3]],
        dtype=np.float64,
    )
    return np.linalg.inv(matrix)


@lru_cache(maxsize=32)
def _quartic_inverse(duration: float) -> np.ndarray:
    t = float(duration)
    matrix = np.asarray([[3.0 * t**2, 4.0 * t**3], [6.0 * t, 12.0 * t**2]], dtype=np.float64)
    return np.linalg.inv(matrix)


def _evaluate(coefficients: np.ndarray, times: np.ndarray, derivative: int = 0) -> np.ndarray:
    coefficient_count = coefficients.shape[-1]
    powers = np.arange(coefficient_count, dtype=np.int64)
    factors = np.ones(coefficient_count, dtype=np.float64)
    for _ in range(derivative):
        factors *= powers
        powers = np.maximum(powers - 1, 0)
    active = np.arange(coefficient_count) >= derivative
    time_powers = np.power(times[:, None], powers[None, :], where=active[None, :])
    time_powers[:, ~active] = 0.0
    basis = time_powers * factors[None, :]
    return np.einsum("...c,hc->...h", coefficients, basis)


def generate_frenet_trajectories(
    initial: FrenetState,
    target_d: np.ndarray,
    target_speed: np.ndarray,
    *,
    horizon_s: float,
    sample_count: int,
) -> np.ndarray:
    """Generate trajectories as ``[..., K, H, (s,d,s_dot,d_dot,s_ddot,d_ddot)]``."""
    if horizon_s <= 0.0:
        raise ValueError("horizon_s must be positive")
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    target_d, target_speed = np.broadcast_arrays(
        np.asarray(target_d, dtype=np.float64),
        np.asarray(target_speed, dtype=np.float64),
    )
    if target_d.ndim < 1:
        raise ValueError("Targets must include a candidate axis")
    target_shape = target_d.shape
    s0 = _broadcast_initial(initial.s, target_shape)
    s_dot0 = _broadcast_initial(initial.s_dot, target_shape)
    s_ddot0 = _broadcast_initial(initial.s_ddot, target_shape)
    d0 = _broadcast_initial(initial.d, target_shape)
    d_dot0 = _broadcast_initial(initial.d_dot, target_shape)
    d_ddot0 = _broadcast_initial(initial.d_ddot, target_shape)

    lateral = _quintic_coefficients(d0, d_dot0, d_ddot0, target_d, horizon_s)
    longitudinal = _quartic_coefficients(s0, s_dot0, s_ddot0, target_speed, horizon_s)
    times = np.linspace(0.0, float(horizon_s), int(sample_count), dtype=np.float64)
    s = _evaluate(longitudinal, times)
    d = _evaluate(lateral, times)
    s_dot = _evaluate(longitudinal, times, derivative=1)
    d_dot = _evaluate(lateral, times, derivative=1)
    s_ddot = _evaluate(longitudinal, times, derivative=2)
    d_ddot = _evaluate(lateral, times, derivative=2)
    return np.stack([s, d, s_dot, d_dot, s_ddot, d_ddot], axis=-1)


def frenet_to_world(
    reference: ReferencePath,
    s: np.ndarray,
    d: np.ndarray,
    *,
    speed: np.ndarray,
    longitudinal_speed: np.ndarray | None = None,
    lateral_speed: np.ndarray | None = None,
    initial_heading_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Map Frenet coordinates using the reference path's positive lateral normal."""
    s, d, speed = np.broadcast_arrays(
        np.asarray(s, dtype=np.float64),
        np.asarray(d, dtype=np.float64),
        np.asarray(speed, dtype=np.float64),
    )
    center_x, center_y, heading = reference.sample_center(s)
    signed_d = reference.lateral_normal_sign * d
    x = center_x - np.sin(heading) * signed_d
    y = center_y + np.cos(heading) * signed_d
    if longitudinal_speed is not None or lateral_speed is not None:
        if longitudinal_speed is None or lateral_speed is None:
            raise ValueError("longitudinal_speed and lateral_speed must be provided together")
        longitudinal_speed, lateral_speed = np.broadcast_arrays(
            np.asarray(longitudinal_speed, dtype=np.float64),
            np.asarray(lateral_speed, dtype=np.float64),
        )
        heading = heading + np.arctan2(reference.lateral_normal_sign * lateral_speed, longitudinal_speed)
    if initial_heading_rad is not None:
        heading = np.array(heading, copy=True)
        heading[..., 0] = np.broadcast_to(
            np.asarray(initial_heading_rad, dtype=np.float64),
            heading[..., 0].shape,
        )
    return np.stack([x, y, heading, speed], axis=-1)
