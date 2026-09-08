"""Simulator-independent trajectory tracking controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import EgoState, PhysicalControl
from .features import wrap_angle


@dataclass(frozen=True)
class TrackerConfig:
    lookahead_points: int = 2
    heading_kp: float = 1.2
    heading_ki: float = 0.0
    heading_kd: float = 0.08
    cross_track_kp: float = 0.35
    speed_kp: float = 1.0
    speed_ki: float = 0.05
    speed_kd: float = 0.0
    max_steering_rad: float = 0.7
    max_accel_mps2: float = 5.0
    max_decel_mps2: float = 8.0

    def __post_init__(self) -> None:
        if self.lookahead_points < 0:
            raise ValueError("lookahead_points must be non-negative")
        if self.max_steering_rad <= 0.0 or self.max_accel_mps2 <= 0.0 or self.max_decel_mps2 <= 0.0:
            raise ValueError("tracker limits must be positive")


class _PID:
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral = 0.0
        self.previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def step(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError("tracker dt must be positive")
        self.integral += float(error) * dt
        derivative = 0.0 if self.previous_error is None else (float(error) - self.previous_error) / dt
        self.previous_error = float(error)
        return self.kp * float(error) + self.ki * self.integral + self.kd * derivative


class TrajectoryPIDTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.reference: np.ndarray | None = None
        self.last_target_index: int | None = None
        self.last_target_point: np.ndarray | None = None
        self._steering_pid = _PID(config.heading_kp, config.heading_ki, config.heading_kd)
        self._speed_pid = _PID(config.speed_kp, config.speed_ki, config.speed_kd)

    def reset(self) -> None:
        self.reference = None
        self.last_target_index = None
        self.last_target_point = None
        self._steering_pid.reset()
        self._speed_pid.reset()

    def set_reference(self, trajectory: np.ndarray) -> None:
        reference = np.asarray(trajectory, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 4 or reference.shape[0] < 2:
            raise ValueError(f"trajectory reference must have shape [H,4] with H >= 2, got {reference.shape}")
        self.reference = reference.copy()
        self.last_target_index = None
        self.last_target_point = None
        self._steering_pid.reset()
        self._speed_pid.reset()

    def _target(self, ego: EgoState) -> tuple[int, np.ndarray]:
        if self.reference is None:
            raise RuntimeError("set_reference() must be called before tracker.step()")
        delta = self.reference[:, :2] - ego.xy
        nearest = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        target_idx = min(nearest + self.config.lookahead_points, self.reference.shape[0] - 1)
        return target_idx, self.reference[target_idx]

    def preview_target(self, ego: EgoState) -> np.ndarray:
        """Return the current lookahead target without changing PID state."""
        _target_index, target = self._target(ego)
        return target[:2].copy()

    def step(self, ego: EgoState, *, dt: float) -> PhysicalControl:
        target_idx, target = self._target(ego)
        self.last_target_index = target_idx
        self.last_target_point = target[:2].copy()
        target_delta = target[:2] - ego.xy
        left_error = -np.sin(ego.heading) * target_delta[0] + np.cos(ego.heading) * target_delta[1]
        heading_error = float(wrap_angle(target[2] - ego.heading))
        cross_track_angle = np.arctan2(self.config.cross_track_kp * left_error, max(abs(ego.speed), 0.5))
        steering_error = heading_error + float(cross_track_angle)
        steering = np.clip(
            self._steering_pid.step(steering_error, dt),
            -self.config.max_steering_rad,
            self.config.max_steering_rad,
        )
        acceleration = np.clip(
            self._speed_pid.step(float(target[3] - ego.speed), dt),
            -self.config.max_decel_mps2,
            self.config.max_accel_mps2,
        )
        return PhysicalControl(float(steering), float(acceleration))
