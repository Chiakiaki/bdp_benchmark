"""Simulator-independent trajectory tracking controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import EgoState, PhysicalControl
from .features import wrap_angle
from .tracking_geometry import project_trajectory, sample_trajectory


STEERING_ERROR_MODES = (
    "combined",
    "lateral_only",
    "lateral_only_no_speed_normalization",
    "heading_only",
)


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
    steering_error_mode: str = "combined"
    integral_reset_on_reference_change: bool = True
    controller_mode: str = "legacy"
    lookahead_time_s: float = 0.5
    min_lookahead_m: float = 2.0
    max_lookahead_m: float = 12.0
    pursuit_gain: float = 1.0
    speed_preview_s: float = 0.4

    def __post_init__(self) -> None:
        if self.controller_mode not in ("legacy", "adaptive_pursuit"):
            raise ValueError("controller_mode must be legacy or adaptive_pursuit")
        values = [self.lookahead_time_s, self.min_lookahead_m, self.max_lookahead_m,
                  self.pursuit_gain, self.speed_preview_s]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("pursuit parameters must be finite and non-negative")
        if self.min_lookahead_m <= 0 or self.max_lookahead_m < self.min_lookahead_m or self.pursuit_gain <= 0:
            raise ValueError("pursuit lookahead bounds and gain must be positive and ordered")
        if self.controller_mode == "adaptive_pursuit" and self.speed_preview_s <= 0:
            raise ValueError("adaptive pursuit speed_preview_s must be positive to accelerate from rest")
        if self.lookahead_points < 0:
            raise ValueError("lookahead_points must be non-negative")
        if self.steering_error_mode not in STEERING_ERROR_MODES:
            raise ValueError(
                f"steering_error_mode must be one of {STEERING_ERROR_MODES}, got {self.steering_error_mode!r}"
            )
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
        self.reset_derivative()

    def reset_derivative(self) -> None:
        self.previous_error = None

    def step(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError("tracker dt must be positive")
        self.integral += float(error) * dt
        derivative = 0.0 if self.previous_error is None else (float(error) - self.previous_error) / dt
        self.previous_error = float(error)
        return self.kp * float(error) + self.ki * self.integral + self.kd * derivative


class TrajectoryPIDTracker:
    def __init__(self, config: TrackerConfig, *, wheelbase: float = 2.5, rear_axle_offset: float = 0.0,
                 sample_dt: float = 0.2) -> None:
        self.config = config
        self.wheelbase = float(wheelbase)
        self.rear_axle_offset = float(rear_axle_offset)
        self.sample_dt = float(sample_dt)
        if not np.isfinite([self.wheelbase, self.rear_axle_offset, self.sample_dt]).all() or (
            self.wheelbase <= 0 or self.rear_axle_offset < 0 or self.sample_dt <= 0
        ):
            raise ValueError("wheelbase/sample_dt must be positive; rear axle offset must be non-negative")
        self._last_speed = None
        self.last_diagnostics: dict[str, float] = {}
        self.reference: np.ndarray | None = None
        self.last_target_index: int | None = None
        self.last_target_point: np.ndarray | None = None
        self._steering_pid = _PID(config.heading_kp, config.heading_ki, config.heading_kd)
        self._speed_pid = _PID(config.speed_kp, config.speed_ki, config.speed_kd)

    def reset(self) -> None:
        self._last_speed = None
        self.last_diagnostics = {}
        self.reference = None
        self.last_target_index = None
        self.last_target_point = None
        self._steering_pid.reset()
        self._speed_pid.reset()

    def set_reference(self, trajectory: np.ndarray) -> None:
        reference = np.asarray(trajectory, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 4 or reference.shape[0] < 2:
            raise ValueError(f"trajectory reference must have shape [H,4] with H >= 2, got {reference.shape}")
        if self.config.controller_mode == "adaptive_pursuit" and not np.isfinite(reference).all():
            raise ValueError("adaptive pursuit requires finite trajectory samples")
        self.reference = reference.copy()
        self.last_target_index = None
        self.last_target_point = None
        if self.config.controller_mode == "adaptive_pursuit":
            return
        if self.config.integral_reset_on_reference_change:
            self._steering_pid.reset()
        else:
            self._steering_pid.reset_derivative()
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
        if self.config.controller_mode == "adaptive_pursuit":
            return self._pursuit_target(ego)[1][:2].copy()
        _target_index, target = self._target(ego)
        return target[:2].copy()

    def _pursuit_target(self, ego):
        if self.reference is None:
            raise RuntimeError("set_reference() must be called before tracker.step()")
        # The reference heading is course, not chassis yaw. Account for the
        # bicycle slip angle before converting the center path to the rear axle.
        rear_reference = self.reference.copy()
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(rear_reference[:, :2], axis=0), axis=1))]
        distinct = np.r_[True, np.diff(arc) > 1e-6]
        curvature = np.zeros(len(arc))
        if np.count_nonzero(distinct) > 1:
            curve_samples = np.gradient(np.unwrap(rear_reference[distinct, 2]), arc[distinct])
            curvature = np.interp(arc, arc[distinct], curve_samples)
        chassis_yaw = rear_reference[:, 2] - np.arcsin(np.clip(self.rear_axle_offset * curvature, -0.95, 0.95))
        rear_reference[:, :2] -= self.rear_axle_offset * np.column_stack(
            (np.cos(chassis_yaw), np.sin(chassis_yaw))
        )
        rear_xy = ego.xy - self.rear_axle_offset * np.array([np.cos(ego.heading), np.sin(ego.heading)])
        s, _, arc = project_trajectory(rear_reference, rear_xy)
        lookahead = np.clip(abs(ego.speed) * self.config.lookahead_time_s,
                            self.config.min_lookahead_m, self.config.max_lookahead_m)
        target_s = min(s + lookahead, arc[-1])
        target = sample_trajectory(rear_reference, arc, target_s)
        return rear_xy, target, min(int(np.searchsorted(arc, target_s)), len(arc) - 1)

    def _adaptive_step(self, ego, dt):
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("tracker dt must be finite and positive")
        rear_xy, target, index = self._pursuit_target(ego)
        delta = target[:2] - rear_xy
        left = -np.sin(ego.heading) * delta[0] + np.cos(ego.heading) * delta[1]
        steering = self.config.pursuit_gain * np.arctan2(2 * self.wheelbase * left, max(float(delta @ delta), 1e-6))
        self.last_target_index = index
        self.last_target_point = target[:2].copy()

        s, _, arc = project_trajectory(self.reference, ego.xy)
        times = np.arange(len(self.reference)) * self.sample_dt
        target_time = np.interp(s, arc, times) + self.config.speed_preview_s
        speed_target = np.interp(target_time, times, self.reference[:, 3])
        error = float(speed_target - ego.speed)
        # Differentiate measurement, avoiding kicks when the selected speed changes.
        derivative = 0.0 if self._last_speed is None else -(ego.speed - self._last_speed) / dt
        self._last_speed = ego.speed
        pid = self._speed_pid
        proposed_integral = pid.integral + error * dt
        output = pid.kp * error + pid.ki * proposed_integral + pid.kd * derivative
        low, high = -self.config.max_decel_mps2, self.config.max_accel_mps2
        if low <= output <= high or (output > high and error < 0) or (output < low and error > 0):
            pid.integral = proposed_integral
        output = pid.kp * error + pid.ki * pid.integral + pid.kd * derivative
        self.last_diagnostics = {
            "steering_controller_saturated": float(abs(steering) >= self.config.max_steering_rad),
            "speed_controller_saturated": float(output >= high or output <= low),
            "target_speed_mps": float(speed_target),
        }
        return PhysicalControl(float(np.clip(steering, -self.config.max_steering_rad, self.config.max_steering_rad)),
                               float(np.clip(output, low, high)))

    def _steering_error(self, *, lateral_error: float, heading_error: float, speed: float) -> float:
        mode = self.config.steering_error_mode
        if mode == "heading_only":
            return heading_error
        if mode == "lateral_only_no_speed_normalization":
            return self.config.cross_track_kp * lateral_error
        cross_track_angle = float(
            np.arctan2(self.config.cross_track_kp * lateral_error, max(abs(speed), 0.5))
        )
        if mode == "lateral_only":
            return cross_track_angle
        return heading_error + cross_track_angle

    def step(self, ego: EgoState, *, dt: float) -> PhysicalControl:
        if self.config.controller_mode == "adaptive_pursuit":
            return self._adaptive_step(ego, dt)
        target_idx, target = self._target(ego)
        self.last_target_index = target_idx
        self.last_target_point = target[:2].copy()
        target_delta = target[:2] - ego.xy
        left_error = -np.sin(ego.heading) * target_delta[0] + np.cos(ego.heading) * target_delta[1]
        heading_error = float(wrap_angle(target[2] - ego.heading))
        steering_error = self._steering_error(
            lateral_error=float(left_error),
            heading_error=heading_error,
            speed=float(ego.speed),
        )
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
