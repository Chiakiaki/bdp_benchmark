# PID Steering Error Modes Design

## Goal

Make Frenet trajectory tracking experiments able to isolate heading and lateral
control errors while preserving PID integral state across frequently replaced
trajectory references. Existing behavior remains the default.

## Controller State

`TrajectoryPIDTracker.set_reference()` continues to replace the trajectory and
clear target bookkeeping. It always clears each PID controller's previous error,
so the derivative term cannot spike from comparing errors belonging to different
trajectories. A new `pid_integral_reset_on_reference_change` option controls
whether the accumulated steering integral is also cleared. The speed PID still
resets completely at every reference change. Full tracker reset at episode
boundaries always clears both integral and derivative state.

The option defaults to `true`, reproducing the current complete reset on every
reference replacement. Setting it to `false` preserves only integral state.

## Steering Error Modes

`pid_steering_error_mode` supports four values:

- `combined`: current behavior, `heading_error + cross_track_angle`.
- `lateral_only`: the current speed-normalized cross-track angle only.
- `lateral_only_no_speed_normalization`: `cross_track_kp * lateral_error`,
  interpreted as the steering PID's angular error, without division by speed.
- `heading_only`: wrapped target-heading error only.

The current speed-normalized term remains:

```text
cross_track_angle = atan2(
    cross_track_kp * lateral_error,
    max(abs(actual_speed), 0.5),
)
```

Only the steering error source changes. Target selection, speed PID, output
limits, candidate generation, nominal-state logic, and simulator actuation are
unchanged. Since the tracker is shared, the options apply identically to
HighwayEnv and MetaDrive Frenet-PID modes; native-controller modes are unaffected.

## Validation

Tests cover all four error formulas, invalid mode rejection, default full-reset
compatibility, integral preservation with derivative reset, episode-level full
reset, YAML/CLI parsing, resolved-config backup, and environment factory wiring.
