# Frenet-PID V3 Legacy Contract

`frenet_pid_v3_legacy` preserves the former `frenet_pid_v3` contract:
distance-retimed lane/circular paths and a cached terminal `v_target` supplied
directly to persistent speed PI. The speed/path retiming implementation is in
[`common/retiming.py`](../bdp_benchmark/common/retiming.py).

Use this explicit mode to reproduce the old execution semantics. Revised
[`frenet_pid_v3`](frenet_pid_v3.md) instead uses independent timed lateral
geometry and an active desired-speed preview. The rename does not change v1
(`frenet_pid`), v2 (`frenet_pid_v2`), or native execution.

## Initial State and Speed Command

Candidate geometry starts from the nominal XY position and heading used by
nominal-origin planning. The scalar speed profile starts from the actual measured
vehicle speed `v0`, not the nominal trajectory's speed. PID feedback continues
to use the actual vehicle state.

The five speed actions use these settings:

```yaml
frenet_speed_command_time_s: 1.0
frenet_speed_delta_rate_mps2: 8.0
frenet_speed_action_scales: [-1, -0.5, 0, 0.5, 1]
```

For an action scale `q`, let `tau` be `frenet_speed_command_time_s`, `rate` be
`frenet_speed_delta_rate_mps2`, and clip to the configured target-speed bounds
(`v_min = 0` in these jobs):

```text
v_target = clip(v0 + tau * rate * q, v_min, v_max)
a = (v_target - v0) / tau

v(t) = v0 + a * t                         for 0 <= t <= tau
v(t) = v_target                           for t > tau

distance(t) = v0 * t + 0.5 * a * t^2       for 0 <= t <= tau
distance(t) = v0 * tau + 0.5 * a * tau^2
              + v_target * (t - tau)      for t > tau
```

The target is clipped before calculating `a`. The ramp lasts until `tau`, then
cruises at `v_target`; clipping can reduce the effective acceleration magnitude.
Distance is the integral of this same speed profile, not `v_target * t` and not
an independently interpolated position curve. The old `frenet_speed_delta_mps`
does not apply to v3_legacy and is omitted from all four legacy jobs.

With `rate = 8 m/s^2` and `tau = 1 s`, the unclipped acceleration commands are
`[-8, -4, 0, 4, 8] m/s^2`. For example, at `v0 = 10 m/s` in the 80 km/h bundle,
the targets are `[2, 6, 10, 14, 18] m/s`. In the 12 m/s bundle at `v0 = 8 m/s`,
clipping produces targets `[0, 4, 8, 12, 12] m/s` and accelerations
`[-8, -4, 0, 4, 4] m/s^2`. Saturation can make different action slots share a
speed profile; it does not remove those slots.

The chosen `tau = 1.0 s` exceeds the steering lookahead time of `0.7 s`. This is
a tuning preference, not a hard requirement: speed-command time and steering
lookahead are independent, and `tau <= pid_lookahead_time_s` is not inherently
invalid. The command time must be finite and positive and fit within the
trajectory horizon (`2.0 s` in these jobs).

## Distance-Based Geometry

Lane paths and optional left/right circular paths use exactly the same speed
profile and integrated distance for a given `v0`, `q`, and speed bounds. Geometry
changes the path, not the meaning of a speed action.

Advance each candidate by its own actual geometric arc length, including the
lateral component of lane changes. Road-longitudinal Frenet `s` alone is not
candidate arc length. Lateral progression therefore follows distance traveled
along the candidate, not elapsed time independently of motion. XY, heading,
speed, and candidate descriptors must all describe that same retimed path.
Here "actual candidate arc length" means the candidate's geometric length,
not the physical vehicle's measured displacement.

If the speed reaches zero and remains zero, distance stops increasing: XY,
heading, and lateral progression freeze. A candidate with zero initial speed
and a zero-speed target has frozen geometry; this is not a physical-vehicle
hard-stop guarantee. Acceleration from rest must still have a nondegenerate
spatial path so that progression can begin as distance increases.

The implementation reuses the existing quintic lateral/quartic longitudinal
polynomials to construct a spatial lane-change shape, using a dimensionless
progress parameter rather than simulation time. Its scale is
`max(actual_speed * trajectory_horizon_s, 2 * current_lane_width)`, so it remains
positive at rest. A constant-offset route-following tail follows the transition.
Dense world polylines are retimed by cumulative Euclidean arc length. Consequently
a slow or stopping candidate may not complete its lane change within two seconds;
it does not move sideways merely to meet the old time deadline.

The internal `[s,d,s_dot,d_dot,s_ddot,d_ddot]` arrays are recomputed from this
retiming. Derivatives are piecewise-consistent within geometric segments;
polyline knots and the acceleration-to-cruise switch are not jerk-smoothed. The
candidate heading is the geometric tangent, and the speed feature uses the
prescribed total speed directly rather than `sqrt(s_dot^2 + d_dot^2)`, which
would miss the curved-road metric. Output remains 11 samples with five normalized
features each.

V3 legacy always uses a connected route reference, including for one-hot candidate
encoding. At the final supplied route endpoint only, a terminal tangent continues
the geometric reference if more path length is needed. This is not a prediction
of a successor road, nor extrapolation of a straight segment across a known bend.
The existing collision, off-road and destination logic judges actual motion.

A backward-facing nominal tangent may initially require negative local Frenet
`s`. V3 legacy uses a local tangent continuation behind the reference origin for
these samples, including its lateral normal, instead of clamping motion to zero.
This is geometric extrapolation behind the local origin, not reconstruction of
previous road segments. Forward route stitching is unchanged. Traveled arc
length remains monotonic even when reference `s` initially decreases; such paths
are not guaranteed to satisfy tire/steering limits.

## Execution and Limits

Cache `v_target` with the candidate that was scored. Selecting that candidate
passes its cached target directly to the persistent speed PI controller, with
actual measured speed as feedback. Do not recompute it after selection, and do
not substitute a future sampled trajectory speed. The PI target is the final
`v_target` immediately, not the intermediate ramp speed `v(t)`.

`pid_speed_preview_s: 0.4` is retained from the sources for configuration
comparability but is ignored in v3_legacy. Steering still uses adaptive pursuit
and its existing lookahead; all PID gains and limits remain unchanged. The direct
speed override must be cleared on reset and must not leak into revised v3, v1,
v2, or native execution.

This is a nominal geometric candidate with a prescribed speed profile, not a
real Bullet prediction. The `8 m/s^2` rate describes candidates; it does not
promise physical acceleration, braking, grip, or successful path tracking.
For example, the inherited `pid_max_accel_mps2` remains `5.0`, independently of
the candidate rate. Circular paths also retain the source steering fraction;
there is no implicit safety filter or steering-limit change.

## Preserved Benchmark Jobs

Each existing bundle contains these explicit legacy copies alongside its active
revised-v3 jobs:

- `metadrive_frenet_pid_v3_legacy_bdp_frenet_benchmark.yaml`
- `metadrive_frenet_pid_v3_legacy_builtin_benchmark.yaml`

| Bundle | Physical ceiling | Target ceiling | Candidates |
| --- | --- | --- | --- |
| [benchmark](../scripts/job_scripts/benchmark/) | 80.0 km/h | 22.222222 m/s | 25 |
| [benchmark_metadrive_12mps](../scripts/job_scripts/benchmark_metadrive_12mps/) | 43.2 km/h | 12.0 m/s | 25 |

Both pairs preserve 15 lane candidates plus 10 left/right circular candidates,
the five existing speed actions, `frenet_include_curvature_candidates: true`,
and `frenet_curvature_steering_fraction: 0.9`. The optional 15/25 ordering is
unchanged; all supplied legacy and revised-v3 jobs use 25.

The copies preserve the old v3 settings exactly, except
`environment.trajectory_execution_mode` becomes `frenet_pid_v3_legacy` and
the run-name `v3` component becomes `v3_legacy`. Comments identify the old
contract. No settings are added: PPO, PID values, horizon and sampling, feature
scales, architecture, map/traffic, observations/rewards, physics, logging, and
diagnostics are unchanged. In particular, the 80 km/h pair keeps tracking
diagnostics enabled, while the 12 m/s pair keeps its inherited defaults.

The active v3 files retain their original names, mode, and hyperparameters and
now describe the revised contract. Existing
`run_metadrive_frenet_pid_v3_comparison.sh` launchers continue to run revised
v3, not these legacy copies. No new launchers are required. No launches or
training are part of this sidecar update.

## Archived Configs and Checkpoints

Old saved configs can still contain the literal
`trajectory_execution_mode: frenet_pid_v3`. That literal now selects revised
v3. To retain old behavior, load the original saved config and checkpoint with
the explicit CLI override:

```text
--trajectory_execution_mode frenet_pid_v3_legacy
```

Do not rewrite archived logs or configs. The parser applies this override to the
effective configuration without modifying the saved file. The original run name
can remain unchanged for historical loading; it does not select execution
semantics. Newly saved resolved configs from the legacy jobs explicitly record
`frenet_pid_v3_legacy` and can be parsed again with that mode.

New v3 checkpoint labels mean independent geometry plus desired-speed preview.
A matching candidate count or feature width does not make an old-v3 checkpoint
a like-for-like revised-v3 result. State observations, rewards, per-sample
feature dimensions, and `critic_based_rl` are not changed by this mode split.

## Focused Sidecar Verification

From the repository root:

```bash
PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}" python3 -m pytest tests/test_v3_legacy_jobs.py -q
```

The tests compare every legacy YAML with its corresponding active-v3 YAML after
normalizing only mode and run name; YAML comments do not participate in field
equality. They also exercise real parser validation, resolved-config backup
roundtrips for both modes, and the old-v3 CLI override while checking the saved
file is unchanged. Missing legacy enum/validation integration is intentionally
an early test failure, not a skip or mocked parser success. No simulator or
training is needed for these configuration tests.
