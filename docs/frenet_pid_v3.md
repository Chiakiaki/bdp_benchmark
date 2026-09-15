# Frenet-PID V3 Contract

This document specifies revised `frenet_pid_v3` under the
[approved independent-geometry plan](plans/2026-09-15-v3-independent-geometry.md).
Lane candidates combine an independent time-based lateral quintic with
longitudinal ramp/cruise geometry. Their speed column is desired `V(t)`, not
speed derived from XY or Frenet derivatives. The persistent speed PI controller
previews this column; it does not receive a direct terminal-speed override.

The former distance-retimed geometry and direct `v_target` PI mechanism are
preserved as [`frenet_pid_v3_legacy`](frenet_pid_v3_legacy.md). V1
(`frenet_pid`), v2 (`frenet_pid_v2`), and native execution are unaffected.

## Initial State and Desired Speed

Candidate position and heading start at the nominal planning pose. The scalar
desired-speed law starts at actual measured vehicle speed `v0`, not speed from
the nominal trajectory. Controller feedback also uses the actual vehicle state.

All four active v3 jobs retain these existing settings:

```yaml
frenet_speed_command_time_s: 1.0
frenet_speed_delta_rate_mps2: 8.0
frenet_speed_action_scales: [-1, -0.5, 0, 0.5, 1]
```

For action scale `q`, command time `tau`, and rate `rate`, clip the target to
the configured bounds before calculating acceleration (`v_min = 0` here):

```text
v_target = clip(v0 + tau * rate * q, v_min, v_max)
a = (v_target - v0) / tau

V(t) = v0 + a * t                         for 0 <= t <= tau
V(t) = v_target                           for t > tau

D(t) = v0 * t + 0.5 * a * t^2             for 0 <= t <= tau
D(t) = v0 * tau + 0.5 * a * tau^2
       + v_target * (t - tau)             for t > tau
```

`D(t)` is the integral of scalar desired speed. Circular candidates use this
distance along their retained circular geometry. Revised lane candidates do
not use it to retime their full XY path by arc length.

With `tau = 1.0 s` and `rate = 8 m/s^2`, unclipped acceleration commands are
`[-8, -4, 0, 4, 8] m/s^2`. At `v0 = 10 m/s` in the 80 km/h bundle, targets
are `[2, 6, 10, 14, 18] m/s`. At `v0 = 8 m/s` in the 12 m/s bundle, targets
are `[0, 4, 8, 12, 12] m/s`, with effective accelerations
`[-8, -4, 0, 4, 4] m/s^2`. Clipping can make action slots share a profile;
those slots are not removed. The old `frenet_speed_delta_mps` does not apply
to either v3 mode and remains absent from these YAMLs.

The three time settings are independent:

| Setting | Value | Meaning |
| --- | --- | --- |
| `frenet_speed_command_time_s` | 1.0 s | Ramp duration `tau` |
| `pid_lookahead_time_s` | 0.7 s | Adaptive-pursuit steering lookahead |
| `pid_speed_preview_s` | 0.4 s | Active desired-speed preview |

`tau` must be finite, positive, and fit inside the 2.0 s trajectory horizon.
Its being greater than 0.7 s is a tuning choice, not a constraint coupling speed
timing to steering. No new settings or gain changes are required.

## Independent Lane Geometry

The original lateral quintic `d(t)` runs on elapsed trajectory time over the
2.0 s horizon, with the projected initial lateral state and the existing lane
target boundary conditions. It is not rescheduled according to traveled
distance or desired speed. Its terminal lateral velocity and acceleration
remain zero.

For longitudinal lane geometry, retain the Frenet projection of initial
velocity at the nominal pose using actual measured speed. Let `u0` denote
that initial longitudinal component and `s0` the initial route coordinate.
Ramp that component to the same scalar `v_target` over `tau`, then cruise:

```text
a_s = (v_target - u0) / tau
s_dot(t) = u0 + a_s * t                   for 0 <= t <= tau
s_dot(t) = v_target                       for t > tau
s(t) = s0 + u0 * t + 0.5 * a_s * t^2     for 0 <= t <= tau
s(t) = s0 + u0 * tau + 0.5 * a_s * tau^2
       + v_target * (t - tau)             for t > tau
```

For a non-aligned pose, `u0` need not equal `v0`: the lane-coordinate
geometry preserves projected initial velocity, while scalar desired `V(t)`
always starts at actual measured speed. On an aligned straight reference,
`u0 = v0` and `s(t) - s0 = D(t)` exactly. Independent lateral motion can
still make full XY arc length differ from `D(t)`.

There is no square-root speed compensation such as choosing longitudinal speed
to cancel lateral speed, and no curvature-based speed compensation. Neither
`sqrt(s_dot^2 + d_dot^2)` nor XY-derived speed replaces desired `V(t)`.

Connected route references remain in use, including for one-hot candidate
encoding. Known route bends are followed; only the final supplied route endpoint
uses a terminal tangent continuation when needed. A backward-facing initial
tangent can require negative local `s`; local tangent continuation behind the
origin preserves that geometry rather than clamping it. These continuations
are not reconstructions of missing roads or guarantees of feasible tracking.

## Candidate Features and Limits

Lane and outer-turn candidates share the same scalar desired-speed law for each
speed action. Revised v3's left/right outer paths follow the configured curvature
until `frenet_speed_command_time_s`, then continue straight at the reached tangent
heading. The cutoff distance is the exact ramp integral
`0.5 * (v0 + v_target) * tau`, including when tau is between sample times. XY and
heading are continuous; speed samples and the steering fraction are unchanged.
No new mode or switch is required. V1/v2/v3_legacy retain their old full-arc paths.
Candidate ordering, the optional 15/25 layout, and feature dimensions
are unchanged: each candidate has 11 samples and five normalized features per
sample. Both supplied bundles enable 25 candidates.

The features represent candidate geometry plus a speed command, not a
time-consistent vehicle-dynamics rollout. In particular:

- XY-derived speed can exceed the desired speed column during lateral motion.
- With `v0 = v_target = 0`, every desired-speed sample is zero, even if future
  lateral quintic positions change sideways.
- If desired speed reaches zero before the horizon ends, future lateral geometry
  may continue changing. It need not freeze as it does in the legacy contract.

Changing future geometry does not predict that a stopped vehicle moves sideways,
and zero desired speed must not be replaced by geometric speed. Equally, zero
desired speed is not a hard-stop or immobilization guarantee: persistent PI
state, tracking error, and Bullet physics still govern actual motion.

## Preview PI Execution

Selecting a revised v3 candidate supplies its sampled reference without the
direct terminal-speed override. The existing controller projects the actual
vehicle position onto the selected reference, maps that projection to
`t_projected`, and interpolates the desired-speed column at:

```text
t_preview = t_projected + pid_speed_preview_s
PI_setpoint = interpolate(sample_times, desired_V, t_preview)
```

Thus `pid_speed_preview_s: 0.4` is ACTIVE: PI previews
`V(t_projected + 0.4 s)`, using endpoint values outside the sampled interval.
It does not immediately command cached `v_target`. Revised candidates leave
`CandidateSet.speed_targets` unset because that field denotes direct controller
overrides; the endpoint desired speed is still present in the raw reference.
For example, at `t_projected = 0`, `v0 = 10`, and `v_target = 18 m/s`,
the preview setpoint is `13.2 m/s`, not `18 m/s`.

Adaptive-pursuit steering, actual-speed feedback, persistent PI state, all PID
gains, and all limits remain unchanged. The candidate rate of `8 m/s^2` is
not a physical acceleration promise; inherited `pid_max_accel_mps2` remains
`5.0`. Preview does not guarantee absence of overshoot, tire slip, or tracking
failure. Observations, rewards, vehicle physics, and `critic_based_rl` are
unchanged.

## Jobs and Checkpoint Compatibility

Each existing bundle retains its v3 BDP and builtin jobs and gains corresponding
v3_legacy copies:

| Bundle | Physical ceiling | Target ceiling | Candidates |
| --- | --- | --- | --- |
| [benchmark](../scripts/job_scripts/benchmark/) | 80.0 km/h | 22.222222 m/s | 25 |
| [benchmark_metadrive_12mps](../scripts/job_scripts/benchmark_metadrive_12mps/) | 43.2 km/h | 12.0 m/s | 25 |

Both use 15 lane candidates plus 10 left/right turn-then-straight candidates, with
`frenet_include_curvature_candidates: true` and steering fraction `0.9`.
The active files remain `metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml`
and `metadrive_frenet_pid_v3_builtin_benchmark.yaml`. Their mode stays
`frenet_pid_v3`, their run names and every hyperparameter stay unchanged,
and only comments change. Legacy copies differ in mode, run name, and comments.

The existing `run_metadrive_frenet_pid_v3_comparison.sh` in each bundle continues
to select the active revised-v3 jobs. No launcher changes or new launchers are
needed. Full benchmark training is not started automatically by this update.

New checkpoints labeled v3 use the revised independent-geometry and preview
semantics. Old saved configs that explicitly contain `frenet_pid_v3` do not
automatically select legacy behavior. When loading an old run, retain its saved
config and checkpoint and supply this CLI override:

```text
--trajectory_execution_mode frenet_pid_v3_legacy
```

Archived logs/configs are not modified. Run names, checkpoint age, and unchanged
feature dimensions do not perform semantic migration. An old checkpoint run
under revised v3 is a transfer evaluation, not a reproduction of its original
execution contract; candidate-count compatibility does not resolve that change.

## Focused Sidecar Verification

From the repository root:

```bash
PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}" python3 -m pytest tests/test_v3_legacy_jobs.py -q
```

These tests compare parsed legacy and active YAMLs, allow only mode/run-name
differences, and exercise real parsing, resolved-config backup roundtrips, and
the historical-v3 CLI override without changing a saved file. Geometry/control
tests are in `tests/test_independent_frenet.py` and `tests/test_v3.py`. The full
suite also performs two-worker train/save/reload checks for both implementations
in both speed bundles. These checks do not establish physical tracking quality.
