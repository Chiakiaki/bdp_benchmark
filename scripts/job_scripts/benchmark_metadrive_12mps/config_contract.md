# MetaDrive 12 m/s Stability Benchmark Contract

The added `metadrive_frenet_pid_v3_*` pair is governed by the
[v3 contract](../../../docs/frenet_pid_v3.md), with actual-speed-based
acceleration-then-cruise commands, independent timed lateral geometry and desired
speed preview. The old delta-speed description below applies to v1/v2, not v3.
`frenet_pid_v3_legacy` copies preserve the former distance-retiming/direct-target
behavior. Old saved v3 configs need an explicit legacy-mode override to preserve
their semantics; see [the legacy contract](../../../docs/frenet_pid_v3_legacy.md).
Existing launchers
still cover the original seven jobs; `run_metadrive_frenet_pid_v3_comparison.sh`
sequentially runs the two additional v3 jobs.

This bundle is the recommended next stability experiment relative to
`scripts/job_scripts/benchmark`. It retains the same maps, traffic, 275-dimensional
state observation, reward formulation, termination behavior, PPO settings, policy
architecture settings, PID configuration, seeds, and training budget. It changes
the ego speed ceiling and, for the four Frenet-PID jobs, enables optional
curvature candidates. It is therefore no longer a speed-only ablation by default.

Both bundles use the same tuned `adaptive_pursuit` controller configuration.
The controller experiment and parameter rationale are documented in
[adaptive pursuit tracking](../../../docs/adaptive_pursuit_tracking.md).

Like the main benchmark bundle, every job that constructs Frenet geometry uses
`candidate_sampler: frenet_route_continuous`. The original lane-center candidates
remain route-continuous; the optional extra curves deliberately do not follow a
lane center. See [paper-writing notes](../benchmark/paper_writing.md#tracking-tire-slip-and-the-12-ms-experiment)
for the motivation and the limits of interpreting tracking failures as tire slip.

## Physical Speed Contract

Every job explicitly overrides the ego vehicle through MetaDrive's flexible per-agent configuration:

```yaml
environment:
  environment_config:
    agent_configs:
      default_agent:
        use_special_color: true
        spawn_lane_index: [">", ">>", 0]
        max_speed_km_h: 43.2
```

`43.2 km/h` equals `12.0 m/s`. MetaDrive requires `max_speed_km_h` under `agent_configs.default_agent`; it is not a valid shared `vehicle_config` key in MetaDrive 0.4.3.

MetaDrive implements this as an engine-force cutoff above the configured speed, not an instantaneous velocity clamp. Brief overshoot or coasting above 12 m/s remains possible.

`metadrive_native_continuous_builtin_benchmark.yaml` preserves the categorical native builtin job's configuration, including this physical limit, while changing only the run name and action space to MetaDrive's normalized continuous steering and throttle/brake `Box(2)`.

## Candidate Speed Contract

All jobs that generate action-conditioned trajectory descriptors use:

```yaml
frenet:
  maximum_target_speed_mps: 12.0
```

The `frenet_pid` and `frenet_pid_v2` jobs additionally use:

```yaml
frenet:
  frenet_speed_delta_mps: 2.5
```

Their five target-speed levels are based on the current or nominal speed plus offsets from `-2.5` to `+2.5 m/s`, clipped to `[0, 12] m/s`.

```text
v_target = clip(v_origin + [-2.5, -1.25, 0, 1.25, 2.5], 0, 12)
```

`v_origin` is measured actual speed for v1 and the nearest previous trajectory
sample's speed for v2. It is not always actual speed in v2. These are terminal
longitudinal speeds at 2 s, not immediate throttle/brake commands or strict
bounds on intermediate total speed during a lateral lane change. Both old and
new candidate families share these five endpoints. Examples:

| Origin speed | Five terminal speeds (m/s) |
| --- | --- |
| 0 | 0, 0, 0, 1.25, 2.5 |
| 6 | 3.5, 4.75, 6, 7.25, 8.5 |
| 12 | 9.5, 10.75, 12, 12, 12 |

The slowest candidate reaches zero only if `v_origin <= 2.5 m/s`. For an aligned
straight path with zero initial acceleration the speed interpolation is
`v(t) = v_origin + (v_target - v_origin) * (3*u^2 - 2*u^3)`, `u=t/2`.
At the current 0.4 s preview, only 10.4% of the endpoint speed change is requested.
For `v_origin = v_actual = 12`, selecting the slowest action initially requests
about 11.74 m/s. With the current PI gains and zero integral, that is only about
6.6% normalized braking, not full braking. If nominal speed is 12 but actual
speed is 10, the same preview can request acceleration. Replanning and nearest
sample projection can repeatedly restart progress along this speed profile.
This explains a possible weak-slowdown mechanism; it does not prove which actions
a trained policy selected. No speed law, preview, gains or braking limits were
changed for the curvature experiment.

For native BDP, the target speed remains a candidate descriptor; native actions still execute MetaDrive steering and throttle/brake directly. The 43.2 km/h ego-vehicle setting is what constrains native execution. Native builtin PPO does not consume trajectory target-speed features.

## Optional Curvature Contract

The four Frenet-PID jobs enable:

```yaml
frenet:
  frenet_include_curvature_candidates: true  # Default false outside these four jobs.
  frenet_curvature_steering_fraction: 0.9
```

Native jobs and upstream MetaDrive/critic_based_rl are unchanged. Both flags are
backed up in effective configuration and support CLI overrides. For the original
15-action geometry use `--no-frenet_include_curvature_candidates`.

- Actions 0..14 are exactly the existing speed-major left/center/right lane grid.
- Actions 15..24 are speed-major left/right circular paths, sharing the five
  target speeds. Positive world curvature is left; feature lateral coordinates
  still use the existing MetaDrive forward/right convention.
- `delta = fraction * min(vehicle_max_steering, pid_max_steering_rad)` (radians).
  Rear curvature is `tan(delta)/wheelbase`; center curvature is
  `rear_curvature / sqrt(1 + (rear_axle_offset * rear_curvature)^2)`.
- Center-path tangent starts from the planning heading. This is nominal heading
  in v2 and actual API heading in v1. It is an idealized geometric reference, not
  a prediction of instantaneous slip or steering transients. The existing PID
  still tracks it using actual pose and speed.
- With the default vehicle, 0.9 gives approximately 36-degree wheel steering and
  a 3.68 m center turning radius. This is not a high-speed feasibility guarantee.
  At 12 m/s a constant-speed two-second arc can exceed one full revolution;
  nearest-point tracking/projection can then be ambiguous. No hidden safety
  filter, horizon truncation or speed reduction is applied.
- Candidate features remain 11 x 5 = 55 values per candidate, with K changing
  from 15 to 25. Normal checkpoint loading requires matching K. For BDP dot/MLP
  visual transfer tests, `--allow_bdp_candidate_count_change` permits changed K
  while preserving state/per-candidate feature spaces and all learned weights.
  This does not apply to builtin categorical heads or establish equivalent
  behavior across changed trajectory timing and speed limits.

The original 15 paths are unchanged, so adding alternatives does not guarantee
selection of the middle lane. Keep BDP/builtin comparisons on the same candidate
set; disable the switch in both when isolating speed-cap effects.

## Experimental Interpretation

This bundle is not paper-speed aligned. It tests whether reducing the distance traveled during the two-second planning horizon improves trajectory tracking and training stability:

```text
12.0 m/s * 2.0 s = approximately 24 m
```

Results from this bundle must be labeled as the 12 m/s stability ablation and reported separately from the 80 km/h benchmark bundle.

The speed ceiling also rescales MetaDrive's existing speed observation/reward
normalization. No new observation or reward term is added. Lower lateral dynamic
demand and improved tracking remain hypotheses to test, not established results.

## Launch

For the current v3-versus-native experiment, use these two queues. Together they
cover the v3 BDP/builtin pair and the three native jobs, with no actual-origin
Frenet-PID training:

```bash
./run_metadrive_frenet_pid_v3_comparison.sh
./run_metadrive_native_comparison.sh
```

The native queue runs BDP-Frenet, discrete builtin PPO, then continuous builtin
PPO. The older `run_metadrive_remaining_comparison.sh` is retained for historical
experiments but includes actual-origin Frenet-PID jobs; it is not this queue.

From this directory:

```bash
./run_metadrive_bdp_frenet_comparison.sh
./run_metadrive_builtin_comparison.sh
./run_metadrive_remaining_comparison.sh
./run_metadrive_frenet_pid_v3_comparison.sh
```

From the repository root:

```bash
./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_bdp_frenet_comparison.sh
./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_builtin_comparison.sh
./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_remaining_comparison.sh
./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_frenet_pid_v3_comparison.sh
```

The third launcher sequentially runs actual-origin Frenet BDP, actual-origin
Frenet builtin, and native continuous builtin. Together the three launchers
cover the original seven training jobs once; the fourth launcher covers the v3
pair. Each forwards optional CLI overrides.
