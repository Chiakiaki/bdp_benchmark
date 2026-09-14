# MetaDrive 12 m/s Stability Benchmark Contract

This bundle is a controlled speed-cap ablation of `scripts/job_scripts/benchmark`. It retains the same maps, traffic, 275-dimensional observation, termination behavior, PPO settings, candidate layouts, policy architectures, PID configuration, seeds, and training budget.

Only speed-related settings and run names differ.

Like the main benchmark bundle, every job that constructs Frenet geometry uses `candidate_sampler: frenet_route_continuous`. The 12 m/s bundle therefore differs from the main bundle in speed settings, not route-reference correctness.

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

For native BDP, the target speed remains a candidate descriptor; native actions still execute MetaDrive steering and throttle/brake directly. The 43.2 km/h ego-vehicle setting is what constrains native execution. Native builtin PPO does not consume trajectory target-speed features.

## Experimental Interpretation

This bundle is not paper-speed aligned. It tests whether reducing the distance traveled during the two-second planning horizon improves trajectory tracking and training stability:

```text
12.0 m/s * 2.0 s = approximately 24 m
```

Results from this bundle must be labeled as the 12 m/s stability ablation and reported separately from the 80 km/h benchmark bundle.

## Launch

From this directory:

```bash
./run_metadrive_bdp_frenet_comparison.sh
./run_metadrive_builtin_comparison.sh
```

From the repository root:

```bash
./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_bdp_frenet_comparison.sh
./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_builtin_comparison.sh
```
