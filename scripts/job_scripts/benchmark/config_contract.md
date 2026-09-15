# MetaDrive Benchmark Configuration Contract

This document records the active MetaDrive benchmark configuration and distinguishes it from both the bundled MetaDrive PPO demo and the configuration stated in the MetaDrive paper. It is the reference for comparing builtin categorical PPO with candidate-scoring BDP policies.

The active contract covers the seven training jobs:

- `metadrive_native_builtin_benchmark.yaml`
- `metadrive_native_continuous_builtin_benchmark.yaml`
- `metadrive_native_bdp_frenet_benchmark.yaml`
- `metadrive_frenet_pid_builtin_benchmark.yaml`
- `metadrive_frenet_pid_bdp_frenet_benchmark.yaml`
- `metadrive_frenet_pid_v2_builtin_benchmark.yaml`
- `metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml`

Files under `scripts/job_scripts/example/` are historical examples and are not required to follow the active observation or PID contract.

## PPO Training Contract

All six active jobs use the paper-aligned PPO optimization settings. The iteration and update counts below assume this repository's fixed budget of 2,000,000 environment transitions.

| Quantity | Earlier benchmark | Current adopted | Paper-aligned target |
| --- | ---: | ---: | ---: |
| Rollout batch | 16,000 | 8,000 | 8,000 |
| PPO iterations over 2M transitions | 125 | 250 | 250 |
| Minibatch size | 1,000 | 100 | 100 |
| Epochs | 10 | 20 | 20 |
| Optimizer updates per iteration | 160 | 1,600 | 1,600 |
| Total optimizer updates | 20,000 | 400,000 | 400,000 |
| Learning rate | `3e-4` | `5e-5` | `5e-5` |

The active decomposition is:

```text
n_envs = 8
n_steps_per_env = 1,000
rollout_batch = 8 * 1,000 = 8,000
minibatches_per_epoch = 8,000 / 100 = 80
updates_per_iteration = 80 * 20 = 1,600
iterations = 2,000,000 / 8,000 = 250
total_updates = 250 * 1,600 = 400,000
```

The remaining shared PPO settings are:

```yaml
algorithm:
  num_timesteps: 2000000
  gamma: 0.99
  gae_lambda: 0.95
  ppo_clip_range: 0.2
  target_kl: null
  schedule: fixed
  vec_env: subproc
  seed: 41
```

The paper states the 8,000-sample batch, 100-sample minibatch, 20 epochs, `5e-5` learning rate, `0.99` discount, `0.95` GAE lambda, and `0.2` clipping parameter. The computed iteration totals above are this benchmark's application of those values to a two-million-transition budget; they are not claimed as an explicit paper training budget.

## Environment Contract

The active jobs use:

```yaml
environment:
  simulator: metadrive
  env_id: MetaDrive-v0
  environment_config:
    use_render: false
    out_of_route_done: true
    on_continuous_line_done: false
    vehicle_config:
      lidar:
        num_others: 4
    agent_configs:
      default_agent:
        use_special_color: true
        spawn_lane_index: [">", ">>", 0]
        max_speed_km_h: 80.0
    map: 3
    traffic_density: 0.1
    random_traffic: false
    num_scenarios: 1000
    start_seed: 5000
    store_map: false
    horizon: 1000
    physics_world_step_size: 0.02
    decision_repeat: 5
```

One policy decision therefore spans:

```text
0.02 s * 5 physics steps = 0.10 s, or 10 Hz
```

`map: 3` denotes three generated road blocks in the public MetaDrive configuration. Internally, MetaDrive also creates a mandatory initial block.

`store_map: false` destroys a procedural map after it is unloaded and regenerates it from its deterministic scenario seed when needed again. This changes reset cost and RAM usage, but not the scenario seed distribution, action semantics, observation values for a given regenerated scenario, or reward definition.

The explicit ego limit of `80.0 km/h` equals MetaDrive's default physical ceiling. MetaDrive 0.4.3 accepts this sampled vehicle-dynamics override under `agent_configs.default_agent`, not under the shared `vehicle_config` mapping.

## Observation Contract

All active root-level MetaDrive jobs explicitly set:

```yaml
environment_config:
  vehicle_config:
    lidar:
      num_others: 4
```

MetaDrive deep-merges this override with its lidar defaults. The resulting state observation has 275 scalar dimensions:

```text
240 lidar ranges
  9 ego, lane, and action-history values
 10 navigation values for two future checkpoints
+ 16 nearby-vehicle values
---
275 dimensions
```

The 16 nearby-vehicle values describe the four closest lidar-detected vehicles. Each contributes normalized ego-relative:

```text
relative_x, relative_y, relative_vx, relative_vy
```

This matches the private observation constructed by MetaDrive's bundled PPO expert. The ordinary MetaDrive default uses `num_others: 0` and therefore exposes 259 dimensions.

For builtin PPO, the policy receives the 275-dimensional state directly. BDP policies receive the same state plus candidate features and a candidate mask. Thus 275 is the shared state dimension, not the complete BDP observation size.

Changing between 259 and 275 dimensions changes the policy architecture. Checkpoints trained under one contract cannot be loaded under the other.

## Speed Contract

The main benchmark retains the paper and demo physical ceiling:

```text
80 km/h = 22.222222 m/s
```

All Frenet-PID jobs cap target trajectories at `22.222222 m/s`. Native BDP applies the same cap to its action-conditioned trajectory descriptors. Native builtin PPO has no target-speed descriptor and is constrained only by the physical vehicle ceiling.

The Frenet speed offset remains `5.0 m/s` in this bundle. Target values are clipped at the physical maximum, so generated references cannot request speeds above what the simulated ego vehicle can produce under power.

The separate `scripts/job_scripts/benchmark_metadrive_12mps` bundle is a controller-stability ablation with a 43.2 km/h physical limit, a 12 m/s trajectory-target limit, and a 2.5 m/s Frenet speed delta.

## Action Contracts

The benchmark intentionally contains different execution MDPs. Comparisons are valid within matched builtin/BDP pairs.

| Execution mode | Candidate/action count | Meaning |
| --- | ---: | --- |
| continuous `native_controller` builtin | 2 continuous dimensions | Normalized steering and throttle/brake in `[-1, 1]` |
| `native_controller` builtin | 25 | Categorical selection from 5 steering by 5 throttle/brake values |
| `native_controller` BDP | 25 | Scores descriptors for the same 25 native actions |
| `frenet_pid` builtin | 15 | Categorical selection from 3 adaptive lane centers by 5 target speeds |
| `frenet_pid` BDP | 15 | Scores the same 15 Frenet trajectories |
| `frenet_pid_v2` builtin | 15 | Same action set, with nominal-state trajectory planning |
| `frenet_pid_v2` BDP | 15 | Scores the same nominal-state trajectories |

The paper's main single-agent PPO benchmark uses continuous normalized steering and throttle/brake. `metadrive_native_continuous_builtin_benchmark.yaml` implements that action contract while preserving the main bundle's other settings. It is an official-style action-space reference, not a like-for-like algorithm ablation against the finite-candidate BDP policies.

## Frenet PID Contract

The four active `frenet_pid` and `frenet_pid_v2` training jobs use the tuned adaptive pursuit tracker:

```yaml
pid:
  pid_controller_mode: adaptive_pursuit
  pid_lookahead_time_s: 0.7
  pid_min_lookahead_m: 3.0
  pid_max_lookahead_m: 12.0
  pid_pursuit_gain: 1.0
  pid_speed_preview_s: 0.4
  pid_speed_kp: 2.0
  pid_speed_ki: 0.2
  pid_speed_kd: 0.0
  pid_max_steering_rad: 0.7
  pid_max_accel_mps2: 5.0
  pid_max_decel_mps2: 8.0
```

Adaptive pursuit uses an interpolated metric lookahead and rear-axle bicycle geometry. Speed PI state persists across replans, with saturation protection and the correct 0.10-second MetaDrive policy interval. `tracking_diagnostics: true` records rollout mean errors in TensorBoard and `diagnostics/tracking_rollouts.csv`; it changes neither observation nor reward. Legacy steering fields retained in the YAML are inactive in this mode. The lateral-only visual preset remains a legacy controller reference.

See [controller measurements and reproduction](../../../docs/adaptive_pursuit_tracking.md) for the experiments and parameter semantics. The 12 m/s bundle retains its earlier legacy tracker and is no longer a speed-only ablation of these newly tuned 80 km/h jobs.

Native-controller jobs do not use this PID block.

`frenet_pid` and `frenet_pid_v2` are configuration-matched apart from run names and trajectory origin. `frenet_pid` generates each candidate set from the current actual ego state. `frenet_pid_v2` projects the actual ego pose onto the previously selected trajectory and generates the next candidate set from that nominal state. Both execute the selected trajectory through PID feedback against the actual vehicle state.

## Route-Continuous Frenet Contract

All active jobs that construct MetaDrive Frenet geometry use:

```yaml
candidate_sampler: frenet_route_continuous
```

This includes native BDP descriptors, Frenet-PID BDP, Frenet-PID-v2 BDP, both builtin Frenet-PID jobs, and the candidate visual check. The setting is interpreted by `bdp_benchmark`; no MetaDrive-specific route logic is added to `critic_based_rl`.

The route-continuous generator consumes the remaining bounded portion of the current route lane, selects geometrically connected successor lanes from `navigation.checkpoints`, and continues until the candidate horizon or route endpoint. Actual-state mode projects `x_actual` onto these bounded route lanes. Nominal-state mode projects `x_nominal`, allowing planning to enter a successor segment before the physical vehicle crosses the boundary.

Candidate count, ordering, trajectory horizon, feature width, and policy architecture remain unchanged. The values differ near segment boundaries, so checkpoints must use the sampler recorded in their effective configuration even though tensor dimensions match.

Legacy `frenet` and `native_action_frenet` retain single-current-lane extrapolation and remain supported for existing checkpoints and historical ablations.

## Reward Contract

The active YAMLs do not override the MetaDrive reward coefficients. They inherit the current `MetaDriveEnv` defaults:

```text
R = 1.0 * longitudinal_displacement
  + 0.1 * current_speed / maximum_speed
  + terminal_reward
```

The terminal outcome replaces the dense reward for that step:

```text
destination reached: +10
out of road:          -5
vehicle collision:    -5
object collision:     -5
```

This is the same reward formulation and coefficient set stated by the paper. It explicitly rewards both displacement and speed, so acceleration toward the vehicle limit can be an optimal learned behavior. There is no default acceleration, high-speed, steering-smoothness, or control-effort penalty.

Because the values are inherited rather than duplicated in every job YAML, the installed MetaDrive revision is part of the reproducibility contract.

## Termination Contract

The active jobs explicitly adopt the interactive demo's route-oriented overrides:

```yaml
out_of_route_done: true
on_continuous_line_done: false
```

This is not the current `MetaDriveEnv` default. Current defaults use `out_of_route_done: false` and `on_continuous_line_done: true`.

Under the adopted contract, an episode terminates when the vehicle leaves its planned route, leaves all drivable lanes, reaches the destination or horizon, or triggers an enabled collision termination. Merely contacting a continuous lane line is not independently terminal.

The paper describes termination for destination success, collision, and traffic-rule or undrivable-area violations, but does not map that description onto all modern MetaDrive boolean flags. The current explicit values should therefore be reported as a demo-aligned benchmark choice rather than an exact paper flag reproduction.

## Paper, Demo, and Current Differences

| Property | Paper | Bundled interactive demo/expert | Current benchmark |
| --- | --- | --- | --- |
| Policy action | Continuous steering and throttle/brake | Continuous PPO expert output | Continuous reference, 25 native actions, or 15 Frenet trajectories |
| State observation | 240 lidar plus ego/navigation state; exact modern total not stated | Private 275-dimensional observation | Explicit 275-dimensional shared state |
| Reward | Progress `1.0`, speed `0.1`, terminal `+10/-5` | Demo inherits MetaDrive reward but does not train | Inherits the same MetaDrive defaults |
| Training maps | Experiments vary training-set size up to 1,000 | Interactive demo requests 10,000 scenarios and `map: 4` | 1,000 scenarios, `map: 3` |
| Lane randomization | Procedural scenario diversity | Lane width and lane count enabled | Default fixed lane width/count |
| Termination flags | Exact modern flags not stated | Route done enabled; continuous-line done disabled | Same explicit demo-style overrides |
| Map retention | Historical implementation retained generated maps | Default behavior | Disabled to bound per-worker RAM |
| PPO settings | Published 8K/100/20/`5e-5` values | Exact expert training config is not packaged with the compressed weights | Adopts published optimizer values |

The interactive demo is an evaluation showcase, not the training recipe for its compressed expert checkpoint. The checkpoint exposes its network and observation contract, but the repository does not retain a complete effective training configuration for it.

## References

- [MetaDrive paper](https://arxiv.org/pdf/2109.12674)
- [MetaDrive action and policy documentation](https://metadrive-simulator.readthedocs.io/en/latest/action.html)
- [MetaDrive reward and termination documentation](https://metadrive-simulator.readthedocs.io/en/latest/reward_cost_done.html)
- [`drive_in_single_agent_env.py`](../../../../metadrive/metadrive/examples/drive_in_single_agent_env.py)
- [`numpy_expert.py`](../../../../metadrive/metadrive/examples/ppo_expert/numpy_expert.py)
- [`metadrive_env.py`](../../../../metadrive/metadrive/envs/metadrive_env.py)
- [`tracking.py`](../../../bdp_benchmark/common/tracking.py)
