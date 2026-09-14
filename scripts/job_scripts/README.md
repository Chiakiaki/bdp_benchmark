# Benchmark Jobs

The active MetaDrive benchmark jobs are grouped under `benchmark/`. The first
matched native-controller comparison is:

```text
highway_native_builtin_benchmark.yaml
highway_native_bdp_frenet_benchmark.yaml
benchmark/metadrive_native_builtin_benchmark.yaml
benchmark/metadrive_native_continuous_builtin_benchmark.yaml
benchmark/metadrive_native_bdp_frenet_benchmark.yaml
```

The active MetaDrive Frenet-PID comparisons are:

```text
benchmark/metadrive_frenet_pid_builtin_benchmark.yaml
benchmark/metadrive_frenet_pid_bdp_frenet_benchmark.yaml
benchmark/metadrive_frenet_pid_v2_builtin_benchmark.yaml
benchmark/metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml
```

Each pair selects the same 15 trajectories: three adaptive lane-center targets
by five target-speed offsets. `frenet_pid` plans from the actual state;
`frenet_pid_v2` plans from the projected nominal state while PID execution and
simulator outcomes remain actual-state based.

The active MetaDrive BDP and Frenet-PID jobs use
`candidate_sampler: frenet_route_continuous`. Their reference centerline follows
connected lane segments along the assigned MetaDrive route instead of
extrapolating the current lane past its endpoint. The legacy example jobs keep
`frenet` and `native_action_frenet` for old-checkpoint compatibility.

Run each matched execution-mode comparison sequentially:

```bash
./scripts/job_scripts/benchmark/run_metadrive_frenet_pid_comparison.sh
./scripts/job_scripts/benchmark/run_metadrive_bdp_frenet_comparison.sh
./scripts/job_scripts/benchmark/run_metadrive_builtin_comparison.sh
```

The first script compares BDP-Frenet with builtin PPO while both use actual-state
Frenet-PID execution. The second compares Frenet-PID v2 with native execution
under BDP-Frenet scoring. The third makes the same execution-mode comparison
under builtin PPO. Additional CLI arguments are forwarded to both jobs within
each script.

The standalone continuous-action reference preserves the 80 km/h native
builtin job's configuration while replacing `Discrete(25)` with MetaDrive's
default normalized `Box(2)` steering and throttle/brake action:

```bash
python3 scripts/train.py --config scripts/job_scripts/benchmark/metadrive_native_continuous_builtin_benchmark.yaml
```

The main bundle explicitly uses the MetaDrive and paper physical ceiling of
`80 km/h` (`22.222222 m/s`). Frenet target trajectories and native BDP
descriptors are capped at the same value. See
[`benchmark/config_contract.md`](benchmark/config_contract.md) for the complete
configuration contract and [`benchmark/paper_writing.md`](benchmark/paper_writing.md)
for the bundled expert's reproducibility limitation.

`benchmark_metadrive_12mps/` is a controlled stability ablation. It preserves
the main configuration while limiting the ego vehicle and candidate targets to
`12 m/s` (`43.2 km/h`) and reducing the Frenet target-speed delta to `2.5 m/s`.
It also includes `metadrive_native_continuous_builtin_benchmark.yaml` as the
continuous-action counterpart to the capped native categorical job.
Launch its matched comparisons with:

```bash
./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_bdp_frenet_comparison.sh
./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_builtin_comparison.sh
```

The mixed-road HighwayEnv comparison uses:

```text
highway_mixed_native_builtin_benchmark.yaml
highway_mixed_native_bdp_frenet_benchmark.yaml
highway_mixed_native_builtin_10hz_benchmark.yaml
highway_mixed_native_bdp_frenet_10hz_benchmark.yaml
highway_mixed_native_bdp_one_hot_10hz_benchmark.yaml
```

All mixed jobs assign `highway-fast-v0`, `merge-v1`, and `roundabout-v1`
to fixed subprocess workers in round-robin order. They use the same normalized
ego-relative `(5, 5)` Kinematics observation.

The original `*_benchmark.yaml` pair preserves each environment's official
1 Hz policy timing. The `*_10hz_benchmark.yaml` pair uses
`simulation_frequency: 20`, `policy_frequency: 10`, `gamma: 0.98`, and five
times as many transitions. Environment duration remains measured in seconds,
so a 10 Hz episode contains ten times as many policy decisions without changing
its simulated travel time.

`highway_mixed_native_bdp_one_hot_10hz_benchmark.yaml` is the controlled BDP
ablation: it replaces `native_action_frenet` candidate features with one-hot
action labels while preserving the BDP-Frenet job's other settings.

Run the complete 10 Hz pair sequentially, BDP-Frenet followed by builtin PPO:

```bash
./scripts/job_scripts/run_highway_mixed_native_comparison.sh
```

Any additional CLI arguments are forwarded to both jobs.

Each simulator's built-in and BDP job uses identical environment, seed, PPO,
rollout, and logging settings. The intended algorithm difference is:

```text
builtin: raw simulator observation/action policy
BDP-Frenet: same native action, plus action-conditioned Frenet candidate rows
```

`native_controller` means the candidate trajectory is a descriptor. The
selected native simulator action remains the executed action. No PID tracker or
Frenet-PID MDP is involved in these four jobs.

The settings are based on the official upstream examples:

- HighwayEnv's SB3 example uses six subprocess environments, `n_steps=128`,
  batch size `64`, ten epochs, learning rate `5e-4`, `gamma=0.8`, and `[256,
  256]` actor/critic layers. Its `2e4` timesteps are a short demonstration;
  these jobs use `2e6` for an actual comparison.
- The [MetaDrive paper](https://arxiv.org/abs/2109.12674) PPO baseline uses an
  8,000-sample train batch, minibatch `100`, 20 epochs, learning rate `5e-5`,
  `gamma=0.99`, `lambda=0.95`, and
  clipping at `0.2`. The active jobs reproduce that optimization schedule with
  `n_steps=2000` and four environments for lower host RAM use. This gives 80
  minibatches and 1,600 optimizer updates per rollout. `log_interval=1` writes
  TensorBoard values every 8,000 transitions. `target_kl` remains unset because
  RLlib's KL coefficient is not equivalent to SB3's early-stop threshold.

Run one job from the benchmark root:

```bash
export PYTHONPATH="/home/dell/project:${PYTHONPATH:-}"
python3 scripts/train.py \
  --config scripts/job_scripts/highway_native_bdp_frenet_benchmark.yaml
```

The two paired runs write TensorBoard/checkpoints below `logs/benchmark`.

Run the mixed BDP comparison with:

```bash
python3 scripts/train.py \
  --config scripts/job_scripts/highway_mixed_native_bdp_frenet_benchmark.yaml
```

During non-visual evaluation, `evaluate_episodes` is interpreted per variant;
the command reports one return summary for each road and one aggregate summary.
For a single rendered check, pass `--visual_check` and
`--visual_check_variant_index N`; the index wraps modulo the three configured
roads.
