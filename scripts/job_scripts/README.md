# Benchmark Jobs

The active jobs in this directory are the first matched native-controller
comparison:

```text
highway_native_builtin_benchmark.yaml
highway_native_bdp_frenet_benchmark.yaml
metadrive_native_builtin_benchmark.yaml
metadrive_native_bdp_frenet_benchmark.yaml
```

The active MetaDrive Frenet-PID comparisons are:

```text
metadrive_frenet_pid_builtin_benchmark.yaml
metadrive_frenet_pid_bdp_frenet_benchmark.yaml
metadrive_frenet_pid_v2_builtin_benchmark.yaml
metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml
```

Each pair selects the same 15 trajectories: three adaptive lane-center targets
by five target-speed offsets. `frenet_pid` plans from the actual state;
`frenet_pid_v2` plans from the projected nominal state while PID execution and
simulator outcomes remain actual-state based.

Run each matched execution-mode comparison sequentially:

```bash
./scripts/job_scripts/run_metadrive_bdp_frenet_comparison.sh
./scripts/job_scripts/run_metadrive_builtin_comparison.sh
```

The first script compares Frenet-PID v2 with native execution under BDP-Frenet
scoring. The second makes the same execution-mode comparison under builtin PPO.
Additional CLI arguments are forwarded to both jobs within each script.

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
- MetaDrive's generalization PPO example uses a 1000-step horizon, five rollout
  workers, approximately 20,000-step training batches, minibatch `256`, ten
  epochs, learning rate `3e-4`, `gamma=0.99`, and `lambda=0.95`. The jobs use
  `n_steps=4000` and five environments, giving a 20,000-sample rollout.

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
