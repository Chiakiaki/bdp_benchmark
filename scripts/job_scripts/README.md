# Benchmark Jobs

The active jobs in this directory are the first matched native-controller
comparison:

```text
highway_native_builtin_benchmark.yaml
highway_native_bdp_frenet_benchmark.yaml
metadrive_native_builtin_benchmark.yaml
metadrive_native_bdp_frenet_benchmark.yaml
```

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
