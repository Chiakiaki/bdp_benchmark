# MetaDrive More-Algorithm Comparisons: 80 km/h

This bundle extends the five TRPO/TRPOBDP comparisons with builtin SB3 DQN
and A2C. The original PPO bundle is unchanged. Algorithm parsing, factories,
loading and score inference are owned by critic_based_rl, not the simulator.

## Jobs

| Algorithm | Job basename | Execution |
| --- | --- | --- |
| TRPOBDP | metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml | Nominal Frenet-PID-v3, BDP scoring |
| TRPO | metadrive_frenet_pid_v3_builtin_benchmark.yaml | Same trajectories, categorical policy |
| TRPOBDP | metadrive_native_bdp_frenet_benchmark.yaml | Native actions, BDP descriptors |
| TRPO | metadrive_native_builtin_benchmark.yaml | Native discrete actions |
| TRPO | metadrive_native_continuous_builtin_benchmark.yaml | Native continuous actions |
| DQN | metadrive_native_dqn_builtin_benchmark.yaml | Native discrete actions |
| DQN | metadrive_frenet_pid_v3_dqn_builtin_benchmark.yaml | Nominal Frenet-PID-v3 discrete actions |
| A2C | metadrive_native_a2c_builtin_benchmark.yaml | Native discrete actions |
| A2C | metadrive_frenet_pid_v3_a2c_builtin_benchmark.yaml | Nominal Frenet-PID-v3 discrete actions |

The new jobs match the corresponding PPO builtin job's environment, 275D
observation, reward, PID, speed limits and candidate geometry. DQN and A2C
do not use BDP scoring. Both select from 25 discrete actions per execution mode.

## Training Contract

Shared settings: fixed learning rate 5e-5, gamma 0.99, 2M transitions,
8 subprocess workers, network widths [256, 256], the source activation,
and unchanged logging/checkpoint frequencies. These are matched data-budget
comparisons, not equally tuned algorithms or equal optimizer-update budgets.

- TRPO: all five existing YAMLs are unchanged. Critic batch 100, 20 critic
  passes, KL 0.01, 1000 steps/worker: 8000 policy samples and 1600 critic
  optimizer steps per rollout. Learning rate controls the critic only.
- A2C: retains 1000 steps/worker and GAE 0.95 as shared rollout settings.
  This means one full-batch actor/critic update per 8000 transitions, or
  250 updates over 2M transitions. This is intentionally NOT the official
  short-rollout default of 5 steps. There is no minibatch, repeated epoch,
  or TRPO critic-pass parameter. RMSprop and advantage normalization are
  explicitly set to SB3 defaults.
- DQN: replay minibatch 100; no GAE or on-policy rollout length.
  Explicit defaults: replay capacity 1M transitions, warmup 100 transitions,
  train frequency 4 vector steps, 1 gradient step, hard target update every
  10000 transitions, epsilon 1.0 to 0.05 over the first 10% of training.
  At 8 workers, each collection block is 32 transitions. Shared gradient
  clipping is 0.5 (overrides DQN's default 10). policy_layers defines the
  Q-network; actor/critic value_layers and feature-sharing do not apply.
  The replay observations alone use about 2.2 GB with 275 float32 dimensions;
  capacity is total across workers, not per worker.

DQN checkpoints save the model, not replay contents. Reloading is not an
exact continuation with the old replay distribution. Visual checks display
DQN Q-values; actor-critic algorithms display categorical logits.

## Launch After Review

From the repository root:

```bash
./scripts/job_scripts/benchmark_more_algorithm/run_metadrive_more_algorithm_comparison.sh
```

From this folder: `./run_metadrive_more_algorithm_comparison.sh`.
The retained `run_metadrive_trpo_comparison.sh` runs only the original five.
Both resolve paths independently of the working directory, forward CLI
overrides, check dependencies before simulator creation, and stop on failure.

Select Python with BDP_BENCHMARK_PYTHON if needed. DQN/A2C are part of SB3;
TRPO requires compatible sb3-contrib. No dependencies are auto-installed.
No benchmark run was launched while preparing these jobs.
