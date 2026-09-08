# BDP Driving Benchmark

This standalone project evaluates `critic_based_rl` candidate-action policies
on HighwayEnv and MetaDrive. It does not modify either simulator repository.

The benchmark has two action MDPs:

- `native_controller`: preserve the simulator action and controller. Frenet
  trajectories are structured action descriptors only.
- `frenet_pid`: select one of five generated Frenet references and execute it
  through a trajectory tracking controller.

See [docs/design.md](docs/design.md) for the complete architecture and fairness
rules.

## Installation

The tested host stack uses Python 3.12, Gymnasium 1.2.x, Stable-Baselines3 2.8,
and Torch 2.10. Install the local simulator checkouts and this package in
editable mode:

```bash
cd /home/dell/project

python3 -m pip install --user --break-system-packages -e HighwayEnv
python3 -m pip install --user --break-system-packages -e metadrive
python3 -m pip install --user --break-system-packages -e bdp_benchmark

export PYTHONPATH="/home/dell/project:${PYTHONPATH:-}"
```

`critic_based_rl` is imported from the common project root. Keep these sibling
directories together, or add their common parent to `PYTHONPATH`:

```text
/home/dell/project/
  bdp_benchmark/
  critic_based_rl/
  HighwayEnv/
  metadrive/
```

## Experiment Matrix

Each simulator supports six controlled comparisons:

| Execution mode | Policy | `candidate_sampler` |
|---|---|---|
| native controller | SB3 categorical | ignored |
| native controller | BDP | `one_hot` |
| native controller | BDP | `native_action_frenet` |
| Frenet PID | SB3 categorical | ignored |
| Frenet PID | BDP | `one_hot` |
| Frenet PID | BDP | `frenet` |

HighwayEnv native mode preserves its five official semantic actions. MetaDrive
native mode preserves its configured steering/throttle grid, which is 25
actions for the default 5 x 5 grid. Both Frenet-PID modes use this order:

```text
0 LANE_LEFT
1 IDLE
2 LANE_RIGHT
3 FASTER
4 SLOWER
```

Comparisons are like-for-like only within the same simulator and execution
mode. The five-action Frenet-PID MetaDrive environment is intentionally not the
same MDP as raw MetaDrive's 25-action environment.

## Train

Run from the benchmark directory:

```bash
cd /home/dell/project/bdp_benchmark
export PYTHONPATH="/home/dell/project:${PYTHONPATH:-}"

python3 scripts/train.py \
  --config scripts/job_scripts/highway_native_bdp_frenet.yaml
```

MetaDrive example:

```bash
python3 scripts/train.py \
  --config scripts/job_scripts/metadrive_frenet_pid_bdp_frenet.yaml
```

The decision and physics rates remain simulator-owned and can be set inside
`environment_config`. HighwayEnv uses:

```yaml
environment_config:
  simulation_frequency: 15  # physics/controller updates per second
  policy_frequency: 1       # calls to env.step() per second
```

MetaDrive uses:

```yaml
environment_config:
  physics_world_step_size: 0.02
  decision_repeat: 5        # env.step() rate is 1 / (0.02 * 5) = 10 Hz
```

In Frenet-PID mode, one high-level action selects a trajectory at each
`env.step()`, while the tracker runs at every internal physics/controller tick.

CLI values override YAML values. For a short smoke run:

```bash
python3 scripts/train.py \
  --config scripts/job_scripts/highway_native_bdp_frenet.yaml \
  --num_timesteps 32 \
  --n_envs 1 \
  --vec_env dummy \
  --trpo_timesteps_per_batch 16 \
  --batch_size 16 \
  --ppo_epochs 1 \
  --save_freq 0
```

Each run stores:

```text
logs/<timestamp>_<experiment>/
  config.yaml             source job YAML
  resolved_config.yaml    YAML plus effective CLI overrides
  reproduction.yaml       arguments, package versions, and Git revision
  monitor/                per-environment episode returns
  models/                 TensorBoard events and SB3 checkpoints
```

## Evaluate

Use the same job YAML that created the checkpoint:

```bash
python3 scripts/evaluate.py \
  --config scripts/job_scripts/highway_native_bdp_frenet.yaml \
  --model_path logs/<run>/models/PPO_BDP_final_model.zip \
  --evaluate_episodes 20 \
  --n_envs 1 \
  --vec_env dummy
```

Add `--render_mode human` for an interactive visual evaluation. MetaDrive must
use `vec_env: subproc` whenever `n_envs > 1`, because its engine is process
global.

## Candidate Features

The shared generator builds lateral quintic and longitudinal quartic curves in
Frenet coordinates, maps them through the current route geometry, and encodes
each sample in the ego-local frame:

```text
relative_x / position_feature_scale_m
relative_y / position_feature_scale_m
sin(relative_heading)
cos(relative_heading)
speed / speed_feature_scale_mps
```

With `trajectory_sample_count: H`, candidate width is `5 * H`. Generation is
vectorized across all candidates and horizon samples. SB3 stacks worker output
as `[E, K, D]`, after which the BDP policy scores it in one Torch batch.

In native MetaDrive mode, all steering magnitudes remain distinct:

```text
target_d = current_d - steering * native_lateral_span_m
target_v = current_speed + throttle * native_speed_span_mps
```

MetaDrive steering is positive to the left while its lane-local lateral value
is positive to the right, which accounts for the minus sign. These are
learnable geometric descriptors, not claims of exact future vehicle dynamics.
The selected native action is still executed unchanged.

## Reproducible Jobs

The `scripts/job_scripts` directory contains all twelve initial configurations:
six policy comparisons for HighwayEnv and six for MetaDrive. To change the
HighwayEnv task, set `env_id` to another compatible five-action environment,
such as `merge-v1` or `roundabout-v1`. MetaDrive currently supports
`MetaDrive-v0`; `ScenarioEnv-v0` is also registered by the adapter but requires
the corresponding scenario dataset configuration.

## Tests

```bash
cd /home/dell/project/bdp_benchmark
export PYTHONPATH="/home/dell/project:${PYTHONPATH:-}"
python3 -m pytest -q
```

The suite includes polynomial constraints, coordinate invariance, HighwayEnv
native-transition parity, both MetaDrive execution modes, all policy wrapping
combinations, and tiny PPO checkpoint smoke tests.
