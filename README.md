# BDP Driving Benchmark

This standalone project evaluates `critic_based_rl` candidate-action policies
on HighwayEnv and MetaDrive. It does not modify either simulator repository.

The benchmark has two action MDPs:

- `native_controller`: preserve the simulator action and controller. Frenet
  trajectories are structured action descriptors only.
- `frenet_pid`: select an executable Frenet reference and follow it through a
  trajectory tracking controller. HighwayEnv exposes five references;
  MetaDrive exposes a 3 lane-center x 5 target-speed grid.
- `frenet_pid_v2`: plan from a nominal pose projected onto the previous selected
  trajectory while PID control still follows the reference using the actual pose.

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

Each simulator supports the six base controlled comparisons, plus an opt-in
nominal-planning v2 variant:

| Execution mode | Policy | `candidate_sampler` |
|---|---|---|
| native controller | SB3 categorical | ignored |
| native controller | BDP | `one_hot` |
| native controller | BDP | `native_action_frenet` |
| Frenet PID | SB3 categorical | ignored |
| Frenet PID | BDP | `one_hot` |
| Frenet PID | BDP | `frenet` |
| Frenet PID v2 | SB3 categorical | ignored |
| Frenet PID v2 | BDP | `one_hot` |
| Frenet PID v2 | BDP | `frenet` |

HighwayEnv native and Frenet-PID modes use the same five semantic actions:

```text
0 LANE_LEFT
1 IDLE
2 LANE_RIGHT
3 FASTER
4 SLOWER
```

MetaDrive native mode preserves its default 5 steering x 5 throttle grid. Its
Frenet-PID modes instead use 15 actions: for each of five target-speed offsets,
choose the left adjacent, current, or right adjacent lane center. Missing
boundary lanes are represented by a virtual center one current-lane width away,
which keeps the action space fixed. Comparisons are like-for-like only within
the same simulator and execution mode. `frenet_pid_v2` uses the same 15-action
space as `frenet_pid` but changes the candidate planning origin.

## Train

Run from the benchmark directory:

```bash
cd /home/dell/project/bdp_benchmark
export PYTHONPATH="/home/dell/project:${PYTHONPATH:-}"

python3 scripts/train.py \
  --config scripts/job_scripts/highway_native_bdp_frenet_benchmark.yaml
```

MetaDrive example:

```bash
python3 scripts/train.py \
  --config scripts/job_scripts/metadrive_frenet_pid_bdp_frenet_benchmark.yaml
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

In `frenet_pid_v2`, reset initializes `x_nominal = x_actual`. After an action
is selected, the next planning pose is the nearest point on that previous
selected trajectory. Candidate generation and features use `x_nominal`; PID
control, reward, and termination use `x_actual`. Legacy `frenet_pid` remains
actual-state based.

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

### Mixed HighwayEnv roads

One HighwayEnv checkpoint can collect rollouts from Highway, Merge, and
Roundabout concurrently:

```yaml
environment:
  simulator: highway
  environment_variants:
    - highway-fast-v0
    - merge-v1
    - roundabout-v1
  environment_config:
    observation:
      type: Kinematics
      vehicles_count: 5
      features: [presence, x, y, vx, vy]
      normalize: true
      absolute: false
      order: sorted
```

Worker rank selects a variant using `rank % len(environment_variants)`, and the
selected road remains fixed for that worker across resets. Omitting
`environment_variants` preserves the existing single-`env_id` behavior. Mixed
training requires at least one worker per variant and warns when workers cannot
be divided evenly. The parser enforces the shared ego-relative observation
contract, and the factory validates final wrapped spaces before subprocesses
start.

The reproducible builtin and BDP jobs are:

```text
scripts/job_scripts/highway_mixed_native_builtin_benchmark.yaml
scripts/job_scripts/highway_mixed_native_bdp_frenet_benchmark.yaml
```

Per-environment Monitor CSVs are grouped below `monitor/<variant>/`. Non-visual
mixed evaluation starts one worker per variant and interprets
`evaluate_episodes` as the number of episodes for each road. A visual check
remains single-environment and uses the first configured variant.

### Reward comparison plots

`scripts/plot_rewards.py` recursively reads every
`monitor/<variant>/rank_<N>/*.monitor.csv` file. Its default `combined` mode
pools all completed episodes from every worker and road into one
episode-weighted curve per run:

```bash
python3 scripts/plot_rewards.py \
  --runs logs/benchmark/<bdp-frenet-run> \
         logs/benchmark/<builtin-run> \
         logs/benchmark/<bdp-one-hot-run> \
  --labels "BDP Frenet" "Builtin PPO" "BDP One-Hot" \
  --curve_mode combined \
  --window_size 100 \
  --output logs/benchmark/highway_mixed_10hz_rewards.png
```

Use `--curve_mode per_variant` with the same arguments to create one comparison
PNG for each road. Given the output above, the generated names are:

```text
highway_mixed_10hz_rewards_highway-fast-v0.png
highway_mixed_10hz_rewards_merge-v1.png
highway_mixed_10hz_rewards_roundabout-v1.png
```

Combined mode weights each completed episode equally, matching the existing
`critic_based_rl.plotting` behavior. Per-variant mode pools only the parallel
workers assigned to that road. Both modes sort episode completions by monitor
time and use cumulative episode lengths as aggregate environment timesteps.

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

## Visual Check

`--visual_check` enables the native simulator renderer and overlays the current
candidate set. It forces one environment and dummy vectorization, and defaults
to deterministic selection. Use `--inference_mode stochastic` to inspect
sampled actions instead.

For a mixed HighwayEnv job, `--visual_check_variant_index` chooses which fixed
road to render. The non-negative index wraps modulo the configured variant
count, so index `4` selects `merge-v1` from the three-road list:

```bash
python3 scripts/evaluate.py \
  --config scripts/job_scripts/highway_mixed_native_bdp_frenet_benchmark.yaml \
  --visual_check \
  --visual_check_variant_index 4 \
  --model_path /path/to/PPO_BDP_final_model.zip
```

The default index is `0`, which preserves the original first-variant behavior.
The selected requested index, resolved index, and environment ID are printed
before evaluation starts.

HighwayEnv example:

```bash
SDL_VIDEODRIVER=x11 python3 scripts/evaluate.py \
  --config scripts/job_scripts/visual_check_highway_bdp.yaml \
  --visual_check \
  --model_path /path/to/PPO_BDP_final_model.zip
```

MetaDrive example:

```bash
python3 scripts/evaluate.py \
  --config scripts/job_scripts/visual_check_metadrive_bdp.yaml \
  --visual_check \
  --model_path /path/to/PPO_BDP_final_model.zip
```

The overlay uses the existing HighwayEnv pygame viewer or MetaDrive Panda3D
renderer. It does not create a separate visualization application. Candidate
colors are red for relatively low scores, neutral near the current-set median,
and blue for relatively high scores. The selected candidate is always green
and thicker. In Frenet-PID mode, a yellow dot marks the exact nearest-plus-
lookahead waypoint used by the PID tracker.

For structured `frenet` or `native_action_frenet` BDP models, the plotted
trajectory features are the same rows passed to the policy. For `one_hot` and
built-in categorical models, the policy receives action-label features/logits;
the plotted trajectories are the corresponding action geometry, not input
geometry that the model processed.

The score helper returns PyTorch categorical distribution scores from the same
forward pass used to select the action. PyTorch exposes these as normalized log
probability logits; their ordering and relative differences are what the color
map visualizes.

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

`relative_y` is positive toward the vehicle's semantic right. Each simulator
adapter converts its native lane-coordinate sign into that shared convention.

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

In MetaDrive Frenet-PID modes, the shared generator instead receives a 3 x 5
target grid. Actual adjacent centers are projected through
the peer lanes associated with `navigation.current_lane.index` in MetaDrive's
road network; at a road boundary, the missing center is extrapolated by the
current lane width. This remains valid while route-reference lanes are updating
at a segment transition. The five speed rows are:

```text
current_speed + [-1.0, -0.5, 0.0, 0.5, 1.0] * frenet_speed_delta_mps
```

Lateral choice changes fastest within each speed row. Structured BDP input is
`[E, 15, 5*H]`; with `H=11`, this is `[E, 15, 55]`. One-hot BDP input is
`[E, 15, 15]`.

Native candidate geometry uses the actual chassis heading and scalar speed to
initialize its Frenet motion direction. This is separate from the simulator's
instantaneous velocity vector, which may have a slip angle during a turn. In
`frenet_pid_v2`, this actual-state rule applies only to the control/tracking
side; candidate planning remains based on `x_nominal`.

In `frenet_pid_v2`, candidate paths are drawn from the nominal planning pose,
while the PID waypoint is computed from the actual vehicle pose. A small gap
between the vehicle and the path is therefore an intentional visualization of
tracking error.

## Reproducible Jobs

The `scripts/job_scripts` directory contains the initial policy comparisons for
HighwayEnv and MetaDrive, plus v2 nominal-planning examples. To change the
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
combinations, tiny PPO checkpoint smoke tests, score extraction, and native
renderer overlay smoke tests.
