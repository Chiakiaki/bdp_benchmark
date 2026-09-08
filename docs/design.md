# BDP Driving Benchmark Design

## Purpose

`bdp_benchmark` evaluates candidate-action reinforcement learning on two public
driving simulators without modifying either upstream repository:

- HighwayEnv
- MetaDrive

The benchmark separates two questions:

1. Does structured candidate scoring outperform an ordinary categorical policy
   when the simulator's existing action/controller semantics are preserved?
2. Does executing explicit Frenet trajectories through a tracking controller
   improve or change that result?

## Project Boundary

The benchmark is a standalone sibling project:

```text
bdp_benchmark/
  common/       simulator-independent trajectory and configuration code
  highway/      HighwayEnv adapter
  metadrive/    MetaDrive adapter
  scripts/      shared train/evaluate entry points and reproducible YAML jobs
  tests/        unit, parity, and simulator smoke tests
```

HighwayEnv, MetaDrive, and `critic_based_rl` remain independently maintained
dependencies. Simulator-specific state extraction and control stay in adapter
modules. Frenet mathematics, candidate contracts, feature encoding, and PID
tracking stay in `common`.

## Orthogonal Experiment Axes

### Simulator

```yaml
simulator: highway  # highway or metadrive
```

### Trajectory execution

```yaml
trajectory_execution_mode: native_controller  # native_controller or frenet_pid
```

`native_controller` preserves the simulator's original control transition. A
trajectory is an action-conditioned descriptor supplied to the BDP scorer, but
the selected raw action is sent to the simulator unchanged.

`frenet_pid` changes the action MDP. Five semantic actions generate five Frenet
reference trajectories, and a low-level tracker follows the selected reference
during the simulator substeps.

### Policy/candidate representation

```yaml
policy_mode: builtin  # builtin or bdp
candidate_sampler: one_hot  # one_hot or native_action_frenet/frenet
```

Each simulator and execution mode supports these comparisons:

| Execution | Policy | Candidate representation |
|---|---|---|
| native | built-in categorical | none |
| native | BDP | one-hot action label |
| native | BDP | action-conditioned Frenet descriptor |
| Frenet PID | built-in categorical | none |
| Frenet PID | BDP | one-hot semantic intent |
| Frenet PID | BDP | executable Frenet reference features |

HighwayEnv native mode has five actions. MetaDrive native mode preserves its
configured steering/throttle grid, which is 5 x 5 and therefore 25 actions by
default. Frenet PID mode exposes five ordered semantic actions in both
simulators: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, and `SLOWER`.

## Candidate Contract

The simulator adapter produces a candidate set with fixed shapes for one
environment:

```text
labels       [K]       raw action associated with each row
mask         [K]       one for valid rows
trajectory   [K,H,S]   internal trajectory state
features     [K,D]     normalized policy input
```

SB3 vector environments stack feature tensors into `[E,K,D]`, and the BDP
policy scores all environments and candidates in one Torch forward pass.

The common generator uses vectorized NumPy operations over `K` and horizon `H`.
It accepts leading batch dimensions so a future prebuilt vector environment can
reuse it. It does not clone simulator worlds, advance traffic, or perform
collision-oracle filtering.

## Action-Conditioned Frenet Descriptors

Candidate trajectories describe action semantics instead of claiming exact
dynamics prediction. The generator starts from the ego Frenet state and builds:

- a quintic lateral polynomial ending at a target lateral displacement;
- a quartic longitudinal polynomial ending at a target speed;
- samples at a configurable horizon and interval.

HighwayEnv semantic actions map to adjacent/current lane centers and discrete
target speeds. An unavailable lane change maps to an IDLE-equivalent target,
matching the upstream controller's effective behavior.

MetaDrive native actions are decoded into normalized steering and throttle.
All steering magnitudes remain distinguishable:

```text
target_d = current_d - steering * native_lateral_span_m
target_v = current_v + throttle * native_speed_span_mps
```

Targets are clipped by configured lateral and speed bounds. The selected raw
action is still executed by MetaDrive's `EnvInputPolicy`.

## Road Geometry and Features

Each adapter provides a local route reference sampled from its current lane or
navigation route. Frenet samples are warped through this geometry, then rotated
into the ego-local frame. This exposes curved-road geometry while avoiding
world-coordinate dependence.

The shared lateral convention is positive to the vehicle's semantic right.
HighwayEnv and MetaDrive use different coordinate handedness, so each adapter
provides the sign that maps its lane-local normal into this common convention.

The v1 feature schema contains, per sampled point:

```text
relative_x / position_scale
relative_y / position_scale
sin(relative_heading)
cos(relative_heading)
speed / speed_scale
```

The flattened candidate width is `D = 5 * H`.

## Frenet PID Execution

In `frenet_pid` mode, the same geometric generator produces five trajectories.
The selected trajectory is retained for one environment decision interval. A
shared tracker computes physical steering and longitudinal acceleration from
cross-track, heading, and speed errors. Simulator adapters convert those
physical commands into their native low-level action units.

HighwayEnv implements this with a sidecar subclass that applies tracker output
at each simulation frame while preserving upstream observation, reward,
termination, and traffic logic. MetaDrive uses a sidecar policy/controller that
persists the selected reference during `decision_repeat` physics steps.

## Fairness Rules

- Compare policies only within the same simulator and execution mode.
- Preserve upstream observations, rewards, traffic, seeds, and terminations.
- Give all policy variants the same raw action meanings and executor.
- Do not use future traffic states or collision filtering in candidate creation.
- Treat the raw MetaDrive environment as a separate reference baseline; its
  native 25-action MDP is not equivalent to the five-action Frenet PID MDP.
- Report that BDP Frenet includes structured action geometry. BDP one-hot is the
  ablation that isolates candidate-scoring architecture from that geometry.

## Configuration and Reproduction

One shared runner uses the precedence:

```text
CLI overrides > job YAML > code defaults
```

Every run stores source and resolved YAML, command arguments, package versions,
and simulator/execution/policy identifiers. Job YAML files cover the six
configurations for each initial simulator smoke environment.

## Validation

Tests cover polynomial boundary conditions, all action-to-target mappings,
ego-local feature invariance, candidate shapes, and invalid configuration.
Native-controller parity tests replay identical seeded action sequences through
the upstream environment and benchmark wrapper and compare observations,
rewards, and termination flags. Simulator smoke tests reset and step all policy
representations. Tiny PPO runs validate built-in, one-hot BDP, and Frenet BDP
training paths.
