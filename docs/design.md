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
trajectory_execution_mode: native_controller  # native_controller, frenet_pid, or frenet_pid_v2
```

`native_controller` preserves the simulator's original control transition. A
trajectory is an action-conditioned descriptor supplied to the BDP scorer, but
the selected raw action is sent to the simulator unchanged.

`frenet_pid` changes the action MDP. HighwayEnv generates five semantic Frenet
references. MetaDrive generates 15 references from three adaptive lane-center
targets and five target-speed offsets. A low-level tracker follows the selected
reference during simulator substeps. `frenet_pid_v2` uses the same
simulator-specific action space but changes only the planning origin to a
nominal projected state.

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
| Frenet PID v2 | built-in categorical | none |
| Frenet PID v2 | BDP | one-hot semantic intent |
| Frenet PID v2 | BDP | executable Frenet reference features |

HighwayEnv native and Frenet-PID modes have five actions. MetaDrive native mode
preserves its configured steering/throttle grid, which is 5 x 5 and therefore
25 actions by default. MetaDrive Frenet-PID modes have 15 actions: three
lane-center targets change fastest within each of five target-speed rows.

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

Normalized steering bounds the lateral shift to the configured span, and target
speed is clipped to configured minimum and maximum values. The selected raw
action is still executed by MetaDrive's `EnvInputPolicy`.

MetaDrive Frenet-PID replaces the five steering-derived lateral targets with
three lane-center targets: left adjacent, current, and right adjacent. Existing
adjacent centers are obtained from the current lane's road-network peers, not
the route-reference list, because current localization and route references can
advance on different frames at a segment transition. At a boundary, a virtual
center is extrapolated by one current-lane width so the action space remains
fixed. These three targets are paired with five normalized speed offsets,
producing 15 executable references. `frenet_pid` and `frenet_pid_v2` share this
action contract.

For actual-state candidate generation, the initial Frenet rates use the direct
chassis heading and scalar speed. This keeps the visible descriptor aligned
with a turned vehicle even when its instantaneous velocity has a slip angle.
The nominal v2 path continues to use only its stored nominal heading and speed.

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

HighwayEnv implements this with a sidecar wrapper that applies tracker output
at each simulation frame while preserving upstream observation, reward,
termination, and traffic logic. MetaDrive uses a sidecar policy/controller that
persists the selected reference during `decision_repeat` physics steps.

## Actual and Nominal Planning States

`native_controller` and legacy `frenet_pid` use `x_actual` for candidate
generation and feature construction. In `frenet_pid_v2`, each environment
stores one previous selected trajectory:

```text
reset:       x_nominal = x_actual
next plan:   x_nominal = nearest_point(previous_selected, x_actual.xy)
planning:    generate candidates/features from x_nominal
control:     PID compares selected reference with x_actual
feedback:    reward and termination remain actual-state based
```

The initial nominal heading may equal the actual heading at reset. Once a
previous trajectory exists, nominal heading and speed come from that trajectory;
the actual chassis heading is not used to generate the next candidate set.
Actual heading remains available to the PID tracking error. Projection is a
nearest-point operation in v1, with no window or tracking-error rejection guard.

## Fairness Rules

- Compare policies only within the same simulator and execution mode.
- Preserve upstream observations, rewards, traffic, seeds, and terminations.
- Give all policy variants the same raw action meanings and executor.
- Do not use future traffic states or collision filtering in candidate creation.
- Treat the raw MetaDrive environment as a separate reference baseline; its
  native 25-action MDP is not equivalent to the 15-action Frenet PID MDP.
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

## Visual Check

The optional `--visual_check` path is evaluation-only. It forces one dummy
environment, uses deterministic categorical selection by default, and can be
switched to stochastic selection with `--inference_mode stochastic`. It calls
the critic-based inference helper once per decision, stores the exact candidate
payload, updates a simulator-native overlay, renders, and only then steps the
environment.

HighwayEnv draws lines through its existing pygame `EnvViewer` callback. MetaDrive
uses its existing Panda3D world line and point drawers. Overlay nodes are cleared
before reset so they cannot interfere with MetaDrive's world cleanup.

Score colors are current-candidate relative: low red, median neutral, high blue;
the selected row is overridden to green. For Frenet-PID, the waypoint marker is
the tracker's pure nearest-plus-lookahead preview and therefore matches the
target used when the action is stepped. No visual score forward pass is added to
training.
