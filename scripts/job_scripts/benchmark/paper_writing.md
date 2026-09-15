# MetaDrive Demo Expert Reproducibility Note

## Scope

MetaDrive distributes a successful continuous-control PPO expert through `metadrive/examples/ppo_expert/expert_weights.npz`. The interactive `drive_in_single_agent_env` demo can execute this policy, but the repository does not contain a complete training bundle that can reproduce the checkpoint from initialization.

This distinction must be stated when comparing models from `bdp_benchmark` with the bundled expert or with results from the MetaDrive paper.

## Available Artifacts

The repository preserves:

- the compressed actor weights;
- the 275-dimensional expert observation construction;
- the two-dimensional continuous action contract;
- an evaluation helper for a historical RLlib checkpoint;
- a conversion script that extracted actor tensors from that checkpoint.

The actor has the following structure:

```text
observation[275]
  -> tanh MLP[256, 256]
  -> mean[2] and log_std[2]
  -> continuous steering and throttle/brake
```

The historical evaluation helper refers to `checkpoint_417/checkpoint-417`, `GeneralizationRacing`, a 1,000-step horizon, and 10,000 evaluation scenarios. That original RLlib checkpoint is not present in the repository.

## Missing Reproduction Information

The compressed `.npz` file does not retain:

- optimizer or critic state;
- training learning rate;
- rollout and minibatch sizes;
- number of PPO epochs;
- total training transitions;
- complete map, traffic, reward, and termination overrides;
- random seeds and software environment;
- the exact effective configuration stored with `checkpoint_417`.

The historical evaluation helper sets `lr=0.0`; this is an inference setting and must not be interpreted as the expert's training learning rate.

## Provenance Limitation

Git history shows that the packaged expert was updated in December 2020. The repository's generalization PPO training example was added in September 2021. The later training example and the hyperparameter table in the MetaDrive paper therefore cannot be asserted to be the recipe that produced the packaged expert.

The paper's PPO parameters can be reproduced as an independent published baseline. They are not checkpoint provenance for `expert_weights.npz`. This repository's `metadrive_native_continuous_builtin_benchmark.yaml` uses the paper-style continuous action contract while holding the local benchmark environment and PPO settings fixed.

## Required Paper Language

Recommended wording:

> MetaDrive provides a pretrained continuous-control PPO expert and its inference-time observation adapter, but does not provide the original complete training checkpoint or effective training configuration. We therefore use the expert as an evaluation-time behavioral reference rather than claim exact reproduction of its training procedure. Our PPO settings follow the hyperparameters reported in the MetaDrive paper as an independent baseline.

Do not write that the bundled expert was trained with this repository's job scripts or that the paper's published PPO table generated the bundled checkpoint.

## Interpretation of Expert Behavior

The expert uses MetaDrive's 80 km/h physical vehicle ceiling but has no explicit target-speed parameter. Its slower motion on curves is learned continuous-control behavior. It may arise from long-horizon optimization over progress, route departure, and collision outcomes, aided by fine-grained throttle control and the private 275-dimensional observation.

This behavior does not establish a hidden configured speed limit. Any comparison should report observed speed statistics separately from the simulator's physical maximum.

## References

- [MetaDrive paper](https://arxiv.org/pdf/2109.12674)
- [Historical PGDrive expert update](https://github.com/decisionforce/pgdrive/pull/181)
- [`numpy_expert.py`](../../../../metadrive/metadrive/examples/ppo_expert/numpy_expert.py)
- [`_evaluate_expert.py`](../../../../metadrive/metadrive/examples/ppo_expert/_evaluate_expert.py)
- [`remove_useless_state.py`](../../../../metadrive/metadrive/examples/ppo_expert/remove_useless_state.py)

## Tracking, Tire Slip, and the 12 m/s Experiment

The adopted vehicle executes front-wheel steering and throttle/brake through
Bullet physics. Continuous and discrete native inputs share the same normalized
[-1, 1] limits, steering-angle limit, and brake-force mapping. The 25-action grid
already contains full braking. Its coarser control resolution must not be
reported as a weaker physical brake or steering bound.

The inspected vehicle execution path has no explicit fixed physical yaw-rate
clamp. There is a clipped yaw-rate magnitude in `StateObservation`; that is an
observation normalization, not a dynamics constraint. Steering imposes an
approximate curvature bound. Even with admissible steering, lateral tire slip
and steering/vehicle transients can prevent exact trajectory tracking at high
speed. In the kinematic approximation, required lateral acceleration is
`a_lat = v^2 * curvature`; steering margin alone does not guarantee tire grip.

Our adopted 275-dimensional observation does not explicitly supply ego lateral
velocity, sideslip angle, or tire-slip state. The unchanged reward has no
dedicated slip penalty: it rewards road progress and normalized speed, with
destination, collision and road-departure outcomes. Road-relative pose, speed,
heading changes and surrounding geometry provide indirect feedback, so it would
be incorrect to claim that slip is entirely unobservable or unpenalized. This is
a limitation of the observation/reward contract used in these experiments, not a
claim that all MetaDrive observation classes or configurable rewards lack these
signals. Tracking error alone does not prove that slip caused a failed episode;
confirming that cause requires measured lateral velocity/sideslip diagnostics.

Recommended wording:

> High-speed tracking is affected by actuator limits and tire dynamics, while
> the adopted observation and reward provide no explicit sideslip supervision.
> We therefore include a 12 m/s stability experiment to reduce lateral dynamic
> demand without adding slip observations or modifying the reward formulation.
> Lower tracking error or higher navigation performance is an experimental
> hypothesis, not guaranteed by the speed ceiling.

Use [the 12 m/s bundle](../benchmark_metadrive_12mps/config_contract.md) as the
recommended next stability experiment; retain this 80 km/h bundle as a separate
reference. Native continuous and categorical policies are unchanged in actuator
authority. All seven 12 m/s jobs share an ego engine-force cutoff of 43.2 km/h;
NPC settings remain unchanged. The speed ceiling is not an instantaneous
velocity clamp, and MetaDrive normalizes observed speed and the speed reward by
that ceiling, so comparisons across ceilings must acknowledge this scaling.

The four 12 m/s Frenet-PID jobs now additionally opt into two constant-curvature
families crossed with the five speed levels (25 actions total). These use 0.9
of the smaller vehicle/PID steering-angle limit. Thus the default 12 m/s bundle
is a speed-and-candidate-coverage experiment, not a speed-only ablation against
the 15-action 80 km/h Frenet jobs. Disable
`frenet_include_curvature_candidates` to isolate the speed effect. Compare BDP
and builtin PPO within each matching execution mode and bundle. Extra extreme
curves do not by themselves guarantee lane keeping, and a 0.9 steering fraction
does not guarantee that all curves can be followed at 12 m/s.
