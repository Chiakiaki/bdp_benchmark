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
