# Highway Mixed-Road Training Design

## Goal

Train one HighwayEnv policy concurrently on three distinct road families while
keeping the observation and action contracts identical across all SB3 workers:

```yaml
environment_variants:
  - highway-fast-v0
  - merge-v1
  - roundabout-v1
```

The mixture targets road-shape robustness. Legacy `v0` and configurable
`generic` duplicates are excluded from this first benchmark. The connected-lane
`v1` variants are used for Merge and Roundabout because their roads contain
multiple connected segments; `highway-fast-v0` is valid because each highway
lane is one continuous segment.

## Configuration Contract

`environment_variants` is an optional list. An empty or omitted list preserves
the existing single-`env_id` behavior. The option is HighwayEnv-only in v1.

Mixed-road training uses one canonical ego-relative observation contract:

```yaml
environment_config:
  observation:
    type: Kinematics
    vehicles_count: 5
    features: [presence, x, y, vx, vy]
    normalize: true
    absolute: false
    order: sorted
```

The parser installs this block when it is omitted and rejects conflicting
observation settings. This prevents equal `(5, 5)` shapes from hiding different
coordinate semantics, particularly Roundabout's absolute-coordinate default.

## Worker Assignment

Each subprocess receives one fixed environment for its lifetime:

```text
environment_id(rank) = environment_variants[rank % variant_count]
```

Fixed assignment avoids reconstructing environments at episode reset and makes
the map distribution deterministic. Training requires at least one worker per
variant. A worker count not divisible by the variant count is allowed with a
warning because the first variants receive one extra worker.

SB3 remains responsible for deriving per-worker seeds from the configured base
seed. Builtin PPO and BDP jobs must use the same variant order, worker count,
and seed.

## Validation And Monitoring

Before subprocess launch, construct one final wrapped environment for each
variant and compare its observation and action spaces. Fail with the conflicting
variant IDs and spaces if they differ.

Every Highway benchmark step exposes the selected `environment_variant` in its
info dictionary. Per-environment Monitor output is grouped by variant, making
episode returns and lengths attributable to a road family. Mixed non-visual
evaluation creates one worker per variant and reports both aggregate and
per-variant returns. Visual checking remains single-environment and displays the
first configured variant unless the variant list is overridden explicitly.

## Reproducible Comparison

Add paired native-controller benchmark jobs with identical simulator, rollout,
seed, and model settings:

- builtin categorical PPO
- BDP with native-action Frenet candidate features

The only intended algorithmic difference is whether candidate-conditioned BDP
scoring is used. Generic geometry configurations and held-out road evaluation
remain a later extension.

