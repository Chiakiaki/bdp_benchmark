# ICRA 2027 PPO vs TRPO: MetaDrive 12 m/s (BDP Frenet-PID-v3)

This bundle holds the two arms behind the ICRA 2027 PPO--TRPO comparison at the
12 m/s speed cap. Both arms are the paper's headline method: the BDP candidate
scorer over Frenet-PID-v3 trajectories (`\pb{}` in the manuscript), and they
differ **only** in the reinforcement-learning algorithm.

| Arm | Job YAML | Algorithm | Purpose |
| --- | --- | --- | --- |
| PPO | `metadrive_frenet_pid_v3_bdp_frenet_ppo_12mps_benchmark.yaml` | PPO | Published baseline arm |
| TRPO | `metadrive_frenet_pid_v3_bdp_frenet_trpo_12mps_benchmark.yaml` | TRPO | Comparison arm added for the training table |

Provenance: the PPO job is copied verbatim from
`scripts/job_scripts/benchmark_metadrive_12mps/metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml`
(the bundle that produced the paper's 12 m/s column); the TRPO job is copied
verbatim from
`scripts/job_scripts/benchmark_metadrive_12mps_more_algorithm/metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml`.
Only `run_name` differs, so each arm is reproducible from the published
contracts. Keep this folder in sync with those sources rather than editing
knobs here.

## Matched training contract

Identical in both arms:

- MetaDrive, 43.2 km/h (`12.0 m/s`) cap via `agent_configs.default_agent.max_speed_km_h`
- 275-D observation, unchanged reward, termination, PID (`adaptive_pursuit`) settings
- `frenet_route_continuous` candidates, `frenet_include_curvature_candidates: true` (K = 25)
- 2,000,000 transitions, 8 subprocess workers, seed 41
- fixed learning rate 5e-5, `gamma 0.99`, `gae_lambda 0.95`, networks `[256, 256]`

Deliberately different (this is the intervention, not a tuning difference):

| Knob | PPO | TRPO |
| --- | --- | --- |
| rollout per worker | 1000 | 1000 |
| optimiser schedule | 20 epochs, clip 0.2 | KL 0.01, CG 10 steps, damping 0.01 |
| critic | minibatch 100 x 20 epochs | 20 full-batch passes |
| learning rate | critic only | critic only (policy step from KL/CG) |

This is a matched **data-budget** comparison (2M transitions each), not matched
optimiser-update counts: PPO takes ~2,000 minibatch epochs while TRPO takes
250 policy steps. Report it as such.

## Resource notes (measured on the reference laptop)

| Quantity | Value |
| --- | --- |
| idle machine baseline | ~6 GB used of 15 GB |
| 8 workers, Frenet-PID-v3 | ~10.6 GB peak used (~4.6 GB above baseline) |
| 1 worker (dummy vec) | ~2.4 GB peak RSS |
| throughput, 8 workers | 300--470 fps on smoke budgets |

Run the arms **sequentially** (the launcher does). Two 8-worker runs at once
would exceed this host's memory. On a smaller host pass `--envs 6` or `--envs 4`;
that changes the wall-clock and the worker count, so record the deviation when
reporting.

## Reproduce

```bash
export PROJECT_ROOT="/path/to/project"      # holds bdp_benchmark/, critic_based_rl/, ...
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT/bdp_benchmark"

# Both arms, sequential, 8 workers (~2-5 h each depending on host)
./scripts/job_scripts/icra2027_trpo_vs_ppo_12mps/run_icra2027_trpo_vs_ppo.sh

# Or one arm directly
python3 scripts/train.py \
  --config scripts/job_scripts/icra2027_trpo_vs_ppo_12mps/metadrive_frenet_pid_v3_bdp_frenet_trpo_12mps_benchmark.yaml
```

Each run writes `logs/benchmark/<timestamp>_<run_name>/` with `config.yaml`,
`resolved_config.yaml`, `reproduction.yaml`, `models/` and `monitor/`.

## Comparative metric

The paper reports a fixed training-tail return (the final 1.6M--2.0M completed
transitions, episode-weighted mean $\pm$ episode SD), not the best episode.
Reproduce that summary with the bundled helper:

```bash
python3 scripts/job_scripts/icra2027_trpo_vs_ppo_12mps/summarize_tail.py \
  logs/benchmark/<timestamp>_icra2027_12mps_bdp_frenet_v3_ppo \
  logs/benchmark/<timestamp>_icra2027_12mps_bdp_frenet_v3_trpo \
  --tail 0.2 --labels PPO-BDP TRPO-BDP
```

`--tail 0.2` = final 20% of transitions (1.6M--2.0M of 2M), matching the
manuscript's interval. For the learning-curve figure, use
`scripts/plot_rewards.py --curve_mode combined` on the same run directories.

## Evaluating trained policies

Use the arm's YAML with the matching checkpoint, per the main README:

```bash
python3 scripts/evaluate.py \
  --config scripts/job_scripts/icra2027_trpo_vs_ppo_12mps/metadrive_frenet_pid_v3_bdp_frenet_trpo_12mps_benchmark.yaml \
  --model_path logs/benchmark/<run>/models/TRPO_BDP_final_model.zip \
  --evaluate_episodes 100 --n_envs 1 --vec_env dummy
```

MetaDrive must use `--vec_env subproc` whenever `n_envs > 1`.

## Recorded results

First matched run of this bundle, executed 2026-11-25 on the reference laptop
(RTX 5060 Laptop 8 GB, 20 cores, 15 GB RAM, 8 subprocess workers, seed 41):

| Arm | Run directory | Wall clock |
| --- | --- | --- |
| PPO | `logs/benchmark/20261125_174049_icra2027_12mps_bdp_frenet_v3_ppo` | 1 h 10 min |
| TRPO | `logs/benchmark/20261125_185040_icra2027_12mps_bdp_frenet_v3_trpo` | 1 h 01 min |

Training-tail return (final 20% of transitions = 1.6M--2.0M, episode-weighted):

| Arm | Episodes | Tail episodes | Tail mean | Tail SD | Tail median |
| --- | --- | --- | --- | --- | --- |
| PPO-BDP | 5,861 | 1,176 | 329.15 | 116.5 | 337.9 |
| TRPO-BDP | 5,998 | 1,118 | 329.21 | 126.4 | 338.7 |

Interpretation: at an equal 2M-transition budget the two algorithms are
effectively tied on the training tail (difference 0.06 return, far inside the
~120 episode SD). The learning curves differ in shape: TRPO rises faster up to
about 0.6M transitions, PPO holds a slightly higher plateau between 0.9M and
1.8M, and both end near 330. This is one training run per arm, so it is a
matched-budget point estimate, not a variance estimate across seeds.

The PPO arm reproduces the manuscript's 12 m/s `\pb{}` figure of 324.2 ± 125.2
within the expected single-run spread.

Figure: `logs/benchmark/icra2027_ppo_vs_trpo_12mps.png`

Regenerate with:

```bash
python3 scripts/plot_rewards.py \
  --runs logs/benchmark/20261125_174049_icra2027_12mps_bdp_frenet_v3_ppo logs/benchmark/20261125_185040_icra2027_12mps_bdp_frenet_v3_trpo \
  --labels "PPO-BDP (12 m/s)" "TRPO-BDP (12 m/s)" \
  --curve_mode combined --window_size 100 \
  --output logs/benchmark/icra2027_ppo_vs_trpo_12mps.png
```
