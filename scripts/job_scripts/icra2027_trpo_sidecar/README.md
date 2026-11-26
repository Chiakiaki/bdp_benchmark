# ICRA 2027 TRPO sidecars for the training figure and table

Fig. 5 and Table VI of the paper report **ten PPO training runs**: five methods
at 80 km/h and the same five at 12 m/s. This bundle documents the matching
**TRPO** sidecar for each of them, so the paper's PPO numbers can be compared
against TRPO at the same data budget.

**No job YAMLs live here.** All ten TRPO jobs already exist in
`benchmark_more_algorithm/` (80 km/h) and
`benchmark_metadrive_12mps_more_algorithm/` (12 m/s), and
`machine_mgr job submit <machine> bdp_benchmark <path/to/yaml>` runs them.
Copying them here would create a second source of truth.

## The ten pairs

Run ids are from `evidence/training_inventory.csv` in the paper repository; the
resolved configs are the ones frozen in `evidence/training_snapshot.zip`.

| # | PPO run (paper) | TRPO yaml | TRPO run name |
|---|---|---|---|
| 1 | `20260915_195317_…frenet_pid_v3_bdp_frenet_curvature25_12mps…` | `benchmark_metadrive_12mps_more_algorithm/metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml` | `metadrive_frenet_pid_v3_trpobdp_frenet_curvature25_12mps_benchmark` |
| 2 | `20260915_210902_…frenet_pid_v3_builtin_curvature25_12mps…` | `benchmark_metadrive_12mps_more_algorithm/metadrive_frenet_pid_v3_builtin_benchmark.yaml` | `metadrive_frenet_pid_v3_trpo_builtin_curvature25_12mps_benchmark` |
| 3 | `20260915_222206_…native_bdp_frenet_12mps…` | `benchmark_metadrive_12mps_more_algorithm/metadrive_native_bdp_frenet_benchmark.yaml` | `metadrive_native_trpobdp_frenet_12mps_benchmark` |
| 4 | `20260915_231757_…native_builtin_12mps…` | `benchmark_metadrive_12mps_more_algorithm/metadrive_native_builtin_benchmark.yaml` | `metadrive_native_trpo_builtin_12mps_benchmark` |
| 5 | `20260916_000709_…native_continuous_builtin_12mps…` | `benchmark_metadrive_12mps_more_algorithm/metadrive_native_continuous_builtin_benchmark.yaml` | `metadrive_native_continuous_trpo_builtin_12mps_benchmark` |
| 6 | `20260915_195318_…frenet_pid_v3_bdp_frenet_adaptive_pursuit…` | `benchmark_more_algorithm/metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml` | `metadrive_frenet_pid_v3_trpobdp_frenet_adaptive_pursuit_benchmark` |
| 7 | `20260915_211048_…frenet_pid_v3_builtin_adaptive_pursuit…` | `benchmark_more_algorithm/metadrive_frenet_pid_v3_builtin_benchmark.yaml` | `metadrive_frenet_pid_v3_trpo_builtin_adaptive_pursuit_benchmark` |
| 8 | `20260915_030757_…native_bdp_frenet_benchmark` | `benchmark_more_algorithm/metadrive_native_bdp_frenet_benchmark.yaml` | `metadrive_native_trpobdp_frenet_benchmark` |
| 9 | `20260914_172247_…native_builtin_benchmark` | `benchmark_more_algorithm/metadrive_native_builtin_benchmark.yaml` | `metadrive_native_trpo_builtin_benchmark` |
| 10 | `20260914_155658_…native_continuous_builtin_benchmark` | `benchmark_more_algorithm/metadrive_native_continuous_builtin_benchmark.yaml` | `metadrive_native_continuous_trpo_builtin_benchmark` |

## Why this is a fair sidecar

Each TRPO yaml was resolved with `bdp_benchmark.config` and diffed key by key
against the frozen PPO `resolved_config.yaml` for the run it pairs with:

- Env, observation (275-D), reward, PID mode (`adaptive_pursuit`), curvature
  candidates (K = 25 at 12 m/s), model widths, seed and budget (2,000,000
  transitions) are **identical**.
- The only differences are `algorithm.*` keys and, for the 80 km/h native runs,
  `frenet.*`/`pid.*` keys that those older configs did not record. Those keys are
  `include_curvature_candidates: false`, `pid_controller_mode: legacy`,
  `speed_action_scales: [-1, -0.5, 0, 0.5, 1]`, … i.e. the defaults the old runs
  used implicitly. They are inert in `native_controller` mode.

## TRPO settings (deliberately different from PPO)

These come from the repository's own TRPO bundles and are not derived from PPO:

```yaml
name: TRPO
num_timesteps: 2000000
learning_rate: 5.0e-05      # critic only; policy step comes from KL/CG
schedule: fixed
trpo_timesteps_per_batch: 1000   # 8 workers -> 8000 samples per rollout
batch_size: 100
gamma: 0.99
gae_lambda: 0.95
target_kl: 0.01
cg_max_steps: 10
cg_damping: 0.01
n_critic_updates: 20
n_envs: 8
vec_env: subproc
seed: 41
```

Matched *data* budget (2M transitions), not matched optimiser-update counts: PPO
takes 20 epochs per rollout, TRPO one KL-constrained step plus 20 critic passes.
Report it that way.

## Submitting the batch

`bdp_benchmark/.machine_mgr.yaml` declares `max_parallel: 1`,
`min_free_gpu_mem_gb: 2` and `min_free_ram_gb: 6`, so the queue runs one job at a
time and refuses to start when the target is still busy or short on memory.

```bash
cd machine_mgr_repo

# 12 m/s first (the paper's primary column)
for y in metadrive_frenet_pid_v3_bdp_frenet metadrive_frenet_pid_v3_builtin \
         metadrive_native_bdp_frenet metadrive_native_builtin \
         metadrive_native_continuous_builtin; do
  ./machine_mgr job queue add alienware-m17 bdp_benchmark \
    "scripts/job_scripts/benchmark_metadrive_12mps_more_algorithm/$y.yaml"
done

# then 80 km/h
for y in metadrive_frenet_pid_v3_bdp_frenet metadrive_frenet_pid_v3_builtin \
         metadrive_native_bdp_frenet metadrive_native_builtin \
         metadrive_native_continuous_builtin; do
  ./machine_mgr job queue add alienware-m17 bdp_benchmark \
    "scripts/job_scripts/benchmark_more_algorithm/$y.yaml"
done

./machine_mgr job queue list
./machine_mgr job queue run --poll 120      # Ctrl-C is safe; rerun to resume
```

Interrupting the scheduler leaves the running job training on the target and the
rest `pending`; `job queue run` continues where it stopped. A run that was
killed can be redone with `--resume-from <run-id>`.

## Cost and follow-up

- The ten PPO runs took **11.3 h wall clock in total** on the original host
  (1.13 h average, from monitor timestamps). Expect a similar total for TRPO,
  possibly a little longer per run; the target is an RTX 3070 Ti laptop.
- Artifacts: `./machine_mgr job fetch <run-id> --artifact logs` pulls
  `logs/benchmark`.
- Evaluation is unchanged and still manual:
  `python3 scripts/evaluate.py --config <same yaml> --model_path <run>/models/TRPO_BDP_final_model.zip`.
  A `--verb evaluate` in the job contract is the natural next step if these
  comparisons become routine.
