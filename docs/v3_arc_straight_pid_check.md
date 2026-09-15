# V3 Arc-to-Straight PID Check

## Change

Active v3 outer candidates turn until `frenet_speed_command_time_s` (tau), then
continue along the terminal tangent. The ramp integral at tau is calculated
analytically, so off-grid cutoffs do not move the junction. Desired-speed values,
25-candidate ordering, state observations, reward, vehicle limits and timestep
settings are unchanged. No new switch was added; older modes retain their paths.

## Diagnostic Setup

Measurements are stored under `logs/tracking_tuning/v3_arc_straight/`, with a
per-step `tracking.csv` and effective-configuration `summary.json` for each run.
The diagnostic script now supports v3 and uses the existing guarded BDP loader.

The frozen selector was:

`logs/benchmark/20260915_003850_metadrive_frenet_pid_v2_bdp_frenet_adaptive_pursuit_benchmark/models/PPO_BDP_final_model.zip`

This is a v2-to-v3 transfer diagnostic, not evaluation of a newly trained v3
policy. The v2 reference uses its own job; v3 uses the requested 80 km/h v3 job
with 25 candidates. PPO is not updated. Five tuning seeds (5000-5004) and ten
separate validation seeds (5010-5019) use a 500-step episode cap. Termination
can end a run earlier; terminal samples are included in the means below.

## Tuning-Seed Results

| Configuration | Mean nominal error m | Mean path error m | Mean actual speed m/s | Steps |
| --- | ---: | ---: | ---: | ---: |
| V2 reference, inherited PID | 0.583 | 0.260 | 8.954 | 988 |
| V3 full-circle geometry, inherited PID | 1.160 | 1.019 | 5.044 | 210 |
| V3 arc-to-straight, inherited PID | 1.705 | 1.459 | 8.807 | 348 |
| Arc-to-straight, steering lookahead 1.0 s | 0.976 | 0.700 | 8.039 | 286 |

Eleven alternatives were tried: lookahead times 0.3, 0.4, 0.5, 0.9, 1.0 and
1.2 s, shorter min/max lookahead bounds, pursuit gains 0.8/1.2, and one 0.2 s
speed-preview variant. The 1.0 s lookahead had the lowest mean path error on
the tuning seeds. The full trial settings and results are in the JSON artifacts.

## Separate-Seed Check

| Bundle and setting | Mean nominal error m | Mean path error m | Mean actual speed m/s | Total distance m |
| --- | ---: | ---: | ---: | ---: |
| 80 km/h, inherited 0.7 s | 1.172 | 0.899 | 8.061 | 477.6 |
| 80 km/h, trial 1.0 s | 1.277 | 1.017 | 8.602 | 564.9 |
| 12 m/s, inherited 0.7 s | 1.142 | 0.866 | 8.620 | 1632.8 |
| 12 m/s, trial 1.0 s | 1.353 | 1.134 | 8.442 | 1870.8 |

The apparent tuning-set improvement did not generalize to these separate seeds.
Some trial runs traveled farther before terminating, but neither cap improved
the requested mean tracking-error metric. This is a closed-loop comparison:
changing controller parameters changes later policy observations and candidate
choices, so it does not isolate a causal controller effect on identical paths.

## Feasible Route Fixture

A separate control-only fixture bypassed the selector and followed the assigned
lane on map `SC`, with no NPCs, seeds 5000-5002 and a 300-step cap. Vehicle,
observation and controller settings otherwise came from the same 80 km/h job.
Initial and requested speeds were fixed at 12 or 20 m/s.

| Requested speed | Inherited mean path error | Trial 1.0 s mean path error | Mean actual speeds |
| --- | ---: | ---: | --- |
| 12 m/s | 0.036 m | 0.072 m | 11.996 / 11.996 m/s |
| 20 m/s | 0.170 m | 0.170 m | 19.966 / 19.966 m/s |

At 20 m/s both time settings reach the existing 12 m lookahead-distance cap.
The inherited controller therefore remains a reasonable starting point for
feasible references; the transferred selector and difficult new references
remain confounding factors in the larger v3 errors. No claim is made that
gains alone can make all high-speed, high-curvature candidates trackable.

Nearest-sample nominal error must not be interpreted as pure cross-track error:
on these fixed-speed fixtures it is about 1.2 m and 2.0 m, respectively, because
the 0.1 s controller step lands near the midpoint of 0.2 s trajectory samples.

## Decision and Reproduction

Retain the inherited PID parameters consistently across the four active v3
jobs. Do not apply the tuning-set winner, which worsened validation tracking.
All v1/v2/v3_legacy parameters remain unchanged. Fresh v3 benchmark training
will test learned behavior under the revised candidate contract.

```bash
python3 scripts/diagnose_tracking.py \
  --config scripts/job_scripts/benchmark/metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml \
  --model_path logs/benchmark/20260915_003850_metadrive_frenet_pid_v2_bdp_frenet_adaptive_pursuit_benchmark/models/PPO_BDP_final_model.zip \
  --allow_bdp_candidate_count_change --device cpu \
  --tracking_output logs/tracking_tuning/v3_arc_straight/recheck \
  --tracking_seeds 5010 5011 5012 5013 5014 5015 5016 5017 5018 5019 \
  --tracking_steps 500
```

Add `--pid_lookahead_time_s 1.0` to reproduce the rejected trial. The pre-change
full-circle baseline was captured before the active v3 implementation changed;
there is deliberately no new geometry toggle for it.
