# Adaptive Pursuit Tracking

The MetaDrive Frenet-PID jobs can select `pid_controller_mode: adaptive_pursuit`.
It combines adaptive pure pursuit steering with persistent longitudinal PI/PID
feedback. `legacy` remains the code default for old checkpoints and configurations.
The four 80 km/h Frenet training jobs now select the measured adaptive settings.
Candidate trajectories, nominal projection, observations, rewards, traffic, speed
ceiling, PPO parameters, and timestep budget are unchanged.

## Controller

The steering target is interpolated along the selected trajectory at a distance
`clip(actual_speed * lookahead_time, min_lookahead, max_lookahead)` beyond its
nearest polyline projection. This avoids whole-sample jumps in the controller.
The planner still uses its existing nearest-sample nominal state.

MetaDrive reports chassis-center poses. The controller converts its reference to
a rear-axle path using the vehicle's actual axle geometry. Trajectory course is
converted to desired chassis yaw using the kinematic bicycle slip angle
`asin(rear_axle_offset * signed_curvature)`. Steering is then
`gain * atan2(2 * wheelbase * target_left, target_distance_squared)`, clipped to
the configured steering limit. Extremely tight, infeasible curvature is bounded
for numerical stability; this does not make the requested path feasible.

The desired speed is interpolated at the projected trajectory time plus
`pid_speed_preview_s`. The sample time is derived from the configured trajectory
horizon and sample count. Positive preview is required so a stationary vehicle
can follow a reference that accelerates from zero. Integral state survives
reference updates; conditional integration prevents saturation windup, and
derivative feedback uses measured speed to avoid reference-change kicks.
Episode resets clear all controller state.

MetaDrive invokes the ego policy once per environment decision. Adaptive mode
uses `physics_world_step_size * decision_repeat` (0.10 seconds here). Legacy
mode retains its former 0.02-second argument for reproducibility.

```yaml
pid:
  pid_controller_mode: adaptive_pursuit
  pid_lookahead_time_s: 0.7
  pid_min_lookahead_m: 3.0
  pid_max_lookahead_m: 12.0
  pid_pursuit_gain: 1.0
  pid_speed_preview_s: 0.4
  pid_speed_kp: 2.0
  pid_speed_ki: 0.2
  pid_speed_kd: 0.0
  pid_max_steering_rad: 0.7
  pid_max_accel_mps2: 5.0
  pid_max_decel_mps2: 8.0
```

The legacy heading gains, `pid_steering_error_mode`, `pid_lookahead_points`, and
`pid_integral_reset_on_reference_change` are used only by the legacy controller.
Acceleration is still converted to MetaDrive's normalized throttle/brake by the
existing scaling. Its m/s-squared naming describes the PID output, not a promise
that engine force produces exactly that acceleration in the physics engine.

## Measurement

The subsequent v3 arc-to-straight check retained these gains after tuning and
separate-seed validation. See [the v3 measurements](v3_arc_straight_pid_check.md)
for the tested alternatives, route fixtures, and limitations of frozen-selector
transfer comparisons.

`tracking_diagnostics: true` adds rollout averages through the existing SB3
callback/logger. They appear under `tracking/` in CLI/TensorBoard and in
`diagnostics/tracking_rollouts.csv`. Only scalar sums and a sample count are
retained, not rollout copies. Terminal steps are included before environment reset.

- `nominal_error_m`: distance to the nearest sampled point on the executed reference.
- `path_error_m`: distance to the nearest interpolated polyline point.
- `heading_error_rad`: chassis yaw versus interpolated trajectory course. Real
  turning slip can make this nonzero even with good tracking.
- `speed_error_mps`: measured speed versus the speed at the path projection.
- `target_speed_mps`: the previewed speed actually requested by adaptive PI.
- `steering_controller_saturated` and `speed_controller_saturated`: PID clipping.
- `steering_saturated` and `throttle_saturated`: normalized actuator saturation.

Nearest-sample nominal error includes longitudinal sample spacing. With 0.2 s
sample spacing at 20 m/s, even a vehicle exactly on a straight path can be 2 m
from its nearest sample. Reducing this metric to zero is not a meaningful control
objective without changing the planner's nominal projection contract.

## Reproduction

Use the frozen September 14 policy only as a controller diagnostic: its training
sampler predates the route-continuous reference. No policy updates occur here.

```bash
cd /home/dell/project/bdp_benchmark
python3 scripts/diagnose_tracking.py \
  --config scripts/job_scripts/benchmark/metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml \
  --model_path logs/benchmark_legacy/20260914_155825_metadrive_frenet_pid_v2_bdp_frenet_benchmark/models/PPO_BDP_final_model.zip \
  --device cpu --tracking_output logs/tracking_tuning/recheck \
  --tracking_seeds 5010 5011 5012 5013 5014 5015 5016 5017 5018 5019 \
  --tracking_steps 500
```

Add `--visual_check` to render candidate scores and the interpolated tracker target.
Each diagnostic records per-step `tracking.csv`, a `summary.json` containing the
effective configuration and episode outcomes, and a screenshot when rendering.
For the former controller, additionally specify `--pid_controller_mode legacy
--pid_speed_kp 1.0 --pid_speed_ki 0.05`.

## Measured Results (2026-09-15)

Gains were selected on seeds 5000-5004. The following separate seeds 5010-5019
were then tested with a deterministic frozen selector and a 500-step episode cap.
All terminal samples are included; trajectories can differ between controllers
because changed motion changes subsequent policy inputs and actions.

| Step-weighted mean | Legacy | Adaptive pursuit |
| --- | ---: | ---: |
| Nearest sampled nominal error (m) | 1.044 | 0.605 |
| Polyline path error (m) | 0.809 | 0.235 |
| Heading/course error (rad) | 0.255 | 0.171 |
| Speed error (m/s) | 0.734 | 0.534 |
| Actual speed (m/s) | 8.861 | 9.552 |

Recorded under `logs/tracking_tuning/heldout_legacy` and `heldout_final`.
Successes were 2/10 and 3/10; this small diagnostic is not evidence that the
newly trained RL policy outperforms the continuous PPO reference.

A separate fixed-route test bypassed the selector: `map: SC`, no NPC traffic,
seeds 5000-5002, 300-step cap, initial and requested speeds 8/12/20 m/s. It kept
the same 11-point/2-second selected-reference shape and used both controllers
on the assigned lane centerline. The speed, traffic, and reference overrides
apply only to this diagnostic, not training.

| Requested speed | Legacy mean path error | Adaptive mean path error |
| --- | ---: | ---: |
| 8 m/s | 0.267 m | 0.017 m |
| 12 m/s | 0.374 m | 0.036 m |
| 20 m/s | 0.400 m | 0.122 m |

Adaptive mean speeds were 7.998, 11.996, and 19.975 m/s. Artifacts are in
`logs/tracking_tuning/lane_legacy_*` and `lane_final_*`. Use
`--tracking_lane_speed_mps` and `--environment_config` to repeat these fixtures.

## Training Queues

The nominal queue trains BDP then builtin PPO:
`scripts/job_scripts/benchmark/run_metadrive_nominal_adaptive_pursuit_comparison.sh`.
The actual-state queue trains Frenet BDP, Frenet builtin, then native BDP:
`scripts/job_scripts/benchmark/run_metadrive_actual_adaptive_pursuit_comparison.sh`.
Both start from fresh models, record their console output under `logs/queues`,
and pass CLI overrides to each job. New Frenet run names include `adaptive_pursuit`.

Native BDP continues to execute direct MetaDrive steering/throttle. Its descriptors
already originate at the actual pose; attaching a PID would change this baseline's
action contract. It is therefore included as the final actual-queue reference.

## References

- [Coulter, Implementation of the Pure Pursuit Path Tracking Algorithm](https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm)
- [Nav2 velocity-scaled lookahead](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/controller_plugins/configuring_regulated_pp/)
