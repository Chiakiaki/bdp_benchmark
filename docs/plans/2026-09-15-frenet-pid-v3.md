# Frenet-PID V3 Implementation Plan

Historical plan: this distance-retimed/direct-target implementation is now
`frenet_pid_v3_legacy`. Revised v3 is specified by
[the independent-geometry plan](2026-09-15-v3-independent-geometry.md).

**Goal:** Implement the approved acceleration-then-cruise speed command and
distance-based geometry progression without changing v1/v2/native behavior.

**Architecture:** A separate common retiming module creates speed targets,
analytically integrated distance/speed/acceleration samples, and distance-retimed
lane geometry. V3 retains nominal XY/heading but initializes longitudinal motion
from actual measured speed. The selected candidate carries a speed-setpoint
metadata value consumed directly by the existing PI controller, never the old
speed preview. Lane and circular candidates share the same five speed commands.

**Config:** `frenet_speed_command_time_s=1.0`,
`frenet_speed_delta_rate_mps2=8.0`,
`frenet_speed_action_scales=[-1,-0.5,0,0.5,1]`. Tau is independent of steering
lookahead; validate finite positive tau <= horizon for v3, but do not require
tau > steering lookahead. The first version retains five configurable scales
and 15/25 actions. Existing configs keep their defaults and behavior.

## Tasks

1. Write failing tests for the new speed law, zero/stopped motion, distance-based
   lateral geometry, coherent features, direct speed control, and config wiring.
2. Implement `common/retiming.py`: exact ramp/cruise integration; retime dense
   lane-path geometry by its own arc length (not road longitudinal distance).
   Retain `[s,d,s_dot,d_dot,s_ddot,d_ddot]` internally and derive heading/speed
   from the same timed geometry. Use vectorized candidate/sample operations.
3. Wire MetaDrive v3 planning, optional candidate speed metadata, action execution,
   PID direct setpoints, config parsing/backup, and visual preview. Preserve
   existing sampling/scoring interfaces and checkpoint behavior in other modes.
4. Create builtin/BDP v3 pairs in `benchmark_metadrive_12mps` and the requested
   new `benchmark_metadrive` directory, using existing 12 m/s and `benchmark`
   80 km/h settings respectively. Do not overwrite existing jobs or launch full
   training. Document the contract and comparison limitations.
5. Run full regression tests, two-worker train/save/reload smokes, and a real
   headless scene check. Review the implementation before completion.

## Constraints and Risks

- No changed state-observation or reward dimensions; features remain 5 per point.
- Planned rates are descriptors, not guarantees of Bullet acceleration or grip.
- The direct PI target is cached with the scored candidate, not recomputed after
  action selection. Reset clears the override; legacy tracker calls omit it.
- Zero distance means no lateral/heading progression. Stopping freezes the pose.
- The lane geometry uses a positive spatial scale derived from measured speed,
  horizon and lane width, so acceleration from rest does not yield a degenerate
  reference. Dense polyline retiming preserves position/speed consistency within
  segments; derivatives are piecewise, not smooth at every geometric knot.
- Reference construction must follow connected route segments; continuation past
  the final route endpoint is explicitly geometric, not a fabricated successor.
- No automatic speed/safety filter or hidden steering-limit change.

## Verification

`PYTHONPATH=/home/dell/project python3 -m pytest tests/test_retiming.py tests/test_v3.py -q`

`PYTHONPATH=/home/dell/project python3 -m pytest tests -q`

## Completed Verification

- All five implementation tasks completed; existing jobs/launchers were not
  redirected and full benchmark training was not launched.
- Full suite: 249 tests passed, including four two-worker v3 PPO train,
  periodic/final checkpoint save, and reload checks.
- A separate 1,000-step real MetaDrive probe exercised acceleration, braking,
  lane targets and both outer curves with finite candidates throughout.
- Review identified a backward-facing nominal-tangent defect; signed geometric
  progress plus matching local negative-s tangent/normal continuation fixed it.
  Three moving-heading regressions passed, followed by the complete suite.
- CLI/YAML/effective-backup, tau bounds, unchanged short-horizon legacy configs,
  action counts and all four source/pair job comparisons pass.
- Tests and numerical descriptors do not establish improved reward, tire-slip
  robustness or real-world tracking performance.

## Subsequent Bundle Wiring

The 80 km/h v3 pair has since been merged into `scripts/job_scripts/benchmark`
and changed to 25 candidates; its separate `benchmark_metadrive` directory was
removed. Both speed bundles now provide `run_metadrive_frenet_pid_v3_comparison.sh`.
Evaluation and visual checks use the existing critic_based_rl compatibility
loader when `--allow_bdp_candidate_count_change` is explicitly supplied.
Verification after this wiring: 256 tests passed; the user's 15-candidate v2
checkpoint completed a 30-step v3 GUI visual smoke with 25 candidate scores.
