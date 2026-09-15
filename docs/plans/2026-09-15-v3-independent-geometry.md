# Revised V3: Independent Geometry and Desired-Speed Preview

## Approved Contract

Preserve distance-retimed paths plus direct terminal-speed PI as
`frenet_pid_v3_legacy`. Revised `frenet_pid_v3` uses independent longitudinal
ramp/cruise motion and the original time-based lateral quintic. Its speed column
is desired `V(t)`, not geometric speed. Reuse the existing speed-preview PI path
by omitting the direct speed override. Keep tau separate from steering lookahead.

Both versions retain nominal position/heading and actual measured initial speed,
route-connected references, existing feature dimensions and 15/25 candidate
ordering. Circle candidates retain their geometry and shared desired-speed law.
No observation, reward, vehicle-physics or critic_based_rl changes.

For non-aligned poses, retain the Frenet projection of initial velocity for lane
coordinate geometry; accelerate that longitudinal component to the common target
over tau. The scalar desired-speed profile still starts at actual measured speed.
This avoids pretending that reference-coordinate speed and scalar vehicle speed
are interchangeable at the initial pose. No square-root or curvature speed
compensation is introduced. Aligned straight-road geometry integrates V(t) exactly.

## Execution Steps

1. Add failing tests for independent timed lateral profiles, sample-speed features,
   zero desired speed despite lateral geometry, active preview, and legacy mode.
2. Add a small vectorized generator reusing the existing quintic and ramp/cruise
   calculations; wire the two explicit mode names in configuration and MetaDrive.
3. Preserve old version job YAMLs under v3_legacy names; update the four active v3
   job descriptions, keeping all other settings. Document speed semantics and
   the explicit override required for old saved configs containing `v3`.
4. Verify v1/v2/native regression tests, both v3 modes, PPO checkpoint roundtrips,
   and a short real scene/visual smoke. Do not launch full benchmark training.

## Deliberately Relaxed Properties

- Independent d(t) may continue changing after desired speed reaches zero; this
  is candidate geometry, not a prediction that a stopped vehicle moves sideways.
- XY-derived speed can exceed the desired speed column; features explicitly
  represent geometry plus a speed command, not a time-consistent dynamics rollout.
- Preview reads the desired speed column, not hypot(s_dot,d_dot). It is not a
  hard immobilization or speed-overshoot guarantee for Bullet physics.
- Historical configs explicitly naming `frenet_pid_v3` need a
  `--trajectory_execution_mode frenet_pid_v3_legacy` override for old semantics.
  Archived logs/configs are not edited automatically.

## Completed Verification

- New geometry and mode tests failed before implementation; targeted tests now
  pass, including desired 6.8 m/s preview versus terminal 2 m/s, and zero desired
  speed despite nonzero lateral geometry.
- Full suite: 285 tests passed, including eight two-worker PPO train/save/reload
  cases spanning revised/legacy v3, builtin/BDP and both speed bundles.
- Pre-change v3 and renamed v3_legacy hashes matched exactly for features,
  trajectories and speed targets in the seed-41, 10 m/s reference scene.
- Revised v3 passed a 1,000-step real-scene probe and a shortened GUI visual
  check using the user's earlier BDP checkpoint with 25 candidates.
- A bounded independent review found no blocking issues. No upstream simulator
  or algorithm code, archived configs, or full training runs were changed.
