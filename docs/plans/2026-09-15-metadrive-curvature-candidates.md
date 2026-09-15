# Opt-In MetaDrive Curvature Candidates

## Approved Scope

Keep native actuators, observations, rewards, existing PID gains, and speed targets
unchanged. Add two constant-curvature families crossed with the five existing
Frenet speed targets. Preserve actions 0..14; append speed-major left/right pairs
as actions 15..24. Use nominal pose/speed in v2 and actual pose/speed in v1.

Expose `frenet_include_curvature_candidates` (false by default) and
`frenet_curvature_steering_fraction` (0.9) through the existing configuration
pipeline. Enable only the four 12 m/s Frenet-PID training jobs. The fraction
multiplies the smaller of vehicle and PID steering limits. Convert rear-axle
curvature to center-path curvature using the existing vehicle wheelbase and rear
offset. Center-path heading is the planning tangent, matching the trajectory
contract and adaptive pursuit's center-to-rear conversion; it is not a prediction
of instantaneous tire slip or steering transients.

## Implementation Sequence

1. Add failing tests for geometry, candidate ordering, configuration and action
   spaces, plus legacy/native regression checks.
2. Reuse the vectorized Frenet longitudinal polynomial for all extra curves;
   generate circular center paths with a stable zero-curvature limit. Reuse the
   existing feature encoder and tracker. Select a benchmark-owned 25-action PID
   policy subclass without modifying MetaDrive or critic_based_rl.
3. Wire and back up the two arguments, validate unsupported native/Highway use,
   and enable the four requested jobs. Add a third launcher for actual-origin
   BDP, actual-origin builtin, and continuous native builtin.
4. Document speed endpoints, delayed speed preview, missing explicit slip
   supervision, and the limitations of a steering margin at high speed. Describe
   the 12 m/s bundle as the recommended next stability experiment, not a proven
   performance improvement or a pure speed ablation with curvature enabled.
5. Run unit/integration tests and short 25-action PPO training/save/load smokes.
   Do not launch full training without a further request.

## Verification Commands

`python3 -m pytest tests/test_curvature_candidates.py tests/test_config.py tests/test_job_scripts.py -q`

`python3 -m pytest tests -q`

Check the new launcher with `bash -n` and a stub Python executable, including
argument forwarding and launch order. Run a small two-worker builtin/BDP PPO
smoke with the configured mixed action set and verify checkpoint reload.

## Risks

0.9 of the steering-angle limit is not a lateral-acceleration guarantee. Extreme
two-second arcs may turn through a full circle at high speed; nearest-point
projection can then be ambiguous. Do not silently shorten paths or change speed
targets. A new action count requires a matching checkpoint/config; old 15-action
jobs continue with the switch omitted or disabled. Adding unsafe choices alone
does not guarantee that the learned policy prefers the center lane.

## Verification Results

- The new feature/config/job tests failed before implementation, while 53
  existing targeted tests passed; all 73 targeted tests now pass.
- Full suite: 202 tests passed, including two-worker train/save/reload smokes
  for builtin and BDP in both actual-origin and nominal-origin modes.
- The remaining-jobs launcher passes shell syntax and stub-execution checks.
- A structured comparison against HEAD confirms that only the two candidate
  settings and run names changed in the four requested job YAMLs.
- The speed-preview probe confirms 11.74 m/s for a slowest 9.5 m/s terminal
  target from a 12 m/s origin, and acceleration if actual speed is only 10 m/s.
- No full training was launched; observations, rewards, actuator mappings and
  tracker code remain unchanged.
