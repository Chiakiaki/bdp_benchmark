# V3 Turn-Then-Straight and PID Check

## Scope

Revise the active v3 outer candidates only: use the existing curvature until
`frenet_speed_command_time_s`, then preserve the reached heading and continue
straight. Keep the same desired-speed law, feature dimensions, candidate count,
observations and rewards. Do not add a mode or configuration switch; preserve
v1/v2 and v3_legacy reference behavior.

## Steps

1. Extend the existing diagnostic entry point to accept the current Frenet modes
   and reuse guarded checkpoint loading. Add a subprocess regression test.
2. Record current PID performance on fixed training/held-out seeds, including
   v2 and pre-change v3 frozen-selector baselines.
3. Add failing geometric tests, then implement exact arc-length cutoff at tau,
   tangent continuation, continuous XY/heading, and unchanged speed samples.
4. Measure post-change v3 with the inherited PID. If tracking is materially
   worse, compare a small parameter sweep on training seeds and confirm the
   selected candidate on separate seeds. Report speed/distance alongside error
   so slowing/stopping cannot masquerade as better tracking.
5. Apply justified parameters consistently to active v3 jobs in both bundles.
   Document measured results and remaining physical/descriptor limitations.
6. Run regression and smoke checks, then launch the two requested v3 shell
   queues in separate GUI terminals with persistent console logs. Verify initial
   training progress, not full completion. Clarify the malformed additional
   non-v3 launcher request before launching duplicate or unspecified jobs.

## Measurement Caveats

Nearest-sample nominal error includes sampling distance. Path error measures
geometric cross-track error; course-vs-chassis heading includes turning slip;
desired-speed error is not a perfect time-consistent trajectory residual. Frozen
policy replay is an exploratory closed-loop comparison: controller changes alter
later observations and selected paths. Infeasible sharp/high-speed references
cannot be made perfectly trackable by PID tuning alone.

## Outcome

- Arc-to-straight implementation completed directly in active v3. All 290 tests
  passed, including off-grid tau and legacy-routing regression checks.
- Diagnostic CLI now supports v3 modes and the existing candidate-count-compatible
  checkpoint loader. Eleven PID alternatives were tested. The tuning-set winner
  worsened separate-seed errors at both speed caps, so inherited gains were kept.
  See `docs/v3_arc_straight_pid_check.md` for measurements and artifacts.
- Started both requested v3 queues in separate GUI terminals at 2026-09-15 19:53.
  BDP PIDs were 851306 (12 m/s) and 851347 (80 km/h); each passed 40,000 steps
  during startup verification. Builtin jobs follow sequentially in each queue.
- Console logs: `logs/queues/20260915_195316_v3_arc_straight_12mps.log` and
  `logs/queues/20260915_195316_v3_arc_straight_80kmh.log`.
- Additional non-v3 launches were not guessed: the user-provided final paragraph
  repeated the v3 paths and contained an unrelated template link. Exact intended
  non-v3 launcher paths were requested asynchronously.
