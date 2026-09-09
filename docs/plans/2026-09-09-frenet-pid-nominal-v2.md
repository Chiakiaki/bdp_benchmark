# Frenet-PID Nominal Planning State Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in `frenet_pid_v2` mode that generates candidate trajectories from a nominal state projected on the previous selected trajectory while the PID tracks them using the actual simulator state.

**Architecture:** Keep `native_controller` and legacy `frenet_pid` actual-state based. Add a small per-environment nominal trajectory state in the benchmark core. At reset, nominal state equals actual state; after each selected trajectory, the next planning nominal state is the nearest point on that previous trajectory. Candidate features use the nominal frame, while PID control, rewards, and termination use actual state.

**Tech Stack:** Python, NumPy, Gymnasium, HighwayEnv, MetaDrive, pytest.

---

### Task 1: Add mode and nominal-state contracts

**Files:**
- Modify: `bdp_benchmark/config.py`
- Modify: `bdp_benchmark/common/contracts.py`
- Create: `bdp_benchmark/common/nominal.py`
- Test: `tests/test_config.py`
- Test: `tests/test_nominal.py`

1. Write failing tests for accepting `frenet_pid_v2`, rejecting unknown modes, and preserving existing defaults.
2. Add a `NominalTrajectoryState` with reset, nearest-point projection, and selected-trajectory commit operations. Store nominal `[x, y, heading, speed]`; use actual XY only for projection and never actual heading for a valid projection.
3. Add tests for reset initialization from actual state, projection heading/speed coming from the previous trajectory, invalid/empty history fallback, and no mutation of the actual state.
4. Run focused tests.

### Task 2: Parameterize common adapter generation by planning state

**Files:**
- Modify: `bdp_benchmark/highway/adapter.py`
- Modify: `bdp_benchmark/metadrive/adapter.py`
- Modify: `bdp_benchmark/common/frenet.py` if needed
- Test: `tests/test_highway_adapter.py`
- Test: `tests/test_metadrive_adapter.py`

1. Write failing tests showing a supplied nominal pose changes candidate origin/features while actual pose remains the PID state.
2. Add an explicit optional planning-state argument to lane/Frenet state construction and candidate generation.
3. For nominal generation, derive Frenet rates from nominal speed/heading and lane tangent; do not read actual chassis heading or measured velocity. Keep actual mode on the current implementation path.
4. Keep direct simulator heading APIs in the actual-state path only. Use lane tangent/previous trajectory tangent for nominal heading.
5. Run focused adapter tests and existing parity tests.

### Task 3: Wire `frenet_pid_v2` into both environments

**Files:**
- Modify: `bdp_benchmark/highway/env.py`
- Modify: `bdp_benchmark/metadrive/env.py`
- Modify: `bdp_benchmark/env_factory.py`
- Test: `tests/test_highway_adapter.py`
- Test: `tests/test_metadrive_smoke.py`

1. Write failing tests for reset -> initial nominal state, first candidate generation from that state, commit of selected trajectory, and next-step nearest projection.
2. Add `frenet_pid_v2` to validation and treat it as a PID execution mode with five actions.
3. In v2 `build_candidate_set()`, read actual state, update nominal state from the previous selected trajectory, and call the adapter with nominal planning state.
4. In v2 `step()`, commit the selected nominal trajectory, then run the existing PID executor against actual state. Keep legacy `frenet_pid` unchanged.
5. Ensure reset/close clear nominal history and visualizer state.
6. Run Highway and MetaDrive smoke tests.

### Task 4: Align visual inspection with the two states

**Files:**
- Modify: `bdp_benchmark/runner.py`
- Modify: `bdp_benchmark/visualization.py`
- Modify: `bdp_benchmark/highway/visualizer.py`
- Modify: `bdp_benchmark/metadrive/visualizer.py`
- Test: `tests/test_visual_check_cli.py`
- Test: `tests/test_visual_candidate_capture.py`

1. Write failing tests for a v2 overlay containing nominal origin, actual pose, selected trajectory, and actual-relative PID target.
2. Keep candidate lines sourced from the exact v2 candidate set. Add optional nominal-origin marker only for v2 to make planning/tracking separation visible.
3. Compute the PID waypoint from actual state projected onto the selected nominal trajectory.
4. Keep native and legacy visual behavior unchanged.
5. Run one-step visual smokes with HighwayEnv and MetaDrive v2 configurations.

### Task 5: Add reproducible v2 jobs and documentation

**Files:**
- Create: `scripts/job_scripts/highway_frenet_pid_v2_bdp_frenet.yaml`
- Create: `scripts/job_scripts/metadrive_frenet_pid_v2_bdp_frenet.yaml`
- Modify: `README.md`
- Modify: `docs/design.md`

1. Add jobs matching existing Frenet-PID BDP jobs except for `trajectory_execution_mode: frenet_pid_v2`.
2. Document the state table, timing, initial reset behavior, nearest projection, and actual-state reward/control boundary.
3. Explain that changing from legacy `frenet_pid` to v2 changes candidate distribution but not observation dimensions.
4. Parse all job files and run complete tests.

### Task 6: Final verification

1. Run the complete benchmark suite.
2. Run the complete `critic_based_rl` suite.
3. Run native Highway parity and legacy Frenet-PID regression tests.
4. Run one Highway and one MetaDrive v2 visual check with existing compatible checkpoints.
5. Confirm upstream HighwayEnv and MetaDrive source trees are unchanged.
