# PID Steering Error Modes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add backward-compatible PID reference-reset control and selectable steering error formulas for Frenet trajectory tracking.

**Architecture:** Extend the simulator-independent `TrackerConfig` and `TrajectoryPIDTracker`, then pass the two settings through the existing YAML/CLI parser and environment factory. Keep all trajectory generation, nominal-state, speed-control, and simulator-specific execution code unchanged.

**Tech Stack:** Python, NumPy, argparse, YAML, pytest, HighwayEnv, MetaDrive.

---

### Task 1: Specify PID state and steering-mode behavior

**Files:**
- Modify: `tests/test_tracking.py`
- Modify: `bdp_benchmark/common/tracking.py`

1. Add failing tests for the four steering error modes and invalid mode validation.
2. Run the focused tests and confirm failures are caused by the missing config fields.
3. Add `steering_error_mode` and `integral_reset_on_reference_change` to `TrackerConfig`.
4. Select the steering error using one private, directly testable helper.
5. Run the focused tests.

### Task 2: Specify reference replacement state handling

**Files:**
- Modify: `tests/test_tracking.py`
- Modify: `bdp_benchmark/common/tracking.py`

1. Add failing tests proving reference replacement always resets derivative history.
2. Add failing tests proving integral state is preserved only when configured, while episode reset always clears it.
3. Split internal PID reset operations into full reset and reference-change reset.
4. Run the focused tests.

### Task 3: Wire YAML and CLI configuration

**Files:**
- Modify: `tests/test_config.py`
- Modify: `bdp_benchmark/config.py`
- Modify: `bdp_benchmark/env_factory.py`

1. Add failing tests for defaults, YAML values, CLI overrides, validation, and resolved-config serialization.
2. Add `--pid_steering_error_mode` with explicit choices.
3. Add a boolean optional `--pid_integral_reset_on_reference_change` flag defaulting to true.
4. Include both values in the `pid` resolved-config section and `TrackerConfig` construction.
5. Run config and tracker tests.

### Task 4: Verify regressions

1. Run all benchmark tests with the sibling project root on `PYTHONPATH`.
2. Confirm current defaults produce the existing combined steering formula and full reference reset.
3. Inspect the diff to ensure no candidate, nominal-state, speed-control, or simulator-specific behavior changed.
