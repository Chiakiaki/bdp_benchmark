# Visual Check Candidate Overlay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `--visual_check` evaluation path that uses HighwayEnv and MetaDrive's native renderers while displaying the exact candidate trajectories, categorical scores, selected action, and PID target waypoint.

**Architecture:** Add a one-pass discrete inference helper to `critic_based_rl`, then keep candidate capture and renderer overlays in `bdp_benchmark`. The benchmark stores the same `CandidateSet` used to construct policy observations, maps raw logits to red-neutral-blue colors with the selected candidate overridden to green, and delegates drawing to each simulator's existing renderer/debug primitives.

**Tech Stack:** Python, NumPy, PyTorch, Gymnasium, Stable-Baselines3, pygame/HighwayEnv rendering, MetaDrive Panda3D debug line and point drawers.

---

### Task 1: Add one-pass discrete score inference to critic_based_rl

**Files:**
- Create: `/home/dell/.config/superpowers/worktrees/critic_based_rl/visual-score-inference/inference.py`
- Test: `/home/dell/.config/superpowers/worktrees/critic_based_rl/visual-score-inference/tests/test_inference.py`
- Modify: `readme.md`

1. Write failing tests using a tiny SB3 categorical model and a tiny BDP model. Verify the helper returns selected action and logits from the same distribution for deterministic and stochastic selection.
2. Implement `predict_discrete_with_scores(model, obs, deterministic)` using the policy distribution, returning NumPy action and detached raw logits. Reject non-discrete action spaces clearly.
3. Keep existing callback extraction behavior unchanged and export the helper without adding any training callback or rollout work.
4. Run the focused tests and the full critic-based suite.

### Task 2: Add score color mapping and overlay state

**Files:**
- Create: `bdp_benchmark/visualization.py`
- Test: `tests/test_visualization.py`

1. Write failing tests for median-centered score normalization, low-score red/high-score blue mapping, invalid-score handling, and selected-candidate green override.
2. Implement immutable overlay data holding candidate trajectories, features, labels, logits, selected index, and optional PID target.
3. Implement a pure color mapper with finite-score validation and a neutral color for scores near the center.
4. Run focused visualization tests.

### Task 3: Capture the exact candidate set without training overhead

**Files:**
- Modify: `bdp_benchmark/common/contracts.py`
- Modify: `bdp_benchmark/highway/env.py`
- Modify: `bdp_benchmark/metadrive/env.py`
- Modify: `bdp_benchmark/env_factory.py`
- Test: `tests/test_visual_candidate_capture.py`

1. Write failing tests proving visual capture is disabled by default and, when enabled, the payload's features exactly match the candidate rows exposed to the BDP observation.
2. Add an opt-in debug-capture flag to benchmark environment construction. Cache the most recent `CandidateSet` only in this mode.
3. Expose a common `get_visual_candidate_set()` and `set_visual_overlay()` adapter interface. For one-hot/built-in modes, generate the same action-conditioned geometry by label while retaining the actual one-hot observation separately.
4. Add a pure PID preview method that returns the exact nearest-plus-lookahead target point without changing PID integrators.
5. Run focused capture and existing benchmark tests.

### Task 4: Implement HighwayEnv native overlay rendering

**Files:**
- Create: `bdp_benchmark/highway/visualizer.py`
- Modify: `bdp_benchmark/highway/env.py`
- Test: `tests/test_highway_visualizer.py`

1. Write failing tests for world-to-surface line construction, selected-line thickness, and PID waypoint marker data.
2. Install an `EnvViewer.set_agent_display()` callback only for visual mode and draw colored candidate polylines on HighwayEnv's existing `WorldSurface`.
3. Draw the PID target waypoint as a small contrasting marker. Clear/rebuild overlay state each rendered frame.
4. Do not alter upstream HighwayEnv files or native reward/transition behavior.
5. Run headless pygame smoke tests with `SDL_VIDEODRIVER=dummy` plus existing Highway parity tests.

### Task 5: Implement MetaDrive native overlay rendering

**Files:**
- Create: `bdp_benchmark/metadrive/visualizer.py`
- Modify: `bdp_benchmark/metadrive/env.py`
- Test: `tests/test_metadrive_visualizer.py`

1. Write failing tests for color/line payload conversion and overlay clearing.
2. Use MetaDrive's existing `engine.make_line_drawer(parent_node=engine.worldNP)` and `make_point_drawer()` primitives. Keep transient nodes owned by the sidecar visualizer.
3. Draw trajectories at a small world-space height, set selected line thickness higher, and redraw after each current-state update.
4. Clear all transient nodes on reset and close. Keep `use_render` and normal MetaDrive camera/window behavior intact.
5. Run headless MetaDrive overlay smoke tests and existing process-isolated tests.

### Task 6: Add benchmark visual-check CLI and loop

**Files:**
- Modify: `bdp_benchmark/config.py`
- Modify: `bdp_benchmark/env_factory.py`
- Modify: `bdp_benchmark/runner.py`
- Modify: `scripts/evaluate.py`
- Test: `tests/test_visual_check_cli.py`

1. Write failing parser tests for `--visual_check`: evaluation mode, one environment, dummy vectorization, human rendering, and deterministic default unless explicitly overridden.
2. Add `--inference_mode {deterministic,stochastic}` and preserve explicit CLI values over visual defaults. Reuse `evaluate_episodes` as the stopping budget.
3. Add a visual evaluation loop that, per step, obtains the exact candidate payload, calls the shared one-pass score helper, previews the PID waypoint, updates the overlay, renders the native environment, then steps the selected raw candidate.
4. Ensure visual-only score extraction is never called from `run_training`.
5. Run one-step Highway and MetaDrive visual checks, plus all existing tests.

### Task 7: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/design.md`
- Create: `scripts/job_scripts/visual_check_highway_bdp.yaml`
- Create: `scripts/job_scripts/visual_check_metadrive_bdp.yaml`

1. Document command examples, deterministic/stochastic behavior, score colors, selected-green override, exact candidate-feature semantics, and PID waypoint meaning.
2. Explain that one-hot models receive one-hot rows while trajectory geometry is an action visualization; structured Frenet models receive the plotted feature descriptors.
3. Parse both visual-check job files and run complete benchmark and critic-based test suites.
4. Verify upstream HighwayEnv and MetaDrive repositories remain unmodified.
