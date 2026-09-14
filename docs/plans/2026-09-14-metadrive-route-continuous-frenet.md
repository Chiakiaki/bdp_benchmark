# MetaDrive Route-Continuous Frenet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a benchmark-owned `frenet_route_continuous` candidate mode that follows MetaDrive's assigned route across lane-segment boundaries while retaining existing `frenet` and `native_action_frenet` behavior for old checkpoints.

**Architecture:** A new MetaDrive route-reference helper will resolve route-connected successor lanes, select the route lane nearest the actual or nominal planning state, and build a cumulative centerline from the remaining current lane through subsequent route edges. `MetaDriveAdapter` will select legacy or route-continuous geometry from an explicit reference mode passed by `bdp_benchmark.env_factory`; `critic_based_rl` remains simulator-agnostic.

**Tech Stack:** Python, NumPy, Gymnasium, MetaDrive lane/navigation APIs, pytest, SB3.

---

### Task 1: Route-reference unit contract

**Files:**
- Create: `tests/test_metadrive_route_reference.py`
- Create: `bdp_benchmark/metadrive/route_reference.py`

1. Add lightweight straight and curved lane doubles with MetaDrive-compatible `position`, `heading_theta_at`, `local_coordinates`, `distance`, `length`, `start`, `end`, and `index` APIs.
2. Add failing tests for straight-to-curve and curve-to-straight stitching, connected-successor selection, nearest route-lane selection for a nominal state beyond the current segment, and route-end clamping.
3. Run `pytest tests/test_metadrive_route_reference.py -q` and verify failure because the helper is absent.
4. Implement successor resolution and cumulative route sampling without modifying MetaDrive.
5. Re-run the focused tests and verify success.

### Task 2: Adapter geometry selection

**Files:**
- Modify: `bdp_benchmark/metadrive/adapter.py`
- Modify: `bdp_benchmark/metadrive/env.py`
- Test: `tests/test_metadrive_adapter.py`
- Test: `tests/test_metadrive_smoke.py`

1. Add failing tests that route-continuous mode bends into a successor lane while legacy mode continues to extrapolate the current lane.
2. Pass a reference mode into `MetaDriveAdapter` and retain `lane_segment` as its default.
3. Compute the initial Frenet state against the nearest lane in the connected route for both `x_actual` and `x_nominal`.
4. Build one route-continuous `ReferencePath` and keep the vectorized `K x H` Frenet-to-world conversion unchanged.
5. Cache route-lane resolution per map, route, and current-lane key; clear it on environment reset.
6. Preserve candidate count, ordering, trajectory shape, and feature width.

### Task 3: Configuration and sampler wiring

**Files:**
- Modify: `bdp_benchmark/config.py`
- Modify: `bdp_benchmark/env_factory.py`
- Modify: `bdp_benchmark/runner.py`
- Test: `tests/test_config.py`
- Test: `tests/test_env_factory.py`
- Test: `tests/test_visual_check_cli.py`

1. Add failing tests accepting `frenet_route_continuous` only for MetaDrive Frenet-capable modes and rejecting incompatible uses.
2. Map `frenet_route_continuous` to the existing benchmark-owned `FrenetCandidateSampler` for BDP while passing `route_continuous` to the MetaDrive adapter.
3. Let builtin Frenet-PID use the same selector for environment geometry without wrapping its observation in candidate fields.
4. Keep `frenet` and `native_action_frenet` mapped to legacy single-lane geometry.
5. Include the new mode in visual candidate-payload validation.

### Task 4: Active benchmark bundles

**Files:**
- Modify: `scripts/job_scripts/benchmark/*.yaml`
- Modify: `scripts/job_scripts/benchmark_metadrive_12mps/*.yaml`
- Modify: `scripts/job_scripts/benchmark/config_contract.md`
- Modify: `scripts/job_scripts/benchmark_metadrive_12mps/config_contract.md`
- Test: `tests/test_job_scripts.py`

1. Add failing tests requiring route-continuous mode in active native BDP, Frenet-PID BDP, Frenet-PID-v2 BDP, and builtin Frenet-PID jobs.
2. Update those active YAMLs to `candidate_sampler: frenet_route_continuous`.
3. Leave native builtin, native continuous builtin, and all `example/` YAMLs unchanged.
4. Document that the new and legacy samplers have identical dimensions but different geometry contracts.

### Task 5: Verification

1. Run focused route-reference, adapter, config, environment, visualization, and job-script tests.
2. Run a tiny MetaDrive BDP smoke train with route-continuous candidates.
3. Run `pytest -q` with the sibling repositories on `PYTHONPATH`.
4. Run `git diff --check` and inspect the final diff for unrelated changes.
