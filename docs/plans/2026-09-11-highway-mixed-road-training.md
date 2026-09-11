# Highway Mixed-Road Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add deterministic mixed HighwayEnv training over Highway, Merge, and Roundabout with one shared ego-relative observation contract.

**Architecture:** Parse an optional list of environment IDs, normalize and validate its shared observation configuration, and select a fixed ID from worker rank. Validate final wrapped Gymnasium spaces before vectorization, annotate monitoring information with the selected variant, and add paired reproducible benchmark jobs.

**Tech Stack:** Python 3.12, Gymnasium, HighwayEnv, Stable-Baselines3, PyYAML, pytest.

---

### Task 1: Configuration contract

**Files:**
- Modify: `bdp_benchmark/config.py`
- Test: `tests/test_config.py`

1. Add failing tests for YAML and CLI `environment_variants`, resolved-config backup, the default single-environment path, Highway-only validation, duplicate rejection, minimum worker count, and the shared ego-relative observation block.
2. Run `python3 -m pytest tests/test_config.py -q` and confirm failures are caused by the missing option.
3. Add the parser option, normalization, validation, warning, and resolved-config entry.
4. Rerun the focused tests until they pass.

### Task 2: Deterministic worker assignment and space validation

**Files:**
- Modify: `bdp_benchmark/env_factory.py`
- Test: `tests/test_env_factory.py`

1. Add failing tests for rank-modulo assignment, fixed per-worker IDs, compatible mixed wrapped spaces, and a clear error for incompatible spaces.
2. Run the focused tests and confirm the expected failures.
3. Add environment-ID selection and pre-vectorization wrapped-space validation without changing single-`env_id` behavior.
4. Rerun the focused tests.

### Task 3: Variant-aware monitoring and evaluation

**Files:**
- Modify: `bdp_benchmark/highway/env.py`
- Modify: `bdp_benchmark/env_factory.py`
- Modify: `bdp_benchmark/runner.py`
- Test: `tests/test_highway_adapter.py`
- Test: `tests/test_runner.py`

1. Add failing tests that terminal info contains `environment_variant`, monitor paths are grouped by variant, and mixed evaluation groups returns by variant.
2. Run the focused tests and confirm the failures.
3. Annotate Highway step info, include it in Monitor output, and aggregate mixed evaluation results while retaining the existing return-list API.
4. Rerun the focused tests.

### Task 4: Reproducible mixed-road jobs and documentation

**Files:**
- Create: `scripts/job_scripts/highway_mixed_native_builtin_benchmark.yaml`
- Create: `scripts/job_scripts/highway_mixed_native_bdp_frenet_benchmark.yaml`
- Modify: `scripts/job_scripts/README.md`
- Modify: `README.md`
- Test: `tests/test_config.py`

1. Add paired jobs using the same three variants, common observation contract, six workers, seed, and PPO settings.
2. Document fixed round-robin assignment, observation enforcement, and evaluation semantics.
3. Run all configuration tests.

### Task 5: Verification

1. Run `python3 -m pytest -q`.
2. Run a tiny builtin mixed-road smoke train.
3. Run a tiny BDP-Frenet mixed-road smoke train.
4. Inspect `git diff --check` and `git status --short` without altering pre-existing user changes.

