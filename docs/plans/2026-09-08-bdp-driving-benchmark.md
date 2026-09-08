# BDP Driving Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone, reproducible benchmark that compares built-in categorical policies and BDP candidate scoring across HighwayEnv and MetaDrive under native-controller and Frenet-PID execution.

**Architecture:** A simulator-independent core owns vectorized Frenet generation, candidate feature encoding, contracts, and tracking. Thin HighwayEnv and MetaDrive adapters preserve upstream environments and expose a common discrete candidate interface. A shared CLI imports algorithm/model training utilities from `critic_based_rl` and applies CLI-over-YAML configuration precedence.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 2.8, PyTorch, HighwayEnv, MetaDrive, PyYAML, pytest, `critic_based_rl`.

---

### Task 1: Extend the generic candidate sampler contract

**Files:**
- Modify: `/home/dell/project/critic_based_rl/samplers.py`
- Modify: `/home/dell/project/critic_based_rl/envs.py`
- Test: `/home/dell/project/critic_based_rl/tests/test_generic_discrete_env.py`

1. Add failing tests for a sampler whose candidate feature width differs from the raw action count and whose output depends on the passed environment.
2. Run the focused test and verify it fails because `GenericDiscreteCandidateEnv` assumes `D == action_space.n` and omits `env=`.
3. Add an optional sampler `candidate_space` contract, implement it for one-hot samplers, and pass the raw environment into `sample()`.
4. Derive candidate observation bounds and width from `candidate_space`, preserving existing one-hot behavior.
5. Run the focused and full critic-based tests.

### Task 2: Scaffold configuration and common contracts

**Files:**
- Create: `/home/dell/project/bdp_benchmark/pyproject.toml`
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/__init__.py`
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/config.py`
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/common/contracts.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_config.py`

1. Add failing tests for CLI-over-YAML precedence, mode combinations, horizon validation, and MetaDrive action-grid counts.
2. Implement dataclasses/enums for simulator, execution mode, candidate mode, Frenet settings, and PID settings.
3. Implement parser construction using shared `critic_based_rl` argument registration plus benchmark arguments.
4. Implement source/resolved config backup.
5. Run configuration tests.

### Task 3: Implement vectorized Frenet generation

**Files:**
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/common/frenet.py`
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/common/features.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_frenet.py`

1. Add failing tests for quintic lateral and quartic longitudinal boundary conditions with batched `K` targets.
2. Implement closed-form coefficient solves and broadcast sampling over arbitrary leading dimensions.
3. Add failing tests for straight and curved reference interpolation and ego-local feature encoding.
4. Implement vectorized route interpolation, Frenet-to-world conversion, heading extraction, ego-frame rotation, normalization, and flattening.
5. Verify output shapes `[K,H,S]` and `[K,5H]`, finite values, and orientation invariance.

### Task 4: Add the common candidate sampler and tracker

**Files:**
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/common/candidates.py`
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/common/tracking.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_candidates.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_tracking.py`

1. Add failing tests for candidate labels/masks, sampler shapes, and environment-dependent sampling.
2. Implement `FrenetCandidateSampler` as a `critic_based_rl.ExternalCandidateSampler` using the environment's candidate-provider protocol.
3. Add failing tests for steering/acceleration signs, clipping, and selected-reference lifetime.
4. Implement a deterministic cross-track/heading/speed tracker returning physical control commands.
5. Run common unit tests.

### Task 5: Implement HighwayEnv integration

**Files:**
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/highway/adapter.py`
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/highway/env.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_highway_adapter.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_highway_parity.py`

1. Add failing tests for the five semantic action targets, lane-boundary no-op behavior, and speed-bin targets.
2. Implement route/lane extraction and action-conditioned candidate generation without mutating the live vehicle.
3. Add a native wrapper that delegates `step()` unchanged and exposes candidate sets.
4. Add a Frenet-PID sidecar environment that keeps upstream reward/observation/termination logic and replaces only action execution.
5. Add seeded native parity and headless reset/step smoke tests.

### Task 6: Implement MetaDrive integration

**Files:**
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/metadrive/adapter.py`
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/metadrive/env.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_metadrive_adapter.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_metadrive_smoke.py`

1. Add failing tests for decoding the 25-action grid and mapping five steering magnitudes to distinct continuous lateral targets.
2. Implement current navigation-lane extraction and canonical lateral-sign conversion.
3. Implement native action descriptors while delegating selected actions unchanged to `EnvInputPolicy`.
4. Implement the five-intent Frenet-PID execution path using a persisted selected reference.
5. Run headless native and Frenet-PID smoke tests, closing every MetaDrive engine instance explicitly.

### Task 7: Build environment factory and SB3 runner

**Files:**
- Create: `/home/dell/project/bdp_benchmark/bdp_benchmark/env_factory.py`
- Create: `/home/dell/project/bdp_benchmark/scripts/train.py`
- Create: `/home/dell/project/bdp_benchmark/scripts/evaluate.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_env_factory.py`
- Test: `/home/dell/project/bdp_benchmark/tests/test_runner.py`

1. Add failing tests for all six simulator/execution/policy combinations and expected action/candidate dimensions.
2. Implement raw, one-hot BDP, and Frenet BDP wrapping with `GenericDiscreteCandidateEnv`.
3. Use `DummyVecEnv` for one environment and configurable `SubprocVecEnv` for process-isolated parallel runs, especially MetaDrive.
4. Reuse `critic_based_rl.train_sb3_model`, model factories, monitoring, and callbacks.
5. Implement deterministic evaluation and resolved-config storage.

### Task 8: Add reproducible jobs and documentation

**Files:**
- Create: `/home/dell/project/bdp_benchmark/scripts/job_scripts/highway_native_builtin.yaml`
- Create: `/home/dell/project/bdp_benchmark/scripts/job_scripts/highway_native_bdp_one_hot.yaml`
- Create: `/home/dell/project/bdp_benchmark/scripts/job_scripts/highway_native_bdp_frenet.yaml`
- Create: `/home/dell/project/bdp_benchmark/scripts/job_scripts/highway_frenet_pid_builtin.yaml`
- Create: `/home/dell/project/bdp_benchmark/scripts/job_scripts/highway_frenet_pid_bdp_one_hot.yaml`
- Create: `/home/dell/project/bdp_benchmark/scripts/job_scripts/highway_frenet_pid_bdp_frenet.yaml`
- Create corresponding six MetaDrive YAML files.
- Create: `/home/dell/project/bdp_benchmark/README.md`

1. Add self-contained installation instructions that keep upstream repos editable and unmodified.
2. Document the experiment matrix, fairness limits, config precedence, output layout, and native-versus-Frenet interpretation.
3. Add train/evaluate commands and short smoke configurations.
4. Validate every YAML parses and maps to a constructible environment.

### Task 9: End-to-end verification

1. Run the complete `bdp_benchmark` unit suite.
2. Run the complete affected `critic_based_rl` suite.
3. Run native parity tests against unwrapped HighwayEnv.
4. Run one reset/step smoke test for both MetaDrive modes.
5. Run tiny PPO training for built-in, one-hot BDP, and Frenet BDP on HighwayEnv.
6. Run a minimal MetaDrive PPO smoke when host graphics/physics initialization permits it.
7. Inspect git status in all touched repositories and confirm upstream HighwayEnv and MetaDrive source trees were not modified.
