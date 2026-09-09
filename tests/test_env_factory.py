from __future__ import annotations

import numpy as np
import pytest

from bdp_benchmark.config import parse_args
from bdp_benchmark.env_factory import make_single_env


@pytest.mark.parametrize("simulator", ["highway", "metadrive"])
@pytest.mark.parametrize("execution_mode", ["native_controller", "frenet_pid", "frenet_pid_v2"])
@pytest.mark.parametrize(
    ("policy_mode", "structured"),
    [("builtin", False), ("one_hot", False), ("frenet", True)],
)
def test_factory_constructs_full_policy_execution_matrix(
    simulator: str,
    execution_mode: str,
    policy_mode: str,
    structured: bool,
) -> None:
    sampler = "one_hot"
    actual_policy_mode = "builtin" if policy_mode == "builtin" else "bdp"
    if structured:
        sampler = "native_action_frenet" if execution_mode == "native_controller" else "frenet"
    env_config = (
        '{"vehicles_count": 0, "initial_lane_id": 1, "duration": 2}'
        if simulator == "highway"
        else '{"use_render": false, "traffic_density": 0.0, "num_scenarios": 1, "map": "S", "horizon": 10, "log_level": 50}'
    )
    args = parse_args(
        [
            "--simulator",
            simulator,
            "--trajectory_execution_mode",
            execution_mode,
            "--policy_mode",
            actual_policy_mode,
            "--candidate_sampler",
            sampler,
            "--trajectory_sample_count",
            "5",
            "--environment_config",
            env_config,
        ]
    )
    env = make_single_env(args)
    try:
        obs, _ = env.reset(seed=0)
        raw_actions = 5 if execution_mode in ("frenet_pid", "frenet_pid_v2") or simulator == "highway" else 25
        assert env.action_space.n == raw_actions
        if actual_policy_mode == "builtin":
            assert isinstance(obs, np.ndarray)
        else:
            assert set(obs) == {"obs", "candidates", "candidate_mask"}
            assert obs["candidates"].shape[0] == raw_actions
            expected_width = 25 if structured else raw_actions
            assert obs["candidates"].shape[1] == expected_width
    finally:
        env.close()


def test_factory_rejects_dummy_vectorization_for_parallel_metadrive() -> None:
    with pytest.raises(ValueError, match="requires vec_env=subproc"):
        parse_args(["--simulator", "metadrive", "--n_envs", "2", "--vec_env", "dummy"])
