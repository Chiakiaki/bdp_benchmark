from __future__ import annotations

import numpy as np
import pytest

from bdp_benchmark.config import parse_args
from bdp_benchmark.env_factory import make_single_env, make_vector_env


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
        raw_actions = 5 if simulator == "highway" else 25 if execution_mode == "native_controller" else 15
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


def test_mixed_highway_worker_assignment_is_fixed_round_robin(tmp_path) -> None:
    args = parse_args(
        [
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--n_envs",
            "6",
            "--vec_env",
            "dummy",
            "--monitor_mode",
            "none",
            "--environment_config",
            '{"vehicles_count":0}',
        ]
    )

    env = make_vector_env(args, log_dir=tmp_path, is_train=True)
    try:
        assert env.get_attr("environment_variant") == [
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
        ]
    finally:
        env.close()


def test_make_single_env_selects_variant_from_rank() -> None:
    args = parse_args(
        [
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--n_envs",
            "6",
            "--monitor_mode",
            "none",
        ]
    )

    env = make_single_env(args, rank=4)
    try:
        assert env.environment_variant == "merge-v1"
    finally:
        env.close()


def test_mixed_highway_validates_final_wrapped_spaces(tmp_path) -> None:
    args = parse_args(
        [
            "--environment_variants",
            "highway-fast-v0",
            "intersection-v0",
            "--n_envs",
            "2",
            "--vec_env",
            "dummy",
            "--monitor_mode",
            "none",
        ]
    )

    with pytest.raises(ValueError, match="incompatible.*intersection-v0"):
        make_vector_env(args, log_dir=tmp_path, is_train=True)


def test_mixed_highway_bdp_spaces_are_compatible(tmp_path) -> None:
    args = parse_args(
        [
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--n_envs",
            "3",
            "--vec_env",
            "dummy",
            "--monitor_mode",
            "none",
            "--policy_mode",
            "bdp",
            "--candidate_sampler",
            "native_action_frenet",
            "--trajectory_sample_count",
            "5",
        ]
    )

    env = make_vector_env(args, log_dir=tmp_path, is_train=True)
    try:
        obs = env.reset()
        assert obs["obs"].shape == (3, 5, 5)
        assert obs["candidates"].shape == (3, 5, 25)
    finally:
        env.close()


def test_mixed_highway_monitor_files_are_grouped_by_variant(tmp_path) -> None:
    args = parse_args(
        [
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--n_envs",
            "3",
            "--vec_env",
            "dummy",
            "--monitor_mode",
            "env",
        ]
    )

    env = make_vector_env(args, log_dir=tmp_path, is_train=True)
    try:
        assert (tmp_path / "monitor" / "highway-fast-v0" / "rank_0" / "env_0.monitor.csv").exists()
        assert (tmp_path / "monitor" / "merge-v1" / "rank_1" / "env_1.monitor.csv").exists()
        assert (tmp_path / "monitor" / "roundabout-v1" / "rank_2" / "env_2.monitor.csv").exists()
    finally:
        env.close()


def test_mixed_highway_vec_monitor_records_variant(tmp_path) -> None:
    args = parse_args(
        [
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--n_envs",
            "3",
            "--vec_env",
            "dummy",
            "--monitor_mode",
            "vec",
        ]
    )

    env = make_vector_env(args, log_dir=tmp_path, is_train=True)
    try:
        assert env.info_keywords == ("environment_variant",)
    finally:
        env.close()


def test_mixed_highway_evaluation_uses_one_worker_per_variant(tmp_path) -> None:
    args = parse_args(
        [
            "--run_mode",
            "evaluate",
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--n_envs",
            "1",
            "--vec_env",
            "dummy",
            "--monitor_mode",
            "none",
        ]
    )

    env = make_vector_env(args, log_dir=tmp_path, is_train=False)
    try:
        assert env.num_envs == 3
        assert env.get_attr("environment_variant") == ["highway-fast-v0", "merge-v1", "roundabout-v1"]
    finally:
        env.close()


def test_mixed_highway_visual_check_stays_single_environment(tmp_path) -> None:
    args = parse_args(
        [
            "--visual_check",
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--monitor_mode",
            "none",
        ]
    )

    env = make_vector_env(args, log_dir=tmp_path, is_train=False)
    try:
        assert env.num_envs == 1
        assert env.get_attr("environment_variant") == ["highway-fast-v0"]
    finally:
        env.close()


def test_mixed_highway_visual_check_wraps_variant_index(tmp_path, capsys) -> None:
    args = parse_args(
        [
            "--visual_check",
            "--environment_variants",
            "highway-fast-v0",
            "merge-v1",
            "roundabout-v1",
            "--visual_check_variant_index",
            "4",
            "--monitor_mode",
            "none",
        ]
    )

    env = make_vector_env(args, log_dir=tmp_path, is_train=False)
    try:
        assert env.num_envs == 1
        assert env.get_attr("environment_variant") == ["merge-v1"]
        assert (
            "Visual check environment: requested_index=4 resolved_index=1 env_id=merge-v1"
            in capsys.readouterr().out
        )
    finally:
        env.close()
