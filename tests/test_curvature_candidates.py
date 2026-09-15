from dataclasses import replace

import numpy as np
import pytest
from gymnasium import spaces

from bdp_benchmark.common.candidates import CandidateGenerationConfig
from bdp_benchmark.common.contracts import EgoState
from bdp_benchmark.common.features import encode_ego_local_features
from bdp_benchmark.common import frenet
from bdp_benchmark.common.tracking import TrackerConfig
from bdp_benchmark.metadrive.env import MetaDriveBenchmarkEnv
from test_metadrive_adapter import _transition_adapter


def test_curvature_rollout_is_mirrored_and_has_constant_radius_and_speed_profile():
    origin = EgoState(0.0, 0.0, 0.0, 6.0)
    curves = frenet.generate_constant_curvature_trajectories(
        origin, np.array([3.5, 3.5, 8.5]), np.array([0.2, -0.2, 0.0]),
        horizon_s=2.0, sample_count=11,
    )
    assert curves.shape == (3, 11, 4)
    np.testing.assert_allclose(curves[:, 0], [[0, 0, 0, 6]] * 3)
    np.testing.assert_allclose(curves[:, -1, 3], [3.5, 3.5, 8.5])
    np.testing.assert_allclose(curves[0, :, 0], curves[1, :, 0])
    np.testing.assert_allclose(curves[0, :, 1:3], -curves[1, :, 1:3])
    np.testing.assert_allclose(np.linalg.norm(curves[0, :, :2] - [0, 5], axis=1), 5)
    np.testing.assert_allclose(curves[2, :, 1:3], 0)
    assert curves[0, 2, 3] == pytest.approx(5.74)  # 0.4 s preview of a 2 s speed transition


def test_curvature_rollout_rotates_with_origin_and_stays_finite_at_zero_speed():
    curves = frenet.generate_constant_curvature_trajectories(
        EgoState(3, 4, np.pi / 2, 0), np.array([0.0, 0.0]), np.array([0.2, -0.2]),
        horizon_s=2, sample_count=11,
    )
    np.testing.assert_allclose(curves, np.broadcast_to([3, 4, np.pi / 2, 0], curves.shape))
    base = frenet.generate_constant_curvature_trajectories(
        EgoState(0, 0, 0, 3), np.array([5.0]), np.array([0.2]), horizon_s=2, sample_count=11,
    )
    rotated = frenet.generate_constant_curvature_trajectories(
        EgoState(3, 4, np.pi / 2, 3), np.array([5.0]), np.array([0.2]), horizon_s=2, sample_count=11,
    )
    np.testing.assert_allclose(rotated[..., 0], 3 - base[..., 1])
    np.testing.assert_allclose(rotated[..., 1], 4 + base[..., 0])
    np.testing.assert_allclose(rotated[..., 2], base[..., 2] + np.pi / 2)


@pytest.mark.parametrize("reference_mode", ["lane_segment", "route_continuous"])
@pytest.mark.parametrize("nominal", [None, EgoState(9, 0.1, 0.3, 5)])
def test_mixed_candidates_preserve_legacy_prefix_and_use_selected_reference(reference_mode, nominal):
    adapter = _transition_adapter(reference_mode)
    vehicle = adapter.vehicle
    vehicle.FRONT_WHEELBASE = 1.0
    vehicle.REAR_WHEELBASE = 1.5
    vehicle.max_steering = 40.0
    adapter.config = replace(adapter.config, include_curvature_candidates=True, maximum_target_speed_mps=12)
    # Keep the same endpoint limits for a bit-exact legacy prefix comparison.
    legacy_adapter = _transition_adapter(reference_mode)
    legacy_adapter.config = replace(legacy_adapter.config, maximum_target_speed_mps=12)
    legacy = legacy_adapter.build_candidate_set("frenet_pid_v2", nominal)
    adapter.env.action_space = spaces.Discrete(25)
    adapter.max_steering_rad = 0.5
    result = adapter.build_candidate_set("frenet_pid_v2", nominal)
    assert result.features.shape == (25, 55)
    np.testing.assert_array_equal(result.features[:15], legacy.features)
    np.testing.assert_array_equal(result.trajectories[:15], legacy.trajectories)
    np.testing.assert_array_equal(result.labels, np.arange(25))
    np.testing.assert_array_equal(result.mask, np.ones(25))
    origin = adapter.ego_state() if nominal is None else nominal
    extra = result.trajectories[15:]
    np.testing.assert_allclose(extra[:, 0], np.tile([origin.x, origin.y, origin.heading, origin.speed], (10, 1)))
    rear_k = np.tan(0.9 * 0.5) / 2.5
    center_k = rear_k / np.sqrt(1 + (1.5 * rear_k)**2)
    expected_speeds = np.repeat(np.clip(origin.speed + np.linspace(-5, 5, 5), 0, 12), 2)
    expected = frenet.generate_constant_curvature_trajectories(
        origin, expected_speeds, np.tile([center_k, -center_k], 5), horizon_s=2, sample_count=11,
    )
    np.testing.assert_allclose(extra, expected, atol=1e-6)
    np.testing.assert_allclose(result.features[15:], encode_ego_local_features(
        expected, ego_xy=origin.xy, ego_heading=origin.heading,
        position_scale_m=50, speed_scale_mps=40, lateral_normal_sign=-1,
    ), atol=1e-6)


@pytest.mark.parametrize("mode", ["frenet_pid", "frenet_pid_v2"])
def test_mixed_metadrive_action_space_and_execution(mode):
    env = MetaDriveBenchmarkEnv(
        execution_mode=mode,
        generation_config=CandidateGenerationConfig(include_curvature_candidates=True),
        tracker_config=TrackerConfig(controller_mode="adaptive_pursuit"),
        env_config={"map": "S", "num_scenarios": 1, "traffic_density": 0, "log_level": 50},
        reference_mode="route_continuous",
    )
    try:
        obs, _ = env.reset(seed=0)
        assert env.action_space.n == env.env.action_space.n == 25
        assert env._pid_policy().get_input_space().n == 25
        candidates = env.build_candidate_set()
        np.testing.assert_equal(candidates.trajectories.shape, (25, 11, 4))
        for index in (15, 16, 23, 24):
            candidates = env.build_candidate_set()
            next_obs, reward, *_ = env.step(index)
            np.testing.assert_array_equal(env._pid_policy().tracker.reference, candidates.trajectories[index])
            assert next_obs.shape == obs.shape
            assert np.isfinite(reward)
    finally:
        env.close()


def test_defaults_preserve_15_actions_and_invalid_fraction_is_rejected():
    assert CandidateGenerationConfig().include_curvature_candidates is False
    for fraction in (0, -0.1, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="curvature_steering_fraction"):
            CandidateGenerationConfig(curvature_steering_fraction=fraction)
