from __future__ import annotations

import numpy as np
import pytest

from bdp_benchmark.common.features import encode_ego_local_features
from bdp_benchmark.common.frenet import FrenetState, ReferencePath, frenet_to_world, generate_frenet_trajectories


def test_vectorized_frenet_polynomials_meet_start_and_terminal_constraints() -> None:
    initial = FrenetState(s=2.0, s_dot=10.0, s_ddot=0.25, d=0.3, d_dot=-0.1, d_ddot=0.05)
    target_d = np.asarray([-3.0, 0.0, 3.0])
    target_speed = np.asarray([5.0, 10.0, 15.0])

    trajectories = generate_frenet_trajectories(
        initial,
        target_d=target_d,
        target_speed=target_speed,
        horizon_s=2.0,
        sample_count=9,
    )

    assert trajectories.shape == (3, 9, 6)
    np.testing.assert_allclose(trajectories[:, 0, 0], initial.s, atol=1e-10)
    np.testing.assert_allclose(trajectories[:, 0, 1], initial.d, atol=1e-10)
    np.testing.assert_allclose(trajectories[:, 0, 2], initial.s_dot, atol=1e-10)
    np.testing.assert_allclose(trajectories[:, 0, 3], initial.d_dot, atol=1e-10)
    np.testing.assert_allclose(trajectories[:, -1, 1], target_d, atol=1e-8)
    np.testing.assert_allclose(trajectories[:, -1, 2], target_speed, atol=1e-8)
    np.testing.assert_allclose(trajectories[:, -1, 3:], 0.0, atol=1e-8)


def test_frenet_generator_supports_environment_and_candidate_batch_axes() -> None:
    initial = FrenetState(
        s=np.asarray([0.0, 4.0]),
        s_dot=np.asarray([8.0, 12.0]),
        s_ddot=np.zeros(2),
        d=np.zeros(2),
        d_dot=np.zeros(2),
        d_ddot=np.zeros(2),
    )
    target_d = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
    target_speed = np.asarray([[8.0, 9.0, 10.0], [10.0, 12.0, 14.0]])

    trajectories = generate_frenet_trajectories(initial, target_d, target_speed, horizon_s=1.5, sample_count=6)

    assert trajectories.shape == (2, 3, 6, 6)
    np.testing.assert_allclose(trajectories[:, :, -1, 1], target_d, atol=1e-8)
    np.testing.assert_allclose(trajectories[:, :, -1, 2], target_speed, atol=1e-8)


def test_frenet_to_world_follows_curved_reference_and_lateral_left_convention() -> None:
    angles = np.linspace(0.0, np.pi / 2.0, 101)
    reference = ReferencePath.from_xy(np.stack([10.0 * np.sin(angles), 10.0 * (1.0 - np.cos(angles))], axis=-1))
    s = np.asarray([[0.0, reference.length * 0.5, reference.length]])
    d = np.asarray([[1.0, 1.0, 1.0]])

    world = frenet_to_world(reference, s, d, speed=np.full_like(s, 5.0))

    assert world.shape == (1, 3, 4)
    assert world[0, 0, 1] > 0.9
    assert world[0, -1, 0] < reference.xy[-1, 0]
    assert world[0, -1, 2] > 1.4


def test_ego_local_features_are_invariant_to_world_translation_and_rotation() -> None:
    trajectory = np.asarray(
        [[[1.0, 0.0, 0.0, 4.0], [2.0, 0.5, 0.2, 5.0], [3.0, 1.0, 0.4, 6.0]]],
        dtype=np.float64,
    )
    original = encode_ego_local_features(
        trajectory,
        ego_xy=np.asarray([0.0, 0.0]),
        ego_heading=0.0,
        position_scale_m=10.0,
        speed_scale_mps=20.0,
    )

    rotation = 1.1
    c, s = np.cos(rotation), np.sin(rotation)
    matrix = np.asarray([[c, -s], [s, c]])
    translated_xy = trajectory[..., :2] @ matrix.T + np.asarray([8.0, -3.0])
    transformed = trajectory.copy()
    transformed[..., :2] = translated_xy
    transformed[..., 2] += rotation
    rotated = encode_ego_local_features(
        transformed,
        ego_xy=np.asarray([8.0, -3.0]),
        ego_heading=rotation,
        position_scale_m=10.0,
        speed_scale_mps=20.0,
    )

    assert original.shape == (1, 15)
    np.testing.assert_allclose(rotated, original, atol=1e-7)


def test_frenet_to_world_can_preserve_direct_initial_heading_when_stationary() -> None:
    reference = ReferencePath.from_xy(np.asarray([[0.0, 0.0], [10.0, 0.0]]))
    s = np.asarray([[0.0, 1.0, 2.0]])
    d = np.zeros_like(s)

    world = frenet_to_world(
        reference,
        s,
        d,
        speed=np.zeros_like(s),
        longitudinal_speed=np.zeros_like(s),
        lateral_speed=np.zeros_like(s),
        initial_heading_rad=np.asarray([1.25]),
    )

    assert world[0, 0, 2] == pytest.approx(1.25)
    assert world[0, 1, 2] == pytest.approx(0.0)
