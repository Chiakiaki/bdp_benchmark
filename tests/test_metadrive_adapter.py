from __future__ import annotations

import numpy as np

from bdp_benchmark.metadrive.adapter import decode_discrete_action_grid, native_action_targets


def test_metadrive_action_grid_matches_env_input_policy_order() -> None:
    steering, throttle = decode_discrete_action_grid(steering_dim=5, throttle_dim=5)

    assert steering.shape == (25,)
    assert throttle.shape == (25,)
    np.testing.assert_allclose(steering[:5], [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(throttle[[0, 5, 10, 15, 20]], [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_all_five_metadrive_steering_bins_have_distinct_continuous_lateral_targets() -> None:
    target_d, target_speed = native_action_targets(
        current_d=0.2,
        current_speed=8.0,
        steering_dim=5,
        throttle_dim=5,
        lateral_span_m=3.5,
        speed_span_mps=4.0,
        minimum_speed_mps=0.0,
        maximum_speed_mps=20.0,
    )

    center_throttle_targets = target_d[10:15]
    assert len(np.unique(center_throttle_targets)) == 5
    # MetaDrive steering is positive left, while its lane-local lateral axis is
    # positive right. The descriptor must account for that sign difference.
    np.testing.assert_allclose(center_throttle_targets, [3.7, 1.95, 0.2, -1.55, -3.3])
    np.testing.assert_allclose(target_speed[10:15], 8.0)
