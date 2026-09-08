from __future__ import annotations

import numpy as np
import pytest

from bdp_benchmark.visualization import CandidateOverlay, candidate_score_colors, make_overlay


def test_candidate_score_colors_are_low_red_high_blue_and_selected_green() -> None:
    colors = candidate_score_colors([0.0, 1.0, 2.0], selected_index=1)

    assert colors[0][0] > colors[0][2]
    assert colors[2][2] > colors[2][0]
    np.testing.assert_allclose(colors[1], (0.0, 1.0, 0.18, 1.0))


def test_candidate_score_colors_use_neutral_center_for_equal_scores() -> None:
    colors = candidate_score_colors([4.0, 4.0, 4.0], selected_index=None)

    assert colors == [(0.88, 0.88, 0.88, 0.55)] * 3


def test_candidate_score_colors_reject_wrong_score_count() -> None:
    with pytest.raises(ValueError, match="candidate count"):
        candidate_score_colors([1.0, 2.0], candidate_count=3)


def test_make_overlay_preserves_exact_features_and_trajectory_payload() -> None:
    features = np.zeros((2, 5), dtype=np.float32)
    trajectories = np.zeros((2, 4, 4), dtype=np.float32)
    overlay = make_overlay(
        labels=np.asarray([0, 1]),
        features=features,
        trajectories=trajectories,
        scores=np.asarray([0.1, 0.2]),
        selected_index=1,
        pid_target_xy=np.asarray([1.0, 2.0]),
    )

    assert isinstance(overlay, CandidateOverlay)
    np.testing.assert_array_equal(overlay.features, features)
    np.testing.assert_array_equal(overlay.trajectories, trajectories)
    np.testing.assert_allclose(overlay.pid_target_xy, [1.0, 2.0])


def test_make_overlay_rejects_mismatched_feature_and_trajectory_axes() -> None:
    with pytest.raises(ValueError, match="candidate axis"):
        make_overlay(
            labels=np.asarray([0, 1]),
            features=np.zeros((2, 5), dtype=np.float32),
            trajectories=np.zeros((3, 4, 4), dtype=np.float32),
            scores=None,
            selected_index=0,
        )
