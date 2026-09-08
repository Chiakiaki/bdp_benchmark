from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bdp_benchmark.runner import _candidate_features_for_observation
from bdp_benchmark.common.contracts import CandidateSet


def _candidate_set() -> CandidateSet:
    return CandidateSet(
        labels=np.asarray([0, 1]),
        mask=np.ones(2, dtype=np.float32),
        trajectories=np.zeros((2, 3, 4), dtype=np.float32),
        features=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )


def test_visual_check_uses_exact_structured_candidate_rows() -> None:
    args = SimpleNamespace(policy_mode="bdp", candidate_sampler="frenet")
    observation = {"candidates": np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)}

    features = _candidate_features_for_observation(args, _candidate_set(), observation)

    np.testing.assert_allclose(features, _candidate_set().features)


def test_visual_check_rejects_structured_feature_mismatch() -> None:
    args = SimpleNamespace(policy_mode="bdp", candidate_sampler="native_action_frenet")
    observation = {"candidates": np.asarray([[[9.0, 9.0], [0.3, 0.4]]], dtype=np.float32)}

    with pytest.raises(RuntimeError, match="does not match"):
        _candidate_features_for_observation(args, _candidate_set(), observation)
