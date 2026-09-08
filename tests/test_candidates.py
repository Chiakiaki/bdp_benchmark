from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from bdp_benchmark.common.candidates import FrenetCandidateSampler
from bdp_benchmark.common.contracts import CandidateSet


class CandidateProviderEnv:
    def __init__(self) -> None:
        self.calls = 0

    def build_candidate_set(self) -> CandidateSet:
        self.calls += 1
        return CandidateSet(
            labels=np.asarray([0, 1, 2], dtype=np.int64),
            mask=np.ones(3, dtype=np.float32),
            trajectories=np.zeros((3, 4, 4), dtype=np.float32),
            features=np.arange(36, dtype=np.float32).reshape(3, 12) / 36.0,
        )


def test_frenet_candidate_sampler_reads_environment_candidate_set() -> None:
    raw_env = CandidateProviderEnv()
    sampler = FrenetCandidateSampler(gym.spaces.Discrete(3), feature_dim=12)

    candidates, mask, labels = sampler.sample(env=raw_env, num_envs=1)

    assert raw_env.calls == 1
    assert sampler.candidate_space.shape == (12,)
    assert candidates.shape == (1, 3, 12)
    assert mask.shape == (1, 3)
    assert labels.shape == (1, 3)
    np.testing.assert_array_equal(labels[0], [0, 1, 2])


def test_frenet_candidate_sampler_rejects_wrong_feature_width() -> None:
    sampler = FrenetCandidateSampler(gym.spaces.Discrete(3), feature_dim=10)
    with pytest.raises(ValueError, match="feature width"):
        sampler.sample(env=CandidateProviderEnv(), num_envs=1)


def test_candidate_set_rejects_inconsistent_candidate_axes() -> None:
    with pytest.raises(ValueError, match="candidate axis"):
        CandidateSet(
            labels=np.asarray([0, 1]),
            mask=np.ones(3),
            trajectories=np.zeros((2, 4, 4)),
            features=np.zeros((2, 10)),
        )
