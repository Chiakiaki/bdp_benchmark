"""Candidate sampler bridge for critic_based_rl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from critic_based_rl.samplers import ExternalCandidateSampler

from .contracts import CandidateSet


@dataclass(frozen=True)
class CandidateGenerationConfig:
    horizon_s: float = 2.0
    sample_count: int = 11
    position_scale_m: float = 50.0
    speed_scale_mps: float = 40.0
    native_lateral_span_m: float = 3.5
    native_speed_span_mps: float = 5.0
    minimum_target_speed_mps: float = 0.0
    maximum_target_speed_mps: float = 40.0
    lane_change_width_scale: float = 1.0
    speed_delta_mps: float = 5.0

    def __post_init__(self) -> None:
        if self.horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if self.position_scale_m <= 0.0 or self.speed_scale_mps <= 0.0:
            raise ValueError("feature scales must be positive")

    @property
    def feature_dim(self) -> int:
        return 5 * self.sample_count


class FrenetCandidateSampler(ExternalCandidateSampler):
    """Read normalized, state-dependent candidate features from a benchmark env."""

    def __init__(self, action_space: spaces.Discrete, *, feature_dim: int):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("FrenetCandidateSampler requires a Discrete action space")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.action_count = int(action_space.n)
        self.feature_dim = int(feature_dim)
        self.candidate_space = spaces.Box(low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32)

    def sample(
        self,
        raw_obs: np.ndarray | None = None,
        *,
        env: Any | None = None,
        num_envs: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del raw_obs
        env_count = 1 if num_envs is None else int(num_envs)
        if env_count != 1:
            raise ValueError("FrenetCandidateSampler expects one raw env per Gymnasium wrapper")
        if env is None or not hasattr(env, "build_candidate_set"):
            raise TypeError("FrenetCandidateSampler requires an env with build_candidate_set()")
        candidate_set = env.build_candidate_set()
        if not isinstance(candidate_set, CandidateSet):
            raise TypeError("build_candidate_set() must return CandidateSet")
        if candidate_set.features.shape[0] != self.action_count:
            raise ValueError(
                f"candidate count {candidate_set.features.shape[0]} does not match action count {self.action_count}"
            )
        if candidate_set.features.shape[1] != self.feature_dim:
            raise ValueError(
                f"candidate feature width {candidate_set.features.shape[1]} does not match expected {self.feature_dim}"
            )
        return (
            np.asarray(candidate_set.features, dtype=np.float32)[None],
            np.asarray(candidate_set.mask, dtype=np.float32)[None],
            np.asarray(candidate_set.labels, dtype=np.int64)[None],
        )
