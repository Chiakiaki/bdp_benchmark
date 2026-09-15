"""Shared data contracts between trajectory code and simulator adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EgoState:
    x: float
    y: float
    heading: float
    speed: float

    @property
    def xy(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float64)


@dataclass(frozen=True)
class CandidateSet:
    labels: np.ndarray
    mask: np.ndarray
    trajectories: np.ndarray
    features: np.ndarray
    # Optional direct controller setpoints; None selects speed from the sampled reference.
    speed_targets: np.ndarray | None = None

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        mask = np.asarray(self.mask)
        trajectories = np.asarray(self.trajectories)
        features = np.asarray(self.features)
        if labels.ndim != 1 or mask.ndim != 1 or trajectories.ndim != 3 or features.ndim != 2:
            raise ValueError("CandidateSet expects labels [K], mask [K], trajectories [K,H,S], and features [K,D]")
        candidate_count = labels.shape[0]
        if not (
            mask.shape[0] == candidate_count
            and trajectories.shape[0] == candidate_count
            and features.shape[0] == candidate_count
        ):
            raise ValueError("CandidateSet arrays must share the same candidate axis")
        if self.speed_targets is not None:
            targets = np.asarray(self.speed_targets)
            if targets.shape != (candidate_count,) or not np.isfinite(targets).all() or np.any(targets < 0):
                raise ValueError("speed_targets must be a finite non-negative [K] vector")


@dataclass(frozen=True)
class PhysicalControl:
    steering_rad: float
    acceleration_mps2: float
