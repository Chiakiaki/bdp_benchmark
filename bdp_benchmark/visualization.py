"""Simulator-independent candidate overlay state and score colors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


NEUTRAL_COLOR = (0.88, 0.88, 0.88, 0.55)
LOW_SCORE_COLOR = (1.00, 0.18, 0.05)
HIGH_SCORE_COLOR = (0.10, 0.25, 1.00)
SELECTED_COLOR = (0.0, 1.0, 0.18, 1.0)


@dataclass(frozen=True)
class CandidateOverlay:
    labels: np.ndarray
    features: np.ndarray
    trajectories: np.ndarray
    scores: np.ndarray | None
    selected_index: int | None
    pid_target_xy: np.ndarray | None = None

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        features = np.asarray(self.features)
        trajectories = np.asarray(self.trajectories)
        if labels.ndim != 1 or features.ndim != 2 or trajectories.ndim != 3:
            raise ValueError("CandidateOverlay expects labels [K], features [K,D], trajectories [K,H,S]")
        if features.shape[0] != labels.shape[0] or trajectories.shape[0] != labels.shape[0]:
            raise ValueError("CandidateOverlay arrays must share the candidate axis")
        if trajectories.shape[-1] < 2:
            raise ValueError("CandidateOverlay trajectories need at least x/y coordinates")
        if self.scores is not None:
            scores = np.asarray(self.scores).reshape(-1)
            if scores.shape[0] != labels.shape[0]:
                raise ValueError("CandidateOverlay scores must share the candidate axis")
            if not np.isfinite(scores).all():
                raise ValueError("CandidateOverlay scores must be finite")
        if self.selected_index is not None and not 0 <= int(self.selected_index) < labels.shape[0]:
            raise ValueError("CandidateOverlay selected_index is outside the candidate set")
        if self.pid_target_xy is not None and np.asarray(self.pid_target_xy).reshape(-1).shape[0] < 2:
            raise ValueError("pid_target_xy must contain x/y")


def _interpolate_color(value: float) -> tuple[float, float, float, float]:
    value = max(-1.0, min(1.0, float(value)))
    if abs(value) < 1.0e-6:
        return NEUTRAL_COLOR
    target = HIGH_SCORE_COLOR if value > 0.0 else LOW_SCORE_COLOR
    amount = abs(value)
    color = tuple((1.0 - amount) * NEUTRAL_COLOR[index] + amount * target[index] for index in range(3))
    alpha = 0.75 if value > 0.0 else 0.90
    return (*color, alpha)


def candidate_score_colors(
    scores: Sequence[float] | np.ndarray | None,
    *,
    candidate_count: int | None = None,
    selected_index: int | None = None,
    score_scale: float = 2.0,
) -> list[tuple[float, float, float, float]] | None:
    """Map relative candidate scores to red-neutral-blue with green selection."""
    if scores is None:
        if candidate_count is None:
            return None
        colors = [NEUTRAL_COLOR] * int(candidate_count)
    else:
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if candidate_count is not None and values.size != int(candidate_count):
            raise ValueError(
                f"score count {values.size} does not match candidate count {int(candidate_count)}"
            )
        if values.size == 0 or not np.isfinite(values).all():
            return None
        center = float(np.median(values))
        scale = max(float(score_scale), 1.0e-6)
        relative = np.tanh((values - center) / scale)
        colors = [_interpolate_color(value) for value in relative]
    if selected_index is not None:
        selected = int(selected_index)
        if not 0 <= selected < len(colors):
            raise ValueError("selected_index is outside the color list")
        colors[selected] = SELECTED_COLOR
    return colors


def make_overlay(
    *,
    labels: np.ndarray,
    features: np.ndarray,
    trajectories: np.ndarray,
    scores: np.ndarray | None,
    selected_index: int | None,
    pid_target_xy: np.ndarray | None = None,
) -> CandidateOverlay:
    return CandidateOverlay(
        labels=np.asarray(labels),
        features=np.asarray(features),
        trajectories=np.asarray(trajectories),
        scores=None if scores is None else np.asarray(scores),
        selected_index=None if selected_index is None else int(selected_index),
        pid_target_xy=None if pid_target_xy is None else np.asarray(pid_target_xy),
    )
