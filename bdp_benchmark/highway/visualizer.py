"""Candidate overlays drawn through HighwayEnv's native pygame viewer."""

from __future__ import annotations

from typing import Any

import numpy as np

from bdp_benchmark.visualization import CandidateOverlay, candidate_score_colors


class HighwayVisualizer:
    def __init__(self, env) -> None:
        self.env = env
        self.overlay: CandidateOverlay | None = None
        self._installed = False

    def set_overlay(self, overlay: CandidateOverlay | None) -> None:
        self.overlay = overlay

    def install(self) -> None:
        # Rendering once creates HighwayEnv's native EnvViewer. The callback is
        # invoked by that viewer every frame after the road is redrawn.
        if getattr(self.env.unwrapped, "viewer", None) is None:
            self.env.render()
        viewer = getattr(self.env.unwrapped, "viewer", None)
        if viewer is None:
            raise RuntimeError("HighwayEnv did not create an EnvViewer for visual check")
        viewer.set_agent_display(self._draw)
        self._installed = True

    @staticmethod
    def surface_primitives(overlay: CandidateOverlay, surface: Any) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
        colors = candidate_score_colors(
            overlay.scores,
            candidate_count=overlay.trajectories.shape[0],
            selected_index=overlay.selected_index,
        )
        lines: list[dict[str, Any]] = []
        for index, trajectory in enumerate(overlay.trajectories):
            points = [surface.pos2pix(float(point[0]), float(point[1])) for point in trajectory]
            color = colors[index] if colors is not None else (0.05, 0.55, 1.0, 0.45)
            lines.append(
                {
                    "points": points,
                    "color": tuple(int(max(0.0, min(1.0, value)) * 255) for value in color[:3]),
                    "alpha": color[3],
                    "width": 3 if overlay.selected_index == index else 1,
                }
            )
        target = None
        if overlay.pid_target_xy is not None:
            target = surface.pos2pix(float(overlay.pid_target_xy[0]), float(overlay.pid_target_xy[1]))
        return lines, target

    def _draw(self, _agent_surface, sim_surface) -> None:
        if self.overlay is None:
            return
        import pygame

        lines, target = self.surface_primitives(self.overlay, sim_surface)
        for line in lines:
            pygame.draw.lines(sim_surface, line["color"], False, line["points"], line["width"])
        if target is not None:
            pygame.draw.circle(sim_surface, (255, 220, 0), target, 4)

    def close(self) -> None:
        from highway_env.envs.common.graphics import EnvViewer

        if EnvViewer.agent_display == self._draw:
            EnvViewer.agent_display = None
        self.overlay = None
