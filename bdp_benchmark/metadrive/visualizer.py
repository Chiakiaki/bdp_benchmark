"""Candidate overlays drawn through MetaDrive's native debug primitives."""

from __future__ import annotations

from typing import Any

from bdp_benchmark.visualization import CandidateOverlay, candidate_score_colors


class MetaDriveVisualizer:
    def __init__(self, env) -> None:
        self.env = env
        self.overlay: CandidateOverlay | None = None
        self._line_drawers: list[Any] = []
        self._point_drawer = None

    def set_overlay(self, overlay: CandidateOverlay | None) -> None:
        self.overlay = overlay
        if overlay is not None:
            self._redraw()

    def install(self) -> None:
        # The standard MetaDrive renderer is configured by the wrapped env. The
        # overlay uses its existing Panda3D world/debug-draw infrastructure.
        if self._point_drawer is None:
            self._point_drawer = self.env.env.engine.make_point_drawer(
                parent_node=self.env.env.engine.worldNP,
                scale=0.8,
            )
        self._redraw()

    @staticmethod
    def world_primitives(overlay: CandidateOverlay) -> tuple[list[dict[str, Any]], list[tuple[float, float, float]]]:
        colors = candidate_score_colors(
            overlay.scores,
            candidate_count=overlay.trajectories.shape[0],
            selected_index=overlay.selected_index,
        )
        lines: list[dict[str, Any]] = []
        for index, trajectory in enumerate(overlay.trajectories):
            lines.append(
                {
                    "points": [(float(point[0]), float(point[1]), 0.15) for point in trajectory],
                    "color": colors[index] if colors is not None else (0.05, 0.55, 1.0, 0.45),
                    "thickness": 4.0 if overlay.selected_index == index else 2.0,
                }
            )
        points = []
        if overlay.pid_target_xy is not None:
            points.append((float(overlay.pid_target_xy[0]), float(overlay.pid_target_xy[1]), 0.2))
        return lines, points

    def _clear(self) -> None:
        for drawer in self._line_drawers:
            drawer.removeNode()
        self._line_drawers.clear()
        if self._point_drawer is not None:
            self._point_drawer.reset()

    def clear(self) -> None:
        """Remove all transient nodes before MetaDrive resets its world."""
        self._clear()
        if self._point_drawer is not None:
            self._point_drawer.removeNode()
            self._point_drawer = None

    def _redraw(self) -> None:
        if self._point_drawer is None:
            return
        self._clear()
        if self.overlay is None:
            return
        lines, points = self.world_primitives(self.overlay)
        engine = self.env.env.engine
        for line in lines:
            drawer = engine.make_line_drawer(parent_node=engine.worldNP, thickness=line["thickness"])
            drawer.draw_lines([line["points"]], [[line["color"]] * (len(line["points"]) - 1)])
            self._line_drawers.append(drawer)
        if points:
            self._point_drawer.draw_points(points, colors=[(1.0, 0.86, 0.0, 1.0) for _ in points])

    def close(self) -> None:
        self.clear()
