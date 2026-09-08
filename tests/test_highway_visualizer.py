from __future__ import annotations

import numpy as np

from bdp_benchmark.highway.visualizer import HighwayVisualizer
from bdp_benchmark.visualization import make_overlay


class FakeSurface:
    def pos2pix(self, x: float, y: float):
        return (round(x * 10.0), round(y * 10.0))


def test_highway_visualizer_converts_world_trajectory_to_surface_primitives() -> None:
    visualizer = HighwayVisualizer.__new__(HighwayVisualizer)
    overlay = make_overlay(
        labels=np.asarray([0, 1]),
        features=np.zeros((2, 4), dtype=np.float32),
        trajectories=np.asarray(
            [
                [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        scores=np.asarray([0.0, 1.0]),
        selected_index=1,
        pid_target_xy=np.asarray([0.0, 1.0]),
    )

    lines, target = visualizer.surface_primitives(overlay, FakeSurface())

    assert lines[0]["points"] == [(0, 0), (10, 0)]
    assert lines[1]["points"] == [(0, 0), (0, 10)]
    assert lines[1]["width"] > lines[0]["width"]
    assert target == (0, 10)
