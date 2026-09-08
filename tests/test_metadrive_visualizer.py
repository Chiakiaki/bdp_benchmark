from __future__ import annotations

import numpy as np

from bdp_benchmark.metadrive.visualizer import MetaDriveVisualizer
from bdp_benchmark.visualization import make_overlay


def test_metadrive_visualizer_builds_world_line_payload() -> None:
    visualizer = MetaDriveVisualizer.__new__(MetaDriveVisualizer)
    overlay = make_overlay(
        labels=np.asarray([0]),
        features=np.zeros((1, 4), dtype=np.float32),
        trajectories=np.asarray([[[1.0, 2.0, 0.0, 1.0], [2.0, 3.0, 0.0, 1.0]]], dtype=np.float32),
        scores=np.asarray([0.0]),
        selected_index=0,
        pid_target_xy=np.asarray([2.0, 3.0]),
    )

    lines, points = visualizer.world_primitives(overlay)

    assert lines[0]["points"] == [(1.0, 2.0, 0.15), (2.0, 3.0, 0.15)]
    assert points == [(2.0, 3.0, 0.2)]
    assert lines[0]["thickness"] > 0.0
