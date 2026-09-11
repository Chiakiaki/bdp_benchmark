from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from bdp_benchmark.plotting import (
    build_curves,
    configure_plot_backend,
    discover_monitor_groups,
    label_for_benchmark_run,
    output_path_for_variant,
)


def _write_monitor(path: Path, rows: list[tuple[float, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['#{"t_start": 0.0}', "r,l,t,environment_variant"]
    variant = path.parents[1].name
    lines.extend(f"{reward},{length},{time_value},{variant}" for reward, length, time_value in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_run(tmp_path: Path, name: str) -> Path:
    run = tmp_path / name
    (run / "resolved_config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump({"runtime": {"run_name": name.removeprefix("timestamp_")}}),
        encoding="utf-8",
    )
    _write_monitor(run / "monitor/highway-fast-v0/rank_0/env_0.monitor.csv", [(1.0, 10, 2.0)])
    _write_monitor(run / "monitor/highway-fast-v0/rank_3/env_3.monitor.csv", [(2.0, 10, 4.0)])
    _write_monitor(run / "monitor/merge-v1/rank_1/env_1.monitor.csv", [(3.0, 20, 1.0)])
    _write_monitor(run / "monitor/roundabout-v1/rank_2/env_2.monitor.csv", [(5.0, 30, 3.0)])
    return run


def test_discover_monitor_groups_finds_nested_workers_by_variant(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "timestamp_bdp")

    groups = discover_monitor_groups(run)

    assert list(groups) == ["highway-fast-v0", "merge-v1", "roundabout-v1"]
    assert len(groups["highway-fast-v0"]) == 2
    assert len(groups["merge-v1"]) == 1
    assert len(groups["roundabout-v1"]) == 1


def test_combined_curve_pools_every_worker_and_weights_each_episode(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "timestamp_bdp")

    curves = build_curves([run], curve_mode="combined", window_size=1)
    curve = curves["combined"][0]

    assert curve.label == "bdp"
    np.testing.assert_allclose(curve.rewards, [3.0, 1.0, 5.0, 2.0])
    np.testing.assert_array_equal(curve.timesteps, [20, 30, 60, 70])


def test_per_variant_curves_pool_only_workers_from_same_road(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "timestamp_bdp")
    run_b = _make_run(tmp_path, "timestamp_builtin")

    curves = build_curves([run_a, run_b], curve_mode="per_variant", window_size=1)

    assert list(curves) == ["highway-fast-v0", "merge-v1", "roundabout-v1"]
    assert [curve.label for curve in curves["merge-v1"]] == ["bdp", "builtin"]
    np.testing.assert_allclose(curves["highway-fast-v0"][0].rewards, [1.0, 2.0])
    np.testing.assert_array_equal(curves["highway-fast-v0"][0].timesteps, [10, 20])


def test_per_variant_curves_require_same_roads_in_every_run(tmp_path: Path) -> None:
    run_a = _make_run(tmp_path, "timestamp_bdp")
    run_b = _make_run(tmp_path, "timestamp_builtin")
    for path in (run_b / "monitor/roundabout-v1").rglob("*.monitor.csv"):
        path.unlink()

    with pytest.raises(ValueError, match="same environment variants"):
        build_curves([run_a, run_b], curve_mode="per_variant", window_size=1)


def test_labels_and_variant_output_paths_are_stable(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "timestamp_bdp")

    assert label_for_benchmark_run(run) == "bdp"
    assert output_path_for_variant(tmp_path / "comparison.png", "roundabout-v1") == (
        tmp_path / "comparison_roundabout-v1.png"
    )


def test_file_only_plotting_forces_noninteractive_backend() -> None:
    configure_plot_backend(show=False)

    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"
