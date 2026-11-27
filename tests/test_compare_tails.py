"""The paper's training-tail statistic (window on completed transitions, ddof=1)."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "job_scripts"
          / "icra2027_trpo_sidecar" / "compare_tails.py")
spec = importlib.util.spec_from_file_location("compare_tails", SCRIPT)
compare_tails = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_tails)


def test_window_is_on_completed_transitions_inclusive():
    # lengths 100 each: ends at 100,200,...   window [300,400] -> ends 300,400
    episodes = [(float(i), float(i), 100.0) for i in range(10)]
    out = compare_tails.window_stats(episodes, 300, 400)
    assert out["n"] == 2
    assert out["mean"] == 2.5                     # ends 300 (ret 2) and 400 (ret 3)


def test_ddof_one_sample_sd():
    episodes = [(0.0, 1.0, 1.0), (1.0, 2.0, 1.0), (2.0, 3.0, 1.0)]
    out = compare_tails.window_stats(episodes, 0, 10, ddof=1)
    assert out["n"] == 3
    assert out["sd"] == 1.0                        # sample SD of 1,2,3
    pop = compare_tails.window_stats(episodes, 0, 10, ddof=0)
    assert pop["sd"] < 1.0                         # population SD is smaller


def test_orders_by_absolute_time_not_input_order():
    episodes = [(50.0, 9.0, 100.0), (10.0, 1.0, 100.0)]   # times out of order
    out = compare_tails.window_stats(episodes, 0, 100)
    assert out["n"] == 1 and out["mean"] == 1.0            # earliest episode kept


def test_monitor_parsing_uses_header_t_start(tmp_path: Path):
    run = tmp_path / "20260101_run"
    monitor = run / "monitor" / "rank_0"
    monitor.mkdir(parents=True)
    path = monitor / "env_0.monitor.csv"
    path.write_text(
        '#' + json.dumps({"t_start": 1000.0}) + "\n" + "r,l,t\n"
        + "5.0,100,1.0\n" + "7.0,100,2.0\n"
    )
    episodes = compare_tails.read_monitor_dir(run)
    assert [t for t, _, _ in episodes] == [1001.0, 1002.0]
    out = compare_tails.window_stats(episodes, 0, 200)
    assert out["n"] == 2 and out["mean"] == 6.0


def test_ppo_episodes_grouping(tmp_path: Path):
    path = tmp_path / "episodes.csv.gz"
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "group", "label", "r", "l", "t", "absolute_t"])
        writer.writerow(["r1", "12mps", "PPO_c", "10", "100", "1", "100.5"])
        writer.writerow(["r1", "12mps", "PPO_c", "20", "100", "2", "101.5"])
    grouped = compare_tails.read_ppo_episodes(path)
    assert list(grouped) == [("12mps", "PPO_c")]
    assert len(grouped[("12mps", "PPO_c")]) == 2


def test_find_trpo_run_picks_latest(tmp_path: Path):
    for name in ("20260101_x_native_trpo_builtin_benchmark",
                 "20260102_x_native_trpo_builtin_benchmark"):
        (tmp_path / name).mkdir()
    found = compare_tails.find_trpo_run(tmp_path, "native_trpo_builtin_benchmark")
    assert found.name.startswith("20260102")
