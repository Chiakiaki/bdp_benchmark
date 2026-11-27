#!/usr/bin/env python3
"""PPO vs TRPO training-tail comparison, using the paper's exact statistic.

Definition (reproduced from evidence/training_stats.csv to 1e-6):

* pool every episode of a run (all workers) ordered by absolute monitor time;
* walk the cumulative completed transitions and keep the episodes whose
  cumulative *end* falls inside [window_start, window_end] (inclusive);
* report the episode-weighted mean and the sample SD (ddof = 1).

PPO side: the parsed `evidence/training_episodes.csv.gz`
(`run_id, group, label, r, l, absolute_t`).
TRPO side: run directories under `logs/benchmark`, reading
`monitor/**/*.monitor.csv` and rebuilding absolute time as `t_start + t`.

Both sources live on the training host after the benchmark runs; the paper
evidence was copied there so the comparison can run on either machine.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

WINDOW_START = 1_600_000
WINDOW_END = 2_000_000

# (ceiling, PPO label in the evidence) -> substring of the TRPO run directory
TRPO_PAIRS = {
    ("80kmh", "PPOBDP_BDP"): "frenet_pid_v3_trpobdp_frenet_adaptive_pursuit",
    ("80kmh", "PPO_c"): "frenet_pid_v3_trpo_builtin_adaptive_pursuit",
    ("80kmh", "PPOBDP_native"): "native_trpobdp_frenet_benchmark",
    ("80kmh", "PPO_native_c"): "native_trpo_builtin_benchmark",
    ("80kmh", "PPO_native_end2end"): "native_continuous_trpo_builtin_benchmark",
    ("12mps", "PPOBDP_BDP"): "frenet_pid_v3_trpobdp_frenet_curvature25_12mps",
    ("12mps", "PPO_c"): "frenet_pid_v3_trpo_builtin_curvature25_12mps",
    ("12mps", "PPOBDP_native"): "native_trpobdp_frenet_12mps",
    ("12mps", "PPO_native_c"): "native_trpo_builtin_12mps",
    ("12mps", "PPO_native_end2end"): "native_continuous_trpo_builtin_12mps",
}

PPO_LABEL_ORDER = ["PPOBDP_BDP", "PPO_c", "PPOBDP_native", "PPO_native_c", "PPO_native_end2end"]


def window_stats(episodes: list[tuple[float, float, float]], start: int, end: int,
                 ddof: int = 1) -> dict:
    """episodes: [(absolute_time, return, length)] -> tail statistics."""
    ordered = sorted(episodes)
    done = 0.0
    kept: list[float] = []
    for _, reward, length in ordered:
        done += length
        if start <= done <= end:
            kept.append(reward)
    if not kept:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "transitions": done}
    mean = stats.fmean(kept)
    sd = stats.stdev(kept) if (ddof == 1 and len(kept) > 1) else stats.pstdev(kept) if len(kept) > 1 else 0.0
    return {"n": len(kept), "mean": mean, "sd": sd, "transitions": done}


def read_ppo_episodes(path: Path) -> dict[tuple[str, str], list[tuple[float, float, float]]]:
    """(group, label) -> episodes from the paper's parsed CSV."""
    out: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["group"], row["label"])
            out[key].append((float(row["absolute_t"]), float(row["r"]), float(row["l"])))
    return out


def read_monitor_dir(run_dir: Path) -> list[tuple[float, float, float]]:
    """All episodes of one benchmark run, absolute time = header t_start + t."""
    episodes: list[tuple[float, float, float]] = []
    for path in sorted(run_dir.glob("monitor/**/*.monitor.csv")):
        with open(path, newline="") as handle:
            header = handle.readline()
            start = 0.0
            if header.startswith("#"):
                try:
                    start = float(json.loads(header[1:]).get("t_start", 0.0))
                except (ValueError, json.JSONDecodeError):
                    start = 0.0
            else:
                handle.seek(0)
            for row in csv.DictReader(handle):
                try:
                    episodes.append((start + float(row["t"]), float(row["r"]), float(row["l"])))
                except (KeyError, TypeError, ValueError):
                    continue
    return episodes


def find_trpo_run(root: Path, needle: str) -> Path | None:
    matches = sorted(p for p in root.glob(f"*{needle}*") if p.is_dir())
    return matches[-1] if matches else None


def load_paper_stats(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                out[(row["group"], row["label"])] = {
                    "n": int(row["episode_count"]),
                    "mean": float(row["mean_reward"]),
                    "sd": float(row["std_reward_ddof1"]),
                }
            except (KeyError, ValueError):
                continue
    return out


def fmt(value: float, width: int = 7, digits: int = 1) -> str:
    return f"{value:{width}.{digits}f}" if value == value else " " * width


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ppo-episodes", type=Path, required=True,
                        help="evidence/training_episodes.csv.gz from the paper")
    parser.add_argument("--trpo-root", type=Path, action="append", default=[],
                        help="directory holding TRPO run dirs (repeatable)")
    parser.add_argument("--window-start", type=int, default=WINDOW_START)
    parser.add_argument("--window-end", type=int, default=WINDOW_END)
    parser.add_argument("--verify-ppo-stats", type=Path, default=None,
                        help="training_stats.csv; recompute PPO and compare against it")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    ppo = read_ppo_episodes(args.ppo_episodes)
    ppo_stats = {k: window_stats(v, args.window_start, args.window_end) for k, v in ppo.items()}

    if args.verify_ppo_stats:
        paper = load_paper_stats(args.verify_ppo_stats)
        worst = 0.0
        for key, paper_stats in paper.items():
            mine = ppo_stats.get(key)
            if not mine or not mine["n"]:
                print(f"MISSING {key}", file=sys.stderr)
                continue
            worst = max(worst, abs(mine["mean"] - paper_stats["mean"]),
                        abs(mine["sd"] - paper_stats["sd"]))
        print(f"PPO self-check vs {args.verify_ppo_stats.name}: "
              f"max |Δ| = {worst:.2e} over {len(paper)} runs "
              f"{'✅' if worst < 1e-6 else '❌'}")

    trpo_root = args.trpo_root[0] if args.trpo_root else None
    rows = []
    for (group, label) in sorted(ppo_stats, key=lambda k: (k[0] != "12mps", k[0],
                                                           PPO_LABEL_ORDER.index(k[1])
                                                           if k[1] in PPO_LABEL_ORDER else 99)):
        p = ppo_stats[(group, label)]
        needle = TRPO_PAIRS.get((group, label))
        t = {"n": 0, "mean": math.nan, "sd": math.nan}
        run_name = ""
        if needle and trpo_root:
            run_dir = find_trpo_run(trpo_root, needle)
            if run_dir:
                run_name = run_dir.name
                t = window_stats(read_monitor_dir(run_dir),
                                 args.window_start, args.window_end)
        rows.append({"ceiling": group, "method": label, "ppo": p, "trpo": t,
                     "trpo_run": run_name})

    print(f"\nTail window: completed transitions [{args.window_start:,}, {args.window_end:,}], "
          f"episode-weighted mean ± sample SD (ddof=1)\n")
    head = (f"{'ceiling':7s} {'method':16s} "
            f"{'PPO n':>6s} {'PPO mean':>9s} {'PPO sd':>8s}   "
            f"{'TRPO n':>6s} {'TRPO mean':>9s} {'TRPO sd':>8s}  {'Δmean':>7s}")
    print(head)
    print("-" * len(head))
    for row in rows:
        p, t = row["ppo"], row["trpo"]
        delta = t["mean"] - p["mean"] if t["n"] else math.nan
        print(f"{row['ceiling']:7s} {row['method']:16s} "
              f"{p['n']:6d} {fmt(p['mean'],9)} {fmt(p['sd'],8)}   "
              f"{t['n']:6d} {fmt(t['mean'],9)} {fmt(t['sd'],8)}  {fmt(delta,7,1)}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ceiling", "method", "ppo_n", "ppo_mean", "ppo_sd",
                             "trpo_n", "trpo_mean", "trpo_sd", "trpo_run"])
            for row in rows:
                p, t = row["ppo"], row["trpo"]
                writer.writerow([row["ceiling"], row["method"], p["n"], p["mean"], p["sd"],
                                 t["n"], t["mean"], t["sd"], row["trpo_run"]])
        print(f"\nwrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
