#!/usr/bin/env python3
"""Summarise the training tail of one or more benchmark runs.

Reads ``<run>/monitor/**/*.monitor.csv`` (the per-worker episode files written
by SB3's Monitor wrapper), orders the episodes by monitor time, and reports the
episode-weighted mean return over the final fraction of completed transitions.

This mirrors the comparison reported in the ICRA 2027 training table: a fixed
training-tail interval rather than the best single episode.

Usage
-----
    python3 summarize_tail.py RUN_DIR [RUN_DIR ...] [--tail 0.2] [--labels A B]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_monitor_files(run_dir: Path) -> list[tuple[float, float, float]]:
    """Return (t, return, length) triples from every monitor CSV below run_dir."""
    rows: list[tuple[float, float, float]] = []
    for path in sorted(run_dir.glob("monitor/**/*.monitor.csv")):
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
            if not first.startswith("#"):
                handle.seek(0)
            else:
                try:
                    json.loads(first[1:])
                except json.JSONDecodeError:
                    pass
            for record in csv.DictReader(handle):
                try:
                    rows.append(
                        (float(record["t"]), float(record["r"]), float(record["l"]))
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    rows.sort(key=lambda item: item[0])
    return rows


def tail_stats(rows: list[tuple[float, float, float]], tail: float) -> dict:
    if not rows:
        return {"episodes": 0, "transitions": 0.0, "mean": float("nan"),
                "std": float("nan"), "tail_mean": float("nan"), "tail_std": float("nan"),
                "tail_episodes": 0}
    total = sum(length for _, _, length in rows)
    cutoff = total * (1.0 - tail)
    done = 0.0
    tail_rows = []
    for _, ret, length in rows:
        done += length
        if done >= cutoff:
            tail_rows.append(ret)
    returns = [ret for _, ret, _ in rows]
    return {
        "episodes": len(rows),
        "transitions": total,
        "mean": statistics.fmean(returns),
        "std": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "tail_mean": statistics.fmean(tail_rows) if tail_rows else float("nan"),
        "tail_std": statistics.pstdev(tail_rows) if len(tail_rows) > 1 else 0.0,
        "tail_episodes": len(tail_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories")
    parser.add_argument("--tail", type=float, default=0.2,
                        help="final fraction of completed transitions (default 0.2)")
    parser.add_argument("--labels", nargs="*", default=None, help="display labels")
    args = parser.parse_args()

    labels = args.labels or [p.name for p in args.runs]
    print(f"tail = final {args.tail:.0%} of completed transitions\n")
    print(f"{'run':44s} {'episodes':>9s} {'tail_eps':>9s} {'tail mean':>10s} {'tail SD':>9s}")
    print("-" * 86)
    results = []
    for label, run_dir in zip(labels, args.runs):
        stats = tail_stats(read_monitor_files(run_dir), args.tail)
        results.append(stats)
        print(f"{label[:44]:44s} {stats['episodes']:9d} {stats['tail_episodes']:9d} "
              f"{stats['tail_mean']:10.1f} {stats['tail_std']:9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
