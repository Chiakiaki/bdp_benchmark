"""Reward-curve aggregation for benchmark monitor directory layouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from critic_based_rl.plotting import (
    RewardCurve,
    average_reward_in_timestep_window,
    format_reward_window_average,
    plot_reward_curves,
    read_monitor_rows,
    rolling_mean_std,
    sanitize_filename_component,
)


def label_for_benchmark_run(run_dir: Path) -> str:
    config_path = Path(run_dir) / "resolved_config.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        run_name = config.get("runtime", {}).get("run_name")
        if run_name:
            return str(run_name)
    return Path(run_dir).name


def discover_monitor_groups(run_dir: Path) -> dict[str, list[Path]]:
    run_dir = Path(run_dir).expanduser().resolve()
    monitor_root = run_dir / "monitor"
    groups: dict[str, list[Path]] = {}
    for monitor_path in sorted(monitor_root.rglob("*.monitor.csv")):
        relative_parts = monitor_path.relative_to(monitor_root).parts
        if len(relative_parts) < 2:
            continue
        variant = relative_parts[0]
        groups.setdefault(variant, []).append(monitor_path)
    if not groups:
        raise ValueError(f"No benchmark monitor files found under {monitor_root}")
    return groups


def _load_curve(
    monitor_paths: Sequence[Path],
    *,
    label: str,
    window_size: int,
    max_timesteps: float | None,
) -> RewardCurve:
    rows = []
    for monitor_path in monitor_paths:
        rows.extend(read_monitor_rows(monitor_path))
    if not rows:
        raise ValueError(f"No complete monitor episodes found for {label}")

    rows.sort(key=lambda item: item[2])
    rewards = np.asarray([item[0] for item in rows], dtype=np.float64)
    lengths = np.asarray([item[1] for item in rows], dtype=np.int64)
    timesteps = np.cumsum(lengths)
    if max_timesteps is not None:
        keep = timesteps <= float(max_timesteps)
        rewards = rewards[keep]
        timesteps = timesteps[keep]
    if not rewards.size:
        raise ValueError(f"No complete monitor episodes remain within max_timesteps for {label}")
    mean, std = rolling_mean_std(rewards, window_size)
    return RewardCurve(label=label, timesteps=timesteps, rewards=rewards, mean=mean, std=std)


def build_curves(
    run_dirs: Sequence[Path],
    *,
    curve_mode: str,
    window_size: int,
    max_timesteps: float | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, list[RewardCurve]]:
    if curve_mode not in ("combined", "per_variant"):
        raise ValueError("curve_mode must be 'combined' or 'per_variant'")
    runs = [Path(path).expanduser().resolve() for path in run_dirs]
    if not runs:
        raise ValueError("At least one benchmark run directory is required")
    if labels and len(labels) != len(runs):
        raise ValueError("The number of labels must match the number of run directories")

    run_labels = list(labels) if labels else [label_for_benchmark_run(run) for run in runs]
    grouped_monitors = [discover_monitor_groups(run) for run in runs]
    if curve_mode == "combined":
        return {
            "combined": [
                _load_curve(
                    [path for paths in groups.values() for path in paths],
                    label=label,
                    window_size=window_size,
                    max_timesteps=max_timesteps,
                )
                for groups, label in zip(grouped_monitors, run_labels)
            ]
        }

    expected_variants = set(grouped_monitors[0])
    if any(set(groups) != expected_variants for groups in grouped_monitors[1:]):
        raise ValueError("per_variant curves require the same environment variants in every run")
    return {
        variant: [
            _load_curve(
                groups[variant],
                label=label,
                window_size=window_size,
                max_timesteps=max_timesteps,
            )
            for groups, label in zip(grouped_monitors, run_labels)
        ]
        for variant in sorted(expected_variants)
    }


def output_path_for_variant(output_path: Path, variant: str) -> Path:
    output_path = Path(output_path)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}_{sanitize_filename_component(variant)}{suffix}")


def configure_plot_backend(*, show: bool) -> None:
    if show:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare episode-weighted reward curves from benchmark runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Benchmark run directories to compare.")
    parser.add_argument("--labels", nargs="*", default=[], help="Optional labels in the same order as --runs.")
    parser.add_argument(
        "--curve_mode",
        choices=("combined", "per_variant"),
        default="combined",
        help="Combine every road into one curve per run, or create one comparison plot per road.",
    )
    parser.add_argument("--window_size", type=int, default=20, help="Rolling reward window in completed episodes.")
    parser.add_argument("--max_timesteps", type=float, default=None)
    parser.add_argument("--avg_start_timesteps", type=float, default=None)
    parser.add_argument("--avg_end_timesteps", type=float, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output PNG, or base PNG for per-variant plots.")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.2, help="Standard-deviation band opacity.")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = build_arg_parser().parse_args(argv)
    configure_plot_backend(show=args.show)
    run_dirs = [Path(path).expanduser().resolve() for path in args.runs]
    curves_by_group = build_curves(
        run_dirs,
        curve_mode=args.curve_mode,
        window_size=args.window_size,
        max_timesteps=args.max_timesteps,
        labels=args.labels,
    )
    default_root = run_dirs[0].parent
    base_output = Path(args.output).expanduser() if args.output else default_root / "reward_comparison.png"

    output_paths = []
    for group, curves in curves_by_group.items():
        output_path = base_output if group == "combined" else output_path_for_variant(base_output, group)
        title = args.title or "Benchmark Reward Comparison"
        if group != "combined":
            title = f"{title}: {group}"
        output_paths.append(plot_reward_curves(curves, output_path, title, args.alpha, args.show))

        if args.avg_start_timesteps is not None or args.avg_end_timesteps is not None:
            if args.avg_start_timesteps is None or args.avg_end_timesteps is None:
                raise SystemExit("--avg_start_timesteps and --avg_end_timesteps must be provided together.")
            print(f"Average raw episode reward ({group}):")
            for curve in curves:
                summary = average_reward_in_timestep_window(
                    curve,
                    start_timesteps=args.avg_start_timesteps,
                    end_timesteps=args.avg_end_timesteps,
                )
                print(format_reward_window_average(summary))

    print("Selected runs:")
    for run_dir in run_dirs:
        print(f"  {run_dir}")
    for output_path in output_paths:
        print(f"Saved plot: {output_path}")
    return output_paths
