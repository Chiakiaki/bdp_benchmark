#!/usr/bin/env python3
"""Convenience entry point for benchmark evaluation and visual checks."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PROJECTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bdp_benchmark.config import parse_args
from bdp_benchmark.runner import run_evaluation


def _has_cli_option(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def apply_visual_check_defaults(args, argv: Sequence[str]) -> None:
    if not args.visual_check:
        return
    if not _has_cli_option(argv, "--inference_mode"):
        args.inference_mode = "deterministic"
    if not _has_cli_option(argv, "--evaluate_episodes"):
        args.evaluate_episodes = 10


def main() -> int:
    argv = sys.argv[1:]
    args = parse_args(argv)
    apply_visual_check_defaults(args, argv)
    args.run_mode = "evaluate"
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
