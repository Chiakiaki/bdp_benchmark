#!/usr/bin/env python3
"""Convenience entry point for deterministic benchmark evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, PROJECTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bdp_benchmark.config import parse_args
from bdp_benchmark.runner import run_evaluation


def main() -> int:
    args = parse_args()
    args.run_mode = "evaluate"
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
