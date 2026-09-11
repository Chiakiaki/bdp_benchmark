#!/usr/bin/env python3
"""Plot reward comparisons from benchmark monitor CSV files."""

from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = BENCHMARK_ROOT.parent
for path in (BENCHMARK_ROOT, PROJECTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bdp_benchmark.plotting import main


if __name__ == "__main__":
    main()
