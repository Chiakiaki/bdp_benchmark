#!/usr/bin/env bash
# Sequentially compare MetaDrive Frenet-PID v2 and native execution using BDP
# Frenet candidate scoring. Optional CLI arguments are forwarded to both runs.

set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"

export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

echo "Starting MetaDrive Frenet-PID v2 BDP-Frenet training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml \
  "$@"

echo "Starting MetaDrive native BDP-Frenet training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/metadrive_native_bdp_frenet_benchmark.yaml \
  "$@"

