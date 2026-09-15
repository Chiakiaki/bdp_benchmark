#!/usr/bin/env bash
# Sequentially train the three 12 m/s native-controller comparisons only:
# BDP-Frenet, discrete builtin PPO, then continuous builtin PPO.
# No actual-origin or nominal-origin Frenet-PID jobs are included.
# From this bundle directory:
#   ./run_metadrive_native_comparison.sh
# From the bdp_benchmark repository root:
#   ./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_native_comparison.sh
# Optional CLI arguments are forwarded to all three training runs.

set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"

export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

echo "Starting 12 m/s native BDP-Frenet training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_native_bdp_frenet_benchmark.yaml \
  "$@"

echo "Starting 12 m/s native discrete builtin PPO training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_native_builtin_benchmark.yaml \
  "$@"

echo "Starting 12 m/s native continuous builtin PPO training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_native_continuous_builtin_benchmark.yaml \
  "$@"
