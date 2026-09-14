#!/usr/bin/env bash
# Sequentially compare BDP-Frenet and builtin categorical PPO with Frenet-PID
# execution. Both jobs generate trajectories from x_actual. Optional CLI
# arguments are forwarded to both runs.
# Launch from this bundle directory:
#   ./run_metadrive_frenet_pid_comparison.sh
# Or launch from the bdp_benchmark repository root:
#   ./scripts/job_scripts/benchmark/run_metadrive_frenet_pid_comparison.sh

set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"

export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

echo "Starting actual-state MetaDrive Frenet-PID BDP-Frenet training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark/metadrive_frenet_pid_bdp_frenet_benchmark.yaml \
  "$@"

echo "Starting actual-state MetaDrive Frenet-PID builtin PPO training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark/metadrive_frenet_pid_builtin_benchmark.yaml \
  "$@"
