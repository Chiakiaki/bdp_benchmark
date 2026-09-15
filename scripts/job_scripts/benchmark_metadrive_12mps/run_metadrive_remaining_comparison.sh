#!/usr/bin/env bash
# Sequentially train the three jobs not covered by the BDP and builtin launchers:
# actual-origin Frenet BDP, actual-origin Frenet builtin, continuous native PPO.
# From this bundle directory:
#   ./run_metadrive_remaining_comparison.sh
# From the bdp_benchmark repository root:
#   ./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_remaining_comparison.sh
# Optional CLI arguments are forwarded to all three runs.

set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"

export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

echo "Starting 12 m/s actual-origin Frenet-PID BDP-Frenet training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_frenet_pid_bdp_frenet_benchmark.yaml \
  "$@"

echo "Starting 12 m/s actual-origin Frenet-PID builtin PPO training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_frenet_pid_builtin_benchmark.yaml \
  "$@"

echo "Starting 12 m/s native continuous builtin PPO training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_native_continuous_builtin_benchmark.yaml \
  "$@"
