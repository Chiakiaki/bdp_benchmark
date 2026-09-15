#!/usr/bin/env bash
# Sequentially train the 12 m/s v3 BDP-Frenet and builtin PPO pair (25 candidates).
# From this bundle directory:
#   ./run_metadrive_frenet_pid_v3_comparison.sh
# From the bdp_benchmark repository root:
#   ./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_frenet_pid_v3_comparison.sh
# Optional CLI arguments are forwarded to both training runs.

set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"

export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

echo "Starting 12 m/s MetaDrive Frenet-PID v3 BDP-Frenet training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml \
  "$@"

echo "Starting 12 m/s MetaDrive Frenet-PID v3 builtin PPO training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_frenet_pid_v3_builtin_benchmark.yaml \
  "$@"
