#!/usr/bin/env bash
# Sequentially compare MetaDrive Frenet-PID v2 and native execution using
# builtin categorical PPO. Optional CLI arguments are forwarded to both runs.
# Launch from this bundle directory:
#   ./run_metadrive_builtin_comparison.sh
# Or launch from the bdp_benchmark repository root:
#   ./scripts/job_scripts/benchmark/run_metadrive_builtin_comparison.sh

set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"

export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

echo "Starting MetaDrive Frenet-PID v2 builtin PPO training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark/metadrive_frenet_pid_v2_builtin_benchmark.yaml \
  "$@"

echo "Starting MetaDrive native builtin PPO training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark/metadrive_native_builtin_benchmark.yaml \
  "$@"
