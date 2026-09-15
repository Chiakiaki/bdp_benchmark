#!/usr/bin/env bash
# From this folder: ./run_metadrive_actual_adaptive_pursuit_comparison.sh
# From repo root: ./scripts/job_scripts/benchmark/run_metadrive_actual_adaptive_pursuit_comparison.sh
# Trains actual-origin Frenet BDP, then builtin PPO, then native BDP as a baseline.
# Native BDP executes direct throttle/steering; it has no trajectory tracker.
# CLI overrides are passed to all runs. Console logs are kept under logs/queues.
set -euo pipefail

BDP_BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_BUNDLE_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
BDP_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"
cd "${BDP_BENCHMARK_ROOT}"
mkdir -p logs/queues
BDP_QUEUE_LOG="logs/queues/$(date +%Y%m%d_%H%M%S)_actual_adaptive_pursuit_$$.log"
exec > >(tee -a "${BDP_QUEUE_LOG}") 2>&1
echo "Actual adaptive pursuit queue PID=$$, log=${BDP_QUEUE_LOG}"

"${BDP_PYTHON_BIN}" scripts/train.py \
  --config "${BDP_BUNDLE_DIR}/metadrive_frenet_pid_bdp_frenet_benchmark.yaml" "$@"
"${BDP_PYTHON_BIN}" scripts/train.py \
  --config "${BDP_BUNDLE_DIR}/metadrive_frenet_pid_builtin_benchmark.yaml" "$@"
"${BDP_PYTHON_BIN}" scripts/train.py \
  --config "${BDP_BUNDLE_DIR}/metadrive_native_bdp_frenet_benchmark.yaml" "$@"
echo "Actual comparison queue completed."
