#!/usr/bin/env bash
# From this folder: ./run_metadrive_nominal_adaptive_pursuit_comparison.sh
# From repo root: ./scripts/job_scripts/benchmark/run_metadrive_nominal_adaptive_pursuit_comparison.sh
# Trains nominal-origin BDP first, then builtin PPO, using the tuned tracker.
# CLI overrides are passed to each training run. Console logs are kept under logs/queues.
set -euo pipefail

BDP_BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_BUNDLE_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
BDP_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"
cd "${BDP_BENCHMARK_ROOT}"
mkdir -p logs/queues
BDP_QUEUE_LOG="logs/queues/$(date +%Y%m%d_%H%M%S)_nominal_adaptive_pursuit_$$.log"
exec > >(tee -a "${BDP_QUEUE_LOG}") 2>&1
echo "Nominal adaptive pursuit queue PID=$$, log=${BDP_QUEUE_LOG}"

"${BDP_PYTHON_BIN}" scripts/train.py \
  --config "${BDP_BUNDLE_DIR}/metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml" "$@"
"${BDP_PYTHON_BIN}" scripts/train.py \
  --config "${BDP_BUNDLE_DIR}/metadrive_frenet_pid_v2_builtin_benchmark.yaml" "$@"
echo "Nominal comparison queue completed."
