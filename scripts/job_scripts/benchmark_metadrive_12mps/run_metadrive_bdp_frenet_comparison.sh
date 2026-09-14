#!/usr/bin/env bash
# Sequentially compare MetaDrive Frenet-PID v2 and native execution using BDP
# Frenet candidate scoring. Optional CLI arguments are forwarded to both runs.
# Run the 12 m/s capped BDP comparison from this bundle directory:
#   ./run_metadrive_bdp_frenet_comparison.sh
# Or launch from the bdp_benchmark repository root:
#   ./scripts/job_scripts/benchmark_metadrive_12mps/run_metadrive_bdp_frenet_comparison.sh

set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"

export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

echo "Starting 12 m/s MetaDrive Frenet-PID v2 BDP-Frenet training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_frenet_pid_v2_bdp_frenet_benchmark.yaml \
  "$@"

echo "Starting 12 m/s MetaDrive native BDP-Frenet training."
"${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
  --config scripts/job_scripts/benchmark_metadrive_12mps/metadrive_native_bdp_frenet_benchmark.yaml \
  "$@"
