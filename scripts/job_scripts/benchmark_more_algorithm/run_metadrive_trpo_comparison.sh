#!/usr/bin/env bash
# Sequentially train the five 80 km/h TRPO/TRPOBDP comparisons.
# From this bundle directory:
#   ./run_metadrive_trpo_comparison.sh
# From the bdp_benchmark repository root:
#   ./scripts/job_scripts/benchmark_more_algorithm/run_metadrive_trpo_comparison.sh
# Optional CLI arguments are forwarded to every job.
# The selected Python environment must already provide sb3-contrib.

set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"

export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

# Fail before creating simulator workers if this Python lacks TRPO support.
"${BDP_BENCHMARK_PYTHON_BIN}" -c \
  'from critic_based_rl.model import get_algorithm_class; get_algorithm_class("TRPO")'

jobs=(
  metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml
  metadrive_frenet_pid_v3_builtin_benchmark.yaml
  metadrive_native_bdp_frenet_benchmark.yaml
  metadrive_native_builtin_benchmark.yaml
  metadrive_native_continuous_builtin_benchmark.yaml
)

for job in "${jobs[@]}"; do
  printf 'Starting TRPO comparison job: %s\n' "${job}"
  "${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
    --config "${BDP_JOB_SCRIPT_DIR}/${job}" \
    "$@"
done
