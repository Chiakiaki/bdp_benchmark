#!/usr/bin/env bash
# Sequentially train all nine 12 m/s TRPO, DQN and A2C comparisons.
# From this folder: ./run_metadrive_more_algorithm_comparison.sh
# From the repository root:
#   ./scripts/job_scripts/benchmark_metadrive_12mps_more_algorithm/run_metadrive_more_algorithm_comparison.sh
# Optional CLI arguments are forwarded to every job.
set -euo pipefail

BDP_JOB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BDP_BENCHMARK_ROOT="$(cd -- "${BDP_JOB_SCRIPT_DIR}/../../.." && pwd)"
BDP_PROJECT_ROOT="$(cd -- "${BDP_BENCHMARK_ROOT}/.." && pwd)"
BDP_BENCHMARK_PYTHON_BIN="${BDP_BENCHMARK_PYTHON:-python3}"
export PYTHONPATH="${BDP_PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${BDP_BENCHMARK_ROOT}"

# Check every algorithm before launching simulation workers.
"${BDP_BENCHMARK_PYTHON_BIN}" -c \
  'from critic_based_rl.model import get_algorithm_class; [get_algorithm_class(name) for name in ("TRPO", "DQN", "A2C")]'

jobs=(
  metadrive_frenet_pid_v3_bdp_frenet_benchmark.yaml
  metadrive_frenet_pid_v3_builtin_benchmark.yaml
  metadrive_native_bdp_frenet_benchmark.yaml
  metadrive_native_builtin_benchmark.yaml
  metadrive_native_continuous_builtin_benchmark.yaml
  metadrive_native_dqn_builtin_benchmark.yaml
  metadrive_frenet_pid_v3_dqn_builtin_benchmark.yaml
  metadrive_native_a2c_builtin_benchmark.yaml
  metadrive_frenet_pid_v3_a2c_builtin_benchmark.yaml
)

for job in "${jobs[@]}"; do
  printf 'Starting algorithm comparison job: %s\n' "${job}"
  "${BDP_BENCHMARK_PYTHON_BIN}" scripts/train.py \
    --config "${BDP_JOB_SCRIPT_DIR}/${job}" \
    "$@"
done
