#!/usr/bin/env bash
# Run the ICRA 2027 PPO-vs-TRPO comparison, sequentially.
#
# Both arms are BDP-scored Frenet-PID-v3 at the 12 m/s (43.2 km/h) speed cap,
# 2M transitions, 8 subprocess workers, seed 41. Sequential execution keeps a
# single 8-worker run in memory at a time; on a 15 GB laptop each run peaks
# near 5 GB RSS above the idle baseline.
#
# Usage:
#   ./run_icra2027_trpo_vs_ppo.sh              # both arms, 8 workers
#   ./run_icra2027_trpo_vs_ppo.sh --envs 6     # fewer workers on a smaller host
#   ./run_icra2027_trpo_vs_ppo.sh --trpo-only  # re-run just the TRPO arm
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "$HERE/../../.." && pwd)"
PROJECT_ROOT="$(cd "$BENCHMARK_ROOT/.." && pwd)"

PYTHON="${PYTHON:-python3}"
ENVS="${ENVS:-8}"
RUN_PPO=1
RUN_TRPO=1

while [ $# -gt 0 ]; do
  case "$1" in
    --envs) ENVS="$2"; shift 2 ;;
    --ppo-only) RUN_TRPO=0; shift ;;
    --trpo-only) RUN_PPO=0; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$BENCHMARK_ROOT"

run_one() {
  local name="$1" config="$2"
  echo "════════ $(date -Iseconds)  $name (n_envs=$ENVS) ════════"
  "$PYTHON" scripts/train.py --config "$config" --n_envs "$ENVS" --vec_env subproc
}

[ "$RUN_PPO" = 1 ] && run_one PPO  "$HERE/metadrive_frenet_pid_v3_bdp_frenet_ppo_12mps_benchmark.yaml"
[ "$RUN_TRPO" = 1 ] && run_one TRPO "$HERE/metadrive_frenet_pid_v3_bdp_frenet_trpo_12mps_benchmark.yaml"

echo "════════ done $(date -Iseconds) — summarise with: ════════"
echo "python3 $HERE/summarize_tail.py logs/benchmark/<ppo-run> logs/benchmark/<trpo-run>"
