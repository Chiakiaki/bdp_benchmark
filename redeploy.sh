#!/usr/bin/env bash
#
# redeploy.sh — build (or verify) the bdp_benchmark runtime on any Ubuntu host.
#
# Tested on: Ubuntu 24.04, 20 cores/15 GB RAM/RTX 5060 (sm_120) and
#            16 cores/14 GB RAM/RTX 3070 Ti (sm_86), driver 580.x.
# No sudo is required: Miniconda and the env live under $HOME.
#
#   ./redeploy.sh                     # full install, idempotent
#   ./redeploy.sh --verify            # check an existing env, install nothing
#   ./redeploy.sh --test              # verify + fast unit-test subset
#   ./redeploy.sh --fetch git         # clone missing sibling repos from GitHub
#   ./redeploy.sh --force             # re-run the pip layer even if up to date
#   ./redeploy.sh --refresh-state     # rewrite .redeploy-state.yaml only
#
# Layout (project root = parent of this repo; the four dirs stay siblings):
#   $PROJECT_ROOT/
#     bdp_benchmark/   critic_based_rl/   HighwayEnv/   metadrive/
#
# Exit codes: 0 ok · 2 human action required (missing repo/credential) · 3 failed.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-bdp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
REPOS="${REPOS:-HighwayEnv metadrive bdp_benchmark critic_based_rl}"
STATE_FILE="$PROJECT_ROOT/.redeploy-state.yaml"

# Repos that are pip-installed editable (critic_based_rl is import path only).
PIP_REPOS="${PIP_REPOS:-HighwayEnv metadrive bdp_benchmark}"
# Fast suite that needs no display; override with REDEPLOY_TESTS.
TEST_TARGETS="${REDEPLOY_TESTS:-tests/test_config.py tests/test_frenet.py tests/test_candidates.py tests/test_job_scripts.py tests/test_highway_adapter.py}"

ENV_DIR="$CONDA_DIR/envs/$ENV_NAME"
PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

MODE=install
DO_TEST=0
FORCE=0
FETCH=none

usage() { sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --verify) MODE=verify; shift ;;
    --test) DO_TEST=1; shift ;;
    --force) FORCE=1; shift ;;
    --refresh-state) MODE=refresh-state; shift ;;
    --fetch) FETCH="${2:-none}"; shift 2 ;;
    --project-root) PROJECT_ROOT="${2:?}"; STATE_FILE="$PROJECT_ROOT/.redeploy-state.yaml"; shift 2 ;;
    --env) ENV_NAME="${2:?}"; ENV_DIR="$CONDA_DIR/envs/$ENV_NAME"; PY="$ENV_DIR/bin/python"; PIP="$ENV_DIR/bin/pip"; shift 2 ;;
    --repos) REPOS="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  warn: %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 3; }
need_human() { printf '\033[33mACTION NEEDED: %s\033[0m\n' "$*" >&2; exit 2; }

repo_path() { echo "$PROJECT_ROOT/$1"; }

declare -A REPO_URLS=(
  [HighwayEnv]="https://github.com/Farama-Foundation/HighwayEnv.git"
  [metadrive]="https://github.com/metadriverse/metadrive.git"
  [bdp_benchmark]="https://github.com/Chiakiaki/bdp_benchmark.git"
  # private: needs a forwarded key or a deploy key on the target
  [critic_based_rl]="git@github.com:Chiakiaki/critic_based_rl.git"
)

# ── checks ──────────────────────────────────────────────────────────────────

preflight() {
  log "Preflight"
  local ok=1
  for tool in git curl; do
    if command -v "$tool" >/dev/null; then printf '  ok   %-10s %s\n' "$tool" "$(command -v "$tool")"
    else printf '  MISS %-10s\n' "$tool"; ok=0; fi
  done
  command -v python3 >/dev/null || { printf '  MISS %-10s\n' python3; ok=0; }
  [ "$ok" = 1 ] || die "install git/curl/python3 first (sudo apt install git curl python3)"

  local free_kb
  free_kb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2{print $4}')
  printf '  ok   %-10s %.1f GB free under %s\n' disk "$(awk -v k="$free_kb" 'BEGIN{print k/1048576}')" "$PROJECT_ROOT"
  [ "$free_kb" -gt 8000000 ] || die "need >= 8 GB free for the CUDA torch wheels"

  for url in https://repo.anaconda.com/miniconda/ https://download.pytorch.org/whl/cu128/ https://pypi.org/simple/; do
    if curl -fsS -o /dev/null -m 12 "$url" 2>/dev/null; then printf '  ok   %-10s %s\n' net "$url"
    else warn "unreachable: $url (install will fail if this stays down)"; fi
  done

  if command -v nvidia-smi >/dev/null; then
    local gpu
    gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    printf '  ok   %-10s %s\n' gpu "${gpu:-driver present, no GPU answered}"
  else
    warn "no nvidia-smi: torch will run CPU-only on this host"
  fi
  printf '  ok   %-10s %s (%s cores)\n' host "$(hostname)" "$(nproc)"
}

ensure_repos() {
  log "Repositories in $PROJECT_ROOT"
  local missing=()
  for repo in $REPOS; do
    if [ -d "$(repo_path "$repo")" ]; then
      local head=""
      [ -d "$(repo_path "$repo")/.git" ] && head="$(git -C "$(repo_path "$repo")" rev-parse --short HEAD 2>/dev/null)"
      printf '  ok   %-16s %s\n' "$repo" "${head:-present (no git)}"
    else
      printf '  --   %-16s missing\n' "$repo"
      missing+=("$repo")
    fi
  done

  [ ${#missing[@]} -eq 0 ] && return 0
  [ "$FETCH" = git ] || need_human "missing repos: ${missing[*]}
  re-run with --fetch git, or clone them into $PROJECT_ROOT yourself"

  for repo in "${missing[@]}"; do
    local url="${REPO_URLS[$repo]:-}"
    [ -n "$url" ] || need_human "no clone URL known for $repo — clone it manually into $PROJECT_ROOT"
    log "Cloning $repo from $url"
    if ! git clone --depth 1 "$url" "$(repo_path "$repo")"; then
      case "$url" in
        git@*) need_human "cannot clone $repo over SSH.
  Private repo: enable agent forwarding (ssh -A) or add a deploy key on the target.
  Then re-run: ./redeploy.sh --fetch git" ;;
        *) die "clone failed for $repo" ;;
      esac
    fi
  done
}

verify() {
  log "Verifying environment at $ENV_DIR"
  [ -x "$PY" ] || need_human "no python at $PY — run without --verify first"
  "$PY" - <<'PYCODE'
import importlib.metadata as md, sys
required = ["torch","gymnasium","stable-baselines3","sb3-contrib","tensorboard",
            "highway-env","metadrive-simulator","bdp-benchmark","numpy","pandas"]
missing=[]
for name in required:
    try: print(f"  {name:24s} {md.version(name)}")
    except md.PackageNotFoundError: missing.append(name)
if missing:
    sys.exit(f"  MISSING: {', '.join(missing)}")
import torch, gymnasium as gym
print(f"  {'cuda available':24s} {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {'device':24s} {torch.cuda.get_device_name(0)}")
    print(f"  {'compute capability':24s} {torch.cuda.get_device_capability(0)}")
    a = torch.randn(256,256, device="cuda"); torch.cuda.synchronize()
    print(f"  {'gpu matmul':24s} ok ({(a@a).sum().item():.1f})")
import highway_env, metadrive  # noqa: F401
print(f"  {'highway envs':24s} {[e for e in gym.registry if 'highway' in e][:2]}")
PYCODE
}

run_tests() {
  log "Fast test subset (no display required)"
  ( cd "$(repo_path bdp_benchmark)" && PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" -m pytest -q $TEST_TARGETS )
}

write_state() {
  local head=""
  for repo in $REPOS; do
    [ -d "$(repo_path "$repo")/.git" ] && \
      head="$head$(printf '  %s: %s' "$repo" "$(git -C "$(repo_path "$repo")" rev-parse HEAD 2>/dev/null)")"$'\n' 
  done
  {
    echo "# written by redeploy.sh — informational, safe to delete"
    echo "generated: $(date -Iseconds)"
    echo "host: $(hostname)"
    echo "project_root: $PROJECT_ROOT"
    echo "env: $ENV_NAME"
    echo "python: $PYTHON_VERSION"
    echo "torch: $TORCH_VERSION+cu128"
    echo "conda: $CONDA_DIR"
    printf 'repos:\n%s\n' "$head"
  } > "$STATE_FILE"
  printf '  ok   %-10s %s\n' state "$STATE_FILE"
}

# ── verify-only path ────────────────────────────────────────────────────────

if [ "$MODE" = refresh-state ]; then
  write_state
  exit 0
fi

if [ "$MODE" = verify ]; then
  verify
  [ "$DO_TEST" = 1 ] && run_tests
  exit $?
fi

# ── install ─────────────────────────────────────────────────────────────────

preflight
ensure_repos

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  log "Installing Miniconda into $CONDA_DIR"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/miniconda.sh" "$MINICONDA_URL" || die "Miniconda download failed"
  bash "$tmp/miniconda.sh" -b -p "$CONDA_DIR" || die "Miniconda install failed"
  rm -rf "$tmp"
else
  log "Miniconda already present: $("$CONDA_DIR/bin/conda" --version)"
fi

# conda-forge only: the Anaconda defaults channel now needs an interactive ToS.
if [ ! -x "$PY" ]; then
  log "Creating conda env '$ENV_NAME' (python $PYTHON_VERSION, conda-forge)"
  "$CONDA_DIR/bin/conda" create -y -n "$ENV_NAME" \
    -c conda-forge --override-channels "python=$PYTHON_VERSION" || die "conda create failed"
else
  log "Conda env already present: $("$PY" --version)"
fi

if [ "$FORCE" = 1 ] || ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  log "Installing torch==$TORCH_VERSION from $CUDA_INDEX"
  "$PIP" install --upgrade pip >/dev/null
  "$PIP" install "torch==$TORCH_VERSION" --index-url "$CUDA_INDEX" || die "torch install failed"
else
  log "torch already installed: $("$PY" -c 'import torch;print(torch.__version__)')"
fi

EDITABLES=()
for repo in $PIP_REPOS; do EDITABLES+=(-e "$(repo_path "$repo")"); done

log "Installing simulators + bdp_benchmark (editable) + RL libraries"
"$PIP" install "${EDITABLES[@]}" \
  "stable-baselines3>=2.7" "sb3-contrib>=2.7" "tensorboard>=2.15" "pytest>=8" \
  || die "pip install failed"

verify || die "verification failed"
write_state
[ "$DO_TEST" = 1 ] && { run_tests || die "test subset failed"; }

log "Done"
cat <<EOF
  source "$CONDA_DIR/etc/profile.d/conda.sh" && conda activate $ENV_NAME
  export PROJECT_ROOT="$PROJECT_ROOT"
  export PYTHONPATH="\$PROJECT_ROOT:\${PYTHONPATH:-}"
EOF
