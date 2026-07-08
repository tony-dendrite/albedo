#!/usr/bin/env bash
# All shell ops in one place:
#   deploy.sh bootstrap                    one-time host prereqs (uv, docker, node+pm2)
#   deploy.sh backend                      CPU host: Postgres, schema, seed, dataset, pm2
#   deploy.sh gpu {eval|sanity} <host> [user]   rsync + install + dataset + pm2 on a GPU box
#   deploy.sh opensearch                   local single-node OpenSearch (real validation in tests)
#   deploy.sh stop                         stop whichever albedo pm2 apps run on this host
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
CMD="${1:?usage: deploy.sh bootstrap|backend|gpu|opensearch|stop}"

# Doppler mode: ALBEDO_DOPPLER=true sources every secret from the configured Doppler
# project instead of a repo .env - commands run under `doppler run --` and GPU boxes
# receive a generated env file (they never get a Doppler token).
DOPPLER="${ALBEDO_DOPPLER:-false}"
RUN=()
if [ "$DOPPLER" = "true" ]; then
  command -v doppler >/dev/null || { echo "ALBEDO_DOPPLER=true but doppler CLI not installed"; exit 1; }
  doppler configure get project >/dev/null 2>&1 || { echo "doppler not configured here (doppler setup)"; exit 1; }
  RUN=(doppler run --)
fi

require_env() {
  if [ "$DOPPLER" = "true" ]; then
    set -a; eval "$(doppler secrets download --no-file --format env-no-quotes)"; set +a
  else
    [ -f .env ] || { echo "create .env first (copy .env.example) or set ALBEDO_DOPPLER=true"; exit 1; }
    set -a; source .env; set +a
  fi
}

case "$CMD" in

bootstrap)
  export PATH="$HOME/.local/bin:$PATH"
  command -v curl >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq curl; }
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
  if ! command -v node >/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y -qq nodejs
  fi
  command -v pm2 >/dev/null || sudo npm install -g pm2
  echo "bootstrap done: $(uv --version), docker $(docker --version | cut -d' ' -f3), pm2 $(pm2 -v)"
  ;;

backend)
  require_env
  echo "== venv + install"
  uv venv --allow-existing
  uv pip install -e . --quiet
  echo "== postgres (docker compose)"
  docker compose up -d
  until docker compose exec -T albedo-postgres pg_isready -U "$ALBEDO_POSTGRES_USER" -d "$ALBEDO_POSTGRES_DB" >/dev/null 2>&1; do
    sleep 1
  done
  echo "== schema + genesis seed (idempotent)"
  "${RUN[@]}" uv run albedo migrate
  "${RUN[@]}" uv run python scripts/ops.py seed-genesis
  if [ -n "${SANITY_DISPATCH_DATASET_ROOT:-}" ] && [ ! -f "$SANITY_DISPATCH_DATASET_ROOT/manifest.json" ]; then
    echo "== dataset (sanity dispatcher samples prompts CPU-side)"
    "${RUN[@]}" uv run python scripts/datasets.py download "$SANITY_DISPATCH_DATASET_ROOT"
  fi
  echo "== pm2: backend + SSH tunnels to both GPU hosts"
  "${RUN[@]}" pm2 start pm2/backend.config.js
  pm2 save
  pm2 status
  echo "== next: ${RUN[*]:-} uv run python scripts/ops.py preflight"
  ;;

gpu)
  ROLE="${2:?usage: deploy.sh gpu eval|sanity <ssh-host> [ssh-user]}"
  HOST="${3:?usage: deploy.sh gpu eval|sanity <ssh-host> [ssh-user]}"
  USER="${4:-root}"
  DEST="/root/albedo-simple"
  case "$ROLE" in
    eval)   PM2_CONFIG="pm2/gpu-eval.config.js" ;;
    sanity) PM2_CONFIG="pm2/gpu-sanity.config.js" ;;
    *) echo "role must be eval or sanity"; exit 1 ;;
  esac
  require_env
  ENV_SRC=.env
  if [ "$DOPPLER" = "true" ]; then
    ENV_SRC="$(mktemp)"
    doppler secrets download --no-file --format env-no-quotes > "$ENV_SRC"
    trap 'rm -f "$ENV_SRC"' EXIT
  fi
  echo "== rsync code -> $USER@$HOST:$DEST"
  rsync -az --delete \
    --exclude '.venv' --exclude '.ruff_cache' --exclude '__pycache__' --exclude '.env' \
    ./ "$USER@$HOST:$DEST/"
  scp -q "$ENV_SRC" "$USER@$HOST:$DEST/.env"
  echo "== remote install + start"
  # ALBEDO_CUDA_TOOLKIT_VERSION must match the cuXXX build torch resolves (default 13-0).
  CUDA_MM="${ALBEDO_CUDA_TOOLKIT_VERSION:-13-0}"
  ssh "$USER@$HOST" bash -s <<REMOTE
set -euo pipefail
cd "$DEST"
export DEBIAN_FRONTEND=noninteractive
set -a; [ -f .env ] && . ./.env; set +a

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v pm2 >/dev/null; then
  # fresh pods ship without node/npm - install via apt, not npm
  apt-get update -qq && apt-get install -y -qq nodejs npm
  npm install -g pm2
fi

# CUDA toolkit (nvcc) is REQUIRED: the Qwen3.x MoE hybrid's gated-delta-net attention is
# served by a flashinfer kernel that JIT-compiles at runtime - without nvcc every
# generation returns 500. (First run still pays a one-time ~8 min compile, then cached.)
if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  . /etc/os-release
  repo="\${ID}\${VERSION_ID//./}"
  wget -qO /tmp/cuda-keyring.deb \
    "https://developer.download.nvidia.com/compute/cuda/repos/\${repo}/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update -qq && apt-get install -y -qq "cuda-toolkit-${CUDA_MM}"
  ln -sfn "/usr/local/cuda-${CUDA_MM/-/.}" /usr/local/cuda
fi
export PATH="/usr/local/cuda/bin:\$PATH"
echo "nvcc: \$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' || echo MISSING)"

uv venv --allow-existing
uv pip install -e . --quiet
uv pip install vllm==0.23.0 --quiet || echo "WARN: install vllm manually (needs matching torch/CUDA)"
if [ "$ROLE" = "eval" ] && [ -n "${ALBEDO_REMOTE_DATASET_ROOT:-}" ] && [ ! -f "${ALBEDO_REMOTE_DATASET_ROOT}/manifest.json" ]; then
  uv run python scripts/datasets.py download "${ALBEDO_REMOTE_DATASET_ROOT}"
fi
pm2 start "$PM2_CONFIG"
pm2 save
pm2 status
REMOTE
  echo "== done. On the backend host: uv run python scripts/ops.py seed-genesis  (re-points hosts)"
  echo "   then: uv run python scripts/ops.py preflight"
  ;;

opensearch)
  NAME="${ALBEDO_OS_CONTAINER:-albedo_opensearch}"
  IMAGE="${ALBEDO_OS_IMAGE:-opensearchproject/opensearch:2}"
  PORT="${ALBEDO_OS_PORT:-9200}"
  if curl -fs "http://localhost:${PORT}/_cluster/health" >/dev/null 2>&1; then
    echo "==> OpenSearch already healthy on 127.0.0.1:${PORT}; reusing it"
    exit 0
  fi
  docker pull "${IMAGE}"
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  for c in $(docker ps -q --filter "publish=${PORT}"); do docker rm -f "$c" >/dev/null 2>&1 || true; done
  docker run -d --name "${NAME}" \
    -p "127.0.0.1:${PORT}:9200" \
    -e discovery.type=single-node \
    -e DISABLE_SECURITY_PLUGIN=true \
    -e DISABLE_INSTALL_DEMO_CONFIG=true \
    -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
    "${IMAGE}" >/dev/null
  echo "==> waiting for cluster health (~30-60s on first boot)"
  for _ in $(seq 1 90); do
    if curl -fs "http://localhost:${PORT}/_cluster/health" >/dev/null 2>&1; then
      echo "==> OpenSearch ready at http://localhost:${PORT}"
      exit 0
    fi
    sleep 2
  done
  echo "!! OpenSearch did not become healthy; check: docker logs ${NAME}" >&2
  exit 1
  ;;

stop)
  for cfg in backend gpu-eval gpu-sanity; do
    pm2 delete "$REPO/pm2/$cfg.config.js" 2>/dev/null && echo "stopped $cfg"
  done
  true
  ;;

*)
  echo "unknown command: $CMD (use bootstrap|backend|gpu|opensearch|stop)"; exit 1 ;;
esac
