#!/usr/bin/env bash
# Deploy AFL model + dashboard to Unraid via rsync + docker compose.
#
# Usage:
#   ./scripts/deploy.sh                # incremental deploy (preserves data/)
#   ./scripts/deploy.sh --with-data    # also sync data/ (use for first deploy)
#   ./scripts/deploy.sh --with-env     # also sync local .env to remote
#   ./scripts/deploy.sh --no-build     # sync code only, don't rebuild image
#   ./scripts/deploy.sh --logs         # tail container logs after deploy
#
# Requires:
#   - SSH access to host alias `grid` (root user)
#   - rsync on both ends
#   - docker + docker compose on the Unraid host
#
# Environment:
#   AFL_REMOTE_HOST     SSH alias  (default: grid)
#   AFL_REMOTE_PATH     Target dir (default: /mnt/user/appdata/afl)
#   ODDS_API_KEY                Passed through to compose on the remote (optional)
#   VISUAL_CROSSING_API_KEY     Passed through to compose on the remote (optional)

set -euo pipefail

REMOTE_HOST="${AFL_REMOTE_HOST:-grid}"
REMOTE_PATH="${AFL_REMOTE_PATH:-/mnt/user/appdata/afl}"

SSH_CMD="${AFL_SSH_CMD:-}"
if [[ -z "$SSH_CMD" ]]; then
  if command -v ssh.exe >/dev/null 2>&1; then
    SSH_CMD="ssh.exe"
  else
    SSH_CMD="ssh"
  fi
fi
RSYNC_RSH="${AFL_RSYNC_RSH:-$SSH_CMD}"

WITH_DATA=0
WITH_ENV=0
DO_BUILD=1
TAIL_LOGS=0

for arg in "$@"; do
  case "$arg" in
    --with-data) WITH_DATA=1 ;;
    --with-env)  WITH_ENV=1 ;;
    --no-build)  DO_BUILD=0 ;;
    --logs)      TAIL_LOGS=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# ---- excludes ----------------------------------------------------------------
# Things we never want on the remote
EXCLUDES=(
  --exclude='.git/'
  --exclude='.claude/'
  --exclude='.venv/'
  --exclude='venv/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='*.pyo'
  --exclude='catboost_info/'
  --exclude='node_modules/'
  --exclude='.next/'
  --exclude='.DS_Store'
  --exclude='*.log'
  --exclude='.env'
  --exclude='.env.local'
)

# Default: protect remote data/ from being overwritten/deleted.
# --with-data overrides this for the initial bootstrap.
if [[ $WITH_DATA -eq 0 ]]; then
  EXCLUDES+=( --exclude='data/' )
fi

# ---- pre-flight --------------------------------------------------------------
echo "==> deploy target:  $REMOTE_HOST:$REMOTE_PATH"
echo "==> with-data:      $([[ $WITH_DATA -eq 1 ]] && echo yes || echo no)"
echo "==> with-env:       $([[ $WITH_ENV -eq 1 ]] && echo yes || echo no)"
echo "==> rebuild image:  $([[ $DO_BUILD -eq 1 ]] && echo yes || echo no)"
echo

if ! $SSH_CMD -o ConnectTimeout=5 "$REMOTE_HOST" "true" 2>/dev/null; then
  echo "ERROR: cannot SSH to $REMOTE_HOST" >&2
  exit 1
fi

# Make sure target path exists
$SSH_CMD "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH'"

# Local env file is optional; remote .env is preserved by default.
LOCAL_ENV="$PROJECT_ROOT/.env"
REMOTE_ENV="$REMOTE_PATH/.env"

# ---- sync --------------------------------------------------------------------
# Prefer rsync (incremental + --delete). Fall back to tar pipe over SSH if
# rsync is not installed locally (common on Windows).
if command -v rsync >/dev/null 2>&1; then
  SYNC_TOOL="rsync"
else
  SYNC_TOOL="tar"
fi
echo "==> sync tool: $SYNC_TOOL"
echo "==> sync project to $REMOTE_HOST:$REMOTE_PATH"

if [[ "$SYNC_TOOL" == "rsync" ]]; then
  rsync -avz --delete --human-readable \
    -e "$RSYNC_RSH" \
    "${EXCLUDES[@]}" \
    ./ "$REMOTE_HOST:$REMOTE_PATH/"
else
  # tar fallback. Translate rsync --exclude=PAT into tar --exclude=PAT.
  # Note: tar does not delete remote files that no longer exist locally —
  # if you remove a tracked file, clean it on the remote manually.
  TAR_EXCLUDES=()
  for ex in "${EXCLUDES[@]}"; do
    case "$ex" in
      --exclude=*) TAR_EXCLUDES+=( "--exclude=${ex#--exclude=}" ) ;;
    esac
  done
  tar -cf - "${TAR_EXCLUDES[@]}" . | \
    $SSH_CMD "$REMOTE_HOST" "mkdir -p '$REMOTE_PATH' && tar -xf - -C '$REMOTE_PATH'"
fi

if [[ $WITH_ENV -eq 1 ]]; then
  if [[ ! -f "$LOCAL_ENV" ]]; then
    echo "ERROR: --with-env was set but no local .env exists at $LOCAL_ENV" >&2
    exit 1
  fi
  echo "==> syncing .env to $REMOTE_HOST:$REMOTE_ENV"
  if [[ "$SYNC_TOOL" == "rsync" ]]; then
    rsync -avz --chmod=600 -e "$RSYNC_RSH" "$LOCAL_ENV" "$REMOTE_HOST:$REMOTE_ENV"
  else
    $SSH_CMD "$REMOTE_HOST" "cat > '$REMOTE_ENV' && chmod 600 '$REMOTE_ENV'" < "$LOCAL_ENV"
  fi
else
  if $SSH_CMD "$REMOTE_HOST" "[ -f '$REMOTE_ENV' ]"; then
    echo "==> preserving remote .env at $REMOTE_ENV"
  elif [[ -f "$LOCAL_ENV" ]]; then
    echo "==> note: local .env exists but will not be copied without --with-env"
  else
    echo "==> note: no remote .env found; compose will rely on shell-passed env vars only"
  fi
fi

# Make scripts executable on the remote
$SSH_CMD "$REMOTE_HOST" "chmod +x '$REMOTE_PATH/scripts/'*.sh 2>/dev/null || true"

# ---- compose -----------------------------------------------------------------
COMPOSE_CMD="cd '$REMOTE_PATH' && docker compose"
if [[ -n "${ODDS_API_KEY:-}" ]]; then
  COMPOSE_CMD="ODDS_API_KEY='$ODDS_API_KEY' $COMPOSE_CMD"
fi
if [[ -n "${VISUAL_CROSSING_API_KEY:-}" ]]; then
  COMPOSE_CMD="VISUAL_CROSSING_API_KEY='$VISUAL_CROSSING_API_KEY' $COMPOSE_CMD"
fi

if [[ $DO_BUILD -eq 1 ]]; then
  echo "==> docker compose up -d --build"
  $SSH_CMD "$REMOTE_HOST" "$COMPOSE_CMD up -d --build"
else
  echo "==> docker compose up -d (no rebuild)"
  $SSH_CMD "$REMOTE_HOST" "$COMPOSE_CMD up -d"
fi

# ---- post --------------------------------------------------------------------
echo
echo "==> container status:"
$SSH_CMD "$REMOTE_HOST" "$COMPOSE_CMD ps"

if [[ $TAIL_LOGS -eq 1 ]]; then
  echo
  echo "==> tailing logs (Ctrl-C to detach)"
  $SSH_CMD -t "$REMOTE_HOST" "$COMPOSE_CMD logs -f --tail=50"
fi

PUBLIC_PORT="$($SSH_CMD "$REMOTE_HOST" "$COMPOSE_CMD port afl-model 3000 2>/dev/null" | awk -F: 'NR==1 {print $NF}')"
PUBLIC_PORT="${PUBLIC_PORT:-3000}"

echo
echo "Done. Dashboard: http://${REMOTE_HOST}:${PUBLIC_PORT}"
