#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/c/Quadruped-PyMPC_linear_osqp_adapter"
SESSION_DIR="$REPO_ROOT/outputs/autotune_session"
MINUTES="${1:-95}"

mkdir -p "$SESSION_DIR"
cd "$REPO_ROOT"
source venv/bin/activate

nohup python -u -m simulation.autotune_linear_osqp \
  --minutes "$MINUTES" \
  --base-dir outputs/autotune_session \
  > "$SESSION_DIR/nohup.log" 2>&1 < /dev/null &

echo $! > "$SESSION_DIR/pid.txt"
cat "$SESSION_DIR/pid.txt"
