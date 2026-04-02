#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${1:-outputs/overnight_linear_session}"
shift || true

cd "$REPO_ROOT"
source venv/bin/activate

pkill -f "simulation\.autotune_linear_osqp" >/dev/null 2>&1 || true
pkill -f "simulation\.overnight_linear_recovery" >/dev/null 2>&1 || true

mkdir -p "$BASE_DIR"
nohup python -u -m simulation.overnight_linear_recovery --minutes 360 --base-dir "$BASE_DIR" "$@" >"$BASE_DIR/nohup.log" 2>&1 &
echo $! | tee "$BASE_DIR/pid.txt"
