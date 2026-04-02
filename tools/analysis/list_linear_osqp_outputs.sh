#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-outputs/linear_osqp_stable}"
if [ ! -d "$ROOT" ]; then
  echo "Directory not found: $ROOT"
  exit 1
fi

echo "[all files under $ROOT]"
find "$ROOT" -maxdepth 5 -type f | sort

echo
echo "[artifact episode dirs]"
find "$ROOT" -maxdepth 2 -type d -name 'episode_*' | sort

echo
echo "[top-level episode_000 contents if present]"
if [ -d "$ROOT/episode_000" ]; then
  find "$ROOT/episode_000" -maxdepth 1 -type f | sort
fi
