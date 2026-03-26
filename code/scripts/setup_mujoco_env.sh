#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/requirements.txt
python -m pip install -r requirements/requirements_mujoco.txt

if [ ! -d mujoco_menagerie ]; then
  git clone https://github.com/google-deepmind/mujoco_menagerie.git
fi

echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  python -m mujoco.viewer --mjcf=./mujoco_menagerie/unitree_a1/scene.xml"
echo "  python phases/runner_mujoco_smoke.py --model ./mujoco_menagerie/unitree_a1/scene.xml --scenario straight_trot"
