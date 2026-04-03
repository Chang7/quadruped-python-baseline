#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -d mujoco_menagerie ]]; then
  git clone https://github.com/google-deepmind/mujoco_menagerie.git
fi

echo "Setup complete."
echo "Activate env: source .venv/bin/activate"
echo "Baseline run: python runners/run_python_baseline.py --scenario straight_trot"
echo "MuJoCo run:  python runners/run_mujoco_quasistatic.py --model ./mujoco_menagerie/unitree_a1/scene.xml --support-enabled --disable-nonfoot-collision"
