#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="./mujoco_menagerie/unitree_a1/scene.xml"

# 1) Standing/support diagnostic: no forward demand, long settle.
python phases/runner_mujoco_phase12_crawl_visual.py \
  --model "$MODEL" \
  --scenario straight_trot \
  --output-dir local_outputs/outputs_mujoco_phase12/stand_diag \
  --disable-nonfoot-collision \
  --settle-time 2.0 \
  --gait-ramp-time 3.0 \
  --crawl-phase-duration 0.50 \
  --swing-duration 0.10 \
  --step-len-front 0.00 \
  --rear-step-scale 0.00 \
  --stance-press-front 0.016 \
  --stance-press-rear 0.006 \
  --stance-drive-front 0.000 \
  --stance-drive-rear 0.000 \
  --front-unload -0.010 \
  --height-k 1.8 \
  --pitch-k 0.025 \
  --pitch-sign -1.0 \
  --roll-k 0.020 \
  --dq-limit 0.05 \
  --demo-speed-cap 0.0 \
  --mpc-load-gain 0.10

# 2) Conservative front-support crawl.
python phases/runner_mujoco_phase12_crawl_visual.py \
  --model "$MODEL" \
  --scenario straight_trot \
  --output-dir local_outputs/outputs_mujoco_phase12/front_support_crawl \
  --disable-nonfoot-collision \
  --settle-time 1.4 \
  --gait-ramp-time 2.4 \
  --crawl-phase-duration 0.44 \
  --swing-duration 0.12 \
  --step-len-front 0.020 \
  --rear-step-scale 0.70 \
  --touchdown-depth-front 0.015 \
  --touchdown-depth-rear 0.020 \
  --stance-press-front 0.016 \
  --stance-press-rear 0.006 \
  --stance-drive-front 0.002 \
  --stance-drive-rear 0.001 \
  --front-unload -0.008 \
  --height-k 1.8 \
  --pitch-k 0.025 \
  --pitch-sign -1.0 \
  --roll-k 0.020 \
  --dq-limit 0.05 \
  --demo-speed-cap 0.06 \
  --mpc-load-gain 0.10
