set -e
cd /mnt/c/Quadruped-PyMPC_linear_osqp_adapter
source venv/bin/activate
run(){
  name="$1"
  shift
  echo "=== $name ==="
  python -m simulation.run_linear_osqp \
    --gait crawl \
    --preset conservative \
    --seconds 5 \
    --artifact-dir "outputs/$name" \
    --speed 0.12 \
    --lateral-speed 0.0 \
    --yaw-rate 0.0 \
    --q-theta 100000 \
    --q-w 10000 \
    --r-u 100 \
    --du-xy-max 1.0 \
    --support-floor-ratio 0.20 \
    --grf-max-scale 0.36 \
    --joint-pd-scale 0.25 \
    --latched-joint-pd-scale 0.25 \
    --gait-step-freq 0.4 \
    --gait-duty-factor 0.9 \
    --stance-target-blend 0.08 \
    --contact-latch-budget-s 0.10 \
    --startup-full-stance-time-s 0.28 \
    --latched-force-scale 0.95 \
    --latched-release-phase-start 0.18 \
    --latched-release-phase-end 0.58 \
    --fy-scale 1.0 \
    --support-centroid-x-gain 2.0 \
    --support-centroid-y-gain 8.0 \
    --pre-swing-lookahead-steps 5 \
    --pre-swing-front-shift-scale 1.5 \
    --pre-swing-rear-shift-scale 1.0 \
    --pre-swing-gate-min-margin 0.025 \
    --front-pre-swing-gate-min-margin 0.025 \
    --rear-pre-swing-gate-min-margin 0.025 \
    --front-late-release-phase-threshold 0.65 \
    --front-late-release-min-margin 0.015 \
    --front-late-release-hold-s 0.04 \
    --front-late-release-forward-scale 0.2 \
    --support-margin-preview-s 0.04 \
    --front-late-release-extra-margin 0.005 \
    --front-late-release-pitch-guard 0.30 \
    --front-late-release-roll-guard 0.35 \
    --front-touchdown-reacquire-hold-s 0.14 \
    --touchdown-reacquire-forward-scale 0.2 \
    --front-touchdown-reacquire-xy-blend 0.7 \
    --front-touchdown-reacquire-extra-depth 0.016 \
    --front-touchdown-reacquire-forward-bias 0.014 \
    --front-touchdown-confirm-hold-s 0.08 \
    --touchdown-confirm-forward-scale 0.25 \
    --front-touchdown-settle-hold-s 0.10 \
    --touchdown-settle-forward-scale 0.35 \
    --touchdown-support-rear-floor-delta 0.10 \
    --touchdown-support-vertical-boost 0.10 \
    --touchdown-support-z-pos-gain-delta 4.0 \
    --touchdown-support-roll-angle-gain-delta 4.0 \
    --touchdown-support-roll-rate-gain-delta 1.0 \
    --touchdown-support-pitch-angle-gain-delta 8.0 \
    --touchdown-support-pitch-rate-gain-delta 2.0 \
    --touchdown-support-rear-joint-pd-scale 0.08 \
    --touchdown-support-anchor-xy-blend 0.75 \
    --touchdown-support-anchor-z-blend 0.5 \
    --pre-swing-gate-hold-s 0.12 \
    --pre-swing-gate-forward-scale 0.35 \
    --z-pos-gain 22.0 \
    --z-vel-gain 5.5 \
    --min-vertical-force-scale 0.55 \
    --reduced-support-vertical-boost 0.05 \
    --roll-angle-gain 32.0 \
    --roll-rate-gain 7.0 \
    --pitch-angle-gain 32.0 \
    --pitch-rate-gain 8.0 \
    --rear-floor-base-scale 0.65 \
    --rear-floor-pitch-gain 0.3 \
    "$@"
}
run focused_contact_seed_bridge3 \
  --rear-swing-bridge-hold-s 0.08 \
  --rear-swing-bridge-forward-scale 0.35 \
  --rear-swing-bridge-roll-threshold 0.55 \
  --rear-swing-bridge-pitch-threshold 0.32 \
  --rear-swing-bridge-height-ratio 0.54 \
  --rear-swing-bridge-recent-front-window-s 0.20 \
  --rear-swing-bridge-lookahead-steps 2
run focused_contact_seed_bridge4 \
  --rear-swing-bridge-hold-s 0.08 \
  --rear-swing-bridge-forward-scale 0.35 \
  --rear-swing-bridge-roll-threshold 0.55 \
  --rear-swing-bridge-pitch-threshold 0.32 \
  --rear-swing-bridge-height-ratio 0.54 \
  --rear-swing-bridge-recent-front-window-s 0.20 \
  --rear-swing-bridge-lookahead-steps 2 \
  --touchdown-support-side-rebalance-delta 0.05 \
  --touchdown-support-roll-angle-gain-delta 6.0 \
  --touchdown-support-roll-rate-gain-delta 2.0 \
  --touchdown-support-rear-floor-delta 0.12
run focused_contact_seed_support_side \
  --touchdown-support-side-rebalance-delta 0.05 \
  --touchdown-support-roll-angle-gain-delta 6.0 \
  --touchdown-support-roll-rate-gain-delta 2.0 \
  --touchdown-support-rear-floor-delta 0.12
