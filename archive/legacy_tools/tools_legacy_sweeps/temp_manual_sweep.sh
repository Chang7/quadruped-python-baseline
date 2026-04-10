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
    --q-theta 100000 \
    --q-w 10000 \
    --r-u 100 \
    --du-xy-max 1.0 \
    --support-floor-ratio 0.20 \
    --grf-max-scale 0.38 \
    --joint-pd-scale 0.25 \
    --latched-joint-pd-scale 0.25 \
    --gait-step-freq 0.38 \
    --gait-duty-factor 0.92 \
    --stance-target-blend 0.08 \
    --contact-latch-budget-s 0.10 \
    --startup-full-stance-time-s 0.28 \
    --latched-force-scale 0.95 \
    --latched-release-phase-start 0.18 \
    --latched-release-phase-end 0.58 \
    --fy-scale 1.0 \
    --support-centroid-x-gain 2.0 \
    --support-centroid-y-gain 10.0 \
    --pre-swing-lookahead-steps 5 \
    --pre-swing-gate-min-margin 0.025 \
    --pre-swing-gate-hold-s 0.12 \
    --min-vertical-force-scale 0.50 \
    --reduced-support-vertical-boost 0.05 \
    --z-pos-gain 22.0 \
    --z-vel-gain 5.5 \
    --roll-angle-gain 30.0 \
    --pitch-angle-gain 30.0 \
    "$@"
}
run manual_touchdown_bridge_a \
  --front-touchdown-reacquire-hold-s 0.08 \
  --front-touchdown-reacquire-xy-blend 0.25 \
  --front-touchdown-reacquire-forward-bias 0.015 \
  --front-touchdown-confirm-hold-s 0.05 \
  --touchdown-confirm-forward-scale 0.15 \
  --front-touchdown-settle-hold-s 0.10 \
  --touchdown-settle-forward-scale 0.15 \
  --touchdown-support-rear-floor-delta 0.10 \
  --touchdown-support-vertical-boost 0.08 \
  --touchdown-support-z-pos-gain-delta 3.0 \
  --touchdown-support-roll-angle-gain-delta 6.0 \
  --touchdown-support-roll-rate-gain-delta 1.5 \
  --touchdown-support-pitch-angle-gain-delta 4.0 \
  --touchdown-support-pitch-rate-gain-delta 1.0 \
  --touchdown-support-rear-joint-pd-scale 0.05 \
  --touchdown-support-anchor-xy-blend 0.25 \
  --touchdown-support-anchor-z-blend 0.10 \
  --rear-swing-bridge-hold-s 0.12 \
  --rear-swing-bridge-forward-scale 0.15 \
  --rear-swing-bridge-roll-threshold 0.45 \
  --rear-swing-bridge-pitch-threshold 0.30 \
  --rear-swing-bridge-height-ratio 0.58 \
  --rear-swing-bridge-recent-front-window-s 0.28 \
  --rear-swing-bridge-lookahead-steps 3
run manual_touchdown_bridge_b \
  --front-touchdown-reacquire-hold-s 0.10 \
  --front-touchdown-reacquire-xy-blend 0.30 \
  --front-touchdown-reacquire-forward-bias 0.020 \
  --front-touchdown-confirm-hold-s 0.06 \
  --touchdown-confirm-forward-scale 0.10 \
  --front-touchdown-settle-hold-s 0.14 \
  --touchdown-settle-forward-scale 0.10 \
  --touchdown-support-rear-floor-delta 0.14 \
  --touchdown-support-vertical-boost 0.12 \
  --touchdown-support-z-pos-gain-delta 5.0 \
  --touchdown-support-roll-angle-gain-delta 8.0 \
  --touchdown-support-roll-rate-gain-delta 2.5 \
  --touchdown-support-pitch-angle-gain-delta 5.0 \
  --touchdown-support-pitch-rate-gain-delta 1.5 \
  --touchdown-support-rear-joint-pd-scale 0.08 \
  --touchdown-support-anchor-xy-blend 0.35 \
  --touchdown-support-anchor-z-blend 0.15 \
  --rear-swing-bridge-hold-s 0.16 \
  --rear-swing-bridge-forward-scale 0.10 \
  --rear-swing-bridge-roll-threshold 0.40 \
  --rear-swing-bridge-pitch-threshold 0.28 \
  --rear-swing-bridge-height-ratio 0.62 \
  --rear-swing-bridge-recent-front-window-s 0.32 \
  --rear-swing-bridge-lookahead-steps 3
run manual_touchdown_support_c \
  --front-touchdown-reacquire-hold-s 0.08 \
  --front-touchdown-reacquire-xy-blend 0.20 \
  --front-touchdown-reacquire-forward-bias 0.012 \
  --front-touchdown-confirm-hold-s 0.04 \
  --touchdown-confirm-forward-scale 0.20 \
  --front-touchdown-settle-hold-s 0.12 \
  --touchdown-settle-forward-scale 0.18 \
  --touchdown-support-rear-floor-delta 0.10 \
  --touchdown-support-vertical-boost 0.10 \
  --touchdown-support-z-pos-gain-delta 4.0 \
  --touchdown-support-roll-angle-gain-delta 7.0 \
  --touchdown-support-roll-rate-gain-delta 2.0 \
  --touchdown-support-pitch-angle-gain-delta 4.0 \
  --touchdown-support-pitch-rate-gain-delta 1.0 \
  --touchdown-support-rear-joint-pd-scale 0.06 \
  --touchdown-support-anchor-xy-blend 0.30 \
  --touchdown-support-anchor-z-blend 0.12
