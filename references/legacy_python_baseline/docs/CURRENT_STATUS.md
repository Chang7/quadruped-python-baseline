# Current status

This cleaned codebase keeps only two executable paths:

1. `runners/run_python_baseline.py`
   - simplified SRB simulator
   - verifies that the high-level MPC loop is working in Python

2. `runners/run_mujoco_quasistatic.py`
   - MuJoCo articulated prototype
   - latest branch with height target fix + gravity compensation
   - best current status: no immediate collapse, positive but slow forward motion, quasi-static crawl state machine runs through STARTUP / SHIFT / SWING / TOUCHDOWN / HOLD

## Latest known meaningful MuJoCo result
- `mean_vx_after_1s ≈ 0.0224 m/s`
- `mean_trunk_height_after_1s ≈ 0.3788 m`
- `collapse_time = None`
- `recovery_count = 0`

## What is intentionally NOT included
Old `phase1`~`phase16` experimental runners are excluded from this cleaned package.
The goal is to continue from a single branch instead of many accumulated patches.
