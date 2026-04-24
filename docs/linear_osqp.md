# linear_osqp Adapter

This repository does not replace the full locomotion stack. It swaps only the
high-level SRB force-planning layer with a compact `linear_osqp` controller
while reusing the stock-style MuJoCo realization path.

## What is reused

- gait / contact schedule generation
- foothold reference generation
- swing trajectory realization
- stance torque application through `tau = -J^T f`
- MuJoCo simulation wrapper and logging flow

## What is custom

- the high-level `linear_osqp` SRB force MPC
- extra transition-side support logic added while debugging the custom path

## Current interpretation

- `trot` is the main benchmark scenario.
- `crawl` is mainly a diagnostic scenario for rear touchdown / recontact and
  late load-transfer seams.

## Quick reproduction

Short custom trot:

```bash
python -m mujoco_sim.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12 --yaw-rate 0.0
```

Current crawl diagnostic:

```bash
python -m mujoco_sim.run_linear_osqp --controller linear_osqp --gait crawl --seconds 20 --speed 0.12 --yaw-rate 0.0
```

## Main files

- `quadruped_pympc/controllers/linear_osqp/linear_baseline_controller.py`
- `quadruped_pympc/helpers/foothold_reference_generator.py`
- `quadruped_pympc/helpers/rear_transition_manager.py`
- `quadruped_pympc/interfaces/wb_interface.py`
- `mujoco_sim/run_linear_osqp.py`
