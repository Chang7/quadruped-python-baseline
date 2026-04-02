# Linear OSQP Adapter for Quadruped-PyMPC

This patch adds a new `cfg.mpc_params['type'] = 'linear_osqp'` mode that reuses the
Quadruped-PyMPC realization stack (gait generator, foothold generator, swing controller,
early-stance detector, stance torque mapping) while swapping the SRBD solver with a
compact 13-state / 12-input linear force MPC solved by OSQP.

## Quick start

```bash
pip install -e .
python simulation/run_linear_osqp.py --gait crawl --seconds 20 --speed 0.30 --render
```

## What is reused from PyMPC
- contact sequence / gait generation
- foothold reference generation
- swing trajectory controller
- early stance detector
- stance torque application through `tau = -J^T f`

## What is replaced
- the SRBD optimal control solve only
