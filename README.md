# Quadruped-PyMPC linear_osqp adapter

## What this is

Integrates a linear QP force MPC (OSQP) as the high-level controller into the
Quadruped-PyMPC MuJoCo stack, replacing only the upper-level planner while
keeping the lower-level realization path (torque mapping, swing control,
contact handling) from the stock codebase.

## Current results

One unified `generic` trot profile covers straight, turn, and disturbance.
Results below are same-scenario local benchmarks on Aliengo (flat ground).

| Scenario | Custom | Stock | Notes |
| --- | ---: | ---: | --- |
| Straight 20 s `mean\|roll\|` | 0.009 | 0.018 | |
| Straight 20 s `mean\|pitch\|` | 0.022 | 0.056 | |
| Turn 10 s `mean\|roll\|` | 0.011 | 0.014 | |
| Turn 10 s `mean\|pitch\|` | 0.024 | 0.045 | |
| Disturb 4 s `mean\|roll\|` | 0.006 | 0.031 | |
| Disturb 4 s `mean\|pitch\|` | 0.030 | 0.059 | |

Go1 results (after robot-specific posture offset retuning):

| Scenario | Go1 Custom | Notes |
| --- | ---: | --- |
| Straight 20 s `mean\|roll\|` | 0.006 | |
| Straight 20 s `mean\|pitch\|` | 0.045 | |
| Turn 10 s `mean\|roll\|` | 0.011 | |
| Turn 10 s `mean\|pitch\|` | 0.041 | |
| Disturb 4 s `mean\|roll\|` | 0.008 | |
| Disturb 4 s `mean\|pitch\|` | 0.052 | |

Solve time (Go1, unified trot): wall-solve mean ~1.2 ms per step.

## Quick start

```bash
pip install -e .
```

Run on Aliengo (default):

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12
```

Run on Go1:

```bash
QUADRUPED_PYMPC_ROBOT=go1 python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12
```

## Benchmarks

Straight 20 s:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 20 --speed 0.12 --artifact-dir outputs/repro_straight
```

Turn 10 s:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 10 --speed 0.10 --yaw-rate 0.3 --artifact-dir outputs/repro_turn
```

Disturbance 4 s:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12 --disturbance-pulse x:0.5:0.25:4.0 --disturbance-pulse x:2.3:0.25:8.0 --artifact-dir outputs/repro_disturb
```

Stock sampling comparison:

```bash
python -m simulation.run_linear_osqp --controller sampling --gait trot --seconds 4 --speed 0.12 --artifact-dir outputs/repro_stock
```

## Repository layout

```text
quadruped_pympc/          controller and helper code
  controllers/linear_osqp/  linear QP MPC (the custom part)
  interfaces/               whole-body interface (stock-based, extended)
simulation/               MuJoCo runners
  run_linear_osqp.py        main entry point
  simulation.py             sim loop
  artifacts.py              logging / summary
  crawl_preset.py           crawl-specific diagnostic parameters
tools/                    benchmark launcher
1.Quadruped-PyMPC-main/   read-only upstream reference
```

## Timing

Each `summary.json` records per-step timing when `linear_osqp` is used:

- `linear_solve_wall_ms_mean` / `max` — OSQP wall-clock solve time
- `linear_solve_total_ms_mean` / `max` — total controller step time

## Notes

- Robot selection: set `QUADRUPED_PYMPC_ROBOT=go1` (or `aliengo`, default).
- Robot-specific posture offsets are in `_robot_posture_offsets()` in
  `run_linear_osqp.py`.
- Crawl is a diagnostic scenario, not a stable benchmark. Stock sampling also
  terminates early (~1.4-2.8 s) in the same crawl setting.
