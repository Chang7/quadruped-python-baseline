# Quadruped-PyMPC linear_osqp adapter

## What this is

This repository is the active working tree for integrating a custom
`linear_osqp` high-level SRB force MPC into a Quadruped-PyMPC-style MuJoCo
locomotion stack.

The goal is not to rebuild the entire quadruped stack from scratch. The goal
is to keep the stock-style MuJoCo realization path as intact as possible while
replacing only the high-level controller and studying where contact-transition
failures still remain.

## Current results

`trot` is the main benchmark. `crawl` is still used mainly as a diagnostic
scenario for rear touchdown / recontact and late load-transfer seams.

The current `linear_osqp` path uses one unified `generic` trot profile for the
three representative trot checks below.

| Scenario | Custom (`linear_osqp`) | Matched stock (`sampling`) |
| --- | ---: | ---: |
| Straight 20 s `mean |roll|` | `0.009` | `0.018` |
| Straight 20 s `mean |pitch|` | `0.022` | `0.056` |
| Turn 10 s `mean |roll|` | `0.011` | `0.014` |
| Turn 10 s `mean |pitch|` | `0.024` | `0.045` |
| Disturb 4 s `mean |roll|` | `0.006` | `0.031` |
| Disturb 4 s `mean |pitch|` | `0.030` | `0.059` |

The current local matched benchmark is:

- Straight: `20 s`, `0.12 m/s`, no yaw command, no disturbance
- Turn: `10 s`, `0.10 m/s`, `yaw-rate = 0.3 rad/s`
- Disturbance: `4 s`, `0.12 m/s`, pulses `x:0.5:0.25:4.0` and `x:2.3:0.25:8.0`

On Go1, the same unified trot controller already runs the same three checks
without termination, but the turn posture is still rougher than on Aliengo and
should be retuned before treating Go1 as a stable benchmark target.

## Current state

- The unified `generic` trot profile covers straight / turn / disturbance
  without the older `straight_tuned` split.
- The current custom trot path completes the three representative trot checks
  without termination and matches or beats the stock sampling reference in
  roll / pitch on the local Aliengo benchmark.
- Go1 now runs through the same three trot checks with the same controller
  structure, which is a good bring-up result, but Go1-specific retuning is
  still expected.
- The main remaining open problems are:
  - Go1 trot retuning, especially for turn posture
  - solve-time characterization for hardware deployment
  - crawl late rear load transfer / post-touchdown stabilization

## Repository layout

```text
quadruped_pympc/          active controller and helper code
simulation/               MuJoCo runners and scenario entry points
tools/                    launchers, report helpers, analysis scripts
outputs/                  local run artifacts and report bundles
docs/                     installation and adapter-specific notes
legacy/                   earlier standalone Python baseline
archive/                  non-active tools, bundles, and local reference assets
references/               notes and copied external references
1.Quadruped-PyMPC-main/   read-only upstream reference tree
```

## Quick start

Install in editable mode:

```bash
pip install -e .
```

Run the custom controller:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12
```

Run the same controller on Go1:

```powershell
$env:QUADRUPED_PYMPC_ROBOT='go1'
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12
```

## How to run benchmarks

Stock short trot:

```bash
python -m simulation.run_linear_osqp --controller sampling --gait trot --seconds 4 --speed 0.12 --yaw-rate 0.0 --artifact-dir outputs/repro_stock_trot
```

Custom short trot turn:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 10 --speed 0.10 --yaw-rate 0.3 --artifact-dir outputs/repro_custom_trot_turn
```

Custom disturbance check:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12 --disturbance-pulse x:0.5:0.25:4.0 --disturbance-pulse x:2.3:0.25:8.0 --artifact-dir outputs/repro_custom_trot_disturb
```

Current crawl diagnostic:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait crawl --seconds 20 --speed 0.12 --yaw-rate 0.0 --artifact-dir outputs/repro_custom_crawl_20s
```

Benchmark suite launcher:

```bash
python tools/launchers/run_trot_benchmark_suite.py --skip-existing
```

## Timing metrics

Each `summary.json` now also records `linear_osqp` timing fields when that
controller is used:

- `linear_solve_total_ms_mean`, `linear_solve_total_ms_max`
- `linear_solve_setup_ms_mean`, `linear_solve_setup_ms_max`
- `linear_solve_wall_ms_mean`, `linear_solve_wall_ms_max`
- `linear_solve_iter_mean`, `linear_solve_iter_max`

A recent Go1 probe on the unified trot controller showed mean total solve times
around `3.7-4.0 ms`, with wall-solve means around `1.2 ms`.

## Where to look first

- `outputs/README.md`
- `outputs/report_progress_explainer/weekly_progress_20260410/README.md`
- `legacy/python_baseline/README.md`
- `docs/linear_osqp.md`
- `docs/install.md`

## Notes

- Save new run artifacts only under `outputs/`.
- Do not use `1.Quadruped-PyMPC-main/` as the active working tree.
- Large local outputs, copied repos, and convenience environments are excluded
  from Git on purpose.
