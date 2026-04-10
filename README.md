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

`trot` is the main benchmark. `crawl` is mainly a diagnostic scenario for rear
touchdown / recontact and late load-transfer seams.

| Scenario | Current custom | Matched stock reference | Notes |
| --- | ---: | ---: | --- |
| Straight 20 s `mean |pitch|` | `0.031` | `0.045` | tuned custom beats current stock reference |
| Turn 10 s `mean |roll|` | `0.018` | `0.014` | improved, but a gap remains |
| Turn 10 s `mean |pitch|` | `0.043` | `0.045` | close to stock |
| Disturb 4 s `mean |roll|` | `0.020` | `0.023` | current custom is slightly better |
| Disturb 4 s `mean |pitch|` | `0.034` | `0.055` | current custom is better |
| Crawl 20 s duration | `13.54 s` | n/a | current custom still fails at the late low-height seam |

## Current state

- Stock `trot` straight / turn / disturbance checks are stable.
- Current `linear_osqp` `trot` runs complete:
  - straight `20 s`
  - turn `10 s`
  - disturbance `4 s`
  without termination.
- Current `crawl` default reaches about `13.54 s` before a late `RL_hip`
  invalid contact.
- The main remaining bottlenecks are:
  - `trot`: remaining turn-roll / tracking quality gap relative to stock
  - `crawl`: late rear load transfer / post-touchdown stabilization

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
