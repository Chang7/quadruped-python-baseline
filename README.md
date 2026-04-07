# Quadruped PyMPC Linear OSQP Adapter

This repository is the active working tree for integrating a custom
`linear_osqp` high-level SRB force MPC into a Quadruped-PyMPC-style MuJoCo
locomotion stack.

The main goal of this workspace is not to reimplement the entire quadruped
stack from scratch, but to:

1. keep the stock-style MuJoCo integration structure as much as possible,
2. replace only the high-level controller with a compact linear OSQP backend,
3. study where touchdown/recontact and load-transfer failures still appear.

## Current Focus

The main remaining bottleneck is not the QP solve alone, but how the planned
forces are carried through contact transition and load transfer, especially
during rear touchdown/recontact.

## Current Status

- The stock MuJoCo integration stack is stable in the reference scenarios used
  here.
- In addition to straight `trot`, the stock sampling controller also completes
  the current short-horizon `trot + turn` and scheduled `trot + disturbance`
  checks without termination.
- The current `linear_osqp` path no longer shows the earlier immediate collapse
  in short-horizon `trot` tests.
- The current `linear_osqp` path now uses two explicit `trot` profiles instead
  of trying to force one dynamic preset to cover every case:
  - the default `auto` selection routes straight-line `trot` commands to the
    `straight_tuned` profile, and routes turning / disturbance checks to the
    `generic` profile;
  - the `generic` profile is meant for all-scenario checks and remains
    viable over the short `trot + turn` and scheduled `trot + disturbance`
    tests;
  - the optional `straight_tuned` profile is meant only for longer straight-line
    `trot` runs, where it improves forward tracking and posture quality over the
    earlier generic long-horizon behavior.
- The current conservative `crawl` default now reaches roughly the 8.7-second
  mark in a 10-second stress test before failure. The most recent improvement
  came from treating the late rear all-contact seam more locally: during the
  rear late all-contact stabilization window, the front touchdown-support alpha
  is reduced while a temporary rear-load floor bias is applied.
- The remaining `crawl` failure should now be read as a late-horizon
  stabilization problem rather than an early rear recontact failure. The robot
  still settles into a low all-contact posture and eventually drifts into a
  rear-hip invalid contact.
- The main remaining gap is now more about motion quality and long-horizon
  contact-transition robustness than basic viability: `trot` still shows a
  larger pitch bias than the stock baseline, and `crawl` still relies on strong
  support/recovery windows.

## Repository Layout

### Active Code

- `quadruped_pympc/`
- `simulation/`
- `tools/`

### Legacy Baseline

- `legacy/python_baseline/`
  - preserved copy of the earlier standalone Python baseline and MuJoCo
    prototype code
  - kept separately so the original baseline work is still readable without
    mixing it into the current stock-stack adapter tree

### Outputs

- `outputs/curated_runs/`
  - milestone runs from the earlier adapter-side debugging history
- `outputs/report_progress_explainer/`
  - current figures, GIFs, and report/email assets
- `outputs/archive/`
  - superseded or raw historical outputs kept for traceability

### References

- `references/notes/`
  - notes kept with this workspace
- `references/external_repos/`
  - local copies of external reference repositories

### Read-Only Stock Reference

- `1.Quadruped-PyMPC-main/`

This folder is kept only as a local upstream reference and should not be used
as the active working tree.

## Quick Start

Install in editable mode:

```bash
pip install -e .
```

Run the custom backend in MuJoCo:

```bash
python -m simulation.run_linear_osqp --gait crawl --controller linear_osqp --seconds 10
```

## Quick Reproduction

Short stock `trot` reference:

```bash
python -m simulation.run_linear_osqp --controller sampling --gait trot --seconds 3 --speed 0.12 --yaw-rate 0.0 --artifact-dir outputs/repro_stock_sampling_trot
```

Short stock `trot + turn` reference:

```bash
python -m simulation.run_linear_osqp --controller sampling --gait trot --seconds 4 --speed 0.12 --yaw-rate 0.4 --artifact-dir outputs/repro_stock_sampling_trot_turn
```

Short stock `trot + disturbance` reference:

```bash
python -m simulation.run_linear_osqp --controller sampling --gait trot --seconds 4 --speed 0.12 --disturbance-pulse x:0.5:0.25:4.0 --disturbance-pulse x:2.3:0.25:8.0 --artifact-dir outputs/repro_stock_sampling_trot_disturb
```

Short custom `linear_osqp` `trot` (default `auto` profile selection):

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 3 --speed 0.12 --yaw-rate 0.0 --artifact-dir outputs/repro_linear_osqp_trot
```

Long straight custom `linear_osqp` `trot` (`straight_tuned` profile):

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --dynamic-trot-profile straight_tuned --seconds 20 --speed 0.12 --yaw-rate 0.0 --artifact-dir outputs/repro_linear_osqp_trot_straight_tuned
```

Short custom `linear_osqp` `trot + turn` (default `auto` profile selection):

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12 --yaw-rate 0.4 --artifact-dir outputs/repro_linear_osqp_trot_turn
```

Short custom `linear_osqp` `trot + disturbance` (default `auto` profile selection):

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12 --disturbance-pulse x:0.5:0.25:4.0 --disturbance-pulse x:2.3:0.25:8.0 --artifact-dir outputs/repro_linear_osqp_trot_disturb
```

Current default `crawl` diagnostic:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait crawl --seconds 4 --speed 0.12 --yaw-rate 0.0 --artifact-dir outputs/repro_linear_osqp_crawl
```

Longer `crawl` stress check:

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait crawl --seconds 10 --speed 0.12 --yaw-rate 0.0 --artifact-dir outputs/repro_linear_osqp_crawl_long
```

Latest locally validated outputs:

- `outputs/curated_runs/crawl_rearallcontact_rearfloor_default_10s/`
- `outputs/curated_runs/trot_generic_turn_profile_4s/`
- `outputs/curated_runs/trot_generic_disturb_profile_4s/`
- `outputs/curated_runs/trot_straight_tuned_profile_20s/`
- `outputs/curated_runs/stock_sampling_trot_turn_4s_y04_recheck/`
- `outputs/curated_runs/stock_sampling_trot_disturb_4s_x48_recheck/`

The main contact-transition logic is currently concentrated in:

- `quadruped_pympc/helpers/rear_transition_manager.py`
- `quadruped_pympc/interfaces/wb_interface.py`
- `simulation/run_linear_osqp.py`

The `--disturbance-pulse` flag injects a smooth scheduled wrench pulse of the
form `axis:time:duration:magnitude`. This is intended as a lightweight local
approximation for paper-style disturbance checks.

## Recommended Reading Order

1. `WORKSPACE_LAYOUT.md`
2. `LINEAR_OSQP_README.md`
3. `outputs/README.md`
4. `outputs/report_progress_explainer/README.md`
5. `legacy/python_baseline/README.md`

## Notes

- New outputs should be saved under `outputs/`, not inside
  `1.Quadruped-PyMPC-main/`.
- Large raw runs, backups, and copied external repositories are intentionally
  excluded from the cleaned Git history.
