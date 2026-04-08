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

The main remaining bottlenecks are now:

1. `crawl`: late rear load transfer / post-touchdown stabilization, after the
   earlier rear recontact seam was narrowed down.
2. `trot`: motion quality in dynamic scenarios, especially pitch bias and
   forward-tracking quality relative to the stock sampling baseline.

## Current Status

- The stock MuJoCo integration stack is stable in the reference scenarios used
  here.
- In addition to straight `trot`, the stock sampling controller also completes
  the current short-horizon `trot + turn` and scheduled `trot + disturbance`
  checks without termination.
- The current `linear_osqp` path no longer shows the earlier immediate collapse
  in short-horizon `trot` tests.
- The earlier `trot` geometry fix in the foothold-reference layer is still part
  of the current custom path: the `linear_osqp` controller freezes the
  world-frame foothold `z` anchors during ongoing swing/stance updates, while
  the stock sampling path keeps the original behavior. This removed a
  front/rear swing-height asymmetry that had been coupling the swing references
  to body bobbing and pitch.
- The most important recent `trot + turn` fix was made in the horizon rollout:
  the `linear_osqp` reference no longer resets yaw toward a fixed zero heading
  at every MPC update. Instead, roll and pitch are reset to their reference
  values while yaw evolves relative to the current base heading. This removed a
  self-opposing yaw-angle feedback loop in turning tests.
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
- After the selective foothold-`z` fix, the most useful additional generic
  `trot` improvement came from splitting the support-reference blend by axis:
  the promoted generic profile now uses `support_reference_mix = 0.85` for the
  vertical component and `support_reference_xy_mix = 1.0` for the horizontal
  components. This keeps the more posture-friendly vertical blend while letting
  the horizontal support reference follow the solved wrench more directly.
- A stronger post-solve `pitch_rebalance_gain` was tested and rejected because
  it caused early dynamic-gait collapse instead of reducing the persistent
  positive pitch bias.
- The most useful follow-up change after that was a small constant
  posture-reference bias inside the `linear_osqp` horizon rollout:
  - `roll_ref_offset = +0.03 rad`
  - `pitch_ref_offset = -0.01 rad`
  This treats the remaining dynamic-gait error as a steady posture bias rather
  than a force-limit problem. The same pair improved the short `turn`,
  scheduled `disturbance`, and long straight-line `trot` checks at once, so it
  is now part of the promoted default dynamic profiles.
- The currently promoted `trot` validations are:
  - `trot_current_straight_default_20s/`: no termination, `mean_vx about 0.050`,
    `mean_base_z about 0.416`, `mean |roll| about 0.013`, `mean |pitch| about 0.052`
  - `trot_current_turn_default_10s/`: no termination, `mean_base_z about 0.410`,
    `mean |roll| about 0.024`, `mean |pitch| about 0.059`, `mean wz about 0.311`
  - `trot_current_disturb_default_10s/`: no termination, `mean_vx about 0.064`,
    `mean_base_z about 0.430`, `mean |roll| about 0.015`, `mean |pitch| about 0.047`
- In the fixed short-horizon benchmark, the turning fix raised the custom
  `linear_osqp` mean yaw rate from roughly `0.065` to `0.270 rad/s` without
  breaking the straight or disturbance checks. The current short-horizon
  benchmark bundle is `outputs/report_progress_explainer/trot_benchmark_suite_20260408_yawref/`.
- The current conservative `crawl` default now reaches roughly the 8.7-second
  mark in a 10-second stress test before failure. The most recent improvement
  came from treating the late rear all-contact seam more locally: during the
  rear late all-contact stabilization window, the front touchdown-support alpha
  is reduced while a temporary rear-load floor bias is applied.
- The remaining `crawl` failure should now be read as a late-horizon
  stabilization problem rather than an early rear recontact failure. The robot
  still settles into a low all-contact posture and eventually drifts into a
  rear-hip invalid contact.
- The main remaining gap is now more about stock-level tracking quality and
  long-horizon contact-transition robustness than basic viability: `trot`
  remains usable across the current straight / turn / disturbance checks, while
  `crawl` still relies on strong support/recovery windows.

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

Fixed trot benchmark suite:

```bash
python tools/launchers/run_trot_benchmark_suite.py --skip-existing
```

This suite standardizes the current main `trot` benchmark around:

- `4 s` straight-line trot
- `4 s` trot + turn
- `4 s` trot + scheduled disturbance
- `20 s` straight-line trot

The suite runner writes a manifest and a compact dashboard so that `trot`
comparisons do not depend on ad-hoc commands or mixed horizons.

Latest locally validated outputs:

- `outputs/curated_runs/crawl_rearallcontact_rearfloor_default_10s/`
- `outputs/curated_runs/trot_current_turn_default_10s/`
- `outputs/curated_runs/trot_current_disturb_default_10s/`
- `outputs/curated_runs/trot_current_straight_default_20s/`
- `outputs/curated_runs/stock_sampling_trot_turn_4s_y04_recheck/`
- `outputs/curated_runs/stock_sampling_trot_disturb_4s_x48_recheck/`
- `outputs/report_progress_explainer/trot_benchmark_suite_20260408_yawref/`

The main active files for the current `trot` / contact-transition work are:

- `quadruped_pympc/controllers/linear_osqp/linear_baseline_controller.py`
- `quadruped_pympc/helpers/foothold_reference_generator.py`
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
