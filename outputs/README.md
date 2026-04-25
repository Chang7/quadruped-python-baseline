# Outputs Summary

All generated results should live under this `outputs/` folder.

Only this summary file is tracked in Git. The actual run artifacts, images,
GIFs, and temporary experiment folders are kept locally and should not be
committed to GitHub.

## Current Structure

- `curated_runs/`
  - the first place to look
  - split by role:
    - `current/`
      - promoted defaults we would rerun today
      - currently:
        - `current/crawl_current_default_20s/`
        - `current/trot_straight_20s_g025_pitchoff003/`
        - `current/trot_turn_10s_g025_pitchoff003/`
        - `current/trot_disturb_4s_g025_pitchoff003/`
    - `stock_baselines/`
      - stock reference runs used for sanity checks and matched comparisons
      - includes both short-horizon stock checks and the weekly same-horizon
        stock references
    - `predecessors/`
      - immediate previous defaults and regression checks kept only because
        they still explain the current code path
  - older milestones were moved to:
    - `archive/curated_milestones/`
- `report_progress_explainer/`
  - small shareable bundles only
  - `stock_vs_linear_analysis_20260408/` is the current stock-vs-custom
    comparison bundle for matched checks
  - `weekly_progress_20260410/` is the current one-stop weekly report bundle
  - `trot_benchmark_suite_20260408_yawref/` is the current fixed trot
    benchmark dashboard after the yaw-reference rollout fix
  - older report bundles were moved under:
    - `report_progress_explainer/archive/`
- `archive/`
  - only raw batches that still explain the current code path are kept
  - grouped by date/theme so the root stays readable:
    - `archive/raw_runs/crawl_20260408/`
    - `archive/raw_runs/crawl_20260409/`
    - `archive/raw_runs/trot_20260408/`
    - `archive/raw_runs/trot_20260409/`
    - `archive/raw_runs/misc_20260408/`
  - recent loose quality sweeps were moved under:
    - `archive/raw_runs/trot_20260409/quality_sweeps/`
  - parking folders that should not be treated as active search batches:
    - `archive/raw_runs/20260409_root_tmp_migration/`
    - `archive/raw_runs/20260409_retired_curated_candidates/`
  - older curated milestone runs were moved into:
    - `archive/curated_milestones/`

## Suggested Reading Order

1. `curated_runs/`
2. `report_progress_explainer/`
3. `archive/`

## Working Rules

- Save new outputs here, not inside `references/upstream_pympc/`.
- Use clearly named folders for meaningful runs.
- If a result is only a temporary probe or a superseded comparison, move it to
  `outputs/archive/` briefly, then prune it if it no longer explains the
  current direction.
- After a tuning pass, keep:
  - one current default,
  - one or two immediate predecessor runs,
  - one stock sanity baseline,
  - and only the smallest raw search batch that still explains the chosen fix.
