# Outputs Summary

All generated results should live under this `outputs/` folder.

Only this summary file is tracked in Git. The actual run artifacts, images,
GIFs, and temporary experiment folders are kept locally and should not be
committed to GitHub.

## Current Structure

- `curated_runs/`
  - only the key milestone runs and current validated defaults are kept here
  - current defaults:
    - `crawl_rearallcontact_rearfloor_default_10s/`
    - `trot_after_rearallcontact_rearfloor_default_3s/`
  - stock-controller sanity checks:
    - `stock_sampling_crawl_4s_s003_isolated_recheck/`
    - `stock_sampling_crawl_4s_s006_isolated_recheck/`
    - `stock_sampling_crawl_4s_s012_isolated_recheck/`
    - `stock_sampling_trot_4s_s012_isolated_recheck/`
  - key historical milestones:
    - `trot_dynamic_gait_fix/`
    - `trot_dynamic_gait_balanced/`
    - `crawl_rear_transition_manager_default_10s/`
    - `crawl_rearallcontact_zcap_default_10s/`
    - `trot_long_after_patch/`
- `report_progress_explainer/`
  - only the current concise shareable summaries are kept here
  - `current_status_20260404/`
  - `trot_stable_compare/`
- `archive/`
  - only the recent raw search batches that still explain the current code path
    are kept
  - `archive/raw_runs/20260405_rear_transition_manager_search/`
  - `archive/raw_runs/tmp_continue/20260405_late_rear_allcontact_followup/`

## Suggested Reading Order

1. `curated_runs/`
2. `report_progress_explainer/`
3. `archive/`

## Working Rules

- Save new outputs here, not inside `1.Quadruped-PyMPC-main/`.
- Use clearly named folders for meaningful runs.
- If a result is only a temporary probe or a superseded comparison, move it to
  `outputs/archive/` briefly, then prune it if it no longer explains the
  current direction.
- After a tuning pass, keep:
  - one clean baseline,
  - one first meaningful breakthrough,
  - one current default validation run,
  and only the smallest raw search batch that still explains the chosen fix.
