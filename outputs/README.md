# Outputs Summary

All generated results should live under this `outputs/` folder.

Only this summary file is tracked in Git. The actual run artifacts, images,
GIFs, and temporary experiment folders are kept locally and should not be
committed to GitHub.

## Current Structure

- `curated_runs/`
  - milestone runs and current validated defaults
  - keep the latest "known-good" crawl/trot runs here so future changes always
    have a clear comparison point
  - current rear-transition-manager-based defaults:
    - `crawl_rear_transition_manager_default_10s/`
    - `trot_rear_transition_manager_default_3s/`
- `report_progress_explainer/`
  - current email/meeting/report assets
- `stock_stack_runs/`
  - raw stock-stack diagnostic history
- `archive/`
  - older raw runs and superseded output folders kept for traceability
  - also the right place for exploratory `tmp_*` run folders once they are no
    longer active
  - `archive/raw_runs/front_recontact_trials/` currently holds the recent
    follow-up probes that tried to push the new crawl default beyond the
    4-second diagnostic horizon
  - `archive/raw_runs/crawl_front_support_trials/` groups the newer 10-second
    crawl experiments around front touchdown-support PD and related rear
    support-window variants
  - `archive/raw_runs/20260404_rear_touchdown_search/` keeps the later
    rear-relatch-focused crawl search, including the rejected rear-leg PD /
    touchdown-damping follow-up probes
  - `archive/raw_runs/20260405_rear_transition_manager_search/` keeps the
    first focused sweep after splitting the rear touchdown / recontact logic
    into a dedicated helper, including stricter vs. looser rear contact
    acceptance and short rear-settle/support variants

## Suggested Reading Order

1. `curated_runs/`
2. `report_progress_explainer/`
3. `stock_stack_runs/`
4. `archive/`

## Working Rules

- Save new outputs here, not inside `1.Quadruped-PyMPC-main/`.
- Use clearly named folders for meaningful runs.
- If a result is only a temporary probe or a superseded comparison, move it to
  `outputs/archive/` instead of leaving it at top level.
- After a tuning pass, keep:
  - one clean baseline,
  - one first meaningful breakthrough,
  - one current default validation run,
  and archive the rest.
