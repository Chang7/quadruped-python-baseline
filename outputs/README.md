# Outputs Summary

All generated results should live under this `outputs/` folder.

Only this summary file is tracked in Git. The actual run artifacts, images,
GIFs, and temporary experiment folders are kept locally and should not be
committed to GitHub.

## Current Structure

- `curated_runs/`
  - the place to start
  - keep only:
    - current defaults that we would rerun today
    - stock sanity references that explain the scenario choice
    - a few milestone runs that explain how the current code path was reached
  - current defaults:
    - `crawl_rearallcontact_rearfloor_default_10s/`
    - `trot_default_turn_profile_10s/`
    - `trot_default_disturb_profile_10s/`
    - `trot_default_straight_profile_20s/`
  - stock sanity references:
    - `stock_sampling_crawl_4s_s003_isolated_recheck/`
    - `stock_sampling_crawl_4s_s006_isolated_recheck/`
    - `stock_sampling_crawl_4s_s012_isolated_recheck/`
    - `stock_sampling_trot_4s_s012_isolated_recheck/`
    - `stock_sampling_trot_turn_4s_y04_recheck/`
    - `stock_sampling_trot_disturb_4s_x48_recheck/`
  - milestone runs kept to avoid repeating the same failed directions:
    - `trot_dynamic_gait_fix/`
    - `trot_dynamic_gait_balanced/`
    - `trot_long_after_patch/`
    - `crawl_rear_transition_manager_default_10s/`
    - `crawl_rearallcontact_zcap_default_10s/`
  - current promoted `trot` path, in plain words:
    - selective linear-only foothold-`z` anchoring fix
    - split support-reference blending:
      `support_reference_mix = 0.85`, `support_reference_xy_mix = 1.0`
    - small constant posture-reference bias:
      `roll_ref_offset = +0.03 rad`, `pitch_ref_offset = -0.01 rad`
- `report_progress_explainer/`
  - small shareable bundles only
  - `stock_vs_linear_analysis_20260408/` is the current stock-vs-custom
    comparison bundle for matched 4 s checks
  - `trot_stable_compare/` is the active email/report comparison bundle
  - `archive/current_status_20260404/` is kept only as an older snapshot
- `archive/`
  - only raw batches that still explain the current code path are kept
  - `archive/raw_runs/20260405_rear_transition_manager_search/`
  - `archive/raw_runs/20260405_crawl_late_rear_allcontact_followup/`
  - `archive/raw_runs/20260408_trot_footholdz_fix/`
  - `archive/raw_runs/20260408_trot_mix_search/`
  - `archive/raw_runs/20260408_trot_xymix_search/`
  - `archive/raw_runs/20260408_trot_posture_bias_followup/`

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
