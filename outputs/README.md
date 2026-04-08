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
    - `trot_default_turn_profile_4s/`
    - `trot_default_disturb_profile_4s/`
    - `trot_straight_tuned_profile_20s/`
  - default trot behavior now selects between the `generic` and
    `straight_tuned` profiles automatically based on the command
  - the current promoted `trot` runs use the selective linear-only
    foothold-`z` anchoring fix in the foothold reference generator
  - the current promoted `generic` trot runs also use
    `support_reference_mix = 0.85` together with
    `support_reference_xy_mix = 1.0`, which improved short-horizon turn and
    disturbance posture quality relative to the earlier generic defaults while
    preserving the separate straight-line `straight_tuned` profile
  - stock-controller sanity checks:
    - `stock_sampling_crawl_4s_s003_isolated_recheck/`
    - `stock_sampling_crawl_4s_s006_isolated_recheck/`
    - `stock_sampling_crawl_4s_s012_isolated_recheck/`
    - `stock_sampling_trot_4s_s012_isolated_recheck/`
    - `stock_sampling_trot_turn_4s_y04_recheck/`
    - `stock_sampling_trot_disturb_4s_x48_recheck/`
  - matching custom-controller short checks:
    - `trot_default_turn_profile_4s/`
    - `trot_default_disturb_profile_4s/`
  - straight-line tuned `trot` profile checks:
    - `trot_straight_tuned_profile_20s/`
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
  - `archive/raw_runs/20260408_trot_footholdz_fix/`
  - `archive/raw_runs/20260408_trot_generic_retune/`
  - `archive/raw_runs/20260408_trot_pitch_rebalance_search/`
  - `archive/raw_runs/20260408_trot_supportcentroid_search/`
  - `archive/raw_runs/20260408_trot_mix_search/`
  - `archive/raw_runs/20260408_trot_mix_vx_search/`
  - `archive/raw_runs/20260408_trot_xymix_search/`
  - `archive/raw_runs/20260408_trot_followup_narrow_search/`
  - `archive/raw_runs/20260408_trot_pitchgain_narrow_search/`
  - `archive/raw_runs/20260408_trot_pitchoffset_search/`
  - `archive/raw_runs/20260408_trot_pitchoffset_vx_search/`
  - `archive/raw_runs/20260408_trot_vxgain_on_mix085_search/`
  - `archive/raw_runs/20260408_pre_mix080_generic_default/`
  - `archive/raw_runs/20260408_pre_mix085_current_defaults/`
  - `archive/raw_runs/20260408_pre_selective_trot_baselines/`

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
