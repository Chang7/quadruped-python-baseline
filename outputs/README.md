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
    - `crawl_current_default_20s/`
    - `crawl_current_lateloadshare_dedicated_default_20s/`
    - `trot_current_turn_default_10s/`
    - `trot_current_disturb_default_10s/`
    - `trot_current_straight_default_20s/`
    - `trot_regression_after_crawl_default_10s/`
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
    - `crawl_rearallcontact_rearfloor_default_10s/`
      - previous crawl default before the later front-tail promotion
    - `crawl_rearallcontact_rearfloor045_default_10s/`
      - immediately previous crawl default before the `rear_floor_delta = 0.50`
        promotion
  - current promoted `trot` path, in plain words:
    - selective linear-only foothold-`z` anchoring fix
    - yaw-reference rollout fix:
      keep yaw relative to the current base heading instead of resetting it
      toward zero at every MPC update
    - split support-reference blending:
      `support_reference_mix = 0.85`, `support_reference_xy_mix = 1.0`
    - small constant posture-reference bias:
      `roll_ref_offset = +0.03 rad`, `pitch_ref_offset = -0.01 rad`
    - rejected follow-ups kept in archive:
      yaw-aware foothold compensation and extra yaw-gain sweeps after the
      rollout fix did not beat the plain yaw-reference fix
  - current promoted `crawl` path, in plain words:
    - keep the delayed front-swing late-recovery hold active for `0.10 s`
    - keep the remaining front stance leg on touchdown-style support for
      `0.10 s` after the opposite front leg actually opens swing
    - keep the targeted rear close-handoff leg-floor support alive for
      `0.22 s` after a late rear re-close
    - keep a separate weak-leg late rear-load-share support window alive for
      `0.20 s`, with `rear_late_load_share_support_leg_floor_scale_delta = 0.10`
      instead of renewing the broader close-handoff timer
    - keep the earlier rear all-contact rear-floor default (`0.55`) underneath
      that front-tail plus rear close-handoff plus dedicated late-load-share logic
    - rejected follow-ups kept in archive:
      `rear_close_handoff_hold_s = 0.18 / 0.20 / 0.24 / 0.26`,
      `rear_all_contact_release_tail_alpha_scale = 0.25 / 0.50`, and
      broader front-planted/postdrop recovery re-arms all failed to beat the
      plain `0.22 / 0.20` close-handoff plus late-load-share setting
    - rejected late weak-leg floor sweeps kept in archive:
      `rear_late_load_share_support_leg_floor_scale_delta = 0.06 / 0.08 / 0.09 / 0.11 / 0.12 / 0.16`
      and extra weak-leg-tail revisits on top of the new default all lost to the
      plain `0.10 / 0.20` dedicated late-load-share setting
- `report_progress_explainer/`
  - small shareable bundles only
  - `stock_vs_linear_analysis_20260408/` is the current stock-vs-custom
    comparison bundle for matched 4 s checks
  - `trot_benchmark_suite_20260408_yawref/` is the current fixed main
    benchmark bundle for short-horizon straight / turn / disturbance plus
    long straight-line trot after the yaw-reference fix
  - `trot_benchmark_suite_20260408/` is kept only as the pre-yaw-reference
    turning baseline
  - `trot_stable_compare/` is the active email/report comparison bundle
  - `archive/current_status_20260404/` is kept only as an older snapshot
- `archive/`
  - only raw batches that still explain the current code path are kept
  - top-level `tmp_*` runs were moved into
    `archive/raw_runs/20260409_root_tmp_migration/` so that `outputs/` stays clean
  - retired former curated crawl candidates were moved into
    `archive/raw_runs/20260409_retired_curated_candidates/`
  - `archive/raw_runs/20260405_rear_transition_manager_search/`
  - `archive/raw_runs/20260405_crawl_late_rear_allcontact_followup/`
  - `archive/raw_runs/20260408_trot_footholdz_fix/`
  - `archive/raw_runs/20260408_trot_benchmark_suite/`
  - `archive/raw_runs/20260408_trot_benchmark_suite_yawref/`
  - `archive/raw_runs/20260408_trot_mix_search/`
  - `archive/raw_runs/20260408_trot_xymix_search/`
  - `archive/raw_runs/20260408_trot_posture_bias_followup/`
  - `archive/raw_runs/20260408_crawl_param_sweeps/`
  - `archive/raw_runs/20260408_crawl_hold010_followup/`
  - `archive/raw_runs/20260408_crawl_rearfloor_refine/`
  - `archive/raw_runs/20260408_crawl_fronttail_followup/`
  - `archive/raw_runs/20260409_crawl_closehandoff_sweep/`
  - `archive/raw_runs/20260409_crawl_lateloadshare_dedicated/`
  - `archive/raw_runs/20260409_crawl_lateloadshare_dedicated_refine/`
  - `archive/raw_runs/20260409_crawl_lateloadshare_local_opt/`
  - `archive/raw_runs/20260409_crawl_fronttrigger_debug/`
  - `archive/raw_runs/20260409_crawl_rearhold_resweep/`
  - `archive/raw_runs/20260409_crawl_closehandoff_refine/`
  - `archive/raw_runs/20260409_crawl_release_tail_refine/`
  - `archive/raw_runs/20260409_crawl_nominalfront_patch/`
  - `archive/raw_runs/20260409_crawl_closehandoff_plateau/`
  - `archive/raw_runs/20260409_crawl_weaktail_revisit/`
  - `archive/raw_runs/20260409_crawl_singleleg_lateload/`
  - `archive/raw_runs/20260409_crawl_current_default_verify/`
  - `archive/raw_runs/20260409_trot_regression_after_crawl/`
  - `archive/raw_runs/20260409_trot_regression_after_crawl_closehandoff014/`
  - `archive/raw_runs/20260408_trot_yaw_ref_fix/`
  - `archive/raw_runs/20260408_trot_yaw_gain_after_ref_fix/`

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
