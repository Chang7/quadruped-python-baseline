# Curated Runs

This folder keeps the milestone runs worth preserving from the earlier
adapter-side debugging history. These are not the full raw sweep history.
They are the short list of runs that capture what changed and why it mattered.

## Recommended Reading Order

1. `01_visual_baseline_mujoco`
2. `02_pre_front_release_lockup`
3. `03_front_release_breakthrough`
4. `04_front_reacquire_confirm_breakthrough`
5. `05_contact_model_improvement`
6. `06_best_duration_supportconfirm`
7. `07_current_front_margin_rescue`
8. `trot_dynamic_gait_fix`
9. `trot_dynamic_gait_pitch_tuned`
10. `trot_dynamic_gait_balanced`
11. `crawl_rear_touchdown_forced_release`
12. `crawl_rear_touchdown_balanced_default`
13. `trot_after_crawl_balanced_default`
14. `crawl_diagnostic_slow_gait_default`
15. `crawl_full_contact_recovery_candidate`
16. `trot_after_crawl_diagnostic_default`

## Milestones

### 01_visual_baseline_mujoco

- Meaning: early MuJoCo baseline where the robot moved forward visually but
  still failed by front-side collapse.
- Why keep it: first useful visual proof that the controller could produce
  recognizable locomotion behavior in MuJoCo.

### 02_pre_front_release_lockup

- Meaning: branch that survived longer but kept the front swing almost fully
  locked.
- Why keep it: useful counterexample showing that "longer duration" did not
  necessarily mean better gait realization.

### 03_front_release_breakthrough

- Meaning: first branch where front actual swing clearly opened.
- Why keep it: marks the transition from "release blocked" to "release is
  possible but unstable later."

### 04_front_reacquire_confirm_breakthrough

- Meaning: first clear milestone where front actual swing and front touchdown
  reacquire/confirm appeared together.
- Why keep it: first meaningful touchdown-transition milestone instead of only
  swing opening.

### 05_contact_model_improvement

- Meaning: run after contact-model adjustments where the failure mode changed
- from immediate trunk collapse toward later rear-side failure.
- Why keep it: documents the point where MuJoCo contact tuning materially
  changed the observed failure mode.

### 06_best_duration_supportconfirm

- Meaning: longest meaningful branch from the older support-confirm direction.
- Why keep it: useful reference for a long-but-imperfect branch with measurable
  gait activity.

### 07_current_front_margin_rescue

- Meaning: latest rescue-direction milestone before shifting attention toward
  the stock-stack integration path.
- Why keep it: captures the last meaningful adapter-side branch and the late
  front-margin rescue idea.

### trot_dynamic_gait_fix

- Meaning: first stock-stack `linear_osqp` trot run that completed the full
  3-second window without termination after adding dynamic-gait-specific
  transition and force-build-up settings.
- Why keep it: marks the point where the early trot collapse was converted into
  a non-terminating run, even though speed and attitude tracking are still
  weaker than the stock sampling controller.

### trot_dynamic_gait_pitch_tuned

- Meaning: follow-up trot run after keeping the dynamic-gait transition fix and
  increasing only the pitch-angle/rate feedback for the dynamic gait preset.
- Why keep it: current safer default for `linear_osqp` trot runs. It keeps the
  non-terminating 3-second behavior while trimming the persistent forward-pitch
  bias slightly compared with the first stable trot milestone.

### trot_dynamic_gait_balanced

- Meaning: follow-up trot run after adding a mild roll-dependent lateral-force
  boost and retuning the dynamic-gait pitch feedback to a more balanced preset.
- Why keep it: current best balanced default for `linear_osqp` trot runs. It
  keeps the non-terminating 3-second behavior while improving mean forward
  speed, lateral drift, roll, and body height relative to the earlier
  pitch-only default, at the cost of a small increase in mean pitch.

### crawl_rear_touchdown_forced_release

- Meaning: first crawl run where the RR leg was actually forced open after the
  planned-swing contact persisted too long.
- Why keep it: marks the point where the crawl failure changed from "RR never
  opens" to "RR opens, but the robot still cannot finish rear recontact."

### crawl_rear_touchdown_balanced_default

- Meaning: current best crawl-oriented conservative preset after combining a
  slightly later rear forced-release timeout, longer/deeper rear touchdown
  reacquire, and a modest reduced-support vertical boost.
- Why keep it: current default crawl diagnostic run. It preserves the RR
  forced-release behavior, increases RR current/actual swing activity, and
  reduces mean roll/pitch relative to the earlier forced-release-only run,
  while still failing later on RR hip contact.

### trot_after_crawl_balanced_default

- Meaning: trot verification run after promoting the new crawl rear-touchdown
  settings to the conservative crawl preset.
- Why keep it: confirms that the crawl-side rear-touchdown changes do not break
  the dynamic-gait path; trot still runs the full 20-second window without
  termination.

### crawl_diagnostic_slow_gait_default

- Meaning: current crawl default after removing the heavy crawl-only
  conservative overrides and keeping only a slower diagnostic gait timing.
- Why keep it: this is the cleanest current crawl diagnostic on the active
  branch. It restores a longer `RR_hip` failure mode without the sticky
  touchdown/recovery support that appeared in more aggressive crawl candidates.

### crawl_full_contact_recovery_candidate

- Meaning: optional crawl candidate that adds only a late full-contact recovery
  window on top of the restored slow-gait crawl default.
- Why keep it: longest crawl run found in the current code state, but it does
  so by leaning heavily on front touchdown/recovery support and therefore is
  better treated as an optional protected-support candidate than the default.

### trot_after_crawl_diagnostic_default

- Meaning: trot verification run after promoting the lighter crawl diagnostic
  preset to the default conservative crawl path.
- Why keep it: confirms that cleaning up the crawl preset did not regress the
  dynamic-gait path; trot still completes the full 5-second check without
  termination.

## Current Code To Watch

- `quadruped_pympc/controllers/linear_osqp/linear_baseline_controller.py`
- `quadruped_pympc/interfaces/wb_interface.py`
- `quadruped_pympc/quadruped_pympc_wrapper.py`
- `quadruped_pympc/config.py`
- `simulation/run_linear_osqp.py`
- `simulation/simulation.py`
- `simulation/artifacts.py`
- `simulation/autotune_linear_osqp.py`

## Interpretation

- These runs show the path from front-swing lockup to partial touchdown
  recovery and later-stage contact-transition issues.
- The remaining bottleneck in this older line of work was already trending
  toward touchdown/recontact stability rather than the QP solve alone.
