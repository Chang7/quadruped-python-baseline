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
