# Report Progress Explainer

This folder contains assets prepared for email updates, meeting slides, and
short report summaries.

## Current Preferred Assets

- `crawl_same_scenario/`
  - Main email-ready diagnostic comparison.
  - Use this when explaining rear touchdown/recontact and load transfer.
- `trot_stable_compare/`
  - Current email-ready trot proof package.
  - Use this when showing that the present `linear_osqp` branch completes the
    short trot scenario without early collapse.
- `trot_same_scenario/`
  - Older same-scenario comparison from the earlier failing trot stage.
  - Keep for traceability only; it is no longer the preferred trot asset set.
- `quadruped_mpc_progress_update.docx`
  - Convenience document version of the current progress summary.

## Archive

- `archive/`
  - Older one-off explainers and combined figures kept for traceability.
  - These are still useful, but they are no longer the preferred top-level
    assets for communication.

## Recommended Use

- For email: start with `crawl_same_scenario/`.
- For strict controller-to-controller comparison: also inspect
  `trot_same_scenario/`.
