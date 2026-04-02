# Outputs Summary

All generated results should live under this `outputs/` folder.

## Current Structure

- `curated_runs/`
  - milestone runs from the earlier adapter-side debugging history
- `report_progress_explainer/`
  - current email/meeting/report assets
- `stock_stack_runs/`
  - raw stock-stack diagnostic history
- `archive/`
  - older raw runs and superseded output folders kept for traceability

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
