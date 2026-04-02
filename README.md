# Quadruped PyMPC Linear OSQP Adapter

This repository is the active working tree for integrating a custom
`linear_osqp` high-level SRB force MPC into a Quadruped-PyMPC-style MuJoCo
locomotion stack.

The main goal of this workspace is not to reimplement the entire quadruped
stack from scratch, but to:

1. keep the stock-style MuJoCo integration structure as much as possible,
2. replace only the high-level controller with a compact linear OSQP backend,
3. study where touchdown/recontact and load-transfer failures still appear.

## Current Focus

The main remaining bottleneck is not the QP solve alone, but how the planned
forces are carried through contact transition and load transfer, especially
during rear touchdown/recontact.

## Repository Layout

### Active Code

- `quadruped_pympc/`
- `simulation/`
- `tools/`

### Outputs

- `outputs/curated_runs/`
  - milestone runs from the earlier adapter-side debugging history
- `outputs/report_progress_explainer/`
  - current figures, GIFs, and report/email assets
- `outputs/stock_stack_runs/`
  - raw stock-stack diagnostic history
- `outputs/archive/`
  - superseded or raw historical outputs kept for traceability

### References

- `references/notes/`
  - notes kept with this workspace
- `references/external_repos/`
  - local copies of external reference repositories

### Read-Only Stock Reference

- `1.Quadruped-PyMPC-main/`

This folder is kept only as a local upstream reference and should not be used
as the active working tree.

## Quick Start

Install in editable mode:

```bash
pip install -e .
```

Run the custom backend in MuJoCo:

```bash
python -m simulation.run_linear_osqp --gait crawl --controller linear_osqp --seconds 10
```

## Recommended Reading Order

1. `WORKSPACE_LAYOUT.md`
2. `LINEAR_OSQP_README.md`
3. `outputs/README.md`
4. `outputs/report_progress_explainer/README.md`

## Notes

- New outputs should be saved under `outputs/`, not inside
  `1.Quadruped-PyMPC-main/`.
- Large raw runs, backups, and copied external repositories are intentionally
  excluded from the cleaned Git history.
