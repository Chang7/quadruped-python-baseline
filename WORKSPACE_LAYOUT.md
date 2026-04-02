# Workspace Layout

This repository now uses a simple split between active code, reference code, and generated outputs.

## Active Code

These are the folders to modify for ongoing work:

- `quadruped_pympc/`
- `simulation/`
- `tools/`
- `pyproject.toml`
- `LINEAR_OSQP_README.md`

## Outputs

All new generated artifacts should go under:

- `outputs/`

Important subfolders:

- `outputs/curated_runs/`
- `outputs/report_progress_explainer/`
- `outputs/stock_stack_runs/`
- `outputs/archive/`

## Stock Reference Repo

The original stock code is kept only as a reference:

- `1.Quadruped-PyMPC-main/Quadruped-PyMPC-main/`

Do not use this folder as the main working tree.
Do not place new reports or experimental outputs there unless there is a very specific reason.

## External References

Reference repositories and imported notes are kept under:

- `references/external_repos/`
- `references/notes/`

Important entries:

- `references/external_repos/Cheetah-Software/`
- `references/external_repos/legged_control/`
- `references/external_repos/Quadruped-PyMPC-upstream/`
- `references/external_repos/matlab_linear_zip/`
- `references/notes/MATLAB_TO_STOCK_PORT_NOTES.md`

## Utility Scripts

Helper scripts are grouped by purpose:

- `tools/analysis/`
- `tools/launchers/`
- `tools/stock_helpers/`
- `tools/report_assets/`
- `tools/legacy_sweeps/`

## Archive

Non-active backups are kept under:

- `archive/backups/`

Temporary extracted discussion files are kept under:

- `references/notes/raw_extracts/`

## Working Rule

Use this rule going forward:

1. Read stock code from `1.Quadruped-PyMPC-main/Quadruped-PyMPC-main/` only as a reference.
2. Edit active code only in the repository root working tree.
3. Save new outputs only under `outputs/`.
4. Move superseded outputs to `outputs/archive/` instead of leaving them loose.
5. Keep external code copies and notes only under `references/`.
