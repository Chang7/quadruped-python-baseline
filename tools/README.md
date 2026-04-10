# Tools Layout

Helper scripts are grouped by purpose.

## Folders

- `analysis/`
  - one-off inspection helpers for existing outputs
- `launchers/`
  - shell launch scripts for longer runs
  - also includes fixed benchmark-suite runners such as
    `run_trot_benchmark_suite.py`
- `report_assets/`
  - scripts that generate figures, GIFs, MP4s, and docx report assets
  - also includes fixed benchmark dashboards such as
    `make_trot_benchmark_dashboard.py`
- `stock_helpers/`
  - stock-stack-specific helper scripts

Older temporary sweep scripts were moved out of the active tools tree and are
now kept under:

- `../archive/legacy_tools/tools_legacy_sweeps/`

## Rule

- Put new helper scripts into one of these subfolders instead of leaving them
  loose under `tools/`.
