# Code Folder Index

This folder is organized so that shared logic is grouped by purpose and runnable entrypoints live next to the modules they belong to.

## Entry Points

- `phases/runner_mujoco_phase1.py` to `phases/runner_mujoco_phase14_forward_supported_visual.py`
- `phases/runner_mujoco_smoke.py`
- `experiments/run_mujoco_clean.py`
- `experiments/run_mujoco_adaptive_clean.py`
- `experiments/run_mujoco_quasistatic_confirmed.py`
- `experiments/run_mujoco_quasistatic_fixed.py`
- `experiments/run_mujoco_quasistatic_heightfix.py`
- `baseline/main.py`

## Folders

- `baseline/`: shared baseline MPC code such as config, model, reference, QP build, plotting
- `phases/`: MuJoCo phase helper modules and visual utilities
- `experiments/`: quasistatic / clean / adaptive helper modules
- `scripts/`: setup and convenience shell / PowerShell scripts
- `tools/`: one-off inspection utilities
- `docs/`: phase notes and Korean walkthrough documents
- `requirements/`: pip requirement files

## Notes

- Generated result folders such as `local_outputs/outputs_mujoco_phase*` are runtime artifacts.
- `mujoco_menagerie/` and `.venv/` are environment assets, not core source files.
- Entry scripts inside `baseline/`, `phases/`, `experiments/`, and `tools/` include a direct-run path fallback.
