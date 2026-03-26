# Code Folder Index

This folder is organized so that runner entrypoints stay easy to spot at the root, while shared logic is grouped by purpose.

## Root Entry Points

- `runner_mujoco_phase1.py` to `runner_mujoco_phase14_forward_supported_visual.py`
- `runner_mujoco_smoke.py`
- `run_mujoco_clean.py`
- `run_mujoco_adaptive_clean.py`
- `run_mujoco_quasistatic_confirmed.py`
- `run_mujoco_quasistatic_fixed.py`
- `run_mujoco_quasistatic_heightfix.py`
- `main.py`

## Folders

- `baseline/`: shared baseline MPC code such as config, model, reference, QP build, plotting
- `phases/`: MuJoCo phase helper modules and visual utilities
- `experiments/`: quasistatic / clean / adaptive helper modules
- `docs/`: phase notes and Korean walkthrough documents
- `requirements/`: pip requirement files

## Notes

- Generated result folders such as `outputs_mujoco_phase*` are runtime artifacts.
- `mujoco_menagerie/` and `.venv/` are environment assets, not core source files.
- If you run a root runner script directly, the new package layout is already reflected in imports.
