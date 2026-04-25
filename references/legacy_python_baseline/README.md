# Legacy Python Baseline

This folder preserves the earlier standalone Python baseline and MuJoCo
prototype that originally lived in the separate
`quadruped-python-baseline` repository.

## Why It Is Kept Here

- to preserve the original high-level MATLAB-to-Python baseline work
- to keep the old MuJoCo prototype readable
- to avoid mixing the old standalone pipeline with the current
  `Quadruped-PyMPC`-style adapter code in the repository root

## Scope

The code here reflects the earlier phase of the project:

- high-level Python MPC baseline core
- standalone MuJoCo helpers and quasi-static prototype runners
- notes and attribution from that stage

This is not the active controller stack anymore. The current working tree is in
the repository root (`quadruped_pympc/`, `simulation/`, `tools/`).
