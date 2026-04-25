# References

This folder collects all read-only reference material — code snapshots, the
upstream package, the legacy Python baseline, and porting notes. Nothing here
is part of the active build path.

## upstream_pympc/

Read-only copy of the DLS Lab Quadruped-PyMPC stock repository (Python). This
is the base we forked when starting the linear-OSQP adapter.

- `upstream_pympc_paper.pdf` — accompanying LTV-MPC paper.

## legacy_python_baseline/

Earlier standalone Python MPC baseline and MuJoCo prototype, kept for
historical comparison.

## external_repos/

Local copies of other relevant repositories used for design comparison only:

- `Cheetah-Software` — MIT Cheetah controller stack (C++).
- `legged_control` — alternative legged-robot controller reference.
- `Quadruped-PyMPC-upstream` — older snapshot kept for diff history.
- `matlab_linear_zip` — early Python snapshot of the MATLAB-to-PyMPC migration
  (despite the name there are no `.m` files; it is the first port).
- `claude-token-efficient` — collaboration-tooling reference.

## notes/

Working notes derived from comparison and porting:

- `MATLAB_TO_STOCK_PORT_NOTES.md`

## Usage

- Do not edit active controller code here.
- Use this folder only for comparison, design reference, and report preparation.
