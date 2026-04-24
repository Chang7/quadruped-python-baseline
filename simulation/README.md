# Compatibility Wrappers

This package keeps older MuJoCo commands such as
`python -m simulation.run_linear_osqp ...` working after the active MuJoCo
backend moved to `mujoco_sim/`.

New code and documentation should import from `mujoco_sim`.

