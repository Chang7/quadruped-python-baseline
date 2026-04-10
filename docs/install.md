# Installation

This project reuses the stock Quadruped-PyMPC integration stack, so the
dependency story is still mostly the upstream one.

## Dependencies

- MuJoCo
- OSQP
- NumPy / SciPy / Matplotlib
- CasADi / acados for the upstream gradient path
- JAX for the upstream sampling path

## Editable install

```bash
pip install -e .
```

## Upstream-style environment setup

If you need the broader upstream stack, see:

- `installation/mamba/`
- `installation/docker/`
- `ros2/`

The original upstream installation flow included:

1. creating the matching conda environment,
2. initializing submodules,
3. building `acados`,
4. exporting the required `ACADOS_SOURCE_DIR` / library path,
5. installing the package in editable mode.

## Quick local check

```bash
python -m simulation.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12
```

## Notes

- The active simulation environment used in this workspace is the MuJoCo path
  under `simulation/`.
- Generated artifacts should always go under `outputs/`.
