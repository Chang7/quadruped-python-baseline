# MuJoCo Direct Backend

This package contains the single-process MuJoCo execution path used for local
controller development, artifact logging, and benchmark reproduction.

Use this entry point for new local runs:

```bash
python -m mujoco_sim.run_linear_osqp --controller linear_osqp --gait trot --seconds 4 --speed 0.12
```

The sibling `simulation/` package is kept only as a compatibility wrapper for
older commands and imports.

