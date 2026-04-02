# Final consolidated bundle

This bundle merges the working pieces that were previously split across multiple zips:

- import-safe `simulation/run_linear_osqp.py`
- artifact logging (`npz/csv/json/png`)
- contact-object-safe artifact recording (`MjContact` no longer crashes logging)
- ffmpeg fallback for top-down motion export
- conservative linear-OSQP controller conditioning (smoothing, slew limits, stance ramp, fy scaling)
- **preset bug fix**: `--preset conservative` now really applies conservative Q/R defaults unless explicit `--q-*` or `--r-u` overrides are passed

## Recommended first run

```bash
pip install -e .
python -m simulation.run_linear_osqp   --gait crawl   --preset conservative   --seconds 10   --artifact-dir outputs/linear_osqp_stable   --recording-path outputs/linear_osqp_stable/h5   --no-mp4
```

If ffmpeg is installed, drop `--no-mp4`.

## Where outputs go

- raw H5: `outputs/.../h5/.../*.h5`
- plots/json/csv/mp4(gif): `outputs/.../episode_000/`

## Important note

This bundle is for **stack validation inside PyMPC/AlienGo first**. It does **not** claim stable locomotion on your original MuJoCo/A1 stack yet.
