# Legacy Consolidated Notes

This note preserves the older consolidated bundle summary that was kept at the
repository root during earlier integration work.

## Summary

- import-safe `simulation/run_linear_osqp.py`
- artifact logging (`npz/csv/json/png`)
- contact-object-safe artifact recording
- ffmpeg fallback for top-down motion export
- conservative `linear_osqp` controller conditioning
- conservative preset bug fix

## Historical first run

```bash
pip install -e .
python -m simulation.run_linear_osqp --gait crawl --preset conservative --seconds 10 --artifact-dir outputs/linear_osqp_stable --recording-path outputs/linear_osqp_stable/h5 --no-mp4
```

## Historical note

This older note belonged to a much earlier stabilization stage and is kept only
for reference. The current promoted runs and report assets live under:

- `outputs/curated_runs/`
- `outputs/report_progress_explainer/`
