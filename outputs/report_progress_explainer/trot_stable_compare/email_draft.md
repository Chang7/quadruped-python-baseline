Subject: Trot Update on linear_osqp Integration

Dear Sriram,

I wanted to share a short follow-up specifically on the trot scenario.

Using the same MuJoCo trot setting (`aliengo`, `flat`, fixed forward command),
the current `linear_osqp` branch now completes the short trot run without early
collapse. For a compact same-scenario comparison, I attached a stock
sampling-controller GIF and a `linear_osqp` GIF together with a small summary
table.

In the attached 3-second trot comparison:

- the stock sampling controller completes the run without termination
- the current `linear_osqp` controller also completes the run without
  termination
- compared with stock, `linear_osqp` still shows a larger pitch bias and lower
  forward-speed tracking, but it is no longer failing at the beginning of trot

I also kept a separate longer verification run for `linear_osqp`, where the
controller completes a 20-second trot simulation without termination:

- `outputs/curated_runs/trot_after_crawl_balanced_default/episode_000/summary.json`

So at this point, I think the main status update is that the earlier "trot
collapses immediately" issue has been resolved, and the remaining gap is now
more about tracking quality than basic trot viability.

Best regards,
Chang Hee Han
