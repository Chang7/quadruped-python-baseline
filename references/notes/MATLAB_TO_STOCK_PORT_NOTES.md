# MATLAB To Stock Quadruped-PyMPC Notes

## What Was Verified

- The stock repo at `references/upstream_pympc/` can run inside the current WSL venv.
- Default `nominal` backend does not run yet because `acados_template` is not installed.
- After installing CPU `jax`, the stock `sampling` backend runs headless.
- Repro script:
  - `tmp_refs/stock_repo_smoke.py`

## Repro Command

From repo root:

```bash
source venv/bin/activate
python tmp_refs/stock_repo_smoke.py --seconds 2.0 --recording-path outputs/stock_repo_sampling_smoke
```

## What To Keep From Stock

- `simulation/simulation.py`
  - MuJoCo environment creation
  - state / jacobian / contact collection
  - torque application loop
- `quadruped_pympc/quadruped_pympc_wrapper.py`
  - `update_state_and_reference -> compute_control -> compute_stance_and_swing_torque`
- `quadruped_pympc/interfaces/wb_interface.py`
  - gait scheduling
  - foothold generation
  - swing / stance realization
  - low-level torque handoff

## What To Replace With MATLAB Semantics

Primary swap point:

- `quadruped_pympc/interfaces/srbd_controller_interface.py`

Why:

- This is the clean backend seam.
- It already switches between `nominal`, `input_rates`, `sampling`, etc.
- A new backend can be added here without rewriting MuJoCo plumbing.

## Why This Is Practical In The Current Repo

- The current adapter already added a `linear_osqp` backend at the same seam:
  - `quadruped_pympc/interfaces/srbd_controller_interface.py`
  - `quadruped_pympc/controllers/linear_osqp/linear_baseline_controller.py`
- That means the migration target is not hypothetical.
- The stock stack and the current adapter already agree on the main contract:
  - `state_current`
  - `ref_state`
  - `contact_sequence`
  - output GRFs / footholds / predicted state
- So the next realistic step is not a full rewrite.
- It is either:
  - port the current `linear_osqp` backend into the verified stock stack, or
  - reimplement the MATLAB convex backend directly at that same seam.

## MATLAB Pieces To Port First

From `Quad_ConvexMPC-main.zip`:

- `Convex_MPC.m`
- `FSM/fcn_FSM.m`
- `refGen/fcn_gen_XdUd.m`
- `getMPC/get_A.m`
- `getMPC/get_B.m`
- `getMPC/get_QP.m`
- `getMPC/sim_MPC.m`
- `robot/dynamics_SRB.m`

## Recommended Port Strategy

1. Keep stock MuJoCo loop unchanged.
2. Keep stock `WBInterface` and `compute_stance_and_swing_torque` unchanged at first.
3. Add a new SRBD backend, for example `type = "convex_osqp"` or `type = "matlab_convex"`.
4. Port only:
   - gait / contact schedule semantics
   - reference rollout
   - SRB linearization
   - QP cost / constraints
   - GRF output shape
5. Return the same data contract expected by the wrapper:
   - GRFs
   - footholds
   - optional predicted state

## Important Lesson

- The previous custom adapter work was not wasted.
- It clarified the high-level convex MPC semantics and the failure modes.
- The stock repo removes the need to rebuild the entire MuJoCo realization layer from scratch.
- The fastest path is:
  - stock demo first
  - high-level backend replacement second
  - touchdown / support refinements only after that
