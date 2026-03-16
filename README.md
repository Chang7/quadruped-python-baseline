# Python baseline for the quadruped LTV-MPC assignment

This repository contains a compact Python baseline that mirrors the **high-level** structure of the provided MATLAB quadruped MPC code.

The assignment goal is not to reproduce the entire quadruped stack, but to:
- understand the provided MATLAB baseline,
- make a limited and explicit Python-side modification,
- and explain the resulting behavior clearly in a short report.

For that reason, this repository keeps the same main loop in a smaller simulator:
- contact scheduling,
- reference rollout,
- SRB-based prediction model,
- QP-based contact-force optimization,
- receding-horizon closed-loop simulation.

## Source attribution

This Python baseline was derived from the MATLAB repository **Quad_ConvexMPC** by **Andrew Zheng** and **Sriram S. K. S. Narayanan**, together with the accompanying project report that was provided for the assignment.

The Python version should be understood as a **high-level baseline translation** of the original control loop rather than as a line-by-line reproduction of the full MATLAB stack. The main retained components are:
- contact scheduling,
- reference rollout,
- force-based MPC/QP construction,
- and closed-loop SRB simulation.

The major simplifications are documented in both this README and the written report so that the scope of the translation remains explicit.

## MATLAB-to-Python mapping

The baseline was translated from the following MATLAB flow:
- `Convex_MPC.m` -> main closed-loop driver
- `fcn_FSM` / `fcn_FSM_bound` -> contact schedule
- `fcn_gen_XdUd` -> desired state and nominal force pattern
- `get_QP` / `sim_MPC` -> QP assembly and solve
- `dynamics_SRB` -> plant propagation

## Supported scenarios

### `straight_trot`
- paper-inspired straight-line trot baseline
- desired forward speed: `0.5 m/s`
- desired yaw: `0 rad`

### `turn_pi_over_4`
- paper-inspired turning baseline
- desired forward speed: `0.35 m/s`
- desired yaw target: `pi/4`
- commanded yaw rate during the turn: `0.45 rad/s`

## What is retained from the MATLAB baseline

- SRB-based high-level prediction structure
- trot contact schedule
- force-based MPC/QP formulation
- receding-horizon closed-loop structure
- straight-line and turning baseline-style verification

## What is intentionally simplified

This repository is a **baseline translation**, not a line-by-line reproduction of the MATLAB stack.
Compared with the MATLAB implementation, the Python version keeps several simplifications:

- single-rate simulation (`dt_sim = dt_mpc = 0.02 s`)
- forward-Euler prediction discretization
- fixed 12-dimensional input vector for four legs
- fixed nominal footholds rotated only by yaw
- no swing-foot touchdown update / no Raibert-style foot placement
- fixed normal-force upper bound in the QP

These choices were kept intentionally so the code remains readable and easy to verify.

## Files

- `config.py` -> model and tuning parameters / scenario factory
- `fsm.py` -> trot contact schedule
- `reference.py` -> horizon reference rollout for straight and turning cases
- `footholds.py` -> nominal foothold geometry
- `model.py` -> reduced prediction model and discretization
- `qp_builder.py` -> sparse stacked QP assembly
- `controller_osqp.py` -> OSQP interface
- `plant.py` -> SRB propagation step
- `plotting.py` -> result figures
- `main.py` -> end-to-end simulation driver

## Setup

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --scenario straight_trot
python main.py --scenario turn_pi_over_4
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --scenario straight_trot
python main.py --scenario turn_pi_over_4
```

## Output

Running the scenarios saves figures under:
- `outputs/straight_trot/`
- `outputs/turn_pi_over_4/`

Typical figures include:
- `fig_leg_fz_subplots.png`
- `fig_velocity_tracking.png`
- `fig_yaw_tracking.png`
- `fig_xy_path.png`

## Why the straight-line GRF figure does not match the ideal curve exactly

The per-leg normal-force plots are meant as a **qualitative sanity check** against the ideal `mg/n` load-sharing trend. Exact agreement is not expected in this baseline because the Python version still omits several MATLAB details, including:
- the swing-foot touchdown update logic in `fcn_FSM`,
- the full nominal-force shaping induced by `Ud` in `fcn_gen_XdUd`,
- the original dual-rate simulation structure,
- and the MATLAB zero-order-hold discretization path.

## GitHub upload

This folder is already organized so it can be pushed to GitHub directly.
A minimal workflow is:

```bash
git init
git add .
git commit -m "Add Python baseline for quadruped LTV-MPC"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

For this assignment, using a **private repository** is the safer default.

## Suggested GitHub structure

A minimal private repository layout is:

```text
<repo root>/
├── config.py
├── fsm.py
├── reference.py
├── footholds.py
├── model.py
├── qp_builder.py
├── controller_osqp.py
├── plant.py
├── plotting.py
├── main.py
├── requirements.txt
├── README.md
└── outputs/
```

The MATLAB zip that was provided for the assignment should not be uploaded to a public repository. The safer default is to keep this Python baseline in a private repository and mention in the README that it was derived from a separately provided MATLAB baseline.
