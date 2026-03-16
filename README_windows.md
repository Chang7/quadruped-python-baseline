# Python baseline for the quadruped LTV-MPC assignment

This folder contains a compact Python baseline that mirrors the **high-level** structure of the provided MATLAB quadruped MPC code:

- contact scheduling (FSM)
- reference rollout
- SRB-based prediction model
- QP-based contact-force optimization
- receding-horizon closed-loop simulation

## Important scope notes

This is a **baseline verification simulator**, not a full reproduction of the MATLAB stack.
Compared with the MATLAB code, the Python version uses several simplifying choices:

- single-rate simulation (`dt_sim = dt_mpc = 0.02 s`)
- forward-Euler discretization
- fixed 12-dimensional input vector for four legs
- fixed nominal footholds rotated only by yaw
- no swing-foot touchdown update / no Raibert-style foot placement
- fixed normal-force upper bound in the QP

These choices were kept intentionally to make the framework readable and easy to verify.

## Setup on Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Output

Running `main.py` saves figures in the local `outputs` folder:

- `fig_velocity_tracking_refined.png`
- `fig_yaw_tracking_refined.png`
- `fig_leg_fz_subplots.png`

The `outputs/` directory is gitignored and is not included in the repository.
