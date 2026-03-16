# Quadruped Python Baseline

Compact Python baseline for a quadruped LTV-MPC assignment. This project mirrors the high-level flow of the provided MATLAB baseline in a smaller and easier-to-read simulator.

The pipeline includes:
- trot contact scheduling
- reference rollout
- SRB-based prediction
- QP-based contact-force optimization
- closed-loop simulation and plotting

## Scenarios

Two example scenarios are included:

- `straight_trot`: forward trot at `0.5 m/s`
- `turn_pi_over_4`: forward motion with a turn toward `pi/4`

## Project Layout

- `main.py`: end-to-end simulation entry point
- `config.py`: model parameters and scenario setup
- `fsm.py`: contact schedule
- `reference.py`: reference trajectory rollout
- `footholds.py`: nominal foothold geometry
- `model.py`: reduced prediction model
- `qp_builder.py`: stacked QP construction
- `controller_osqp.py`: OSQP solve wrapper
- `plant.py`: SRB propagation
- `plotting.py`: result plotting
- `docs/`: notes and reference material

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

## Outputs

Running a scenario generates figures locally under:

- `outputs/straight_trot/`
- `outputs/turn_pi_over_4/`

Typical plots include velocity tracking, yaw tracking, XY path, and per-leg normal force.
The `outputs/` directory is gitignored and is not included in the repository.

## Notes

This repository is a high-level baseline translation, not a line-by-line reproduction of the original MATLAB stack. A few details are intentionally simplified to keep the implementation compact and easy to inspect.

Source attribution is described in `ATTRIBUTION.md`.
