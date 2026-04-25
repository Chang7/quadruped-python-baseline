from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'core'
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from config import make_config, IDX_G
from fsm import rollout_contact_schedule
from reference import rollout_reference
from footholds import nominal_foot_positions_world
from model import build_prediction_model
from qp_builder import build_qp
from controller_osqp import MPCControllerOSQP
from plant import step_srb
from plotting import plot_logs
import numpy as np


def make_initial_state(cfg) -> np.ndarray:
    x0 = np.zeros(cfg.nx, dtype=float)
    x0[2] = cfg.nominal_height
    x0[IDX_G] = cfg.g
    return x0


def run_simulation(cfg) -> dict:
    controller = MPCControllerOSQP(verbose=False)
    x = make_initial_state(cfg)

    log = {
        't': [],
        'x': [],
        'u': [],
        'contact': [],
        'x_ref0': [],
    }

    n_steps = int(cfg.sim_time / cfg.dt_sim)
    t = 0.0

    for _ in range(n_steps):
        contact_schedule = rollout_contact_schedule(t, cfg)
        x_ref = rollout_reference(t, x, cfg)
        foot_rel_world = nominal_foot_positions_world(x, cfg)

        Ad_list, Bd_list = build_prediction_model(x_ref, foot_rel_world, cfg)
        qp = build_qp(
            x_init=x,
            x_ref=x_ref,
            Ad_list=Ad_list,
            Bd_list=Bd_list,
            contact_schedule=contact_schedule,
            cfg=cfg,
        )

        u0, _ = controller.solve(qp)
        x = step_srb(x, u0, foot_rel_world, cfg, cfg.dt_sim)

        log['t'].append(t)
        log['x'].append(x.copy())
        log['u'].append(u0.copy())
        log['contact'].append(contact_schedule[0].copy())
        log['x_ref0'].append(x_ref[0].copy())

        t += cfg.dt_sim

    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quadruped LTV-MPC Python baseline')
    parser.add_argument('--scenario', default='straight_trot', choices=['straight_trot', 'turn_pi_over_4'])
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs' / 'baseline'))
    args = parser.parse_args()

    cfg = make_config(args.scenario)
    logs = run_simulation(cfg)
    out = Path(args.output_dir) / args.scenario
    saved = plot_logs(logs, cfg, output_dir=str(out))
    print(f'Simulation finished for scenario: {args.scenario}')
    print('Saved figures:')
    for path in saved:
        print(f' - {path}')
