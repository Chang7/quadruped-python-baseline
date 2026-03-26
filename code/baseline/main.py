import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import traceback
import numpy as np

from baseline.config import make_config, IDX_G
from baseline.fsm import rollout_contact_schedule
from baseline.reference import rollout_reference
from baseline.footholds import nominal_foot_positions_world
from baseline.model import build_prediction_model
from baseline.qp_builder import build_qp
from baseline.controller_osqp import MPCControllerOSQP
from baseline.plant import step_srb
from baseline.plotting import plot_logs


def make_initial_state(cfg) -> np.ndarray:
    x0 = np.zeros(cfg.nx, dtype=float)
    x0[2] = cfg.nominal_height
    x0[IDX_G] = cfg.g
    return x0


def run_simulation(cfg) -> dict:
    controller = MPCControllerOSQP(verbose=False)
    x = make_initial_state(cfg)

    log = {
        "t": [],
        "x": [],
        "u": [],
        "contact": [],
        "x_ref0": [],
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

        log["t"].append(t)
        log["x"].append(x.copy())
        log["u"].append(u0.copy())
        log["contact"].append(contact_schedule[0].copy())
        log["x_ref0"].append(x_ref[0].copy())

        t += cfg.dt_sim

    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruped LTV-MPC Python baseline")
    parser.add_argument(
        "--scenario",
        default="straight_trot",
        choices=["straight_trot", "turn_pi_over_4"],
        help="Scenario to simulate",
    )
    args = parser.parse_args()

    try:
        cfg = make_config(args.scenario)
        logs = run_simulation(cfg)
        saved = plot_logs(logs, cfg, output_dir=f"local_outputs/outputs/{args.scenario}")
        print(f"Simulation finished for scenario: {args.scenario}")
        print("Saved figures:")
        for path in saved:
            print(f" - {path}")
    except Exception:
        print("Simulation failed. Full traceback below:")
        traceback.print_exc()
