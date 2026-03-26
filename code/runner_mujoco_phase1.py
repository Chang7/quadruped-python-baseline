from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

from baseline.config import make_config
from baseline.fsm import rollout_contact_schedule
from baseline.reference import rollout_reference
from baseline.model import build_prediction_model
from baseline.qp_builder import build_qp
from baseline.controller_osqp import MPCControllerOSQP
from baseline.plotting import plot_logs
from phases.mujoco_phase1_helpers import (
    discover_model_bindings,
    foot_point_world,
    foot_rel_world,
    mujoco_to_x,
    force_to_qfrc,
    actual_contact_state,
    store_home_joint_qpos,
    store_swing_lift_dirs,
    build_ctrl_targets_with_cfg,
    print_binding_summary,
)

# Keep the same approximate contact point used in the smoke test.
FOOT_LOCAL_OFFSET = np.array([0.0, 0.0, -0.20], dtype=float)


def resolve_home_ctrl(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    if m.nu == 0:
        return np.zeros(0, dtype=float)

    # Best effort: if the model uses position-style actuators and the reset state contains one target per actuator,
    # use those; otherwise preserve the current ctrl vector.
    if d.qpos.shape[0] >= 7 + m.nu:
        return d.qpos[7:7 + m.nu].copy()
    return d.ctrl.copy()


def run_mujoco_phase1(
    cfg,
    model_path: str,
    viewer: bool = True,
    output_dir: str | None = None,
    swing_amp: float = 0.22,
) -> tuple[dict, list[str]]:
    controller = MPCControllerOSQP(verbose=False)

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)

    bindings = discover_model_bindings(m)
    print(print_binding_summary(bindings))

    home_ctrl = resolve_home_ctrl(m, d)

    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    store_home_joint_qpos(d, bindings)
    store_swing_lift_dirs(m, d, bindings, FOOT_LOCAL_OFFSET)

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)
    scheduled_contact_hold = np.zeros(4, dtype=bool)

    log = {
        "t": [],
        "x": [],
        "u": [],
        "contact": [],
        "contact_actual": [],
        "x_ref0": [],
    }

    def one_step() -> bool:
        nonlocal next_mpc_time, u_hold, x_ref0_hold, scheduled_contact_hold

        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

        if d.time >= next_mpc_time - 1e-12:
            x = mujoco_to_x(d, cfg, bindings.base_body_name)
            feet = foot_rel_world(d, bindings.base_body_name, bindings.leg_bindings, FOOT_LOCAL_OFFSET)

            contact_schedule = rollout_contact_schedule(float(d.time), cfg)
            x_ref = rollout_reference(float(d.time), x, cfg)
            Ad_list, Bd_list = build_prediction_model(x_ref, feet, cfg)

            qp = build_qp(
                x_init=x,
                x_ref=x_ref,
                Ad_list=Ad_list,
                Bd_list=Bd_list,
                contact_schedule=contact_schedule,
                cfg=cfg,
            )
            u_hold, _ = controller.solve(qp)
            x_ref0_hold = x_ref[0].copy()
            scheduled_contact_hold = contact_schedule[0].copy()
            next_mpc_time += cfg.dt_mpc

        if m.nu > 0:
            d.ctrl[:] = build_ctrl_targets_with_cfg(
                d=d,
                bindings=bindings,
                home_ctrl=home_ctrl,
                scheduled_contact=scheduled_contact_hold,
                swing_amp=swing_amp,
                cfg=cfg,
            )

        d.qfrc_applied[:] = 0.0
        for leg_i, binding in enumerate(bindings.leg_bindings):
            if not bool(scheduled_contact_hold[leg_i]):
                continue
            f_world = u_hold[3 * leg_i: 3 * leg_i + 3]
            point_world = foot_point_world(d, binding.calf_body_name, FOOT_LOCAL_OFFSET)
            d.qfrc_applied[:] += force_to_qfrc(m, d, binding.calf_body_name, point_world, f_world)

        mujoco.mj_step(m, d)

        x_now = mujoco_to_x(d, cfg, bindings.base_body_name)
        contact_actual = actual_contact_state(m, d, bindings)

        log["t"].append(float(d.time))
        log["x"].append(x_now.copy())
        log["u"].append(u_hold.copy())
        log["contact"].append(scheduled_contact_hold.copy())
        log["contact_actual"].append(contact_actual.copy())
        log["x_ref0"].append(x_ref0_hold.copy())

        return d.time < cfg.sim_time

    if viewer:
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running() and d.time < cfg.sim_time:
                keep_going = one_step()
                v.sync()
                if not keep_going:
                    break
    else:
        while d.time < cfg.sim_time:
            if not one_step():
                break

    saved = []
    if output_dir is not None:
        saved = plot_logs(log, cfg, output_dir=output_dir)

    # Simple console diagnostics.
    if len(log["contact_actual"]):
        sched = np.asarray(log["contact"], dtype=bool)
        act = np.asarray(log["contact_actual"], dtype=bool)
        mismatch = (sched != act).mean(axis=0)
        print("Contact mismatch ratio per leg [FL, FR, RL, RR]:", np.round(mismatch, 3))
        print("Mean mismatch ratio:", float(np.mean(mismatch)))

    return log, saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo phase-1 baseline: contact diagnostics + simple swing-lift control")
    parser.add_argument("--scenario", default="straight_trot", choices=["straight_trot", "turn_pi_over_4"])
    parser.add_argument(
        "--model",
        default="./mujoco_menagerie/unitree_a1/scene.xml",
        help="Path to a MuJoCo MJCF scene (Menagerie scene.xml recommended)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer")
    parser.add_argument("--output-dir", default=None, help="Directory for saved plots")
    parser.add_argument("--swing-amp", type=float, default=0.22, help="Joint-space swing-lift target amplitude")
    args = parser.parse_args()

    try:
        cfg = make_config(args.scenario)
        outdir = args.output_dir or f"outputs_mujoco_phase1/{args.scenario}"
        _, saved = run_mujoco_phase1(
            cfg,
            model_path=args.model,
            viewer=not args.headless,
            output_dir=outdir,
            swing_amp=args.swing_amp,
        )
        print(f"MuJoCo phase-1 run finished for scenario: {args.scenario}")
        print(f"Model: {Path(args.model).resolve()}")
        if saved:
            print("Saved figures:")
            for p in saved:
                print(f" - {p}")
    except Exception:
        print("MuJoCo phase-1 run failed. Full traceback below:")
        traceback.print_exc()
