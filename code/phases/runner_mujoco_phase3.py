from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from phases.mujoco_phase3_helpers import (
    FALLBACK_FOOT_LOCAL_OFFSET,
    discover_model_bindings,
    foot_point_world,
    foot_rel_world,
    mujoco_to_x,
    force_to_qfrc,
    actual_contact_state,
    store_home_joint_qpos,
    compute_swing_delta_maps,
    resolve_home_ctrl,
    update_swing_anchors,
    build_ctrl_targets_phase3,
    print_binding_summary,
)


def run_mujoco_phase3(
    cfg,
    model_path: str,
    viewer: bool = True,
    output_dir: str | None = None,
    clearance: float = 0.05,
    step_len: float | None = None,
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
    compute_swing_delta_maps(m, d, bindings)

    next_mpc_time = 0.0
    u_hold = np.zeros(cfg.nu, dtype=float)
    x_ref0_hold = np.zeros(cfg.nx, dtype=float)
    scheduled_contact_hold = np.zeros(4, dtype=bool)
    prev_sched = np.ones(4, dtype=bool)

    if step_len is None:
        step_len = max(0.03, min(0.10, 0.55 * cfg.desired_speed * cfg.swing_time))

    log = {
        "t": [],
        "x": [],
        "u": [],
        "contact": [],
        "contact_actual": [],
        "x_ref0": [],
    }

    def one_step() -> bool:
        nonlocal next_mpc_time, u_hold, x_ref0_hold, scheduled_contact_hold, prev_sched

        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

        if d.time >= next_mpc_time - 1e-12:
            x = mujoco_to_x(d, cfg, bindings.base_body_name)
            feet = foot_rel_world(d, bindings.base_body_name, bindings.leg_bindings, FALLBACK_FOOT_LOCAL_OFFSET)

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

        update_swing_anchors(d, bindings, prev_sched, scheduled_contact_hold)

        if m.nu > 0:
            d.ctrl[:] = build_ctrl_targets_phase3(
                m=m,
                d=d,
                bindings=bindings,
                home_ctrl=home_ctrl,
                scheduled_contact=scheduled_contact_hold,
                cfg=cfg,
                clearance=clearance,
                step_len=step_len,
            )

        d.qfrc_applied[:] = 0.0
        for leg_i, binding in enumerate(bindings.leg_bindings):
            if not bool(scheduled_contact_hold[leg_i]):
                continue
            f_world = u_hold[3 * leg_i : 3 * leg_i + 3]
            point_world = foot_point_world(d, binding, FALLBACK_FOOT_LOCAL_OFFSET)
            d.qfrc_applied[:] += force_to_qfrc(m, d, binding.calf_body_name, point_world, f_world)

        mujoco.mj_step(m, d)
        prev_sched = scheduled_contact_hold.copy()

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

    if len(log["contact_actual"]):
        sched = np.asarray(log["contact"], dtype=bool)
        act = np.asarray(log["contact_actual"], dtype=bool)
        mismatch = (sched != act).mean(axis=0)
        print("Contact mismatch ratio per leg [FL, FR, RL, RR]:", np.round(mismatch, 3))
        print("Mean mismatch ratio:", float(np.mean(mismatch)))

    print(f"Phase-3 params: clearance={clearance:.3f} m, step_len={step_len:.3f} m")
    return log, saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo phase-3 baseline: swing anchor + Jacobian-based foot-space swing arc")
    parser.add_argument("--scenario", default="straight_trot", choices=["straight_trot", "turn_pi_over_4"])
    parser.add_argument(
        "--model",
        default="./mujoco_menagerie/unitree_a1/scene.xml",
        help="Path to a MuJoCo MJCF scene (Menagerie scene.xml recommended)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer")
    parser.add_argument("--output-dir", default=None, help="Directory for saved plots")
    parser.add_argument("--clearance", type=float, default=0.05, help="Swing foot clearance [m]")
    parser.add_argument("--step-len", type=float, default=None, help="Swing foot forward placement [m]")
    args = parser.parse_args()

    try:
        cfg = make_config(args.scenario)
        outdir = args.output_dir or f"local_outputs/outputs_mujoco_phase3/{args.scenario}"
        _, saved = run_mujoco_phase3(
            cfg,
            model_path=args.model,
            viewer=not args.headless,
            output_dir=outdir,
            clearance=args.clearance,
            step_len=args.step_len,
        )
        print(f"MuJoCo phase-3 run finished for scenario: {args.scenario}")
        print(f"Model: {Path(args.model).resolve()}")
        if saved:
            print("Saved figures:")
            for p in saved:
                print(f" - {p}")
    except Exception:
        print("MuJoCo phase-3 run failed. Full traceback below:")
        traceback.print_exc()
